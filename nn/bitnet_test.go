package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// Paper CONFIRMED (arXiv:2402.17764 absmean quantizer): γ = mean(|W|),
// W̃ = clamp(round(W/γ), −1, +1), effective weight γ·W̃ — so the effective weight
// takes at most the 3 distinct values {−γ, 0, +γ}, and a known W maps exactly.
func TestBitLinearTernaryStructural(t *testing.T) {
	// mean|W| = (0.4+0.2+1.0+1.2+0.1+0.7)/6 = 0.6
	w := tensor.FromFloat64(tensor.Shape{3, 2}, []float64{0.4, -0.2, 1.0, -1.2, 0.1, 0.7})
	weff, gamma, err := nn.BitQuantizeWeight(backend.NewContext(), w)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(gamma-0.6) > 1e-12 {
		t.Fatalf("γ = %g, want mean(|W|) = 0.6", gamma)
	}
	// round(W/0.6) clamped: 0.4/0.6→1, −0.2/0.6→0, 1.0/0.6→2→clamp 1, −1.2/0.6→−2→clamp −1,
	// 0.1/0.6→0, 0.7/0.6→1
	want := []float64{0.6, 0, 0.6, -0.6, 0, 0.6}
	for i, wv := range want {
		c := tensor.Unravel(i, weff.Shape())
		if math.Abs(weff.AtF64(c...)-wv) > 1e-12 {
			t.Errorf("Wq[%v] = %g, want %g", c, weff.AtF64(c...), wv)
		}
	}
	// arbitrary weights land on at most 3 distinct values, all in {−γ, 0, +γ}
	l := nn.NewBitLinear(tensor.F64, 6, 5, 7)
	weff, gamma, err = nn.BitQuantizeWeight(backend.NewContext(), l.W)
	if err != nil {
		t.Fatal(err)
	}
	levels := map[float64]bool{}
	for i := range weff.Numel() {
		v := weff.AtF64(tensor.Unravel(i, weff.Shape())...)
		levels[v] = true
		if math.Abs(v) > 1e-12 && math.Abs(math.Abs(v)-gamma) > 1e-12 {
			t.Errorf("effective weight %g not in {−γ, 0, +γ} with γ=%g", v, gamma)
		}
	}
	if len(levels) > 3 {
		t.Errorf("effective weight has %d distinct values, want at most 3", len(levels))
	}
}

// STE gradient (official BitNet code: w + (quant(w) − w).detach()): the analytic
// tape gradient on the FULL-PRECISION latent W equals the gradient the matmul
// would give the effective weight, passed through as identity — for loss = Σy,
// ∂L/∂W[k,o] = Σ_b xq[b,k] where xq is the quantized, normalized input. Finite
// differences are INVALID through round, so (as in the LSQ tests) we verify the
// tape against the defined straight-through closed form.
func TestBitLinearSTEGradient(t *testing.T) {
	l := nn.NewBitLinear(tensor.F64, 3, 2, 11)
	x := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{0.8, -1.3, 0.2, -0.4, 2.1, 1.7})

	// forward-only replica of the input pipeline to get xq
	plain := backend.NewContext()
	h, err := l.Norm.Forward(plain, x)
	if err != nil {
		t.Fatal(err)
	}
	xq, err := nn.BitQuantizeAct8(plain, h)
	if err != nil {
		t.Fatal(err)
	}

	tape := autograd.NewTape()
	y, err := l.Forward(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{y}, backend.ReduceAttrs{Axes: []int{0, 1}})
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}

	gw := tape.Grad(l.W)
	if gw == nil {
		t.Fatal("no gradient reached the full-precision latent W")
	}
	nonzero := false
	for k := range 3 {
		want := xq.AtF64(0, k) + xq.AtF64(1, k) // Σ_b xq[b,k], same for every out column
		for o := range 2 {
			if got := gw.AtF64(k, o); math.Abs(got-want) > 1e-9 {
				t.Errorf("∂L/∂W[%d,%d] = %g, want straight-through %g", k, o, got, want)
			} else if got != 0 {
				nonzero = true
			}
		}
	}
	if !nonzero {
		t.Error("gradient on W is identically zero")
	}
	// bias: ∂L/∂B[o] = batch; norm gain must also receive a (nonzero) gradient
	gb := tape.Grad(l.B)
	if gb == nil {
		t.Fatal("no gradient reached the bias")
	}
	for o := range 2 {
		if math.Abs(gb.AtF64(o)-2) > 1e-9 {
			t.Errorf("∂L/∂B[%d] = %g, want 2 (batch size)", o, gb.AtF64(o))
		}
	}
	gg := tape.Grad(l.Norm.Gamma)
	if gg == nil {
		t.Fatal("no gradient reached the norm gain")
	}
	var gsum float64
	for i := range 3 {
		gsum += math.Abs(gg.AtF64(i))
	}
	if gsum == 0 {
		t.Error("gradient on the norm gain is identically zero")
	}
}

// Numeric gradcheck on the SMOOTH configuration (both quantizers off, norm on):
// with the rounds removed the layer is differentiable everywhere, so central
// finite differences must match the tape gradient — pinning the tape wiring the
// STE path reuses.
func TestBitLinearGradcheckFullPrecision(t *testing.T) {
	opts := []nn.BitLinearOption{nn.WithBitLinearWeightQuant(false), nn.WithBitLinear8BitAct(false)}
	l := nn.NewBitLinear(tensor.F64, 3, 2, 13, opts...)
	x := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{0.8, -1.3, 0.2, -0.4, 2.1, 1.7})

	lossAt := func() float64 {
		y, err := l.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		var s float64
		for i := range y.Numel() {
			s += y.AtF64(tensor.Unravel(i, y.Shape())...)
		}
		return s
	}

	tape := autograd.NewTape()
	y, err := l.Forward(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{y}, backend.ReduceAttrs{Axes: []int{0, 1}})
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	gw := tape.Grad(l.W)
	if gw == nil {
		t.Fatal("no gradient on W")
	}
	const h = 1e-6
	for i := range l.W.Numel() {
		c := tensor.Unravel(i, l.W.Shape())
		orig := l.W.AtF64(c...)
		l.W.SetF64(orig+h, c...)
		up := lossAt()
		l.W.SetF64(orig-h, c...)
		down := lossAt()
		l.W.SetF64(orig, c...)
		numeric := (up - down) / (2 * h)
		if math.Abs(gw.AtF64(c...)-numeric) > 1e-5 {
			t.Errorf("∂L/∂W[%v]: tape %g vs numeric %g", c, gw.AtF64(c...), numeric)
		}
	}
}

// With weight-quant, act-quant AND the norm disabled, BitLinear must reduce to a
// plain nn.Linear on the same (same-seed) weights — pins the wiring.
func TestBitLinearCollapsesToLinear(t *testing.T) {
	const seed = 99
	bl := nn.NewBitLinear(tensor.F64, 4, 3, seed,
		nn.WithBitLinearWeightQuant(false), nn.WithBitLinear8BitAct(false), nn.WithBitLinearNorm(false))
	ref := nn.NewLinear(tensor.F64, 4, 3, seed)
	x := tensor.FromFloat64(tensor.Shape{2, 4}, []float64{1, -0.5, 0.25, 2, -1, 0.75, 0.5, -0.25})
	got, err := bl.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	want, err := ref.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	for i := range got.Numel() {
		c := tensor.Unravel(i, got.Shape())
		if d := math.Abs(got.AtF64(c...) - want.AtF64(c...)); d > 1e-10 {
			t.Errorf("y[%v]: BitLinear %g vs Linear %g (Δ=%g)", c, got.AtF64(c...), want.AtF64(c...), d)
		}
	}
	if len(bl.Params()) != len(ref.Params()) {
		t.Errorf("full-precision BitLinear has %d params, Linear has %d", len(bl.Params()), len(ref.Params()))
	}
}

// Mechanism demo at toy scale: a 2-layer BitNet MLP (ternary weights + 8-bit
// activations end to end) trained with Adam on a fixed synthetic regression task
// at least halves its loss — QAT through the STE converges even though every
// forward matmul only ever sees {−γ, 0, +γ} weights.
func TestBitLinearTrains(t *testing.T) {
	l1 := nn.NewBitLinear(tensor.F64, 4, 16, 1)
	l2 := nn.NewBitLinear(tensor.F64, 16, 2, 2)

	// deterministic inputs and targets: t0 = x0 − x1, t1 = 0.5·(x2 + x3)
	const n = 8
	x := tensor.New(tensor.F64, tensor.Shape{n, 4})
	target := tensor.New(tensor.F64, tensor.Shape{n, 2})
	for i := range n {
		for j := range 4 {
			x.SetF64(math.Sin(float64(3*i+2*j+1)), i, j)
		}
		target.SetF64(x.AtF64(i, 0)-x.AtF64(i, 1), i, 0)
		target.SetF64(0.5*(x.AtF64(i, 2)+x.AtF64(i, 3)), i, 1)
	}

	params := append(l1.Params(), l2.Params()...)
	opt := nn.NewAdam(params, 0.02)
	var first, last float64
	for step := range 300 {
		tape := autograd.NewTape()
		ctx := tape.Context()
		h, err := l1.Forward(ctx, x)
		if err != nil {
			t.Fatal(err)
		}
		if h, err = mustOp(ctx, backend.OpReLU, h); err != nil {
			t.Fatal(err)
		}
		y, err := l2.Forward(ctx, h)
		if err != nil {
			t.Fatal(err)
		}
		loss, err := nn.MSE(ctx, y, target)
		if err != nil {
			t.Fatal(err)
		}
		if step == 0 {
			first = loss.AtF64()
		}
		last = loss.AtF64()
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		if err := opt.Step(tape.Grad); err != nil {
			t.Fatal(err)
		}
	}
	if !(last < first/2) {
		t.Fatalf("BitNet MLP did not train: loss %.6f → %.6f (want at least halved)", first, last)
	}
	t.Logf("BitNet QAT loss: %.6f → %.6f", first, last)
}

// Storage: a ternary weight carries log2(3) ≈ 1.585 bits (the "1.58" in b1.58) —
// a >20× reduction vs f32 masters at inference/export time (the masters stay
// full-precision during training; the win is at deployment).
func TestBitNetBitsPerWeight(t *testing.T) {
	bits := nn.BitNetBitsPerWeight()
	if math.Abs(bits-math.Log2(3)) > 1e-15 {
		t.Fatalf("BitNetBitsPerWeight = %g, want log2(3) = %g", bits, math.Log2(3))
	}
	if ratio := 32 / bits; ratio < 20 {
		t.Errorf("f32→ternary compression = %.2fx, want > 20x", ratio)
	}
}

func TestBitLinearErrors(t *testing.T) {
	l := nn.NewBitLinear(tensor.F64, 4, 3, 5)
	if _, err := l.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{2, 3, 4})); err == nil {
		t.Error("rank-3 x: want error")
	}
	if _, err := l.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{2, 5})); err == nil {
		t.Error("in-dim mismatch: want error")
	}
	if _, err := nn.BitQuantizeAct8(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{3})); err == nil {
		t.Error("rank-1 activation: want error")
	}
	if _, _, err := nn.BitQuantizeWeight(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{0, 2})); err == nil {
		t.Error("empty weight: want error")
	}
}

func ExampleBitLinear() {
	// A BitNet b1.58 layer: full-precision masters, ternary {−γ,0,+γ} weights and
	// 8-bit activations in the forward pass.
	l := nn.NewBitLinear(tensor.F64, 4, 3, 42)
	x := tensor.FromFloat64(tensor.Shape{2, 4}, []float64{1, -0.5, 0.25, 2, -1, 0.75, 0.5, -0.25})
	y, _ := l.Forward(backend.NewContext(), x)

	weff, gamma, _ := nn.BitQuantizeWeight(backend.NewContext(), l.W)
	levels := map[float64]bool{}
	for i := range weff.Numel() {
		levels[weff.AtF64(tensor.Unravel(i, weff.Shape())...)] = true
	}
	fmt.Println("out shape:", y.Shape())
	fmt.Println("trainable params:", len(l.Params()))
	fmt.Println("distinct effective weight values <= 3:", len(levels) <= 3, "| gamma > 0:", gamma > 0)
	fmt.Printf("bits per ternary weight: %.2f\n", nn.BitNetBitsPerWeight())
	// Output:
	// out shape: (2, 3)
	// trainable params: 3
	// distinct effective weight values <= 3: true | gamma > 0: true
	// bits per ternary weight: 1.58
}
