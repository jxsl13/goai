package nn_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §V16 tier-1: GatedAttention maps [T,dim]→[T,dim] and exposes every learnable
// tensor: 3 Linears × (W,B) = 6 (fused in-proj, gate W_θ, out-proj W_O).
func TestGatedAttentionShapes(t *testing.T) {
	const dim, heads, seq = 8, 2, 5
	g, err := nn.NewGatedAttention(tensor.F64, dim, heads, 7)
	if err != nil {
		t.Fatal(err)
	}
	if g.HeadDim != dim/heads {
		t.Errorf("HeadDim=%d, want %d", g.HeadDim, dim/heads)
	}
	if !g.Gate.W.Shape().Equal(tensor.Shape{dim, dim}) {
		t.Errorf("elementwise gate W shape %v, want [%d %d]", g.Gate.W.Shape(), dim, dim)
	}
	rng := rand.New(rand.NewPCG(1, 2))
	x := randMat(rng, seq, dim)
	out, err := g.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	if !out.Shape().Equal(tensor.Shape{seq, dim}) {
		t.Fatalf("output shape %v, want [%d %d]", out.Shape(), seq, dim)
	}
	if got, want := len(g.Params()), 6; got != want {
		t.Errorf("Params count %d, want %d", got, want)
	}
}

// §V16: the constructor rejects a dim not divisible by heads and non-positive
// dim/heads; Forward rejects a wrong input width and a wrong rank.
func TestGatedAttentionErrors(t *testing.T) {
	if _, err := nn.NewGatedAttention(tensor.F64, 7, 2, 1); err == nil {
		t.Error("dim%heads != 0: want error")
	}
	if _, err := nn.NewGatedAttention(tensor.F64, 0, 1, 1); err == nil {
		t.Error("dim=0: want error")
	}
	if _, err := nn.NewGatedAttention(tensor.F64, 8, 0, 1); err == nil {
		t.Error("heads=0: want error")
	}
	if _, err := nn.NewGatedAttention(tensor.F64, 8, -2, 1); err == nil {
		t.Error("negative heads: want error")
	}
	g, _ := nn.NewGatedAttention(tensor.F64, 8, 2, 1)
	if _, err := g.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{3, 4})); err == nil {
		t.Error("wrong input width: want error")
	}
	if _, err := g.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{2, 3, 8})); err == nil {
		t.Error("rank-3 input: want error")
	}
}

// gatedAttnGradcheck compares every analytic gradient of every parameter of g
// against central finite differences of the summed forward output (F64 only —
// the h=1e-6 stencil drowns in F32 rounding).
func gatedAttnGradcheck(t *testing.T, g *nn.GatedAttention, x *tensor.Tensor) {
	t.Helper()
	const h = 1e-6
	tape := autograd.NewTape()
	out, err := g.Forward(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := sumAll(tape.Context(), out)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	sumLoss := func() float64 {
		o, err := g.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		var s float64
		for i := range o.Numel() {
			s += o.AtF64(tensor.Unravel(i, o.Shape())...)
		}
		return s
	}
	check := func(name string, p *tensor.Tensor) {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("%s: nil grad — no gradient reached this parameter", name)
		}
		var nz bool
		for idx := range p.Numel() {
			coord := tensor.Unravel(idx, p.Shape())
			o := p.AtF64(coord...)
			p.SetF64(o+h, coord...)
			lp := sumLoss()
			p.SetF64(o-h, coord...)
			lm := sumLoss()
			p.SetF64(o, coord...)
			num, ana := (lp-lm)/(2*h), g.AtF64(coord...)
			if math.Abs(ana) > 1e-9 {
				nz = true
			}
			if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("%s grad%v: numeric %.8g vs analytic %.8g", name, coord, num, ana)
			}
		}
		if !nz {
			t.Errorf("%s: all analytic grads zero — parameter not exercised", name)
		}
	}
	check("InProj.W", g.InProj.W) // fused Q/K/V — collects grads from all three slices
	check("InProj.B", g.InProj.B)
	check("Gate.W", g.Gate.W) // W_θ — the gate branch must be on the tape
	check("Gate.B", g.Gate.B)
	check("OutProj.W", g.OutProj.W) // W_O
	check("OutProj.B", g.OutProj.B)
}

// §V16 (c) GRADIENT FLOW: central finite differences confirm gradients reach
// EVERY parameter — the fused Q/K/V in-proj (through OpMHA), the gate
// projection W_θ (through sigmoid and the elementwise multiply), and the
// out-proj W_O — in both gate granularities.
func TestGatedAttentionGradcheck(t *testing.T) {
	const dim, heads, seq = 4, 2, 3
	rng := rand.New(rand.NewPCG(29, 7))
	x := randMat(rng, seq, dim)
	g, err := nn.NewGatedAttention(tensor.F64, dim, heads, 3)
	if err != nil {
		t.Fatal(err)
	}
	gatedAttnGradcheck(t, g, x)
}

// §V16: the HEADWISE variant (one sigmoid scalar per head, expanded over the
// head's channels by slice/broadcast/concat) is equally differentiable —
// same full-parameter gradcheck through the expansion path.
func TestGatedAttentionHeadwiseGradcheck(t *testing.T) {
	const dim, heads, seq = 4, 2, 3
	rng := rand.New(rand.NewPCG(31, 9))
	x := randMat(rng, seq, dim)
	g, err := nn.NewGatedAttention(tensor.F64, dim, heads, 3, nn.WithGatedAttentionHeadwiseGate())
	if err != nil {
		t.Fatal(err)
	}
	if !g.Gate.W.Shape().Equal(tensor.Shape{dim, heads}) {
		t.Fatalf("headwise gate W shape %v, want [%d %d]", g.Gate.W.Shape(), dim, heads)
	}
	gatedAttnGradcheck(t, g, x)
}

// §V16 (d) GATE STRUCTURE: the gate really multiplies the SDPA output before
// W_O. Zeroing W_θ and its bias pins the gate at exactly σ(0)=0.5, so the
// output must satisfy out_half − b_O = 0.5·(out_ungated − b_O), where
// out_ungated is the SAME block's attention pushed through W_O without the
// gate (recomputed here from the block's own InProj/OpMHA/OutProj pieces).
// The default (random W_θ) output must differ from the pinned-gate one — the
// gate is data-dependent, not a constant.
func TestGatedAttentionGateEffect(t *testing.T) {
	const dim, heads, seq, seed = 8, 2, 5, 11
	rng := rand.New(rand.NewPCG(5, 6))
	x := randMat(rng, seq, dim)
	g, err := nn.NewGatedAttention(tensor.F64, dim, heads, seed)
	if err != nil {
		t.Fatal(err)
	}
	outDefault, err := g.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	nn.Zeros(g.Gate.W) // gate logits ≡ 0 → gate ≡ 0.5 exactly
	nn.Zeros(g.Gate.B)
	outHalf, err := g.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	// Ungated reference from the block's own pieces: InProj → slice → OpMHA → OutProj.
	ctx := backend.NewContext()
	p, err := g.InProj.Forward(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	var qkv [3]*tensor.Tensor
	for i := range 3 {
		s, err := backend.Execute(ctx, backend.OpSlice, []*tensor.Tensor{p},
			backend.SliceAttrs{Axis: 1, Start: i * dim, End: (i + 1) * dim})
		if err != nil {
			t.Fatal(err)
		}
		qkv[i] = s[0]
	}
	attn, err := backend.Execute(ctx, backend.OpMHA, []*tensor.Tensor{qkv[0], qkv[1], qkv[2]},
		backend.AttnAttrs{Heads: heads, Causal: true})
	if err != nil {
		t.Fatal(err)
	}
	ungated, err := g.OutProj.Forward(ctx, attn[0])
	if err != nil {
		t.Fatal(err)
	}
	var maxRel, maxDiff float64
	for i := range seq {
		for j := range dim {
			b := g.OutProj.B.AtF64(j)
			half, full := outHalf.AtF64(i, j)-b, ungated.AtF64(i, j)-b
			if d := math.Abs(half - 0.5*full); d > maxRel {
				maxRel = d
			}
			if d := math.Abs(outDefault.AtF64(i, j) - outHalf.AtF64(i, j)); d > maxDiff {
				maxDiff = d
			}
		}
	}
	if maxRel > 1e-12 {
		t.Errorf("pinned gate 0.5: out−b ≠ 0.5·(ungated−b), maxdiff %.3g — gate not multiplying the SDPA output", maxRel)
	}
	if maxDiff < 1e-9 {
		t.Errorf("random-W_θ output identical to pinned-gate output (maxdiff %.3g) — gate not data-dependent", maxDiff)
	}
	// The gate branch is on the tape: a non-nil, non-zero gradient reaches W_θ.
	tape := autograd.NewTape()
	g2, err := nn.NewGatedAttention(tensor.F64, dim, heads, seed)
	if err != nil {
		t.Fatal(err)
	}
	out, err := g2.Forward(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := sumAll(tape.Context(), out)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	gw := tape.Grad(g2.Gate.W)
	if gw == nil {
		t.Fatal("nil grad on Gate.W — gate branch not on the tape")
	}
	var nz bool
	for i := range gw.Numel() {
		if math.Abs(gw.AtF64(tensor.Unravel(i, gw.Shape())...)) > 1e-12 {
			nz = true
			break
		}
	}
	if !nz {
		t.Error("all-zero grad on Gate.W — gate not exercised")
	}
}

// §V16 (e) F32 PARITY: gradcheck is F64-only, so the F32 path is pinned
// separately — with identical weights (copied from the F64 block) the F32
// forward must equal the F64 forward to float32 accuracy. OpMHA and the
// elementwise ops accumulate in f64 (§V10), so the only divergence is f32
// storage rounding.
func TestGatedAttentionF32Parity(t *testing.T) {
	const dim, heads, seq, seed = 8, 2, 5, 13
	g64, err := nn.NewGatedAttention(tensor.F64, dim, heads, seed)
	if err != nil {
		t.Fatal(err)
	}
	g32, err := nn.NewGatedAttention(tensor.F32, dim, heads, seed)
	if err != nil {
		t.Fatal(err)
	}
	p64, p32 := g64.Params(), g32.Params()
	if len(p64) != len(p32) {
		t.Fatalf("param count mismatch: %d vs %d", len(p64), len(p32))
	}
	for i := range p64 { // copy the F64 weights into the F32 block (rounded once)
		for j := range p64[i].Numel() {
			c := tensor.Unravel(j, p64[i].Shape())
			p32[i].SetF64(p64[i].AtF64(c...), c...)
		}
	}
	rng := rand.New(rand.NewPCG(17, 3))
	x64 := randMat(rng, seq, dim)
	x32 := tensor.New(tensor.F32, tensor.Shape{seq, dim})
	for i := range seq {
		for j := range dim {
			x32.SetF64(x64.AtF64(i, j), i, j)
		}
	}
	out64, err := g64.Forward(backend.NewContext(), x64)
	if err != nil {
		t.Fatal(err)
	}
	out32, err := g32.Forward(backend.NewContext(), x32)
	if err != nil {
		t.Fatal(err)
	}
	if out32.Dtype() != tensor.F32 {
		t.Fatalf("F32 forward returned dtype %v, want F32", out32.Dtype())
	}
	for i := range seq {
		for j := range dim {
			a, b := out64.AtF64(i, j), out32.AtF64(i, j)
			if d := math.Abs(a - b); d > 1e-4*math.Max(1, math.Abs(a)) {
				t.Errorf("F32 parity at (%d,%d): f64 %.8g vs f32 %.8g (diff %.3g)", i, j, a, b, d)
			}
		}
	}
}

func ExampleGatedAttention() {
	// A tiny gated attention layer: causal 2-head SDPA whose raw output is
	// multiplied by a sigmoid gate σ(x·W_θ) before the output projection.
	g, _ := nn.NewGatedAttention(tensor.F64, 4, 2, 1)
	x := tensor.New(tensor.F64, tensor.Shape{3, 4})
	for i := range 3 {
		x.SetF64(1, i, i) // arbitrary non-zero input
	}
	out, _ := g.Forward(backend.NewContext(), x)
	fmt.Printf("%v params=%d\n", out.Shape(), len(g.Params()))
	// Output: (3, 4) params=6
}
