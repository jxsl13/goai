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

// §V16 tier-1: SoftpickAttention maps [T,dim]→[T,dim] and exposes every
// learnable tensor: 4 Linears × (W,B) = 8 (Q, K, V, O projections).
func TestSoftpickAttentionShapes(t *testing.T) {
	const dim, heads, seq = 8, 2, 5
	s, err := nn.NewSoftpickAttention(tensor.F64, dim, heads, 7)
	if err != nil {
		t.Fatal(err)
	}
	if s.HeadDim != dim/heads {
		t.Errorf("HeadDim=%d, want %d", s.HeadDim, dim/heads)
	}
	rng := rand.New(rand.NewPCG(1, 2))
	x := randMat(rng, seq, dim)
	out, err := s.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	if !out.Shape().Equal(tensor.Shape{seq, dim}) {
		t.Fatalf("output shape %v, want [%d %d]", out.Shape(), seq, dim)
	}
	if got, want := len(s.Params()), 8; got != want {
		t.Errorf("Params count %d, want %d", got, want)
	}
}

// §V16: the constructor rejects a dim not divisible by heads and non-positive
// dim/heads; Forward rejects a wrong input width and a wrong rank.
func TestSoftpickAttentionErrors(t *testing.T) {
	if _, err := nn.NewSoftpickAttention(tensor.F64, 7, 2, 1); err == nil {
		t.Error("dim%heads != 0: want error")
	}
	if _, err := nn.NewSoftpickAttention(tensor.F64, 0, 1, 1); err == nil {
		t.Error("dim=0: want error")
	}
	if _, err := nn.NewSoftpickAttention(tensor.F64, 8, 0, 1); err == nil {
		t.Error("heads=0: want error")
	}
	if _, err := nn.NewSoftpickAttention(tensor.F64, 8, -2, 1); err == nil {
		t.Error("negative heads: want error")
	}
	s, _ := nn.NewSoftpickAttention(tensor.F64, 8, 2, 1)
	if _, err := s.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{3, 4})); err == nil {
		t.Error("wrong input width: want error")
	}
	if _, err := s.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{2, 3, 8})); err == nil {
		t.Error("rank-3 input: want error")
	}
}

// softpickGradcheck compares every analytic gradient of every parameter of s
// against central finite differences of the summed forward output (F64 only —
// the h=1e-6 stencil drowns in F32 rounding). Returns the entry count and max
// relative error for the report.
func softpickGradcheck(t *testing.T, s *nn.SoftpickAttention, x *tensor.Tensor) (int, float64) {
	t.Helper()
	const h = 1e-6
	tape := autograd.NewTape()
	out, err := s.Forward(tape.Context(), x)
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
		o, err := s.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		var acc float64
		for i := range o.Numel() {
			acc += o.AtF64(tensor.Unravel(i, o.Shape())...)
		}
		return acc
	}
	var entries int
	var maxRel float64
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
			entries++
			if math.Abs(ana) > 1e-9 {
				nz = true
			}
			rel := math.Abs(num-ana) / math.Max(1, math.Abs(num))
			if rel > maxRel {
				maxRel = rel
			}
			if rel > 1e-4 {
				t.Errorf("%s grad%v: numeric %.8g vs analytic %.8g", name, coord, num, ana)
			}
		}
		if !nz {
			t.Errorf("%s: all analytic grads zero — parameter not exercised", name)
		}
	}
	check("QProj.W", s.QProj.W)
	check("QProj.B", s.QProj.B)
	check("KProj.W", s.KProj.W)
	check("KProj.B", s.KProj.B)
	check("VProj.W", s.VProj.W)
	check("VProj.B", s.VProj.B)
	check("OProj.W", s.OProj.W)
	check("OProj.B", s.OProj.B)
	return entries, maxRel
}

// §V16 (b) GRADCHECK: central finite differences confirm gradients reach EVERY
// parameter — Q/K/V/O projections — through the augmented softmax (OpConcat →
// OpSoftmax → OpSlice) and the value aggregation, in both the bidirectional and
// causal masks.
func TestSoftpickAttentionGradcheck(t *testing.T) {
	const dim, heads, seq = 4, 2, 3
	rng := rand.New(rand.NewPCG(29, 7))
	x := randMat(rng, seq, dim)
	s, err := nn.NewSoftpickAttention(tensor.F64, dim, heads, 3)
	if err != nil {
		t.Fatal(err)
	}
	n, rel := softpickGradcheck(t, s, x)
	t.Logf("bidirectional gradcheck: %d entries, max rel err %.3g", n, rel)

	sc, err := nn.NewSoftpickAttention(tensor.F64, dim, heads, 5, nn.WithSoftpickCausal())
	if err != nil {
		t.Fatal(err)
	}
	n, rel = softpickGradcheck(t, sc, x)
	t.Logf("causal gradcheck: %d entries, max rel err %.3g", n, rel)
}

// §V16 (d) CAUSAL CORRECTNESS: with WithSoftpickCausal, output row i must depend
// only on input rows ≤ i. Perturbing a strictly-future input row leaves earlier
// output rows bit-identical; the bidirectional layer, by contrast, changes.
func TestSoftpickAttentionCausal(t *testing.T) {
	const dim, heads, seq = 6, 3, 5
	rng := rand.New(rand.NewPCG(41, 13))
	x := randMat(rng, seq, dim)
	causal, err := nn.NewSoftpickAttention(tensor.F64, dim, heads, 9, nn.WithSoftpickCausal())
	if err != nil {
		t.Fatal(err)
	}
	base, err := causal.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	// Perturb the LAST input row (position seq-1): no earlier output may move.
	x2 := randMat(rng, seq, dim)
	for i := range seq { // identical rows except the final one
		for j := range dim {
			if i < seq-1 {
				x2.SetF64(x.AtF64(i, j), i, j)
			}
		}
	}
	pert, err := causal.Forward(backend.NewContext(), x2)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < seq-1; i++ {
		for j := range dim {
			if d := math.Abs(base.AtF64(i, j) - pert.AtF64(i, j)); d > 1e-12 {
				t.Errorf("causal leak: row %d col %d moved %.3g when only future input changed", i, j, d)
			}
		}
	}
	// Sanity: the bidirectional layer DOES see the future — earlier rows change.
	bidir, err := nn.NewSoftpickAttention(tensor.F64, dim, heads, 9)
	if err != nil {
		t.Fatal(err)
	}
	b1, _ := bidir.Forward(backend.NewContext(), x)
	b2, _ := bidir.Forward(backend.NewContext(), x2)
	var moved bool
	for j := range dim {
		if math.Abs(b1.AtF64(0, j)-b2.AtF64(0, j)) > 1e-9 {
			moved = true
		}
	}
	if !moved {
		t.Error("bidirectional row 0 unchanged by a future-row edit — mask leaked the other way")
	}
}

// §V16: SoftpickAttention.Forward is deterministic — same weights, same input,
// bit-identical output across calls.
func TestSoftpickAttentionDeterminism(t *testing.T) {
	const dim, heads, seq = 8, 2, 4
	rng := rand.New(rand.NewPCG(2, 3))
	x := randMat(rng, seq, dim)
	s, err := nn.NewSoftpickAttention(tensor.F64, dim, heads, 7, nn.WithSoftpickCausal())
	if err != nil {
		t.Fatal(err)
	}
	a, err := s.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	b, err := s.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	for i := range seq {
		for j := range dim {
			if a.AtF64(i, j) != b.AtF64(i, j) {
				t.Fatalf("non-deterministic at (%d,%d): %v vs %v", i, j, a.AtF64(i, j), b.AtF64(i, j))
			}
		}
	}
}

// §V16 (e) F32 PARITY: gradcheck is F64-only, so the F32 path is pinned
// separately — with identical weights copied from the F64 block the F32 forward
// must equal the F64 forward to float32 accuracy (softmax/matmul accumulate in
// f64 per §V10; the only divergence is f32 storage rounding).
func TestSoftpickAttentionF32Parity(t *testing.T) {
	const dim, heads, seq, seed = 8, 2, 5, 13
	s64, err := nn.NewSoftpickAttention(tensor.F64, dim, heads, seed, nn.WithSoftpickCausal())
	if err != nil {
		t.Fatal(err)
	}
	s32, err := nn.NewSoftpickAttention(tensor.F32, dim, heads, seed, nn.WithSoftpickCausal())
	if err != nil {
		t.Fatal(err)
	}
	p64, p32 := s64.Params(), s32.Params()
	if len(p64) != len(p32) {
		t.Fatalf("param count mismatch: %d vs %d", len(p64), len(p32))
	}
	for i := range p64 {
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
	out64, err := s64.Forward(backend.NewContext(), x64)
	if err != nil {
		t.Fatal(err)
	}
	out32, err := s32.Forward(backend.NewContext(), x32)
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

func ExampleSoftpickAttention() {
	// A tiny softmax-off-by-one attention layer: causal 2-head attention whose
	// weights use softmax_1 (the +1-in-the-denominator "quiet" softmax), so a
	// query can attend to nothing.
	s, _ := nn.NewSoftpickAttention(tensor.F64, 4, 2, 1, nn.WithSoftpickCausal())
	x := tensor.New(tensor.F64, tensor.Shape{3, 4})
	for i := range 3 {
		x.SetF64(1, i, i) // arbitrary non-zero input
	}
	out, _ := s.Forward(backend.NewContext(), x)
	fmt.Printf("%v params=%d\n", out.Shape(), len(s.Params()))
	// Output: (3, 4) params=8
}
