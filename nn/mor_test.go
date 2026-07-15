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

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// morMLP is a tiny concrete RecursionBlock for the tests: Linear→tanh→Linear,
// width-preserving [k,d]→[k,d].
type morMLP struct {
	l1, l2 *nn.Linear
}

func newMorMLP(dtype tensor.Dtype, d, hidden int, seed uint64) *morMLP {
	return &morMLP{
		l1: nn.NewLinear(dtype, d, hidden, seed),
		l2: nn.NewLinear(dtype, hidden, d, seed+1),
	}
}

func (b *morMLP) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := b.l1.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	o, err := backend.Execute(ctx, backend.OpTanh, []*tensor.Tensor{h}, nil)
	if err != nil {
		return nil, err
	}
	return b.l2.Forward(ctx, o[0])
}

func (b *morMLP) Params() []*tensor.Tensor {
	return append(b.l1.Params(), b.l2.Params()...)
}

// morRecorder is an identity RecursionBlock that records every input it sees,
// so tests can observe which token rows reach the shared block at each step.
type morRecorder struct {
	inputs []*tensor.Tensor
}

func (b *morRecorder) Forward(_ *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	b.inputs = append(b.inputs, x)
	return x, nil
}

func (b *morRecorder) Params() []*tensor.Tensor { return nil }

func morExec(t *testing.T, ctx *backend.Context, op backend.Op, in ...*tensor.Tensor) *tensor.Tensor {
	t.Helper()
	o, err := backend.Execute(ctx, op, in, nil)
	if err != nil {
		t.Fatal(err)
	}
	return o[0]
}

// Forward maps [T,d]→[T,d]; Params is the shared block's tensors once plus one
// weight per per-step router; StepSizes follows the shrinking-active-set
// capacity convention round(c·|active|) clamped to ≥1.
func TestMoRShapes(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 3))
	const seq, d, hidden, nr = 8, 4, 6, 3
	block := newMorMLP(tensor.F64, d, hidden, 3)
	m, err := nn.NewMixtureOfRecursions(block, nr, 0.5, tensor.F64, d, 7)
	if err != nil {
		t.Fatal(err)
	}
	out, err := m.Forward(backend.NewContext(), randMat(rng, seq, d))
	if err != nil {
		t.Fatal(err)
	}
	if out.Ndim() != 2 || out.Shape()[0] != seq || out.Shape()[1] != d {
		t.Fatalf("Forward shape %v, want [%d %d]", out.Shape(), seq, d)
	}
	if got, want := len(m.Params()), len(block.Params())+nr; got != want {
		t.Errorf("Params() len %d, want block %d + routers %d = %d", got, len(block.Params()), nr, want)
	}
	sizes := m.StepSizes(seq)
	want := []int{4, 2, 1} // 8→round(.5·8)=4→round(.5·4)=2→round(.5·2)=1
	for r := range want {
		if sizes[r] != want[r] {
			t.Errorf("StepSizes(8)[%d]=%d want %d", r, sizes[r], want[r])
		}
	}
}

func TestMoRErrors(t *testing.T) {
	block := nn.NewLinear(tensor.F64, 3, 3, 1)
	if _, err := nn.NewMixtureOfRecursions(nil, 2, 0.5, tensor.F64, 3, 1); err == nil {
		t.Error("nil block: want error")
	}
	if _, err := nn.NewMixtureOfRecursions(block, 0, 0.5, tensor.F64, 3, 1); err == nil {
		t.Error("maxRecursion 0: want error")
	}
	if _, err := nn.NewMixtureOfRecursions(block, 2, 0, tensor.F64, 3, 1); err == nil {
		t.Error("capacity 0: want error")
	}
	if _, err := nn.NewMixtureOfRecursions(block, 2, 1.5, tensor.F64, 3, 1); err == nil {
		t.Error("capacity 1.5: want error")
	}
	if _, err := nn.NewMixtureOfRecursions(block, 2, 0.5, tensor.F64, 0, 1); err == nil {
		t.Error("d 0: want error")
	}
	if _, err := nn.NewMixtureOfRecursions(block, 2, 0.5, tensor.F64, 3, 1, nn.WithMoRCapacities(0.5)); err == nil {
		t.Error("capacity schedule wrong length: want error")
	}
	if _, err := nn.NewMixtureOfRecursions(block, 2, 0.5, tensor.F64, 3, 1, nn.WithMoRCapacities(0.5, 2)); err == nil {
		t.Error("capacity schedule entry >1: want error")
	}
	m, err := nn.NewMixtureOfRecursions(block, 2, 0.5, tensor.F64, 3, 1)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{2, 3, 1})); err == nil {
		t.Error("rank-3 input: want error")
	}
	if _, err := m.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{4, 5})); err == nil {
		t.Error("width mismatch: want error")
	}
}

// Full-parameter gradcheck: central differences through the SHARED block
// parameters, EVERY per-step router, and x (the discrete top-k selection is
// constant, as in MoD).
func TestMoRGradcheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 11))
	const seq, d, hidden, nr = 6, 3, 4, 3
	block := newMorMLP(tensor.F64, d, hidden, 17)
	m, err := nn.NewMixtureOfRecursions(block, nr, 0.5, tensor.F64, d, 23)
	if err != nil {
		t.Fatal(err)
	}
	x := randMat(rng, seq, d)
	fwd := func(ctx *backend.Context) (*tensor.Tensor, error) {
		out, err := m.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		return sumAll(ctx, out)
	}
	tape := autograd.NewTape()
	loss, err := fwd(tape.Context())
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	const h = 1e-6
	scalarFwd := func() float64 {
		l, err := fwd(backend.NewContext())
		if err != nil {
			t.Fatal(err)
		}
		return l.AtF64()
	}
	check := func(name string, p *tensor.Tensor) {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("%s: nil grad", name)
		}
		allZero := true
		for idx := range p.Numel() {
			coord := tensor.Unravel(idx, p.Shape())
			if g.AtF64(coord...) != 0 {
				allZero = false
			}
			o := p.AtF64(coord...)
			p.SetF64(o+h, coord...)
			lp := scalarFwd()
			p.SetF64(o-h, coord...)
			lm := scalarFwd()
			p.SetF64(o, coord...)
			if num, ana := (lp-lm)/(2*h), g.AtF64(coord...); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("%s grad%v: numeric %.8g vs analytic %.8g", name, coord, num, ana)
			}
		}
		if allZero {
			t.Errorf("%s: gradient is identically zero", name)
		}
	}
	check("x", x)
	for i, p := range m.Params() {
		check(fmt.Sprintf("param[%d]", i), p)
	}
}

// F32 parity: with weights copied from the F64 model, the F32 forward matches
// the F64 forward to ~1e-4 relative (gradcheck itself is F64-only).
func TestMoRF32Parity(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 4))
	const seq, d, hidden, nr = 5, 3, 4, 2
	b64 := newMorMLP(tensor.F64, d, hidden, 31)
	m64, err := nn.NewMixtureOfRecursions(b64, nr, 0.5, tensor.F64, d, 37)
	if err != nil {
		t.Fatal(err)
	}
	b32 := newMorMLP(tensor.F32, d, hidden, 31)
	m32, err := nn.NewMixtureOfRecursions(b32, nr, 0.5, tensor.F32, d, 37)
	if err != nil {
		t.Fatal(err)
	}
	p64, p32 := m64.Params(), m32.Params()
	if len(p64) != len(p32) {
		t.Fatalf("param count mismatch %d vs %d", len(p64), len(p32))
	}
	for i := range p64 {
		for idx := range p64[i].Numel() {
			c := tensor.Unravel(idx, p64[i].Shape())
			p32[i].SetF64(p64[i].AtF64(c...), c...)
		}
	}
	x64 := randMat(rng, seq, d)
	x32 := tensor.New(tensor.F32, tensor.Shape{seq, d})
	for i := range seq {
		for j := range d {
			x32.SetF64(x64.AtF64(i, j), i, j)
		}
	}
	o64, err := m64.Forward(backend.NewContext(), x64)
	if err != nil {
		t.Fatal(err)
	}
	o32, err := m32.Forward(backend.NewContext(), x32)
	if err != nil {
		t.Fatal(err)
	}
	for i := range seq {
		for j := range d {
			a, b := o64.AtF64(i, j), o32.AtF64(i, j)
			if math.Abs(a-b) > 1e-4*math.Max(1, math.Abs(a)) {
				t.Errorf("out[%d,%d]: f64 %.8g vs f32 %.8g", i, j, a, b)
			}
		}
	}
}

// Reduction 1: maxRecursion==1 IS a single Mixture-of-Depths pass — a hand-built
// Route→block→Combine with the same router weights matches to 1e-12.
func TestMoRSingleStepIsMoD(t *testing.T) {
	rng := rand.New(rand.NewPCG(13, 8))
	const seq, d, hidden = 6, 3, 5
	block := newMorMLP(tensor.F64, d, hidden, 41)
	m, err := nn.NewMixtureOfRecursions(block, 1, 0.5, tensor.F64, d, 43)
	if err != nil {
		t.Fatal(err)
	}
	mod := nn.NewMixtureOfDepths(tensor.F64, d, 0.5)
	for j := range d {
		mod.Router.SetF64(m.Routers[0].Router.AtF64(j, 0), j, 0)
	}
	x := randMat(rng, seq, d)
	got, err := m.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	ctx := backend.NewContext()
	g, w, sel, err := mod.Route(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	p, err := block.Forward(ctx, g)
	if err != nil {
		t.Fatal(err)
	}
	want, err := mod.Combine(ctx, x, p, w, sel)
	if err != nil {
		t.Fatal(err)
	}
	for i := range seq {
		for j := range d {
			if diff := math.Abs(got.AtF64(i, j) - want.AtF64(i, j)); diff > 1e-12 {
				t.Errorf("out[%d,%d]: MoR %.16g vs MoD %.16g (diff %g)", i, j, got.AtF64(i, j), want.AtF64(i, j), diff)
			}
		}
	}
}

// Reduction 2: capacity==1 selects every token every step, so MoR IS the shared
// block applied Nr times to all tokens in the same routed-residual form
// h ← h + (h·w_r) ⊙ B(h) — a plain weight-tied recursive stack.
func TestMoRFullCapacityIsRecursiveStack(t *testing.T) {
	rng := rand.New(rand.NewPCG(19, 6))
	const seq, d, hidden, nr = 4, 3, 5, 3
	block := newMorMLP(tensor.F64, d, hidden, 47)
	m, err := nn.NewMixtureOfRecursions(block, nr, 1.0, tensor.F64, d, 53)
	if err != nil {
		t.Fatal(err)
	}
	x := randMat(rng, seq, d)
	got, err := m.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	ctx := backend.NewContext()
	h := x
	for r := range nr {
		logits := morExec(t, ctx, backend.OpMatMul, h, m.Routers[r].Router) // [seq,1]
		p, err := block.Forward(ctx, h)
		if err != nil {
			t.Fatal(err)
		}
		wp := morExec(t, ctx, backend.OpMul, p, logits) // broadcast [seq,d]*[seq,1]
		h = morExec(t, ctx, backend.OpAdd, h, wp)
	}
	for i := range seq {
		for j := range d {
			if diff := math.Abs(got.AtF64(i, j) - h.AtF64(i, j)); diff > 1e-10 {
				t.Errorf("out[%d,%d]: MoR %.16g vs manual stack %.16g (diff %g)", i, j, got.AtF64(i, j), h.AtF64(i, j), diff)
			}
		}
	}
}

// Structural: with capacity<1 and Nr>1 the shared block sees geometrically
// fewer tokens at deeper steps, step 1 processes exactly the router's top-k
// tokens, and changing the router changes WHICH tokens are selected.
func TestMoRDepthRoutingVaries(t *testing.T) {
	const seq, d, nr = 8, 2, 3
	x := toT([][]float64{{7, 0}, {1, 1}, {5, -1}, {2, 3}, {8, 2}, {3, 0}, {6, 1}, {4, -2}})
	run := func(routerCol []float64) *morRecorder {
		rec := &morRecorder{}
		m, err := nn.NewMixtureOfRecursions(rec, nr, 0.5, tensor.F64, d, 59)
		if err != nil {
			t.Fatal(err)
		}
		setCol(m.Routers[0].Router, routerCol)
		if _, err := m.Forward(backend.NewContext(), x); err != nil {
			t.Fatal(err)
		}
		wantSizes := m.StepSizes(seq) // [4 2 1]
		if len(rec.inputs) != nr {
			t.Fatalf("block called %d times, want %d", len(rec.inputs), nr)
		}
		for r, in := range rec.inputs {
			if in.Shape()[0] != wantSizes[r] {
				t.Errorf("step %d processed %d tokens, want %d", r, in.Shape()[0], wantSizes[r])
			}
			if r > 0 && in.Shape()[0] >= rec.inputs[r-1].Shape()[0] {
				t.Errorf("step %d processed %d tokens, not fewer than step %d's %d", r, in.Shape()[0], r-1, rec.inputs[r-1].Shape()[0])
			}
		}
		return rec
	}
	checkStep1 := func(rec *morRecorder, wantIdx []int) {
		t.Helper()
		for i, p := range wantIdx {
			for j := range d {
				if got, want := rec.inputs[0].AtF64(i, j), x.AtF64(p, j); got != want {
					t.Errorf("step-1 row %d col %d = %g, want x[%d,%d]=%g", i, j, got, p, j, want)
				}
			}
		}
	}
	// router = +e0: step-1 logit is x[:,0] ⇒ top-4 = tokens {7,8,5,6} at positions 0,2,4,6.
	checkStep1(run([]float64{1, 0}), []int{0, 2, 4, 6})
	// router = −e0 flips the ranking ⇒ the OTHER four tokens are selected.
	checkStep1(run([]float64{-1, 0}), []int{1, 3, 5, 7})
}

func ExampleMixtureOfRecursions() {
	// One weight-shared Linear applied up to 2 times; capacity ½ halves the
	// active token set at every recursion step (4 → 2 → 1).
	block := nn.NewLinear(tensor.F64, 2, 2, 1)
	mor, _ := nn.NewMixtureOfRecursions(block, 2, 0.5, tensor.F64, 2, 42)
	x := toT([][]float64{{1, 0}, {0, 1}, {2, 2}, {-1, 1}})
	out, _ := mor.Forward(backend.NewContext(), x)
	// Params: the shared block's W and B once, plus one router per step.
	fmt.Println(mor.StepSizes(4), len(mor.Params()), out.Shape())
	// Output:
	// [2 1] 4 (4, 2)
}
