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

// ExampleVeRALinear adapts a frozen weight with a VeRA adapter built over the shared
// frozen random matrices — at init b=0, so the output equals the frozen base x·W.
func ExampleVeRALinear() {
	w := toT([][]float64{{1, 0}, {0, 1}}) // frozen base weight [in=2, out=2]
	m, _ := nn.NewVeRAMatrices(2, 2, 2, w.Dtype(), 1)
	v, _ := nn.NewVeRA(w, m, nn.VeRADefaultDInit)
	y, _ := v.Forward(backend.NewContext(), toT([][]float64{{3, 4}}))
	fmt.Printf("%.1f %.1f (trainable vectors: %d)\n", y.AtF64(0, 0), y.AtF64(0, 1), len(v.Params()))
	// Output:
	// 3.0 4.0 (trainable vectors: 2)
}

// §V16 tier-1 (arXiv:2310.11454 Eq.): the VeRA forward is y = x·W + b ⊙ (((x·A)⊙d)·B).
// Verified against an INDEPENDENT hand computation.
func TestVeRAForwardParity(t *testing.T) {
	// W = I; A[2,1]=[0.5;1.0]; B[1,2]=[2,-1]; d=[0.1]; b=[1,3]; x=[1,2].
	// s = x·A = 1·0.5+2·1 = 2.5; s·d = 0.25; ·B = [0.5,-0.25]; ⊙b = [0.5,-0.75];
	// + x·W = [1,2] ⇒ y = [1.5, 1.25].
	v := &nn.VeRALinear{
		W:  toT([][]float64{{1, 0}, {0, 1}}),
		A:  toT([][]float64{{0.5}, {1.0}}),
		B:  toT([][]float64{{2.0, -1.0}}),
		D:  vec([]float64{0.1}),
		Bv: vec([]float64{1.0, 3.0}),
		R:  1,
	}
	y, err := v.Forward(backend.NewContext(), toT([][]float64{{1, 2}}))
	if err != nil {
		t.Fatal(err)
	}
	for j, want := range []float64{1.5, 1.25} {
		if got := y.AtF64(0, j); math.Abs(got-want) > 1e-12 {
			t.Errorf("y[0,%d] = %.12g, want %g", j, got, want)
		}
	}
}

// §V16 tier-1 (init): b=0 at construction ⇒ ΔW=0, so a fresh adapter reproduces the
// frozen base x·W exactly regardless of the random A, B and the nonzero d.
func TestVeRAIdentityAtInit(t *testing.T) {
	w := toT([][]float64{{1, 2, 3}, {4, 5, 6}}) // [2,3]
	m, err := nn.NewVeRAMatrices(2, 3, 2, w.Dtype(), 42)
	if err != nil {
		t.Fatal(err)
	}
	v, err := nn.NewVeRA(w, m, 0) // dInit≤0 → default 0.1
	if err != nil {
		t.Fatal(err)
	}
	x := toT([][]float64{{1, -1}, {0.5, 2}})
	got, err := v.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	want, err := backend.Execute(backend.NewContext(), backend.OpMatMul, []*tensor.Tensor{x, w}, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i := range got.Numel() {
		c := tensor.Unravel(i, got.Shape())
		if math.Abs(got.AtF64(c...)-want[0].AtF64(c...)) > 1e-12 {
			t.Errorf("init not identity at %v: %g vs %g", c, got.AtF64(c...), want[0].AtF64(c...))
		}
	}
}

// §V16 tier-1 (parameter efficiency): only the vectors d[r] and b[out] train — the
// shared matrices A, B and the base W are NOT returned as parameters.
func TestVeRAParamFootprint(t *testing.T) {
	w := toT([][]float64{{1, 2, 3, 4}, {5, 6, 7, 8}}) // in=2, out=4
	m, _ := nn.NewVeRAMatrices(2, 4, 2, w.Dtype(), 7)
	v, _ := nn.NewVeRA(w, m, 0)
	ps := v.Params()
	if len(ps) != 2 {
		t.Fatalf("VeRA should expose 2 trainable vectors, got %d", len(ps))
	}
	if n := ps[0].Numel(); n != 2 { // d has length r=2
		t.Errorf("d length %d, want r=2", n)
	}
	if n := ps[1].Numel(); n != 4 { // b has length out=4
		t.Errorf("b length %d, want out=4", n)
	}
	trainable := ps[0].Numel() + ps[1].Numel() // 6
	loraLike := 2*2 + 2*4                      // A[in,r]+B[r,out] = 12 for LoRA
	if trainable >= loraLike {
		t.Errorf("VeRA footprint %d not smaller than LoRA-like %d", trainable, loraLike)
	}
}

// §V-GRAD: finite-difference check of the gradient into the trainable vectors d, b.
func TestVeRAGradCheck(t *testing.T) {
	w := toT([][]float64{{0.5, -1}, {2, 0.3}})
	m, _ := nn.NewVeRAMatrices(2, 2, 2, w.Dtype(), 3)
	v, _ := nn.NewVeRA(w, m, 0.2)
	// give b a nonzero value so both scaling paths are exercised.
	v.Bv.SetF64(0.7, 0)
	v.Bv.SetF64(-0.4, 1)
	x := toT([][]float64{{1, -2}, {0.5, 1.5}})

	sumLoss := func(ctx *backend.Context) (*tensor.Tensor, error) {
		y, err := v.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		return exec1(ctx, backend.OpSum, backend.ReduceAttrs{Axes: []int{0, 1}}, y)
	}
	tape := autograd.NewTape()
	loss, err := sumLoss(tape.Context())
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	const eps = 1e-6
	check := func(name string, p *tensor.Tensor) {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("%s: nil gradient", name)
		}
		for i := range p.Numel() {
			o := p.AtF64(i)
			p.SetF64(o+eps, i)
			lp, _ := sumLoss(backend.NewContext())
			p.SetF64(o-eps, i)
			lm, _ := sumLoss(backend.NewContext())
			p.SetF64(o, i)
			num, ana := (lp.AtF64()-lm.AtF64())/(2*eps), g.AtF64(i)
			if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("%s grad[%d]: numeric %.8g vs analytic %.8g", name, i, num, ana)
			}
		}
	}
	check("d", v.D)
	check("b", v.Bv)
}

// §V-PROP: a gradient-descent step on an MSE-to-target objective reduces the loss —
// the tiny d,b vectors are enough to move the output.
func TestVeRATrains(t *testing.T) {
	w := toT([][]float64{{1, 0}, {0, 1}})
	m, _ := nn.NewVeRAMatrices(2, 2, 2, w.Dtype(), 11)
	v, _ := nn.NewVeRA(w, m, 0.1)
	x := toT([][]float64{{1, 1}})
	target := toT([][]float64{{2, -2}})

	mse := func(ctx *backend.Context) (*tensor.Tensor, error) {
		y, err := v.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		diff, err := exec1(ctx, backend.OpSub, nil, y, target)
		if err != nil {
			return nil, err
		}
		sq, err := exec1(ctx, backend.OpMul, nil, diff, diff)
		if err != nil {
			return nil, err
		}
		return exec1(ctx, backend.OpMean, nil, sq)
	}
	tape := autograd.NewTape()
	loss0, err := mse(tape.Context())
	if err != nil {
		t.Fatal(err)
	}
	l0 := loss0.AtF64()
	if err := tape.Backward(loss0); err != nil {
		t.Fatal(err)
	}
	const lr = 0.1
	for _, p := range v.Params() {
		g := tape.Grad(p)
		for i := range p.Numel() {
			p.SetF64(p.AtF64(i)-lr*g.AtF64(i), i)
		}
	}
	loss1, err := mse(backend.NewContext())
	if err != nil {
		t.Fatal(err)
	}
	if loss1.AtF64() >= l0 {
		t.Errorf("VeRA GD step did not reduce loss: %g → %g", l0, loss1.AtF64())
	}
}

func TestVeRAErrors(t *testing.T) {
	w := toT([][]float64{{1, 2, 3}, {4, 5, 6}}) // [2,3]
	if _, err := nn.NewVeRAMatrices(2, 3, 5, w.Dtype(), 1); err == nil {
		t.Error("rank > dims: want error")
	}
	m, _ := nn.NewVeRAMatrices(2, 3, 2, w.Dtype(), 1)
	// matrices sized for [2,3] must not fit a [3,3] base.
	if _, err := nn.NewVeRA(toT([][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}), m, 0); err == nil {
		t.Error("mismatched shared matrices: want error")
	}
}

// exec1 runs a single-output op and returns its result (test helper).
func exec1(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}
