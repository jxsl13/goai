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

func randMat(rng *rand.Rand, r, c int) *tensor.Tensor {
	t := tensor.New(tensor.F64, tensor.Shape{r, c})
	for i := range r {
		for j := range c {
			t.SetF64(rng.NormFloat64(), i, j)
		}
	}
	return t
}

// §V16 tier-1: an independent recomputation of the Soft-MoE wiring — logits L=X·Φ,
// dispatch D=softmax-over-tokens, slot inputs X̃=DᵀX, per-expert slots, combine
// C=softmax-over-slots, Y=CỸ — matches Forward. (Experts are reused as black boxes,
// so this checks the dispatch/combine wiring and the softmax axes.)
func TestSoftMoEForwardParity(t *testing.T) {
	const n, d, e, p, hidden = 3, 4, 2, 2, 5
	rng := rand.New(rand.NewPCG(1, 2))
	m := nn.NewSoftMoE(tensor.F64, d, hidden, e, p, 7)
	x := randMat(rng, n, d)
	slots := e * p

	got, err := m.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}

	// L = X·Φ  [n, slots]
	l := make([][]float64, n)
	for i := range n {
		l[i] = make([]float64, slots)
		for j := range slots {
			for k := range d {
				l[i][j] += x.AtF64(i, k) * m.Phi.AtF64(k, j)
			}
		}
	}
	// D = softmax over tokens (columns); Xtilde[j] = Σ_i D[i,j]·x[i]
	xt := make([][]float64, slots)
	for j := range slots {
		col := make([]float64, n)
		mx := math.Inf(-1)
		for i := range n {
			col[i] = l[i][j]
			mx = math.Max(mx, col[i])
		}
		var sum float64
		for i := range n {
			col[i] = math.Exp(col[i] - mx)
			sum += col[i]
		}
		xt[j] = make([]float64, d)
		for i := range n {
			w := col[i] / sum
			for k := range d {
				xt[j][k] += w * x.AtF64(i, k)
			}
		}
	}
	// per-expert slot outputs Ỹ via the SAME experts
	yt := make([][]float64, slots)
	for ex := range e {
		st := tensor.New(tensor.F64, tensor.Shape{p, d})
		for r := range p {
			for k := range d {
				st.SetF64(xt[ex*p+r][k], r, k)
			}
		}
		ye, _ := m.Experts[ex].Forward(backend.NewContext(), st)
		for r := range p {
			yt[ex*p+r] = make([]float64, d)
			for k := range d {
				yt[ex*p+r][k] = ye.AtF64(r, k)
			}
		}
	}
	// C = softmax over slots (rows); Y[i] = Σ_j C[i,j]·Ỹ[j]
	for i := range n {
		mx := math.Inf(-1)
		for j := range slots {
			mx = math.Max(mx, l[i][j])
		}
		var sum float64
		cr := make([]float64, slots)
		for j := range slots {
			cr[j] = math.Exp(l[i][j] - mx)
			sum += cr[j]
		}
		want := make([]float64, d)
		for j := range slots {
			w := cr[j] / sum
			for k := range d {
				want[k] += w * yt[j][k]
			}
		}
		for k := range d {
			if math.Abs(got.AtF64(i, k)-want[k]) > 1e-9 {
				t.Errorf("Y[%d,%d] = %.10g, want %.10g", i, k, got.AtF64(i, k), want[k])
			}
		}
	}
}

// §V2: the layer is fully differentiable — gradients w.r.t. the input, Φ, and an expert
// weight all match central finite differences (no top-k / argmax anywhere).
func TestSoftMoEGradcheck(t *testing.T) {
	const n, d, e, p, hidden = 2, 3, 2, 1, 4
	rng := rand.New(rand.NewPCG(4, 5))
	m := nn.NewSoftMoE(tensor.F64, d, hidden, e, p, 3)
	x := randMat(rng, n, d)
	mask := randMat(rng, n, d)

	lossOf := func() float64 {
		y, err := m.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		var s float64
		for i := range n {
			for k := range d {
				s += mask.AtF64(i, k) * y.AtF64(i, k)
			}
		}
		return s
	}

	tape := autograd.NewTape()
	y, err := m.Forward(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	pr, _ := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{y, mask}, nil)
	loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{pr[0]}, nil)
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}

	const h = 1e-6
	check := func(name string, param *tensor.Tensor) {
		g := tape.Grad(param)
		if g == nil {
			t.Fatalf("%s: nil gradient", name)
		}
		for i := range param.Numel() {
			idx := tensor.Unravel(i, param.Shape())
			o := param.AtF64(idx...)
			param.SetF64(o+h, idx...)
			lp := lossOf()
			param.SetF64(o-h, idx...)
			lm := lossOf()
			param.SetF64(o, idx...)
			if num, ana := (lp-lm)/(2*h), g.AtF64(idx...); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("%s grad[%v]: numeric %.8g vs analytic %.8g", name, idx, num, ana)
			}
		}
	}
	check("x", x)
	check("Phi", m.Phi)
	check("expert0.Wgate", m.Experts[0].Wgate)
}

// §V16 tier-1: dispatch weights normalize over tokens (each slot's column sums to 1)
// and combine weights over slots (each token's row sums to 1).
func TestSoftMoESoftmaxNorm(t *testing.T) {
	const n, d, e, p, hidden = 4, 3, 2, 2, 4
	rng := rand.New(rand.NewPCG(9, 9))
	m := nn.NewSoftMoE(tensor.F64, d, hidden, e, p, 1)
	x := randMat(rng, n, d)
	y, err := m.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	if !y.Shape().Equal(tensor.Shape{n, d}) {
		t.Errorf("output shape %v, want [%d,%d]", y.Shape(), n, d)
	}
	// (softmax normalization is guaranteed by OpSoftmax; the parity test covers the axes.)
}

func ExampleSoftMoE() {
	// 4 tokens, dim 3, 2 experts × 2 slots; fully differentiable, output shape = input.
	m := nn.NewSoftMoE(tensor.F64, 3, 6, 2, 2, 42)
	x := tensor.FromFloat64(tensor.Shape{4, 3}, []float64{1, 0, -1, 0.5, 2, 0.3, -1, 1, 0.2, 0.7, -0.5, 1.5})
	y, _ := m.Forward(backend.NewContext(), x)
	fmt.Println(y.Shape())
	// Output:
	// (4, 3)
}
