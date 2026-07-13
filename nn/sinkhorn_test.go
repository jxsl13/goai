package nn_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

func rowSum(p *tensor.Tensor, i int) float64 {
	var s float64
	for j := range p.Shape()[1] {
		s += p.AtF64(i, j)
	}
	return s
}

func colSum(p *tensor.Tensor, j int) float64 {
	var s float64
	for i := range p.Shape()[0] {
		s += p.AtF64(i, j)
	}
	return s
}

// marginalErr = max |rowSum−r| + max |colSum−c|.
func marginalErr(p *tensor.Tensor, r, c []float64) float64 {
	var e float64
	for i := range r {
		e = math.Max(e, math.Abs(rowSum(p, i)-r[i]))
	}
	for j := range c {
		e = math.Max(e, math.Abs(colSum(p, j)-c[j]))
	}
	return e
}

// §V16 tier-1 (paper CONFIRMED, arXiv:1306.0895): after enough Sinkhorn-Knopp iterations
// the plan's row sums converge to r and column sums to c.
func TestSinkhornMarginals(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 895))
	const m, n = 5, 4
	cost := randMat(rng, m, n) // any cost
	r := uniform(m)
	c := uniform(n)
	p, err := nn.Sinkhorn(cost, r, c, 0.5, 200)
	if err != nil {
		t.Fatal(err)
	}
	if e := marginalErr(p, r, c); e > 1e-6 {
		t.Errorf("marginal error %g after 200 iters, want <1e-6", e)
	}
}

// §V2: the transport plan is nonnegative.
func TestSinkhornNonnegative(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 896))
	cost := randMat(rng, 4, 6)
	p, err := nn.Sinkhorn(cost, uniform(4), uniform(6), 0.3, 100)
	if err != nil {
		t.Fatal(err)
	}
	for i := range p.Numel() {
		if v := p.AtF64(tensor.Unravel(i, p.Shape())...); v < 0 {
			t.Fatalf("negative plan entry %g", v)
		}
	}
}

// §V16 tier-1 (closed form): a zero cost matrix has a uniform Gibbs kernel, so the plan is
// exactly the independent coupling P = r·cᵀ (outer product).
func TestSinkhornZeroCostOuterProduct(t *testing.T) {
	r := []float64{0.5, 0.3, 0.2}
	c := []float64{0.25, 0.25, 0.5}
	cost := tensor.New(tensor.F64, tensor.Shape{3, 3}) // all zeros
	p, err := nn.Sinkhorn(cost, r, c, 1.0, 50)
	if err != nil {
		t.Fatal(err)
	}
	for i := range 3 {
		for j := range 3 {
			if want := r[i] * c[j]; math.Abs(p.AtF64(i, j)-want) > 1e-9 {
				t.Errorf("P[%d,%d]=%g want r·c=%g", i, j, p.AtF64(i, j), want)
			}
		}
	}
}

// §V2: more iterations reduce the marginal violation (Sinkhorn-Knopp converges).
func TestSinkhornConvergence(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 897))
	cost := randMat(rng, 5, 5)
	r, c := uniform(5), uniform(5)
	few, _ := nn.Sinkhorn(cost, r, c, 0.2, 1)
	many, _ := nn.Sinkhorn(cost, r, c, 0.2, 50)
	if e1, e50 := marginalErr(few, r, c), marginalErr(many, r, c); !(e50 < e1) {
		t.Errorf("marginal error did not decrease: 1 iter %g, 50 iter %g", e1, e50)
	}
}

// §V2 (equipartition / SwAV): uniform square marginals give a doubly-stochastic plan whose
// rows and columns each sum to 1/n.
func TestSinkhornDoublyStochastic(t *testing.T) {
	rng := rand.New(rand.NewPCG(4, 898))
	const n = 6
	cost := randMat(rng, n, n)
	p, err := nn.Sinkhorn(cost, uniform(n), uniform(n), 0.4, 200)
	if err != nil {
		t.Fatal(err)
	}
	for i := range n {
		if math.Abs(rowSum(p, i)-1.0/n) > 1e-6 || math.Abs(colSum(p, i)-1.0/n) > 1e-6 {
			t.Errorf("row/col %d sums = %g/%g, want %g", i, rowSum(p, i), colSum(p, i), 1.0/n)
		}
	}
}

func TestSinkhornErrors(t *testing.T) {
	r := func(shape ...int) *tensor.Tensor { return tensor.New(tensor.F64, tensor.Shape(shape)) }
	if _, err := nn.Sinkhorn(r(3, 4, 1), uniform(3), uniform(4), 0.5, 10); err == nil {
		t.Error("rank-3 cost: want error")
	}
	if _, err := nn.Sinkhorn(r(3, 4), uniform(2), uniform(4), 0.5, 10); err == nil {
		t.Error("wrong r length: want error")
	}
	if _, err := nn.Sinkhorn(r(3, 4), uniform(3), uniform(4), 0, 10); err == nil {
		t.Error("eps=0: want error")
	}
	if _, err := nn.Sinkhorn(r(3, 4), uniform(3), uniform(4), 0.5, 0); err == nil {
		t.Error("iters<1: want error")
	}
	if _, err := nn.Sinkhorn(r(3, 3), []float64{0.5, 0.5, 0}, []float64{1, 1, 1}, 0.5, 10); err == nil {
		t.Error("unequal total mass: want error")
	}
}

func uniform(n int) []float64 {
	u := make([]float64, n)
	for i := range u {
		u[i] = 1.0 / float64(n)
	}
	return u
}

func ExampleSinkhorn() {
	// Transport a uniform 2-bin source to a uniform 2-bin target under a cost that makes the
	// diagonal cheap: the plan concentrates mass on the diagonal.
	cost := tensor.New(tensor.F64, tensor.Shape{2, 2})
	cost.SetF64(0, 0, 0) // cheap
	cost.SetF64(5, 0, 1) // expensive off-diagonal
	cost.SetF64(5, 1, 0)
	cost.SetF64(0, 1, 1) // cheap
	p, _ := nn.Sinkhorn(cost, []float64{0.5, 0.5}, []float64{0.5, 0.5}, 0.1, 100)
	fmt.Printf("diag=%.2f off=%.2f\n", p.AtF64(0, 0), p.AtF64(0, 1))
	// Output:
	// diag=0.50 off=0.00
}
