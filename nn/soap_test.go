package nn_test

import (
	"fmt"
	"math"
	"testing"

	intlinalg "github.com/jxsl13/goai/internal/linalg"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// §V16 tier-1: for a 1×1 parameter the eigenbasis is trivial (Q=[±1], rotation cancels),
// so SOAP reduces EXACTLY to Adam. We run SOAP against an independent Adam recurrence on
// the same convex quadratic and require identical trajectories.
func TestSOAPReducesToAdam1x1(t *testing.T) {
	const b1, b2, eps, lr = 0.9, 0.999, 1e-8, 0.05
	const target = 2.0
	w := tensor.FromFloat64(tensor.Shape{1, 1}, []float64{5})
	opt := nn.NewSOAP([]*tensor.Tensor{w}, lr, nn.WithSOAPBetas(b1, b2), nn.WithSOAPEps(eps), nn.WithSOAPFreq(3))

	// independent Adam on the same scalar
	wa, m, v := 5.0, 0.0, 0.0
	for step := 1; step <= 30; step++ {
		if err := opt.Step(func(p *tensor.Tensor) *tensor.Tensor {
			g := tensor.New(tensor.F64, p.Shape())
			g.SetF64(2*(p.AtF64(0, 0)-target), 0, 0)
			return g
		}); err != nil {
			t.Fatal(err)
		}
		ga := 2 * (wa - target)
		m = b1*m + (1-b1)*ga
		v = b2*v + (1-b2)*ga*ga
		mh := m / (1 - math.Pow(b1, float64(step)))
		vh := v / (1 - math.Pow(b2, float64(step)))
		wa -= lr * mh / (math.Sqrt(vh) + eps)
		if math.Abs(w.AtF64(0, 0)-wa) > 1e-9 {
			t.Fatalf("step %d: SOAP %.12g vs Adam %.12g", step, w.AtF64(0, 0), wa)
		}
	}
}

// §V16 tier-1: one SOAP step on a matrix matches an independent recomputation of the full
// pipeline (L/R EMA → eigenbasis → rotate → Adam → rotate back), using the shared
// eigendecomposition and manual matrix multiplies.
func TestSOAPOneStepParity(t *testing.T) {
	const b1, b2, eps, lr = 0.9, 0.99, 1e-8, 0.1
	m, n := 3, 2
	w0 := []float64{1, -2, 0.5, 3, -1, 0.25}
	w := tensor.FromFloat64(tensor.Shape{m, n}, w0)
	G := [][]float64{{0.5, -1}, {2, 0.3}, {-0.4, 1.1}}
	opt := nn.NewSOAP([]*tensor.Tensor{w}, lr, nn.WithSOAPBetas(b1, b2), nn.WithSOAPEps(eps))
	if err := opt.Step(func(p *tensor.Tensor) *tensor.Tensor { return matTo(G) }); err != nil {
		t.Fatal(err)
	}

	// independent recompute (step 1): L=(1−β2)GGᵀ, R=(1−β2)GᵀG, Q=eig, M=(1−β1)G', V=(1−β2)G'²
	l := scaleOuter(G, false, 1-b2) // (1−β2)·GGᵀ  [m×m]
	r := scaleOuter(G, true, 1-b2)  // (1−β2)·GᵀG  [n×n]
	ql, qr := eigBasis(l), eigBasis(r)
	gp := rotFwd(ql, G, qr)
	nprime := make([][]float64, m)
	for i := range m {
		nprime[i] = make([]float64, n)
		for j := range n {
			mm := (1 - b1) * gp[i][j]
			vv := (1 - b2) * gp[i][j] * gp[i][j]
			nprime[i][j] = (mm / (1 - b1)) / (math.Sqrt(vv/(1-b2)) + eps)
		}
	}
	upd := rotBack(ql, nprime, qr)
	for i := range m {
		for j := range n {
			want := w0[i*n+j] - lr*upd[i][j]
			// relative tolerance: amd64 and arm64 libm/FMA differ in the last
			// digit of the eigendecomposition chain (CI finding, §T565)
			if math.Abs(w.AtF64(i, j)-want) > 1e-8*math.Max(1, math.Abs(want)) {
				t.Errorf("W[%d,%d] = %.10g, want %.10g", i, j, w.AtF64(i, j), want)
			}
		}
	}
}

// §V2/convergence: SOAP minimizes a convex quadratic on a matrix parameter (crossing
// several eigenbasis refreshes).
func TestSOAPConverges(t *testing.T) {
	w := tensor.FromFloat64(tensor.Shape{3, 3}, []float64{4, -2, 5, 1, -3, 2, 0, 6, -1})
	target := make([]float64, 9)
	for i := range target {
		target[i] = 1
	}
	opt := nn.NewSOAP([]*tensor.Tensor{w}, 0.1, nn.WithSOAPFreq(5))
	loss := func() float64 {
		var s float64
		for i := range 9 {
			idx := tensor.Unravel(i, w.Shape())
			d := w.AtF64(idx...) - target[i]
			s += d * d
		}
		return s
	}
	first := loss()
	for range 500 {
		if err := opt.Step(func(p *tensor.Tensor) *tensor.Tensor {
			g := tensor.New(tensor.F64, p.Shape())
			for i := range 9 {
				idx := tensor.Unravel(i, p.Shape())
				g.SetF64(2*(p.AtF64(idx...)-target[i]), idx...)
			}
			return g
		}); err != nil {
			t.Fatal(err)
		}
	}
	if last := loss(); last > 0.01*first {
		t.Errorf("did not converge: loss %.4g → %.4g", first, last)
	}
}

func ExampleSOAP() {
	w := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, 1, 1, 1})
	opt := nn.NewSOAP([]*tensor.Tensor{w}, 0.1)
	// One step toward target 0 (grad = 2w). Unlike Adam, SOAP concentrates the update in
	// the dominant eigen-direction of the (rank-1) gradient and spreads it on rotate-back,
	// so each entry moves by 0.05 rather than the Adam-like 0.1.
	_ = opt.Step(func(p *tensor.Tensor) *tensor.Tensor {
		g := tensor.New(tensor.F64, p.Shape())
		for i := range 2 {
			for j := range 2 {
				g.SetF64(2*p.AtF64(i, j), i, j)
			}
		}
		return g
	})
	fmt.Printf("%.3f\n", w.AtF64(0, 0))
	// Output:
	// 0.950
}

// --- independent test helpers (manual, not the implementation's) ---

func matTo(g [][]float64) *tensor.Tensor {
	r, c := len(g), len(g[0])
	t := tensor.New(tensor.F64, tensor.Shape{r, c})
	for i := range r {
		for j := range c {
			t.SetF64(g[i][j], i, j)
		}
	}
	return t
}

// scaleOuter returns s·GᵀG (transpose=true) or s·GGᵀ (false).
func scaleOuter(g [][]float64, transpose bool, s float64) [][]float64 {
	m, n := len(g), len(g[0])
	if transpose {
		out := make([][]float64, n)
		for i := range n {
			out[i] = make([]float64, n)
			for j := range n {
				var acc float64
				for k := range m {
					acc += g[k][i] * g[k][j]
				}
				out[i][j] = s * acc
			}
		}
		return out
	}
	out := make([][]float64, m)
	for i := range m {
		out[i] = make([]float64, m)
		for j := range m {
			var acc float64
			for k := range n {
				acc += g[i][k] * g[j][k]
			}
			out[i][j] = s * acc
		}
	}
	return out
}

func eigBasis(mat [][]float64) [][]float64 {
	_, vecs := intlinalg.SymEig(mat)
	n := len(mat)
	q := make([][]float64, n)
	for i := range n {
		q[i] = make([]float64, n)
	}
	for k := range n {
		for i := range n {
			q[i][k] = vecs[k][i]
		}
	}
	return q
}

func rotFwd(ql, g, qr [][]float64) [][]float64 { // Q_Lᵀ G Q_R
	m, n := len(ql), len(qr)
	out := make([][]float64, m)
	for k := range m {
		out[k] = make([]float64, n)
		for l := range n {
			var acc float64
			for i := range m {
				for j := range n {
					acc += ql[i][k] * g[i][j] * qr[j][l]
				}
			}
			out[k][l] = acc
		}
	}
	return out
}

func rotBack(ql, nm, qr [][]float64) [][]float64 { // Q_L N Q_Rᵀ
	m, n := len(ql), len(qr)
	out := make([][]float64, m)
	for i := range m {
		out[i] = make([]float64, n)
		for j := range n {
			var acc float64
			for k := range m {
				for l := range n {
					acc += ql[i][k] * nm[k][l] * qr[j][l]
				}
			}
			out[i][j] = acc
		}
	}
	return out
}
