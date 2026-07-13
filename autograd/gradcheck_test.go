package autograd_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// Systematic numeric gradient checking (§V2): for every differentiable L1 op,
// the analytic gradient from Backward must match central finite differences of
// a scalar loss within rel 1e-5 (spec bound 1e-4). Loss = Sum(op(...)) so the
// Sum VJP participates everywhere; inputs avoid kinks (relu 0) and ties (max).

type gradCase struct {
	name   string
	inputs [][]float64    // raw data per input
	shapes []tensor.Shape // shape per input
	fwd    func(ctx *backend.Context, xs []*tensor.Tensor) (*tensor.Tensor, error)
}

func exec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, xs ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, xs, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

func opCase(name string, op backend.Op, attrs backend.Attrs, shapes []tensor.Shape, inputs ...[]float64) gradCase {
	return gradCase{
		name: name, inputs: inputs, shapes: shapes,
		fwd: func(ctx *backend.Context, xs []*tensor.Tensor) (*tensor.Tensor, error) {
			return exec(ctx, op, attrs, xs...)
		},
	}
}

func cases() []gradCase {
	v3 := tensor.Shape{3}
	a := []float64{0.8, -1.2, 2.1}  // generic, away from relu kink
	b := []float64{1.7, 0.9, -2.3}  // generic
	pos := []float64{0.6, 1.4, 3.2} // log domain
	m23 := tensor.Shape{2, 3}
	m34 := tensor.Shape{3, 4} // distinct values → no max/min ties
	am := []float64{1.1, -2.2, 0.7, 3.3, -0.4, 1.9}
	bm := []float64{0.5, 1.5, -1.0, 2.0, 0.3, -0.8, 1.2, -2.1, 0.9, 1.8, -1.4, 0.6}

	return []gradCase{
		opCase("div", backend.OpDiv, nil, []tensor.Shape{v3, v3}, a, b),
		opCase("exp", backend.OpExp, nil, []tensor.Shape{v3}, a),
		opCase("log", backend.OpLog, nil, []tensor.Shape{v3}, pos),
		opCase("tanh", backend.OpTanh, nil, []tensor.Shape{v3}, a),
		opCase("relu", backend.OpReLU, nil, []tensor.Shape{v3}, a),
		opCase("gelu", backend.OpGELU, nil, []tensor.Shape{v3}, a),
		opCase("sigmoid", backend.OpSigmoid, nil, []tensor.Shape{v3}, a),
		opCase("matmul", backend.OpMatMul, nil, []tensor.Shape{m23, m34}, am, bm),
		opCase("dot", backend.OpDot, nil, []tensor.Shape{v3, v3}, a, b),
		opCase("nrm2", backend.OpNrm2, nil, []tensor.Shape{v3}, a),
		opCase("axpy", backend.OpAXPY, backend.AXPYAttrs{Alpha: 2.5}, []tensor.Shape{v3, v3}, a, b),
		opCase("sum_all", backend.OpSum, nil, []tensor.Shape{m23}, am),
		opCase("sum_axis0", backend.OpSum, backend.ReduceAttrs{Axes: []int{0}}, []tensor.Shape{m23}, am),
		opCase("sum_axis1_keep", backend.OpSum, backend.ReduceAttrs{Axes: []int{1}, KeepDims: true}, []tensor.Shape{m23}, am),
		opCase("mean_all", backend.OpMean, nil, []tensor.Shape{m23}, am),
		opCase("mean_axis1", backend.OpMean, backend.ReduceAttrs{Axes: []int{1}}, []tensor.Shape{m23}, am),
		opCase("max_axis0", backend.OpMax, backend.ReduceAttrs{Axes: []int{0}}, []tensor.Shape{m23}, am),
		opCase("min_axis1", backend.OpMin, backend.ReduceAttrs{Axes: []int{1}}, []tensor.Shape{m23}, am),
		// CV (§T24b): distinct values → no max-pool ties near h
		opCase("conv2d", backend.OpConv2D, backend.ConvAttrs{Stride: 1, Pad: 0},
			[]tensor.Shape{{1, 2, 3, 3}, {2, 2, 2, 2}, {2}}, cvX, cvW, []float64{0.3, -0.6}),
		opCase("conv2d_s2p1", backend.OpConv2D, backend.ConvAttrs{Stride: 2, Pad: 1},
			[]tensor.Shape{{1, 2, 3, 3}, {2, 2, 2, 2}}, cvX, cvW),
		opCase("maxpool", backend.OpMaxPool2D, backend.PoolAttrs{Kernel: 2},
			[]tensor.Shape{{1, 1, 4, 4}}, cvPool),
		opCase("avgpool", backend.OpAvgPool2D, backend.PoolAttrs{Kernel: 2, Stride: 1},
			[]tensor.Shape{{1, 1, 4, 4}}, cvPool),
		// transformer training (§T31): layernorm over last axis, grads for x,γ,β
		opCase("layernorm", backend.OpLayerNorm, backend.NormAttrs{Eps: 1e-5},
			[]tensor.Shape{{2, 4}, {4}, {4}}, lnX, lnGamma, lnBeta),
		opCase("softmax", backend.OpSoftmax, nil, []tensor.Shape{{2, 4}}, lnX),
		// fused attention (§T32): grads for Q,K,V, both mask modes
		opCase("mha", backend.OpMHA, backend.AttnAttrs{Heads: 2, Causal: false},
			[]tensor.Shape{{3, 4}, {3, 4}, {3, 4}}, mhaQ, mhaK, mhaV),
		opCase("mha_causal", backend.OpMHA, backend.AttnAttrs{Heads: 2, Causal: true},
			[]tensor.Shape{{3, 4}, {3, 4}, {3, 4}}, mhaQ, mhaK, mhaV),
		// llama-family (§T38): RMSNorm (x,γ) grads; RoPE (q) grad
		opCase("rmsnorm", backend.OpRMSNorm, backend.NormAttrs{Eps: 1e-5},
			[]tensor.Shape{{2, 4}, {4}}, rmsX, rmsGamma),
		opCase("rope", backend.OpRoPE, backend.RoPEAttrs{Base: 10000.0},
			[]tensor.Shape{{3, 4}}, ropeQ),
		opCase("silu", backend.OpSiLU, nil, []tensor.Shape{{5}}, []float64{-1.5, -0.3, 0.1, 0.7, 2.1}),
		// GQA (§T38b): 4 query heads, 2 kv heads, dk=3
		opCase("gqa", backend.OpMHA, backend.AttnAttrs{Heads: 4, KVHeads: 2},
			[]tensor.Shape{{3, 12}, {3, 6}, {3, 6}}, ramp(36, 1), ramp(18, 2), ramp(18, 3)),
	}
}

func ramp(n, seed int) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = math.Sin(float64(i*7+seed) * 0.3) // deterministic, well-varied
	}
	return out
}

var (
	rmsX     = []float64{0.5, -1.2, 2.1, 0.3, 1.4, -0.7, 0.9, -2.0}
	rmsGamma = []float64{1.1, 0.8, 1.3, 0.6}
	ropeQ    = []float64{0.3, -0.7, 1.1, 0.5, -1.2, 0.8, 0.2, -0.4, 0.9, 1.3, -0.6, 0.1}
)

var (
	mhaQ = []float64{0.3, -0.7, 1.1, 0.5, -1.2, 0.8, 0.2, -0.4, 0.9, 1.3, -0.6, 0.1}
	mhaK = []float64{0.6, 0.2, -0.9, 1.0, 0.4, -1.1, 0.7, 0.3, -0.5, 0.8, 1.2, -0.2}
	mhaV = []float64{-0.3, 1.4, 0.6, -0.8, 0.9, 0.1, -1.0, 0.5, 0.2, -0.7, 1.1, 0.4}
)

var (
	lnX     = []float64{0.5, -1.2, 2.1, 0.3, 1.4, -0.7, 0.9, -2.0}
	lnGamma = []float64{1.1, 0.8, 1.3, 0.6}
	lnBeta  = []float64{0.2, -0.4, 0.1, 0.5}
)

var (
	cvX = []float64{0.5, -1.1, 0.8, 1.9, -0.4, 0.2, -1.6, 0.7, 1.2,
		-0.9, 1.4, 0.1, -0.3, 2.1, -1.8, 0.6, 0.9, -0.2}
	cvW = []float64{0.7, -0.5, 1.1, 0.4, -1.2, 0.9, 0.3, -0.8,
		1.5, -0.1, 0.6, -1.4, 0.2, 1.0, -0.7, 0.8}
	cvPool = []float64{1, 7, 3, 12, 5, 2, 9, 4, 14, 6, 11, 8, 10, 15, 13, 16}
)

// scalarLoss runs fwd and reduces to a scalar via Sum if needed.
func scalarLoss(ctx *backend.Context, c gradCase, xs []*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := c.fwd(ctx, xs)
	if err != nil {
		return nil, err
	}
	if out.Numel() == 1 && out.Ndim() == 0 {
		return out, nil
	}
	return exec(ctx, backend.OpSum, nil, out)
}

func buildInputs(c gradCase) []*tensor.Tensor {
	xs := make([]*tensor.Tensor, len(c.inputs))
	for i, d := range c.inputs {
		xs[i] = tensor.FromFloat64(c.shapes[i], append([]float64(nil), d...))
	}
	return xs
}

func TestGradCheckAllOps(t *testing.T) {
	const h = 1e-6
	const rtol = 1e-5 // spec bound is 1e-4 (§V2); we hold a stricter line

	for _, c := range cases() {
		t.Run(c.name, func(t *testing.T) {
			// analytic
			tape := autograd.NewTape()
			ctx := tape.Context()
			xs := buildInputs(c)
			loss, err := scalarLoss(ctx, c, xs)
			if err != nil {
				t.Fatal(err)
			}
			if err := tape.Backward(loss); err != nil {
				t.Fatal(err)
			}

			// numeric: perturb every element of every input
			eval := func(data [][]float64) float64 {
				exs := make([]*tensor.Tensor, len(data))
				for i, d := range data {
					exs[i] = tensor.FromFloat64(c.shapes[i], d)
				}
				out, err := scalarLoss(backend.NewContext(), c, exs)
				if err != nil {
					t.Fatal(err)
				}
				return out.AtF64()
			}
			for i := range c.inputs {
				grad := tape.Grad(xs[i])
				if grad == nil {
					t.Fatalf("input %d: nil grad", i)
				}
				for j := range c.inputs[i] {
					plus := cloneAll(c.inputs)
					minus := cloneAll(c.inputs)
					plus[i][j] += h
					minus[i][j] -= h
					numeric := (eval(plus) - eval(minus)) / (2 * h)
					idx := tensor.Unravel(j, c.shapes[i])
					analytic := grad.AtF64(idx...)
					if math.Abs(numeric-analytic) > rtol*math.Max(1, math.Abs(numeric)) {
						t.Errorf("input %d elem %d: numeric %.10g vs analytic %.10g", i, j, numeric, analytic)
					}
				}
			}
		})
	}
}

func cloneAll(in [][]float64) [][]float64 {
	out := make([][]float64, len(in))
	for i, d := range in {
		out[i] = append([]float64(nil), d...)
	}
	return out
}

// argmax is non-differentiable: its input receives no gradient, and Backward
// through it must not fail.
func TestArgmaxNonDifferentiable(t *testing.T) {
	tape := autograd.NewTape()
	ctx := tape.Context()
	x := tensor.FromFloat64(tensor.Shape{3}, []float64{1, 5, 2})
	am, err := exec(ctx, backend.OpArgMax, nil, x)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(am); err != nil {
		t.Fatal(err)
	}
	if tape.Grad(x) != nil {
		t.Error("argmax input must receive nil grad")
	}
}
