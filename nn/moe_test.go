package nn_test

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

func loadMoE(t *testing.T) (alpha float64, shape []int, logits, assign []float64, loss float64) {
	t.Helper()
	raw, err := os.ReadFile("testdata/moe.json")
	if err != nil {
		t.Fatalf("read golden (run make golden): %v", err)
	}
	var g struct {
		Alpha       float64   `json:"alpha"`
		Shape       []int     `json:"shape"`
		Logits      []float64 `json:"logits"`
		Assignments []float64 `json:"assignments"`
		Loss        float64   `json:"loss"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g.Alpha, g.Shape, g.Logits, g.Assignments, g.Loss
}

// §V1 + §V16: the fused MoE load-balancing loss matches an INDEPENDENT numpy
// α·N·Σ f_i·P_i computation (Switch Transformer eq.4) @ f64 rtol 1e-12.
func TestMoEBalanceParity(t *testing.T) {
	alpha, shape, logits, assign, want := loadMoE(t)
	gl := tensor.FromFloat64(tensor.Shape{shape[0], shape[1]}, logits)
	as := tensor.FromFloat64(tensor.Shape{shape[0]}, assign)
	got, err := nn.MoEBalanceLoss(backend.NewContext(), gl, as, alpha)
	if err != nil {
		t.Fatal(err)
	}
	if v := got.AtF64(); math.Abs(v-want) > 1e-12*math.Max(1, math.Abs(want)) {
		t.Errorf("MoE balance loss = %.17g, want %.17g", v, want)
	}
}

// §V2: central finite differences of the loss w.r.t. the gate logits match the
// analytic VJP (which flows only through the differentiable P_i); the hard
// dispatch assignments get no gradient.
func TestMoEBalanceGradCheck(t *testing.T) {
	alpha, shape, logits, assign, _ := loadMoE(t)
	m, n := shape[0], shape[1]
	as := tensor.FromFloat64(tensor.Shape{m}, assign)

	lossAt := func(ld []float64) float64 {
		out, err := nn.MoEBalanceLoss(backend.NewContext(), tensor.FromFloat64(tensor.Shape{m, n}, ld), as, alpha)
		if err != nil {
			t.Fatal(err)
		}
		return out.AtF64()
	}

	gl := tensor.FromFloat64(tensor.Shape{m, n}, logits)
	tape := autograd.NewTape()
	loss, err := nn.MoEBalanceLoss(tape.Context(), gl, as, alpha)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	if tape.Grad(as) != nil {
		t.Error("assignments must be non-differentiable (nil gradient)")
	}
	ggl := tape.Grad(gl)
	const h = 1e-6
	work := append([]float64(nil), logits...)
	for k := range logits {
		o := work[k]
		work[k] = o + h
		lp := lossAt(work)
		work[k] = o - h
		lm := lossAt(work)
		work[k] = o
		if num, ana := (lp-lm)/(2*h), ggl.AtF64(k/n, k%n); math.Abs(num-ana) > 1e-4*math.Max(1e-6, math.Abs(num)) {
			t.Errorf("grad[%d]: numeric %.8g vs analytic %.8g", k, num, ana)
		}
	}
}

// §R61: the loss equals α at perfectly uniform routing (its minimum) and rises
// toward α·N when both the router and the dispatch concentrate on one expert.
func TestMoEBalanceUniformIsMinimum(t *testing.T) {
	const N = 4
	// balanced: uniform logits (P=1/N) + one token per expert (f=1/N) → loss = α
	balLogits := tensor.New(tensor.F64, tensor.Shape{N, N}) // all zeros → uniform softmax
	balAssign := tensor.FromFloat64(tensor.Shape{N}, []float64{0, 1, 2, 3})
	bal, _ := nn.MoEBalanceLoss(backend.NewContext(), balLogits, balAssign, 1.0)
	if math.Abs(bal.AtF64()-1.0) > 1e-12 {
		t.Errorf("balanced loss = %.12g, want 1 (= α·N·1/N minimum)", bal.AtF64())
	}

	// concentrated: router certain on expert 0 + all tokens dispatched to 0
	concLogits := tensor.New(tensor.F64, tensor.Shape{N, N})
	for tk := range N {
		concLogits.SetF64(20, tk, 0) // huge logit on expert 0 → P≈[1,0,0,0]
	}
	concAssign := tensor.New(tensor.F64, tensor.Shape{N}) // all 0
	conc, _ := nn.MoEBalanceLoss(backend.NewContext(), concLogits, concAssign, 1.0)
	if conc.AtF64() <= bal.AtF64() {
		t.Errorf("concentrated loss %.4f must exceed balanced %.4f", conc.AtF64(), bal.AtF64())
	}
	if math.Abs(conc.AtF64()-float64(N)) > 0.01 { // approaches α·N = 4
		t.Errorf("concentrated loss %.4f should approach α·N = %d", conc.AtF64(), N)
	}
}

// TopKGating selects the k highest-probability experts (highest first) and returns
// gate weights that sum to 1 (renormalized over the selection).
func TestTopKGating(t *testing.T) {
	logits := []float64{1.0, 3.0, 0.5, 2.0} // order by prob: 1 (3.0), 3 (2.0), 0, 2
	experts, weights := nn.TopKGating(logits, 2)
	if len(experts) != 2 || experts[0] != 1 || experts[1] != 3 {
		t.Errorf("top-2 experts = %v, want [1 3]", experts)
	}
	var s float64
	for _, w := range weights {
		s += w
	}
	if math.Abs(s-1.0) > 1e-12 {
		t.Errorf("gate weights sum = %.12g, want 1 (renormalized)", s)
	}
	// higher-logit expert gets the larger weight
	if weights[0] <= weights[1] {
		t.Errorf("weights not descending: %v", weights)
	}
	// weights equal softmax(top-k logits) renormalized: exp(3)/(exp(3)+exp(2)) …
	w0 := math.Exp(3) / (math.Exp(3) + math.Exp(2))
	if math.Abs(weights[0]-w0) > 1e-12 {
		t.Errorf("weight[0] = %.12g, want softmax-over-topk %.12g", weights[0], w0)
	}
}
