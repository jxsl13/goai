package nn_test

import (
	"encoding/json"
	"fmt"
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

type cpoGolden struct {
	SumW  []float64 `json:"sum_w"`
	SumL  []float64 `json:"sum_l"`
	Beta  float64   `json:"beta"`
	Alpha float64   `json:"alpha"`
	CPO   float64   `json:"cpo"`
}

func loadCPO(t *testing.T) cpoGolden {
	t.Helper()
	raw, err := os.ReadFile("testdata/cpo.json")
	if err != nil {
		t.Fatalf("read golden (run `make golden`): %v", err)
	}
	var g cpoGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

// §V16 tier-1: CPO matches the independent numpy reference at f64 1e-12
// (Xu et al. 2024, §R121; summed log-probs, β=0.1, α=1).
func TestCPOParity(t *testing.T) {
	g := loadCPO(t)
	out, err := nn.CPO(backend.NewContext(), vec(g.SumW), vec(g.SumL), nn.Beta(g.Beta), nn.CPOAlpha(g.Alpha))
	if err != nil {
		t.Fatal(err)
	}
	if v := out.AtF64(); math.Abs(v-g.CPO) > 1e-12*math.Max(1, math.Abs(g.CPO)) {
		t.Errorf("CPO = %.17g, want %.17g", v, g.CPO)
	}
}

// §V16 tier-1: CPO with α=0 reduces to the bare reference-free preference term
// −log σ(β(lw−ll)) = softplus(−β(lw−ll)), verified against a hand recomputation.
func TestCPOAlphaZeroIsPreferenceOnly(t *testing.T) {
	g := loadCPO(t)
	out, err := nn.CPO(backend.NewContext(), vec(g.SumW), vec(g.SumL), nn.Beta(g.Beta), nn.CPOAlpha(0))
	if err != nil {
		t.Fatal(err)
	}
	var want float64
	for i := range g.SumW {
		z := g.Beta * (g.SumW[i] - g.SumL[i])
		want += math.Log1p(math.Exp(-z)) // softplus(−z), z can be small → stable enough here
	}
	want /= float64(len(g.SumW))
	if v := out.AtF64(); math.Abs(v-want) > 1e-12*math.Max(1, math.Abs(want)) {
		t.Errorf("CPO(α=0) = %.17g, want preference-only %.17g", v, want)
	}
}

// §V2: finite differences of the CPO loss w.r.t. the chosen and rejected summed
// log-probs match the analytic VJP.
func TestCPOGradCheck(t *testing.T) {
	g := loadCPO(t)
	loss := func(ctx *backend.Context, w, l *tensor.Tensor) (*tensor.Tensor, error) {
		return nn.CPO(ctx, w, l, nn.Beta(g.Beta), nn.CPOAlpha(g.Alpha))
	}
	wData := append([]float64(nil), g.SumW...)
	lData := append([]float64(nil), g.SumL...)
	lossAt := func() float64 {
		out, err := loss(backend.NewContext(), vec(wData), vec(lData))
		if err != nil {
			t.Fatal(err)
		}
		return out.AtF64()
	}
	wT, lT := vec(wData), vec(lData)
	tape := autograd.NewTape()
	out, err := loss(tape.Context(), wT, lT)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(out); err != nil {
		t.Fatal(err)
	}
	const h = 1e-6
	sides := []struct {
		name string
		d    []float64
		grad *tensor.Tensor
	}{{"w", wData, tape.Grad(wT)}, {"l", lData, tape.Grad(lT)}}
	for _, side := range sides {
		for i := range side.d {
			o := side.d[i]
			side.d[i] = o + h
			lp := lossAt()
			side.d[i] = o - h
			lm := lossAt()
			side.d[i] = o
			if num, ana := (lp-lm)/(2*h), side.grad.AtF64(i); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("cpo %s grad[%d]: numeric %.8g vs analytic %.8g", side.name, i, num, ana)
			}
		}
	}
}

// §V16 tier-1 (behavioral): a gradient step on the CPO loss must raise the chosen
// summed log-prob relative to the rejected one AND push the chosen NLL down
// (both the preference term and the BC regularizer act in the right direction).
func TestCPOTrainsPreference(t *testing.T) {
	w0 := []float64{-9.0, -12.0}
	l0 := []float64{-9.5, -11.0}
	wT, lT := vec(append([]float64(nil), w0...)), vec(append([]float64(nil), l0...))
	tape := autograd.NewTape()
	out, err := nn.CPO(tape.Context(), wT, lT)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(out); err != nil {
		t.Fatal(err)
	}
	gw, gl := tape.Grad(wT), tape.Grad(lT)
	// gradient descent raises chosen log-prob (−grad > 0) and lowers rejected.
	for i := range w0 {
		if gw.AtF64(i) >= 0 {
			t.Errorf("chosen grad[%d]=%g should be negative (descent raises logπ_chosen)", i, gw.AtF64(i))
		}
		if gl.AtF64(i) <= 0 {
			t.Errorf("rejected grad[%d]=%g should be positive (descent lowers logπ_rejected)", i, gl.AtF64(i))
		}
	}
}

func TestCPOErrors(t *testing.T) {
	// mismatched batch lengths
	if _, err := nn.CPO(backend.NewContext(), vec([]float64{-1, -2}), vec([]float64{-1})); err == nil {
		t.Error("mismatched chosen/rejected lengths must error")
	}
	// rank-2 input
	bad := tensor.FromFloat64(tensor.Shape{2, 1}, []float64{-1, -2})
	if _, err := nn.CPO(backend.NewContext(), bad, bad); err == nil {
		t.Error("non rank-1 input must error")
	}
}

// CPO (Xu et al. 2024) aligns a model from preference pairs with NO reference
// model, adding an NLL behavior-cloning term on the chosen response. Here the
// chosen response has summed log-prob −2 vs the rejected −4; with β=0.1 the
// preference term is softplus(−0.1·2)=softplus(−0.2) and the BC term is −(−2)=2.
func ExampleCPO() {
	chosen := tensor.FromFloat64(tensor.Shape{1}, []float64{-2.0})
	rejected := tensor.FromFloat64(tensor.Shape{1}, []float64{-4.0})

	loss, _ := nn.CPO(backend.NewContext(), chosen, rejected, nn.Beta(0.1), nn.CPOAlpha(1))
	fmt.Printf("%.4f\n", loss.AtF64())
	// Output: 2.5981
}
