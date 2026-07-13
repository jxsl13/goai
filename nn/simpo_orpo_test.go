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

type simpoOrpoGolden struct {
	AvgW   []float64 `json:"avg_w"`
	AvgL   []float64 `json:"avg_l"`
	Beta   float64   `json:"beta"`
	Gamma  float64   `json:"gamma"`
	Lambda float64   `json:"lambda"`
	SimPO  float64   `json:"simpo"`
	ORPO   float64   `json:"orpo"`
}

func loadSimPOORPO(t *testing.T) simpoOrpoGolden {
	t.Helper()
	raw, err := os.ReadFile("testdata/simpo_orpo.json")
	if err != nil {
		t.Fatalf("read golden (run `make golden`): %v", err)
	}
	var g simpoOrpoGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

// §V16 tier-1: SimPO and ORPO match the independent numpy reference at f64 1e-12.
func TestSimPOORPOParity(t *testing.T) {
	g := loadSimPOORPO(t)
	sp, err := nn.SimPO(backend.NewContext(), vec(g.AvgW), vec(g.AvgL), nn.Beta(g.Beta), nn.Gamma(g.Gamma))
	if err != nil {
		t.Fatal(err)
	}
	if v := sp.AtF64(); math.Abs(v-g.SimPO) > 1e-12*math.Max(1, math.Abs(g.SimPO)) {
		t.Errorf("SimPO = %.17g, want %.17g", v, g.SimPO)
	}
	or, err := nn.ORPO(backend.NewContext(), vec(g.AvgW), vec(g.AvgL), nn.Lambda(g.Lambda))
	if err != nil {
		t.Fatal(err)
	}
	if v := or.AtF64(); math.Abs(v-g.ORPO) > 1e-12*math.Max(1, math.Abs(g.ORPO)) {
		t.Errorf("ORPO = %.17g, want %.17g", v, g.ORPO)
	}
}

// §V2: finite differences of both losses w.r.t. the chosen and rejected average
// log-probs match the analytic VJPs.
func TestSimPOORPOGradCheck(t *testing.T) {
	g := loadSimPOORPO(t)
	cases := []struct {
		name string
		loss func(ctx *backend.Context, w, l *tensor.Tensor) (*tensor.Tensor, error)
	}{
		{"simpo", func(ctx *backend.Context, w, l *tensor.Tensor) (*tensor.Tensor, error) {
			return nn.SimPO(ctx, w, l, nn.Beta(g.Beta), nn.Gamma(g.Gamma))
		}},
		{"orpo", func(ctx *backend.Context, w, l *tensor.Tensor) (*tensor.Tensor, error) {
			return nn.ORPO(ctx, w, l, nn.Lambda(g.Lambda))
		}},
	}
	for _, tc := range cases {
		wData := append([]float64(nil), g.AvgW...)
		lData := append([]float64(nil), g.AvgL...)
		lossAt := func() float64 {
			out, err := tc.loss(backend.NewContext(), vec(wData), vec(lData))
			if err != nil {
				t.Fatal(err)
			}
			return out.AtF64()
		}
		wT, lT := vec(wData), vec(lData)
		tape := autograd.NewTape()
		out, err := tc.loss(tape.Context(), wT, lT)
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
			g    *tensor.Tensor
		}{{"w", wData, tape.Grad(wT)}, {"l", lData, tape.Grad(lT)}}
		for _, side := range sides {
			for i := range side.d {
				o := side.d[i]
				side.d[i] = o + h
				lp := lossAt()
				side.d[i] = o - h
				lm := lossAt()
				side.d[i] = o
				if num, ana := (lp-lm)/(2*h), side.g.AtF64(i); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
					t.Errorf("%s %s grad[%d]: numeric %.8g vs analytic %.8g", tc.name, side.name, i, num, ana)
				}
			}
		}
	}
}

// SimPO (Meng et al. 2024) aligns a model from preference pairs WITHOUT a
// reference model, using the length-normalized average log-probability as the
// reward. Here the chosen response is more likely (avg log-prob −0.5) than the
// rejected (−1.5); with β=2 and margin γ=1 the reward gap 2·(−0.5+1.5)=2 clears
// the margin, giving a small loss.
func ExampleSimPO() {
	avgChosen := tensor.FromFloat64(tensor.Shape{1}, []float64{-0.5})
	avgRejected := tensor.FromFloat64(tensor.Shape{1}, []float64{-1.5})

	loss, _ := nn.SimPO(backend.NewContext(), avgChosen, avgRejected, nn.Beta(2), nn.Gamma(1))
	fmt.Printf("%.4f\n", loss.AtF64())
	// Output: 0.3133
}
