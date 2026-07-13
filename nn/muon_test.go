package nn_test

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

type muonGolden struct {
	R, C     int
	Momentum float64     `json:"momentum"`
	LR       float64     `json:"lr"`
	WD       float64     `json:"wd"`
	NSSteps  int         `json:"ns_steps"`
	P0       []float64   `json:"p0"`
	Grads    [][]float64 `json:"grads"`
	Traj     [][]float64 `json:"traj"`
}

func loadMuon(t *testing.T) muonGolden {
	t.Helper()
	raw, err := os.ReadFile("testdata/muon.json")
	if err != nil {
		t.Fatalf("read golden (run `make golden`): %v", err)
	}
	var g muonGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

// §V16: the Muon trajectory (lerp momentum + Newton-Schulz orthogonalization +
// √max(1,R/C) scale + decoupled weight decay) reproduces the independent f64 numpy
// reference step by step at rtol 1e-12.
func TestMuonGoldenParity(t *testing.T) {
	g := loadMuon(t)
	p := tensor.FromFloat64(tensor.Shape{g.R, g.C}, append([]float64(nil), g.P0...))
	opt := nn.NewMuon([]*tensor.Tensor{p}, g.LR,
		nn.WithMuonMomentum(g.Momentum), nn.WithMuonWeightDecay(g.WD), nn.WithMuonNSSteps(g.NSSteps))

	step := 0
	fn := func(q *tensor.Tensor) *tensor.Tensor {
		if q != p {
			return nil
		}
		return tensor.FromFloat64(p.Shape(), g.Grads[step])
	}
	for s := range g.Grads {
		if err := opt.Step(fn); err != nil {
			t.Fatal(err)
		}
		for k := range p.Numel() {
			idx := tensor.Unravel(k, p.Shape())
			if got, want := p.AtF64(idx...), g.Traj[s][k]; math.Abs(got-want) > 1e-12*math.Max(1, math.Abs(want)) {
				t.Errorf("step %d p[%d]: got %.14g want %.14g", s, k, got, want)
			}
		}
		step++
	}
}

// Muon requires 2-D parameters.
func TestMuonRejectsNon2D(t *testing.T) {
	p := tensor.New(tensor.F64, tensor.Shape{4})
	opt := nn.NewMuon([]*tensor.Tensor{p}, 0.02)
	if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return tensor.New(tensor.F64, tensor.Shape{4}) }); err == nil {
		t.Error("Muon should reject a 1-D parameter")
	}
}

// Muon orthogonalizes the momentum before stepping. Feeding the identity as the
// gradient — already an orthogonal matrix — leaves the Newton-Schulz output close
// to the identity (it drives every singular value toward 1), so each diagonal
// weight steps down by ≈ the learning rate (1 → ≈0.89) while the off-diagonal
// stays put.
func ExampleMuon() {
	w := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, 0, 0, 1})
	opt := nn.NewMuon([]*tensor.Tensor{w}, 0.1, nn.WithMuonMomentum(0))
	_ = opt.Step(func(*tensor.Tensor) *tensor.Tensor {
		return tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, 0, 0, 1})
	})
	fmt.Printf("%.2f %.2f\n", w.AtF64(0, 0), w.AtF64(0, 1))
	// Output: 0.89 0.00
}
