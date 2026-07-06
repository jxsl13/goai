package nn_test

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

type optimGolden struct {
	Optim struct {
		P0    []float64   `json:"p0"`
		Grads [][]float64 `json:"grads"`
		SGD   struct {
			LR    float64     `json:"lr"`
			Steps [][]float64 `json:"steps"`
		} `json:"sgd"`
		SGDMomentum struct {
			LR       float64     `json:"lr"`
			Momentum float64     `json:"momentum"`
			Steps    [][]float64 `json:"steps"`
		} `json:"sgd_momentum"`
		Adam struct {
			LR    float64     `json:"lr"`
			Beta1 float64     `json:"beta1"`
			Beta2 float64     `json:"beta2"`
			Eps   float64     `json:"eps"`
			Steps [][]float64 `json:"steps"`
		} `json:"adam"`
		AdamW struct {
			LR    float64     `json:"lr"`
			WD    float64     `json:"wd"`
			Steps [][]float64 `json:"steps"`
		} `json:"adamw"`
	} `json:"optim"`
}

func loadOptimGolden(t *testing.T) optimGolden {
	t.Helper()
	raw, err := os.ReadFile("testdata/optim.json")
	if err != nil {
		t.Fatalf("read golden (run `make golden`): %v", err)
	}
	var g optimGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

// gradSeq feeds a fixed gradient sequence as a GradFn, one row per Step call.
func gradSeq(p *tensor.Tensor, grads [][]float64) (nn.GradFn, func()) {
	step := 0
	fn := func(q *tensor.Tensor) *tensor.Tensor {
		if q != p {
			return nil
		}
		return tensor.FromFloat64(p.Shape(), grads[step])
	}
	advance := func() { step++ }
	return fn, advance
}

func assertTraj(t *testing.T, name string, p *tensor.Tensor, want []float64) {
	t.Helper()
	for i := range want {
		if got := p.AtF64(i); math.Abs(got-want[i]) > 1e-12*math.Max(1, math.Abs(want[i])) {
			t.Errorf("%s[%d]: got %.17g want %.17g", name, i, got, want[i])
		}
	}
}

// §V1: SGD, SGD+momentum and Adam reproduce the Python-reference trajectories
// (torch update formulas) at f64 rtol 1e-12, step by step.
func TestOptimizerGoldenParity(t *testing.T) {
	g := loadOptimGolden(t).Optim

	// plain SGD
	p := tensor.FromFloat64(tensor.Shape{3}, append([]float64(nil), g.P0...))
	opt := nn.NewSGD([]*tensor.Tensor{p}, g.SGD.LR, 0)
	fn, adv := gradSeq(p, g.Grads)
	for s := range g.Grads {
		if err := opt.Step(fn); err != nil {
			t.Fatal(err)
		}
		assertTraj(t, "sgd", p, g.SGD.Steps[s])
		adv()
	}

	// SGD + momentum
	pm := tensor.FromFloat64(tensor.Shape{3}, append([]float64(nil), g.P0...))
	optm := nn.NewSGD([]*tensor.Tensor{pm}, g.SGDMomentum.LR, g.SGDMomentum.Momentum)
	fnm, advm := gradSeq(pm, g.Grads)
	for s := range g.Grads {
		if err := optm.Step(fnm); err != nil {
			t.Fatal(err)
		}
		assertTraj(t, "sgd_momentum", pm, g.SGDMomentum.Steps[s])
		advm()
	}

	// Adam
	pa := tensor.FromFloat64(tensor.Shape{3}, append([]float64(nil), g.P0...))
	opta := nn.NewAdam([]*tensor.Tensor{pa}, g.Adam.LR)
	fna, adva := gradSeq(pa, g.Grads)
	for s := range g.Grads {
		if err := opta.Step(fna); err != nil {
			t.Fatal(err)
		}
		assertTraj(t, "adam", pa, g.Adam.Steps[s])
		adva()
	}

	// AdamW (§T33, decoupled weight decay vs real torch)
	pw := tensor.FromFloat64(tensor.Shape{3}, append([]float64(nil), g.P0...))
	optw := nn.NewAdamW([]*tensor.Tensor{pw}, g.AdamW.LR, g.AdamW.WD)
	fnw, advw := gradSeq(pw, g.Grads)
	for s := range g.Grads {
		if err := optw.Step(fnw); err != nil {
			t.Fatal(err)
		}
		assertTraj(t, "adamw", pw, g.AdamW.Steps[s])
		advw()
	}
}

// §T33: global-norm gradient clipping scales all grads by maxNorm/norm.
func TestClipGradNorm(t *testing.T) {
	p1 := tensor.New(tensor.F64, tensor.Shape{2})
	p2 := tensor.New(tensor.F64, tensor.Shape{2})
	// grads: [3,0] and [0,4] → global norm = 5
	grads := map[*tensor.Tensor][]float64{p1: {3, 0}, p2: {0, 4}}
	grad := func(p *tensor.Tensor) *tensor.Tensor {
		return tensor.FromFloat64(tensor.Shape{2}, grads[p])
	}
	clipped, norm := nn.ClipGradNorm([]*tensor.Tensor{p1, p2}, grad, 1.0)
	if math.Abs(norm-5) > 1e-12 {
		t.Errorf("reported norm %v, want 5", norm)
	}
	// after clipping to 1.0, each grad scaled by 1/5
	g1 := clipped(p1)
	if math.Abs(g1.AtF64(0)-0.6) > 1e-12 { // 3/5
		t.Errorf("clipped g1[0] = %v, want 0.6", g1.AtF64(0))
	}
	// clipped global norm must equal maxNorm
	var sq float64
	for _, p := range []*tensor.Tensor{p1, p2} {
		g := clipped(p)
		for i := range 2 {
			sq += g.AtF64(i) * g.AtF64(i)
		}
	}
	if math.Abs(math.Sqrt(sq)-1) > 1e-12 {
		t.Errorf("post-clip norm %v, want 1", math.Sqrt(sq))
	}
	// norm below maxNorm → unchanged (same fn)
	_, n2 := nn.ClipGradNorm([]*tensor.Tensor{p1}, func(*tensor.Tensor) *tensor.Tensor {
		return tensor.FromFloat64(tensor.Shape{2}, []float64{0.1, 0.1})
	}, 10)
	if n2 > 10 {
		t.Errorf("small norm %v should be < maxNorm", n2)
	}
}

// §T33: warmup+cosine schedule — linear up, cosine down to minLR.
func TestWarmupCosine(t *testing.T) {
	const warmup, total, base, mn = 10, 100, 1.0, 0.0
	if lr := nn.WarmupCosine(4, warmup, total, base, mn); math.Abs(lr-0.5) > 1e-12 {
		t.Errorf("warmup step 4: lr %v, want 0.5", lr) // (4+1)/10
	}
	if lr := nn.WarmupCosine(9, warmup, total, base, mn); math.Abs(lr-1.0) > 1e-12 {
		t.Errorf("warmup end: lr %v, want 1.0", lr)
	}
	if lr := nn.WarmupCosine(10, warmup, total, base, mn); math.Abs(lr-1.0) > 1e-12 {
		t.Errorf("cosine start: lr %v, want baseLR 1.0", lr) // cos(0)=1
	}
	mid := nn.WarmupCosine(55, warmup, total, base, mn) // progress 0.5 → cos(π/2)=0
	if math.Abs(mid-0.5) > 1e-12 {
		t.Errorf("cosine midpoint: lr %v, want 0.5", mid)
	}
	if lr := nn.WarmupCosine(100, warmup, total, base, mn); math.Abs(lr-mn) > 1e-12 {
		t.Errorf("cosine end: lr %v, want minLR %v", lr, mn)
	}
}

// End-to-end sanity: Linear+MSE trained by each optimizer strictly decreases the
// loss (preview of §T18 convergence).
func TestOptimizerTrainingDecreasesLoss(t *testing.T) {
	x := tensor.FromFloat64(tensor.Shape{4, 2}, []float64{1, 2, -1, 0.5, 3, -2, 0, 1})
	yt := tensor.FromFloat64(tensor.Shape{4, 1}, []float64{3, -0.5, 1, 2})

	train := func(name string, mk func(ps []*tensor.Tensor) nn.Optimizer) {
		l := nn.NewLinear(tensor.F64, 2, 1, 11)
		opt := mk(l.Params())
		var prev float64
		for step := range 5 {
			tape := autograd.NewTape()
			ctx := tape.Context()
			pred, err := l.Forward(ctx, x)
			if err != nil {
				t.Fatal(err)
			}
			loss, err := nn.MSE(ctx, pred, yt)
			if err != nil {
				t.Fatal(err)
			}
			lv := loss.AtF64()
			if step > 0 && lv >= prev {
				t.Fatalf("%s: loss did not decrease at step %d (%v → %v)", name, step, prev, lv)
			}
			prev = lv
			if err := tape.Backward(loss); err != nil {
				t.Fatal(err)
			}
			if err := opt.Step(tape.Grad); err != nil {
				t.Fatal(err)
			}
		}
	}
	train("sgd", func(ps []*tensor.Tensor) nn.Optimizer { return nn.NewSGD(ps, 0.05, 0) })
	train("adam", func(ps []*tensor.Tensor) nn.Optimizer { return nn.NewAdam(ps, 0.05) })
}

// nil grads are skipped; shape mismatch errors.
func TestOptimizerEdge(t *testing.T) {
	p := tensor.FromFloat64(tensor.Shape{2}, []float64{1, 2})
	opt := nn.NewSGD([]*tensor.Tensor{p}, 0.1, 0)
	if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return nil }); err != nil {
		t.Fatal(err)
	}
	if p.AtF64(0) != 1 || p.AtF64(1) != 2 {
		t.Error("nil grad must leave params unchanged")
	}
	bad := tensor.New(tensor.F64, tensor.Shape{3})
	if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return bad }); err == nil {
		t.Error("shape mismatch must error")
	}
}
