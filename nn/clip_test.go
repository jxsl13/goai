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

// §V16 tier-1 (§R165 CLIP form): CLIPLoss with a frozen log-scale equals InfoNCE with temperature
// 1/exp(LogScale) — the learned-scale CLIP loss reduces to the verified InfoNCE at any fixed scale.
func TestCLIPLossEqualsInfoNCE(t *testing.T) {
	for _, appliedScale := range []float64{1.0, 5.0, 1.0 / 0.07} {
		sc, err := nn.NewCLIPLogitScale(tensor.F64, appliedScale, 200)
		if err != nil {
			t.Fatal(err)
		}
		got, err := nn.CLIPLoss(backend.NewContext(), toT(nceA), toT(nceB), sc)
		if err != nil {
			t.Fatal(err)
		}
		want, _ := nn.InfoNCE(backend.NewContext(), toT(nceA), toT(nceB), 1.0/appliedScale, true)
		if math.Abs(got.AtF64()-want.AtF64()) > 1e-9 {
			t.Errorf("scale=%g: CLIPLoss %.10g != InfoNCE(τ=1/scale) %.10g", appliedScale, got.AtF64(), want.AtF64())
		}
	}
}

// §V2: CLIPLoss is differentiable w.r.t. the image embeddings, the text embeddings AND the learned
// log-scale parameter (the gradient reaches the scale through the broadcast multiply).
func TestCLIPLossGradcheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(21, 7))
	const n, d = 4, 3
	img := randMat(rng, n, d)
	txt := randMat(rng, n, d)
	sc, _ := nn.NewCLIPLogitScale(tensor.F64, 4.0, 200)
	tape := autograd.NewTape()
	loss, err := nn.CLIPLoss(tape.Context(), img, txt, sc)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	const h = 1e-6
	fd := func(p *tensor.Tensor, idx ...int) float64 {
		o := p.AtF64(idx...)
		p.SetF64(o+h, idx...)
		lp, _ := nn.CLIPLoss(backend.NewContext(), img, txt, sc)
		p.SetF64(o-h, idx...)
		lm, _ := nn.CLIPLoss(backend.NewContext(), img, txt, sc)
		p.SetF64(o, idx...)
		return (lp.AtF64() - lm.AtF64()) / (2 * h)
	}
	checkMat := func(name string, p *tensor.Tensor) {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("%s: nil grad", name)
		}
		for i := range n {
			for j := range d {
				if num, ana := fd(p, i, j), g.AtF64(i, j); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
					t.Errorf("%s grad[%d,%d]: numeric %.8g vs analytic %.8g", name, i, j, num, ana)
				}
			}
		}
	}
	checkMat("image", img)
	checkMat("text", txt)
	// the learned log-scale (rank-0)
	gs := tape.Grad(sc.LogScale)
	if gs == nil {
		t.Fatal("nil log-scale grad")
	}
	if num, ana := fd(sc.LogScale), gs.AtF64(); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
		t.Errorf("logScale grad: numeric %.8g vs analytic %.8g", num, ana)
	}
}

// §V16 tier-1: Clamp caps the applied scale to [1, MaxScale] (open_clip clamps logit_scale to
// [0, log(MaxScale)]), leaving in-range values untouched.
func TestCLIPClamp(t *testing.T) {
	sc, _ := nn.NewCLIPLogitScale(tensor.F64, 10, 100)
	sc.LogScale.SetF64(math.Log(500)) // above ceiling
	sc.Clamp()
	if got := sc.Scale(); math.Abs(got-100) > 1e-9 {
		t.Errorf("clamp above: scale = %g, want 100", got)
	}
	sc.LogScale.SetF64(-2) // below floor (exp<1)
	sc.Clamp()
	if got := sc.LogScale.AtF64(); got != 0 {
		t.Errorf("clamp below: logScale = %g, want 0 (scale 1)", got)
	}
	sc.LogScale.SetF64(math.Log(5)) // in range
	sc.Clamp()
	if got := sc.Scale(); math.Abs(got-5) > 1e-9 {
		t.Errorf("clamp in-range: scale = %g, want 5 (unchanged)", got)
	}
}

func TestCLIPInitAndParams(t *testing.T) {
	sc, _ := nn.NewCLIPLogitScale(tensor.F64, 0, 0) // defaults
	if math.Abs(sc.Scale()-1.0/0.07) > 1e-9 {
		t.Errorf("default init scale = %g, want 1/0.07", sc.Scale())
	}
	if len(sc.Params()) != 1 || sc.Params()[0] != sc.LogScale {
		t.Error("Params should return the single log-scale tensor")
	}
	if _, err := nn.NewCLIPLogitScale(tensor.F64, 200, 100); err == nil {
		t.Error("initScale > maxScale: want error")
	}
}

func TestCLIPLossErrors(t *testing.T) {
	sc, _ := nn.NewCLIPLogitScale(tensor.F64, 10, 100)
	a := tensor.New(tensor.F64, tensor.Shape{2, 3})
	if _, err := nn.CLIPLoss(backend.NewContext(), a, tensor.New(tensor.F64, tensor.Shape{3, 3}), sc); err == nil {
		t.Error("shape mismatch: want error")
	}
	if _, err := nn.CLIPLoss(backend.NewContext(), a, a, nil); err == nil {
		t.Error("nil scale: want error")
	}
}

func ExampleCLIPLogitScale() {
	// Two identical, well-separated image/text embeddings ⇒ near-zero CLIP loss.
	emb := tensor.FromFloat64(tensor.Shape{3, 3}, []float64{3, 0, 0, 0, 3, 0, 0, 0, 3})
	sc, _ := nn.NewCLIPLogitScale(tensor.F64, 20, 100)
	loss, _ := nn.CLIPLoss(backend.NewContext(), emb, emb, sc)
	fmt.Printf("%.4f\n", loss.AtF64())
	// Output:
	// 0.0000
}
