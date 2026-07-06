//go:build metal && darwin && cgo

package metal_test

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

type gptCfg struct {
	Config struct {
		Vocab, Ctx, Dim, Heads, Layers int
		Eps                            float64
	} `json:"config"`
	Tokens  []int     `json:"tokens"`
	Targets []float64 `json:"targets"`
}

// loadGPTf32 loads the golden GPT and casts every weight to f32 (metal is
// f32-only), so its GEMMs dispatch to the GPU.
func loadGPTf32(t *testing.T) (*nlp.GPT, gptCfg) {
	t.Helper()
	raw, err := os.ReadFile("../../nlp/testdata/gpt.json")
	if err != nil {
		t.Fatalf("read gpt.json (run make golden): %v", err)
	}
	var c gptCfg
	if err := json.Unmarshal(raw, &c); err != nil {
		t.Fatal(err)
	}
	ts, _, err := safetensors.LoadFile("../../nlp/testdata/gpt.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	f32 := make(map[string]*tensor.Tensor, len(ts))
	for k, v := range ts {
		f32[k] = v.Cast(tensor.F32)
	}
	model, err := nlp.FromSafetensors(nlp.GPTConfig{
		Vocab: c.Config.Vocab, Ctx: c.Config.Ctx, Dim: c.Config.Dim,
		Heads: c.Config.Heads, Layers: c.Config.Layers, Eps: c.Config.Eps,
	}, f32)
	if err != nil {
		t.Fatal(err)
	}
	return model, c
}

// §T20/priority: LLM INFERENCE on the GPU — the GPT's projection/FFN/head GEMMs
// run on Metal; per-position argmax must match the cpu run and logits stay close
// (f32 through a deep net; compounding within the documented cross-tol).
func TestMetalGPTInference(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal: no MPS GPU — skipping (§V4)")
	}
	model, c := loadGPTf32(t)
	metalB, _ := backend.Get("metal")
	cpuB, _ := backend.Get("cpu")

	run := func(be backend.Backend) *tensor.Tensor {
		out, err := model.Forward(backend.NewContext().WithBackend(be), c.Tokens)
		if err != nil {
			t.Fatal(err)
		}
		return out
	}
	lg := run(metalB)
	lc := run(cpuB)
	seq, vocab := lg.Shape()[0], lg.Shape()[1]
	for i := range seq {
		am, ac := 0, 0
		for v := 1; v < vocab; v++ {
			if lg.AtF64(i, v) > lg.AtF64(i, am) {
				am = v
			}
			if lc.AtF64(i, v) > lc.AtF64(i, ac) {
				ac = v
			}
		}
		if am != ac {
			t.Errorf("pos %d: GPU argmax %d != CPU argmax %d", i, am, ac)
		}
		for v := range vocab {
			if math.Abs(lg.AtF64(i, v)-lc.AtF64(i, v)) > 1e-3*math.Max(1, math.Abs(lc.AtF64(i, v))) {
				t.Errorf("pos %d cls %d: GPU %.5f vs CPU %.5f", i, v, lg.AtF64(i, v), lc.AtF64(i, v))
			}
		}
	}
	t.Log("GPT inference runs on GPU with matching predictions")
}

// §T30/priority: LLM TRAINING on the GPU — the GPT's GEMMs (forward + backward)
// run on Metal via a metal-backed tape; CrossEntropy must drop, tracking the cpu
// run.
func TestMetalGPTTraining(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal: no MPS GPU — skipping (§V4)")
	}
	metalB, _ := backend.Get("metal")
	cpuB, _ := backend.Get("cpu")

	train := func(be backend.Backend) (float64, float64) {
		model, c := loadGPTf32(t)
		targets := tensor.New(tensor.F32, tensor.Shape{len(c.Targets)})
		for i, v := range c.Targets {
			targets.SetF64(v, i)
		}
		opt := nn.NewAdamW(model.Params(), 0.01, 0.01)
		var first, last float64
		for step := range 60 {
			tape := autograd.NewTapeOn(be)
			ctx := tape.Context()
			logits, err := model.Forward(ctx, c.Tokens)
			if err != nil {
				t.Fatal(err)
			}
			loss, err := nn.CrossEntropy(ctx, logits, targets)
			if err != nil {
				t.Fatal(err)
			}
			if step == 0 {
				first = loss.AtF64()
			}
			last = loss.AtF64()
			if err := tape.Backward(loss); err != nil {
				t.Fatal(err)
			}
			cl, _ := nn.ClipGradNorm(model.Params(), tape.Grad, 1.0)
			if err := opt.Step(cl); err != nil {
				t.Fatal(err)
			}
		}
		return first, last
	}
	gf, gl := train(metalB)
	cf, cl := train(cpuB)
	if gl >= gf*0.4 {
		t.Fatalf("GPU GPT training did not converge: %.4f → %.4f", gf, gl)
	}
	if math.Abs(gl-cl) > 0.2 {
		t.Errorf("GPU vs CPU GPT training diverged: gpu %.4f vs cpu %.4f", gl, cl)
	}
	t.Logf("GPU GPT training: CE %.4f → %.4f (cpu %.4f → %.4f)", gf, gl, cf, cl)
}
