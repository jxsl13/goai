//go:build darwin && cgo

package metal_test

import (
	"encoding/json"
	"math"
	"math/rand/v2"
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
	metalB, _ := backend.Get(backend.Metal)
	cpuB, _ := backend.Get(backend.CPU)

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
	metalB, _ := backend.Get(backend.Metal)
	cpuB, _ := backend.Get(backend.CPU)

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
	_, cl := train(cpuB)
	if gl >= gf*0.4 {
		t.Fatalf("GPU GPT training did not converge: %.4f → %.4f", gf, gl)
	}
	if math.Abs(gl-cl) > 0.2 {
		t.Errorf("GPU vs CPU GPT training diverged: gpu %.4f vs cpu %.4f", gl, cl)
	}
	t.Logf("GPU GPT training converged: CE %.4f → %.4f (cpu final %.4f)", gf, gl, cl)
}

// randGPTf32 builds a GPT of the given config with deterministic random f32 weights
// (norms init to γ=1/β=0, projections small) — a realistically-sized model for the
// end-to-end throughput benchmark, since the golden fixture (dim 8, 2 layers) is far
// too small to show real GPU utilization. ffn is the FFN inner width.
func randGPTf32(tb testing.TB, cfg nlp.GPTConfig, ffn int) *nlp.GPT {
	tb.Helper()
	rng := rand.New(rand.NewPCG(1, 2))
	small := func(shape ...int) *tensor.Tensor {
		t := tensor.New(tensor.F32, tensor.Shape(shape))
		d := t.Storage().F32()
		for i := range d {
			d[i] = float32(rng.NormFloat64()) * 0.02
		}
		return t
	}
	ones := func(n int) *tensor.Tensor {
		t := tensor.New(tensor.F32, tensor.Shape{n})
		d := t.Storage().F32()
		for i := range d {
			d[i] = 1
		}
		return t
	}
	zeros := func(n int) *tensor.Tensor { return tensor.New(tensor.F32, tensor.Shape{n}) }
	ts := map[string]*tensor.Tensor{
		"tok_emb":   small(cfg.Vocab, cfg.Dim),
		"pos_emb":   small(cfg.Ctx, cfg.Dim),
		"head":      small(cfg.Dim, cfg.Vocab),
		"lnf.gamma": ones(cfg.Dim),
		"lnf.beta":  zeros(cfg.Dim),
	}
	for l := range cfg.Layers {
		p := func(s string) string { return "blocks." + string(rune('0'+l)) + "." + s }
		if l >= 10 { // rune trick only covers 0..9; keep the bench configs small
			tb.Fatalf("randGPTf32 supports <10 layers, got %d", cfg.Layers)
		}
		ts[p("ln1.gamma")] = ones(cfg.Dim)
		ts[p("ln1.beta")] = zeros(cfg.Dim)
		ts[p("attn.wq")] = small(cfg.Dim, cfg.Dim)
		ts[p("attn.wk")] = small(cfg.Dim, cfg.Dim)
		ts[p("attn.wv")] = small(cfg.Dim, cfg.Dim)
		ts[p("attn.wo")] = small(cfg.Dim, cfg.Dim)
		ts[p("ln2.gamma")] = ones(cfg.Dim)
		ts[p("ln2.beta")] = zeros(cfg.Dim)
		ts[p("ffn.w1")] = small(cfg.Dim, ffn)
		ts[p("ffn.b1")] = zeros(ffn)
		ts[p("ffn.w2")] = small(ffn, cfg.Dim)
		ts[p("ffn.b2")] = zeros(cfg.Dim)
	}
	model, err := nlp.FromSafetensors(cfg, ts)
	if err != nil {
		tb.Fatal(err)
	}
	return model
}

// BenchmarkGPTForward times a full transformer forward pass — the REAL workload
// (§C3/§B10: optimize against a model, not a micro-benchmark) — on cpu vs metal.
// Reports tokens/s. A realistic small-GPT config exercises embed→(LN→attention→
// LN→FFN)×L→LN→head, i.e. ~15 GPU dispatches per layer, so it also shows whether
// per-op dispatch latency (§B42) dominates a real multi-op forward.
func BenchmarkGPTForward(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal: no MPS GPU — skipping (§V4)")
	}
	cfg := nlp.GPTConfig{Vocab: 4096, Ctx: 256, Dim: 512, Heads: 8, Layers: 6, Eps: 1e-5}
	const seq = 256
	model := randGPTf32(b, cfg, 4*cfg.Dim)
	tokens := make([]int, seq)
	for i := range tokens {
		tokens[i] = i % cfg.Vocab
	}
	for _, name := range []backend.Name{backend.CPU, backend.Metal} {
		be, ok := backend.Get(name)
		if !ok {
			continue
		}
		ctx := backend.NewContext().WithBackend(be)
		if _, err := model.Forward(ctx, tokens); err != nil { // warmup (pipelines, pools)
			b.Fatal(err)
		}
		b.Run(string(name), func(b *testing.B) {
			for range b.N {
				if _, err := model.Forward(ctx, tokens); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(seq)*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
		})
	}
}

// BenchmarkGPTTrainingStep times a full forward + cross-entropy + backward pass — the
// training workload (LOOP.md priority: Training UND Inferenz) — on cpu vs metal, the
// companion to BenchmarkGPTForward. The optimizer step is excluded (it is a cheap
// elementwise update; the GEMMs + their backward dominate). Reports tokens/s.
func BenchmarkGPTTrainingStep(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal: no MPS GPU — skipping (§V4)")
	}
	cfg := nlp.GPTConfig{Vocab: 4096, Ctx: 256, Dim: 512, Heads: 8, Layers: 6, Eps: 1e-5}
	const seq = 256
	model := randGPTf32(b, cfg, 4*cfg.Dim)
	tokens := make([]int, seq)
	targets := tensor.New(tensor.F32, tensor.Shape{seq})
	for i := range tokens {
		tokens[i] = i % cfg.Vocab
		targets.SetF64(float64(i%cfg.Vocab), i)
	}
	step := func(be backend.Backend) {
		tape := autograd.NewTapeOn(be)
		ctx := tape.Context()
		logits, err := model.Forward(ctx, tokens)
		if err != nil {
			b.Fatal(err)
		}
		loss, err := nn.CrossEntropy(ctx, logits, targets)
		if err != nil {
			b.Fatal(err)
		}
		if err := tape.Backward(loss); err != nil {
			b.Fatal(err)
		}
	}
	for _, name := range []backend.Name{backend.CPU, backend.Metal} {
		be, ok := backend.Get(name)
		if !ok {
			continue
		}
		step(be) // warmup
		b.Run(string(name), func(b *testing.B) {
			for range b.N {
				step(be)
			}
			b.ReportMetric(float64(seq)*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
		})
	}
}

// BenchmarkGPTDecode times autoregressive DECODE (one token per step with a KV cache) —
// the real inference workload, distinct from the full-sequence forward of §T350. Each
// step's ops are tiny (seq=1), so per-op GPU dispatch/waitUntilCompleted latency
// dominates: §T360 measured Metal ~2.3× SLOWER than cpu here, the opposite of prefill.
func BenchmarkGPTDecode(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal: no MPS GPU — skipping (§V4)")
	}
	cfg := nlp.GPTConfig{Vocab: 4096, Ctx: 512, Dim: 512, Heads: 8, Layers: 6, Eps: 1e-5}
	model := randGPTf32(b, cfg, 4*cfg.Dim)
	for _, name := range []backend.Name{backend.CPU, backend.Metal} {
		be, ok := backend.Get(name)
		if !ok {
			continue
		}
		ctx := backend.NewContext().WithBackend(be)
		cache := model.NewCache()
		for p := 0; p < 8; p++ { // warm the pipelines + prefill a short context
			if _, err := model.DecodeStep(ctx, cache, p%cfg.Vocab, p); err != nil {
				b.Fatal(err)
			}
		}
		b.Run(string(name), func(b *testing.B) {
			pos := 8
			for range b.N {
				if _, err := model.DecodeStep(ctx, cache, pos%cfg.Vocab, pos); err != nil {
					b.Fatal(err)
				}
				pos++
				if pos >= cfg.Ctx { // reset the cache before it overflows the context
					cache = model.NewCache()
					pos = 0
				}
			}
			b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "tok/s")
		})
	}
}
