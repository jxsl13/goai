//go:build darwin && cgo

package llamagpu_test

import (
	"math"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/metal"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// §T405: the productionized batched Decoder reproduces the library's own per-op decode. Drives a
// real nlp.Llama (GQA + SwiGLU) through llamagpu.Decoder and confirms the logits match m.DecodeStep
// on the reference backend across an autoregressive run (next token = argmax).
func TestDecoderMatchesReference(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal: no gpu")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 48, Ctx: 32, Dim: 64, Heads: 8, KVHeads: 2, Layers: 3,
		Hidden: 176, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 7)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := llamagpu.New(m)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Release()

	refCtx := backend.NewContext().WithBackend(backend.Reference())
	cache := m.NewCache()
	tok := 5
	argmax := func(v []float32) int {
		bi := 0
		for i, x := range v {
			if x > v[bi] {
				bi = i
			}
		}
		return bi
	}
	for pos := range 6 {
		got, err := dec.Step(tok, pos)
		if err != nil {
			t.Fatal(err)
		}
		refT, err := m.DecodeStep(refCtx, cache, tok, pos)
		if err != nil {
			t.Fatal(err)
		}
		for j := range got {
			want := refT.AtF64(0, j)
			// IsNaN guard: NaN fails every > comparison, so a NaN logit would silently "pass" (§T428).
			if math.IsNaN(float64(got[j])) || math.Abs(float64(got[j])-want) > 3e-2*math.Max(1, math.Abs(want)) {
				t.Fatalf("pos %d tok %d logit[%d]: decoder %v vs reference %v", pos, tok, j, got[j], want)
			}
		}
		tok = argmax(got) // autoregressive: same token feeds both next step
	}
	t.Logf("llamagpu.Decoder matches the reference nlp.Llama across an autoregressive run (GQA %d:%d, SwiGLU, %d layers)", cfg.Heads, cfg.KVHeads, cfg.Layers)
}

// §T406: llamagpu.Decoder.Generate produces the SAME greedy token sequence as nlp.Llama.Generate.
// Greedy (argmax) is deterministic, so the batched decode and the library's per-op decode must agree
// token-for-token when their logits match.
func TestGenerateMatchesReference(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal: no gpu")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 48, Ctx: 40, Dim: 64, Heads: 8, KVHeads: 2, Layers: 3,
		Hidden: 176, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 3)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := llamagpu.New(m)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Release()

	prompt := []int{5, 12, 3}
	const maxNew = 20
	gpuOut, err := dec.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	refOut, err := m.Generate(prompt, maxNew, nlp.Greedy(), nlp.WithBackend(backend.Reference()))
	if err != nil {
		t.Fatal(err)
	}
	if len(gpuOut) != len(refOut) {
		t.Fatalf("length: gpu %d vs ref %d", len(gpuOut), len(refOut))
	}
	for i := range gpuOut {
		if gpuOut[i] != refOut[i] {
			t.Fatalf("token[%d]: gpu %d vs ref %d\ngpu=%v\nref=%v", i, gpuOut[i], refOut[i], gpuOut, refOut)
		}
	}
	t.Logf("llamagpu.Decoder.Generate == nlp.Llama.Generate greedy: %d tokens (prompt %d + %d new)", len(gpuOut), len(prompt), len(gpuOut)-len(prompt))
}

// §T407: a GGUF-format Llama drives the batched Decoder correctly. Round-tripping through the GGUF
// metadata/tensor maps yields F32 weights (GGUF's storage dtype) — all other tests use NewLlama's F64
// — so this confirms the model-file path (load GGUF → nlp.Llama → llamagpu.Decoder → fast generate)
// works for the F32 weights real models carry.
func TestGGUFLoadedDecoderGenerate(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal: no gpu")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 48, Ctx: 40, Dim: 64, Heads: 8, KVHeads: 2, Layers: 3,
		Hidden: 176, Eps: 1e-5, RopeBase: 10000,
	}
	orig, err := nlp.NewLlama(cfg, 9)
	if err != nil {
		t.Fatal(err)
	}
	meta, ts := nlp.LlamaToGGUF(orig)
	m, err := nlp.LlamaFromGGUF(meta, ts) // F32 weights, as loaded from a real .gguf
	if err != nil {
		t.Fatal(err)
	}
	dec, err := llamagpu.New(m)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Release()

	prompt := []int{2, 9, 4}
	const maxNew = 18
	gpuOut, err := dec.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	refOut, err := m.Generate(prompt, maxNew, nlp.Greedy(), nlp.WithBackend(backend.Reference()))
	if err != nil {
		t.Fatal(err)
	}
	if len(gpuOut) != len(refOut) {
		t.Fatalf("length: gpu %d vs ref %d", len(gpuOut), len(refOut))
	}
	for i := range gpuOut {
		if gpuOut[i] != refOut[i] {
			t.Fatalf("token[%d]: gpu %d vs ref %d", i, gpuOut[i], refOut[i])
		}
	}
	t.Logf("GGUF-loaded (F32) Llama through llamagpu.Decoder.Generate == reference: %d tokens", len(gpuOut))
}

// §T415: the QUANTIZED batched Decoder — every projection a device-resident Q8_0 weight consumed by
// record-mode QMatMulResident — matches the QuantLlama's own per-op decode: logits at each step and
// the full greedy generation. The memory win of quantization + the batched-decode speedup, verified.
func TestQuantDecoderMatchesReference(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal: no gpu")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 48, Ctx: 40, Dim: 64, Heads: 8, KVHeads: 2, Layers: 3,
		Hidden: 192, Eps: 1e-5, RopeBase: 10000, // hidden multiple of 32 (Q8_0 block)
	}
	fm, err := nlp.NewLlama(cfg, 11)
	if err != nil {
		t.Fatal(err)
	}
	qm, err := nlp.QuantizeLlama(fm, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	defer qm.Close()

	dec, err := llamagpu.NewQuant(qm)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Release()

	// per-step logits vs the QuantLlama's own per-op DecodeStep (reference backend).
	refCtx := backend.NewContext().WithBackend(backend.Reference())
	cache := qm.NewCache()
	tok := 7
	for pos := range 5 {
		got, err := dec.Step(tok, pos)
		if err != nil {
			t.Fatal(err)
		}
		refT, err := qm.DecodeStep(refCtx, cache, tok, pos)
		if err != nil {
			t.Fatal(err)
		}
		for j := range got {
			want := refT.AtF64(0, j)
			if math.Abs(float64(got[j])-want) > 3e-2*math.Max(1, math.Abs(want)) {
				t.Fatalf("pos %d logit[%d]: quant decoder %v vs reference %v", pos, j, got[j], want)
			}
		}
		tok = (tok*7 + 3) % cfg.Vocab
	}

	// full greedy generation must match token-for-token.
	prompt := []int{2, 9, 4}
	gpuOut, err := dec2Gen(t, qm, prompt)
	if err != nil {
		t.Fatal(err)
	}
	refOut, err := qm.Generate(prompt, 15, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	for i := range refOut {
		if gpuOut[i] != refOut[i] {
			t.Fatalf("token[%d]: quant decoder %d vs reference %d", i, gpuOut[i], refOut[i])
		}
	}
	t.Logf("quantized (Q8_0) batched Decoder == QuantLlama reference: logits @3e-2 and %d greedy tokens", len(refOut))
}

// dec2Gen runs Generate on a FRESH quant decoder (the logits test above consumed KV positions).
func dec2Gen(t *testing.T, qm *nlp.QuantLlama, prompt []int) ([]int, error) {
	t.Helper()
	dec, err := llamagpu.NewQuant(qm)
	if err != nil {
		return nil, err
	}
	defer dec.Release()
	return dec.Generate(prompt, 15, nlp.Greedy())
}

// §T416 (§C3, the §T404 analog for the quantized path): real-model decode throughput — the batched
// QUANT decoder vs the QuantLlama's own per-op DecodeStep on the same metal backend, plus the f32
// batched decoder for comparison (Q8_0 reads 4× fewer weight bytes per matmul; decode matmuls are
// memory-bound, so quant should be at least as fast per token while using ~4× less weight memory).
func TestQuantDecodeThroughput(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal: no gpu")
	}
	metalBE, ok := backend.Get(backend.Metal)
	if !ok {
		t.Skip("metal backend not registered")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 1024, Ctx: 128, Dim: 512, Heads: 8, KVHeads: 2, Layers: 6,
		Hidden: 1376, Eps: 1e-5, RopeBase: 10000,
	}
	fm, err := nlp.NewLlama(cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	qm, err := nlp.QuantizeLlama(fm, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	defer qm.Close()

	const steps = 32
	run := func(step func(tok, pos int) error) float64 {
		if err := step(0, 0); err != nil { // warmup (pos 0 reused; correctness untested here)
			t.Fatal(err)
		}
		s := time.Now()
		for pos := range steps {
			if err := step(pos%cfg.Vocab, pos); err != nil {
				t.Fatal(err)
			}
		}
		return float64(steps) / time.Since(s).Seconds()
	}

	qdec, err := llamagpu.NewQuant(qm)
	if err != nil {
		t.Fatal(err)
	}
	defer qdec.Release()
	quantBatched := run(func(tok, pos int) error { _, e := qdec.Step(tok, pos); return e })

	fdec, err := llamagpu.New(fm)
	if err != nil {
		t.Fatal(err)
	}
	defer fdec.Release()
	f32Batched := run(func(tok, pos int) error { _, e := fdec.Step(tok, pos); return e })

	mctx := backend.NewContext().WithBackend(metalBE)
	cache := qm.NewCache()
	perOp := run(func(tok, pos int) error {
		if pos == 0 {
			cache = qm.NewCache() // fresh cache when the warmup restarts positions
		}
		_, e := qm.DecodeStep(mctx, cache, tok, pos)
		return e
	})

	t.Logf("quant decode D=%d GQA%d:%d %d-layer Q8_0: batched-quant %.1f tok/s | batched-f32 %.1f tok/s | QuantLlama per-op(metal) %.1f tok/s | quant speedup %.1fx",
		cfg.Dim, cfg.Heads, cfg.KVHeads, cfg.Layers, quantBatched, f32Batched, perOp, quantBatched/perOp)
}

// §T418: StepN — a whole multi-token step (prompt prefill / speculative verify) in ONE recorded
// command buffer — produces the same per-position logits as sequential single-token Steps, and is
// measured against the sequential prefill.
func TestStepNMatchesSequentialSteps(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal: no gpu")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 48, Ctx: 40, Dim: 64, Heads: 8, KVHeads: 2, Layers: 3,
		Hidden: 176, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 5)
	if err != nil {
		t.Fatal(err)
	}
	tokens := []int{7, 3, 11, 0, 5, 22, 9}

	// batched: one StepN over all tokens.
	dN, err := llamagpu.New(m)
	if err != nil {
		t.Fatal(err)
	}
	defer dN.Release()
	all, err := dN.StepN(tokens, 0)
	if err != nil {
		t.Fatal(err)
	}

	// sequential: one Step per token on a fresh decoder.
	d1, err := llamagpu.New(m)
	if err != nil {
		t.Fatal(err)
	}
	defer d1.Release()
	for pos, tok := range tokens {
		want, err := d1.Step(tok, pos)
		if err != nil {
			t.Fatal(err)
		}
		got := all[pos*cfg.Vocab : (pos+1)*cfg.Vocab]
		for j := range want {
			if math.Abs(float64(got[j]-want[j])) > 2e-3*math.Max(1, math.Abs(float64(want[j]))) {
				t.Fatalf("pos %d logit[%d]: StepN %v vs Step %v", pos, j, got[j], want[j])
			}
		}
	}
	t.Logf("StepN(%d tokens) == %d sequential Steps at every position", len(tokens), len(tokens))
}

// §T418 prefill throughput: StepN vs the old token-at-a-time prefill at a realistic size.
func TestPrefillThroughput(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal: no gpu")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 1024, Ctx: 128, Dim: 512, Heads: 8, KVHeads: 2, Layers: 6,
		Hidden: 1376, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := llamagpu.New(m)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Release()
	prompt := make([]int, 64)
	for i := range prompt {
		prompt[i] = (i * 7) % cfg.Vocab
	}
	// batched prefill (positions restart at 0 each iter; cache rows are overwritten — fine for timing)
	if _, err := dec.StepN(prompt, 0); err != nil {
		t.Fatal(err)
	}
	const it = 10
	s := time.Now()
	for range it {
		if _, err := dec.StepN(prompt, 0); err != nil {
			t.Fatal(err)
		}
	}
	batched := time.Since(s).Seconds() / it
	// sequential prefill
	s = time.Now()
	for range it {
		for pos, tok := range prompt {
			if _, err := dec.Step(tok, pos); err != nil {
				t.Fatal(err)
			}
		}
	}
	seq := time.Since(s).Seconds() / it
	t.Logf("prefill %d tokens D=%d 6-layer: StepN %.2f ms (%.0f tok/s) | sequential Steps %.2f ms (%.0f tok/s) | speedup %.1fx",
		len(prompt), cfg.Dim, batched*1e3, float64(len(prompt))/batched, seq*1e3, float64(len(prompt))/seq, seq/batched)
}

// §T419: speculative decoding on the batched decoders — with a greedy sampler the accept/reject
// math makes the output EXACTLY the target's greedy sequence regardless of the draft, so the test
// is token-for-token equality with the plain batched Generate, while the stats prove real
// acceptance (the speedup mechanism) with a different-seed draft model.
func TestSpeculativeGenerateMatchesTarget(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal: no gpu")
	}
	tcfg := nlp.LlamaConfig{
		Vocab: 48, Ctx: 64, Dim: 64, Heads: 8, KVHeads: 2, Layers: 3,
		Hidden: 176, Eps: 1e-5, RopeBase: 10000,
	}
	dcfg := tcfg
	dcfg.Layers = 1 // small draft
	tm, err := nlp.NewLlama(tcfg, 3)
	if err != nil {
		t.Fatal(err)
	}
	dm, err := nlp.NewLlama(dcfg, 99) // different weights entirely
	if err != nil {
		t.Fatal(err)
	}
	target, err := llamagpu.New(tm)
	if err != nil {
		t.Fatal(err)
	}
	defer target.Release()
	draft, err := llamagpu.New(dm)
	if err != nil {
		t.Fatal(err)
	}
	defer draft.Release()

	prompt := []int{5, 12, 3}
	const maxNew = 24
	got, stats, err := llamagpu.SpeculativeGenerate(target, draft, prompt, maxNew, 4, nlp.Greedy(), 7)
	if err != nil {
		t.Fatal(err)
	}
	ref, err := llamagpu.New(tm) // fresh target for the plain generate
	if err != nil {
		t.Fatal(err)
	}
	defer ref.Release()
	want, err := ref.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("length: speculative %d vs plain %d\nspec=%v\nplain=%v", len(got), len(want), got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token[%d]: speculative %d vs plain %d\nspec=%v\nplain=%v", i, got[i], want[i], got, want)
		}
	}
	if stats.Proposed == 0 {
		t.Fatal("no draft tokens proposed")
	}
	t.Logf("speculative == plain greedy target: %d tokens; draft proposed %d, accepted %d (rate %.0f%%)",
		len(got), stats.Proposed, stats.Accepted, 100*stats.AcceptanceRate())
}

// §T419: with the draft sharing the target's weights, greedy acceptance is ~100% — this exercises
// the FULL-ACCEPT path (bonus token, pos advancing by the whole window, draft-KV completeness on
// full accept) that the different-models test (0% acceptance) never reaches.
func TestSpeculativeFullAcceptPath(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal: no gpu")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 48, Ctx: 64, Dim: 64, Heads: 8, KVHeads: 2, Layers: 2,
		Hidden: 176, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 3)
	if err != nil {
		t.Fatal(err)
	}
	target, err := llamagpu.New(m)
	if err != nil {
		t.Fatal(err)
	}
	defer target.Release()
	draft, err := llamagpu.New(m) // same weights → greedy draft always matches
	if err != nil {
		t.Fatal(err)
	}
	defer draft.Release()

	prompt := []int{5, 12, 3}
	const maxNew = 24
	got, stats, err := llamagpu.SpeculativeGenerate(target, draft, prompt, maxNew, 4, nlp.Greedy(), 7)
	if err != nil {
		t.Fatal(err)
	}
	ref, err := llamagpu.New(m)
	if err != nil {
		t.Fatal(err)
	}
	defer ref.Release()
	want, err := ref.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token[%d]: speculative %d vs plain %d", i, got[i], want[i])
		}
	}
	if stats.AcceptanceRate() < 0.99 {
		t.Fatalf("self-draft acceptance %.0f%%, want ~100%% (full-accept path not exercised)", 100*stats.AcceptanceRate())
	}
	t.Logf("full-accept path: %d tokens, acceptance %.0f%% (%d/%d)", len(got), 100*stats.AcceptanceRate(), stats.Accepted, stats.Proposed)
}

// §T420 (§V22): the speculative cost structure, measured honestly. A real speedup needs a TRAINED
// related draft (random models accept 0%, §T419), so instead of faking acceptance this measures the
// real mechanics — t_draft (1-layer step), t_target (6-layer step), t_verify (StepN k) — and reports
// the round arithmetic: tokens/round = 1 + a·(k-1), time/round = (k-1)·t_draft + t_verify. With
// dispatch-bound steps t_verify ≈ t_target, so speedup ≈ k·a / ((k-1)·r + 1) with r = t_draft/t_target.
func TestSpeculativeCostStructure(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal: no gpu")
	}
	tcfg := nlp.LlamaConfig{
		Vocab: 1024, Ctx: 128, Dim: 512, Heads: 8, KVHeads: 2, Layers: 6,
		Hidden: 1376, Eps: 1e-5, RopeBase: 10000,
	}
	dcfg := tcfg
	dcfg.Layers = 1
	tm, _ := nlp.NewLlama(tcfg, 1)
	dm, _ := nlp.NewLlama(dcfg, 2)
	target, err := llamagpu.New(tm)
	if err != nil {
		t.Fatal(err)
	}
	defer target.Release()
	draft, err := llamagpu.New(dm)
	if err != nil {
		t.Fatal(err)
	}
	defer draft.Release()

	const it = 30
	timeStep := func(d *llamagpu.Decoder) float64 {
		if _, err := d.Step(0, 0); err != nil {
			t.Fatal(err)
		}
		s := time.Now()
		for i := range it {
			if _, err := d.Step(i%tcfg.Vocab, i%64); err != nil {
				t.Fatal(err)
			}
		}
		return time.Since(s).Seconds() / it * 1e3
	}
	tTarget := timeStep(target)
	tDraft := timeStep(draft)
	const k = 4
	win := []int{1, 2, 3, 4}
	if _, err := target.StepN(win, 0); err != nil {
		t.Fatal(err)
	}
	s := time.Now()
	for range it {
		if _, err := target.StepN(win, 0); err != nil {
			t.Fatal(err)
		}
	}
	tVerify := time.Since(s).Seconds() / it * 1e3

	r := tDraft / tTarget
	roundMs := float64(k-1)*tDraft + tVerify
	for _, a := range []float64{0.5, 0.8, 1.0} {
		toks := 1 + a*float64(k-1)
		t.Logf("acceptance %.0f%%: %.2f tok/round / %.2f ms = %.0f tok/s (plain %.0f) → speedup %.2fx",
			100*a, toks, roundMs, toks/roundMs*1e3, 1/tTarget*1e3, toks/roundMs*tTarget)
	}
	t.Logf("measured: t_target %.2f ms | t_draft(1-layer) %.2f ms (ratio %.2f) | t_verify(StepN %d) %.2f ms",
		tTarget, tDraft, r, k, tVerify)
}

// randGPT builds a small deterministic f32 GPT via the safetensors naming convention (nlp has no
// direct constructor; mirrors backend/metal's randGPTf32 with non-trivial norms/biases so the
// LayerNorm-beta and FFN-bias paths are actually exercised).
func randGPT(t *testing.T, cfg nlp.GPTConfig) *nlp.GPT {
	t.Helper()
	seed := 0
	small := func(shape ...int) *tensor.Tensor {
		x := tensor.New(tensor.F32, tensor.Shape(shape))
		d := x.Storage().F32()
		for i := range d {
			seed++
			d[i] = float32((seed*7)%23-11) * 0.02
		}
		return x
	}
	gain := func(n int) *tensor.Tensor {
		x := tensor.New(tensor.F32, tensor.Shape{n})
		d := x.Storage().F32()
		for i := range d {
			d[i] = 0.9 + float32(i%5)*0.02
		}
		return x
	}
	bias := func(n int) *tensor.Tensor {
		x := tensor.New(tensor.F32, tensor.Shape{n})
		d := x.Storage().F32()
		for i := range d {
			d[i] = float32(i%3-1) * 0.05
		}
		return x
	}
	ffn := 4 * cfg.Dim
	ts := map[string]*tensor.Tensor{
		"tok_emb":   small(cfg.Vocab, cfg.Dim),
		"pos_emb":   small(cfg.Ctx, cfg.Dim),
		"head":      small(cfg.Dim, cfg.Vocab),
		"lnf.gamma": gain(cfg.Dim),
		"lnf.beta":  bias(cfg.Dim),
	}
	for l := range cfg.Layers {
		if l >= 10 {
			t.Fatalf("randGPT supports <10 layers")
		}
		p := func(x string) string { return "blocks." + string(rune('0'+l)) + "." + x }
		ts[p("ln1.gamma")] = gain(cfg.Dim)
		ts[p("ln1.beta")] = bias(cfg.Dim)
		ts[p("attn.wq")] = small(cfg.Dim, cfg.Dim)
		ts[p("attn.wk")] = small(cfg.Dim, cfg.Dim)
		ts[p("attn.wv")] = small(cfg.Dim, cfg.Dim)
		ts[p("attn.wo")] = small(cfg.Dim, cfg.Dim)
		ts[p("ln2.gamma")] = gain(cfg.Dim)
		ts[p("ln2.beta")] = bias(cfg.Dim)
		ts[p("ffn.w1")] = small(cfg.Dim, ffn)
		ts[p("ffn.b1")] = bias(ffn)
		ts[p("ffn.w2")] = small(ffn, cfg.Dim)
		ts[p("ffn.b2")] = bias(cfg.Dim)
	}
	m, err := nlp.FromSafetensors(cfg, ts)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// §T422: the GPTDecoder — the GPT-2-style sibling (LayerNorm, learned pos-emb, biased GELU MLP) —
// matches nlp.GPT's own per-op DecodeStep on the reference backend across an autoregressive run,
// and its batched decode is measured against the per-op path on metal.
func TestGPTDecoderMatchesReference(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal: no gpu")
	}
	cfg := nlp.GPTConfig{Vocab: 48, Ctx: 32, Dim: 64, Heads: 8, Layers: 3, Eps: 1e-5}
	m := randGPT(t, cfg)
	dec, err := llamagpu.NewGPT(m)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Release()

	refCtx := backend.NewContext().WithBackend(backend.Reference())
	cache := m.NewCache()
	tok := 5
	for pos := range 6 {
		got, err := dec.Step(tok, pos)
		if err != nil {
			t.Fatal(err)
		}
		refT, err := m.DecodeStep(refCtx, cache, tok, pos)
		if err != nil {
			t.Fatal(err)
		}
		bi := 0
		for j := range got {
			want := refT.AtF64(0, j)
			if math.Abs(float64(got[j])-want) > 3e-2*math.Max(1, math.Abs(want)) {
				t.Fatalf("pos %d logit[%d]: gpt decoder %v vs reference %v", pos, j, got[j], want)
			}
			if got[j] > got[bi] {
				bi = j
			}
		}
		tok = bi // autoregressive
	}
	t.Logf("GPTDecoder matches nlp.GPT DecodeStep across an autoregressive run (%d layers)", cfg.Layers)
}

// §T422 (§C3): batched GPT decode vs nlp per-op DecodeStep on metal, real size.
func TestGPTDecodeThroughput(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal: no gpu")
	}
	metalBE, ok := backend.Get(backend.Metal)
	if !ok {
		t.Skip("metal backend not registered")
	}
	cfg := nlp.GPTConfig{Vocab: 1024, Ctx: 128, Dim: 512, Heads: 8, Layers: 6, Eps: 1e-5}
	m := randGPT(t, cfg)
	dec, err := llamagpu.NewGPT(m)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Release()

	const steps = 32
	if _, err := dec.Step(0, 0); err != nil {
		t.Fatal(err)
	}
	s := time.Now()
	for pos := range steps {
		if _, err := dec.Step(pos%cfg.Vocab, pos); err != nil {
			t.Fatal(err)
		}
	}
	batched := float64(steps) / time.Since(s).Seconds()

	mctx := backend.NewContext().WithBackend(metalBE)
	cache := m.NewCache()
	if _, err := m.DecodeStep(mctx, cache, 0, 0); err != nil {
		t.Fatal(err)
	}
	cache = m.NewCache()
	s = time.Now()
	for pos := range steps {
		if _, err := m.DecodeStep(mctx, cache, pos%cfg.Vocab, pos); err != nil {
			t.Fatal(err)
		}
	}
	perOp := float64(steps) / time.Since(s).Seconds()
	t.Logf("GPT decode D=%d %d-layer: batched %.1f tok/s | nlp per-op(metal) %.1f tok/s | speedup %.2fx",
		cfg.Dim, cfg.Layers, batched, perOp, batched/perOp)
}

// §T423: GPT StepN — the multi-token GPT step (with the broadcast AddBias record op) — matches
// sequential single-token Steps at every position; Generate's prefill now uses it.
func TestGPTStepNMatchesSequentialSteps(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal: no gpu")
	}
	cfg := nlp.GPTConfig{Vocab: 48, Ctx: 32, Dim: 64, Heads: 8, Layers: 3, Eps: 1e-5}
	m := randGPT(t, cfg)
	tokens := []int{7, 3, 11, 0, 5, 22}

	dN, err := llamagpu.NewGPT(m)
	if err != nil {
		t.Fatal(err)
	}
	defer dN.Release()
	all, err := dN.StepN(tokens, 0)
	if err != nil {
		t.Fatal(err)
	}
	d1, err := llamagpu.NewGPT(m)
	if err != nil {
		t.Fatal(err)
	}
	defer d1.Release()
	for pos, tok := range tokens {
		want, err := d1.Step(tok, pos)
		if err != nil {
			t.Fatal(err)
		}
		got := all[pos*cfg.Vocab : (pos+1)*cfg.Vocab]
		for j := range want {
			if math.Abs(float64(got[j]-want[j])) > 2e-3*math.Max(1, math.Abs(float64(want[j]))) {
				t.Fatalf("pos %d logit[%d]: StepN %v vs Step %v", pos, j, got[j], want[j])
			}
		}
	}
	t.Logf("GPT StepN(%d tokens) == sequential Steps at every position", len(tokens))
}

// §T424: SpeculativeGenerate over the Stepper interface drives GPT decoders too — self-draft
// (~100% acceptance, full-accept path) produces the target's exact greedy sequence.
func TestGPTSpeculativeGenerate(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal: no gpu")
	}
	cfg := nlp.GPTConfig{Vocab: 48, Ctx: 64, Dim: 64, Heads: 8, Layers: 2, Eps: 1e-5}
	m := randGPT(t, cfg)
	target, err := llamagpu.NewGPT(m)
	if err != nil {
		t.Fatal(err)
	}
	defer target.Release()
	draft, err := llamagpu.NewGPT(m) // same weights → greedy full accept
	if err != nil {
		t.Fatal(err)
	}
	defer draft.Release()

	prompt := []int{5, 12, 3}
	const maxNew = 20
	got, stats, err := llamagpu.SpeculativeGenerate(target, draft, prompt, maxNew, 4, nlp.Greedy(), 7)
	if err != nil {
		t.Fatal(err)
	}
	ref, err := llamagpu.NewGPT(m)
	if err != nil {
		t.Fatal(err)
	}
	defer ref.Release()
	want, err := ref.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token[%d]: speculative %d vs plain %d", i, got[i], want[i])
		}
	}
	if stats.AcceptanceRate() < 0.99 {
		t.Fatalf("self-draft acceptance %.0f%%, want ~100%%", 100*stats.AcceptanceRate())
	}
	t.Logf("GPT speculative == plain greedy: %d tokens, acceptance %.0f%%", len(got), 100*stats.AcceptanceRate())
}

// §T426: prompt-lookup speculative decoding — no draft model, guesses copied from the sequence's
// own history — is lossless: token-for-token equal to plain greedy Generate. A prompt with a
// repeated pattern gives real n-gram acceptance (the speedup mechanism); a random prompt still
// matches (pure-rejection safety).
func TestPromptLookupGenerateMatchesPlain(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal: no gpu")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 48, Ctx: 96, Dim: 64, Heads: 8, KVHeads: 2, Layers: 2,
		Hidden: 176, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 3)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name   string
		prompt []int
	}{
		{"repetitive", []int{7, 8, 9, 10, 7, 8, 9, 10, 7, 8, 9, 10, 7, 8}}, // n-grams recur → real acceptance possible
		{"random", []int{5, 12, 3, 40, 21}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			target, err := llamagpu.New(m)
			if err != nil {
				t.Fatal(err)
			}
			defer target.Release()
			const maxNew = 24
			got, stats, err := llamagpu.PromptLookupGenerate(target, tc.prompt, maxNew, 3, 6, nlp.Greedy(), 7)
			if err != nil {
				t.Fatal(err)
			}
			ref, err := llamagpu.New(m)
			if err != nil {
				t.Fatal(err)
			}
			defer ref.Release()
			want, err := ref.Generate(tc.prompt, maxNew, nlp.Greedy())
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(want) {
				t.Fatalf("length: lookup %d vs plain %d", len(got), len(want))
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("token[%d]: lookup %d vs plain %d\nlookup=%v\nplain=%v", i, got[i], want[i], got, want)
				}
			}
			t.Logf("%s: %d tokens ok; proposed %d accepted %d (rate %.0f%%)",
				tc.name, len(got), stats.Proposed, stats.Accepted, 100*stats.AcceptanceRate())
		})
	}
}

// §T428 (§V22 value-check for a quantized KV cache): does batched decode time grow materially with
// context length? A Q8 KV cache would cut attention's cache reads 4× — worth building only if those
// reads are a real fraction. Measures Step time at short vs long positions on a long-context model.
func TestDecodeCostVsContextLength(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal: no gpu")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 1024, Ctx: 2048, Dim: 512, Heads: 8, KVHeads: 2, Layers: 6,
		Hidden: 1376, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := llamagpu.New(m)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Release()

	// fill the cache to near-Ctx with batched prefill windows.
	pos := 0
	win := make([]int, 128)
	for i := range win {
		win[i] = (i * 7) % cfg.Vocab
	}
	for pos+128 <= 1920 {
		if _, err := dec.StepN(win, pos); err != nil {
			t.Fatal(err)
		}
		pos += 128
	}

	timeAt := func(p int) float64 {
		if _, err := dec.Step(0, p); err != nil {
			t.Fatal(err)
		}
		const it = 30
		s := time.Now()
		for range it {
			if _, err := dec.Step(1, p); err != nil {
				t.Fatal(err)
			}
		}
		return time.Since(s).Seconds() / it * 1e3
	}
	short := timeAt(16)
	long := timeAt(1920)
	kvMB := float64(2*1920*cfg.KVHeads*(cfg.Dim/cfg.Heads)*4*cfg.Layers) / 1e6
	t.Logf("batched decode step D=%d 6-layer: @pos16 %.2f ms | @pos1920 %.2f ms (cache reads %.1f MB f32) | growth %.0f%%",
		cfg.Dim, short, long, kvMB, 100*(long-short)/short)
}

// §T431 (§V22 follow-up to §T428): StepN prefill windows against a LONG cache use the two-pass mha
// kernel (sq=k>1, so the §T428 decode fast path does not apply). Measure whether late prefill
// windows also hit a serial cliff.
func TestPrefillWindowCostVsPosition(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal: no gpu")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 1024, Ctx: 2048, Dim: 512, Heads: 8, KVHeads: 2, Layers: 6,
		Hidden: 1376, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := llamagpu.New(m)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Release()
	win := make([]int, 128)
	for i := range win {
		win[i] = (i * 7) % cfg.Vocab
	}
	timeWin := func(pos int) float64 {
		if _, err := dec.StepN(win, pos); err != nil {
			t.Fatal(err)
		}
		const it = 10
		s := time.Now()
		for range it {
			if _, err := dec.StepN(win, pos); err != nil {
				t.Fatal(err)
			}
		}
		return time.Since(s).Seconds() / it * 1e3
	}
	// fill so the late window has a full cache behind it
	for p := 0; p+128 <= 1792; p += 128 {
		if _, err := dec.StepN(win, p); err != nil {
			t.Fatal(err)
		}
	}
	early := timeWin(0)
	late := timeWin(1792)
	t.Logf("StepN(128) window D=%d 6-layer: @pos0 %.2f ms | @pos1792 %.2f ms | growth %.0f%%",
		cfg.Dim, early, late, 100*(late-early)/early)
}
