//go:build cuda && cgo && (linux || windows)

package llamagpu

import (
	"os"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
)

const glModelPath = "../models/tinyllama-1.1b-chat-q8_0.gguf"

// generateEager decodes greedily using stepEager for EVERY token (no graph) — the reference the
// graph path must reproduce exactly.
func (gd *GraphLlamaDecoder) generateEager(prompt []int, maxNew int) ([]int, error) {
	out := append([]int(nil), prompt...)
	var logits []float32
	for i, tok := range prompt {
		l, err := gd.stepEager(tok, i)
		if err != nil {
			return nil, err
		}
		logits = l
	}
	pos := len(prompt)
	for range maxNew {
		if pos >= gd.maxLen {
			break
		}
		next := glTestArgmax(logits)
		out = append(out, next)
		l, err := gd.stepEager(next, pos)
		if err != nil {
			return nil, err
		}
		logits = l
		pos++
	}
	return out, nil
}

// TestGraphLlamaDecodeMatchesEager proves the CUDA-graph decode path reproduces pure-eager execution
// token-for-token: both run the identical forwardBody op chain, so graph replay (capture-once +
// pos.Set/emb/Launch per token) must equal eager launches. A divergence means the graph capture or
// the device-pos buffer update is wrong. Dims are multiples of 256 (Q4_K super-block).
func TestGraphLlamaDecodeMatchesEager(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 256, Ctx: 128, Dim: 256, Heads: 8, KVHeads: 2, Layers: 4,
		Hidden: 512, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 5)
	if err != nil {
		t.Fatal(err)
	}
	prompt := []int{1, 9, 42, 17}
	const maxNew = 24

	// eager reference
	gdE, err := NewLlamaQ4KGraphCUDA(m, cfg.Ctx)
	if err != nil {
		t.Fatal(err)
	}
	eager, err := gdE.generateEager(prompt, maxNew)
	gdE.Release()
	if err != nil {
		t.Fatal(err)
	}

	// graph path (Generate uses stepGraph for decode)
	gdG, err := NewLlamaQ4KGraphCUDA(m, cfg.Ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer gdG.Release()
	graph, err := gdG.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}

	if len(graph) != len(eager) {
		t.Fatalf("length mismatch: graph %d, eager %d", len(graph), len(eager))
	}
	for i := range graph {
		if graph[i] != eager[i] {
			t.Fatalf("token %d: graph=%d eager=%d (graph decode diverges from eager)", i, graph[i], eager[i])
		}
	}
	t.Logf("GraphLlamaDecoder: graph decode == eager decode over %d tokens (capture/replay correct)", len(graph))
}

func glTestArgmax(v []float32) int {
	best, bi := v[0], 0
	for i, x := range v {
		if x > best {
			best, bi = x, i
		}
	}
	return bi
}

// TestGraphLlamaDecodePureSpeed times ONLY the captured-graph decode replays on real TinyLlama (prefill
// + capture excluded), the apples-to-apples match for llama.cpp llama-bench tg (Vulkan Q8 = 244 tok/s).
// This is the production form of the q4kGraphDecoder POC (#885, 253.5 tok/s).
func TestGraphLlamaDecodePureSpeed(t *testing.T) {
	if testing.Short() {
		t.Skip("model-loading integration test")
	}
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	if _, err := os.Stat(glModelPath); err != nil {
		t.Skipf("model not present (%s)", glModelPath)
	}
	f, err := gguf.ReadFile(glModelPath)
	if err != nil {
		t.Fatal(err)
	}
	model, err := nlp.LlamaFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	const prefill, steps = 8, 64
	gd, err := NewLlamaQ4KGraphCUDA(model, prefill+steps+4)
	if err != nil {
		t.Fatal(err)
	}
	defer gd.Release()
	for i := 0; i < prefill; i++ {
		if _, err := gd.stepEager((i*131+1)%model.Config.Vocab, i); err != nil {
			t.Fatal(err)
		}
	}
	if err := gd.captureGraph(); err != nil {
		t.Fatal(err)
	}
	// warm the graph once, then time pure decode replays with GPU argmax (1-int transfer/token).
	tk := 1
	if err := gd.pos.Set(prefill); err != nil {
		t.Fatal(err)
	}
	if err := gd.emb.EmbedInto([]int32{int32(tk)}, gd.dx); err != nil {
		t.Fatal(err)
	}
	must0(t, gd.graph.Launch())
	must0(t, cuda.GraphSync())
	t0 := time.Now()
	for d := 0; d < steps; d++ {
		tk = gd.logits.Argmax()
		must0(t, gd.pos.Set(prefill+1+d))
		must0(t, gd.emb.EmbedInto([]int32{int32(tk)}, gd.dx))
		must0(t, gd.graph.Launch())
		must0(t, cuda.GraphSync())
	}
	tps := float64(steps) / time.Since(t0).Seconds()
	t.Logf("production GraphLlamaDecoder pure graph decode: %.1f tok/s (POC 253.5; llama.cpp Vulkan Q8 244)", tps)
	if tps < 200 {
		t.Fatalf("graph decode slower than expected: %.1f tok/s", tps)
	}
}

func must0(t *testing.T, err error) {
	if err != nil {
		t.Helper()
		t.Fatal(err)
	}
}

// TestGraphLlamaBatchedPrefillSpeed times batched prefill vs the naive per-token eager prefill for a
// realistic prompt length on real TinyLlama — the batched pass reads each weight ONCE across M rows
// (WMMA tensor-core GEMM at M>=48) instead of M weight-bandwidth-bound M=1 GEMVs.
func TestGraphLlamaBatchedPrefillSpeed(t *testing.T) {
	if testing.Short() {
		t.Skip("model-loading integration test")
	}
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	if _, err := os.Stat(glModelPath); err != nil {
		t.Skipf("model not present (%s)", glModelPath)
	}
	f, err := gguf.ReadFile(glModelPath)
	if err != nil {
		t.Fatal(err)
	}
	model, err := nlp.LlamaFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	const M = 128
	prompt := make([]int, M)
	for i := range prompt {
		prompt[i] = (i*131 + 1) % model.Config.Vocab
	}
	// batched
	gdB, err := NewLlamaQ4KGraphCUDA(model, M+8)
	if err != nil {
		t.Fatal(err)
	}
	must0(t, func() error { _, e := gdB.prefillForward(prompt); return e }())
	must0(t, cuda.GraphSync())
	t0 := time.Now()
	_, err = gdB.prefillForward(prompt)
	must0(t, err)
	must0(t, cuda.GraphSync())
	batched := time.Since(t0)
	gdB.Release()
	// eager per-token
	gdE, err := NewLlamaQ4KGraphCUDA(model, M+8)
	if err != nil {
		t.Fatal(err)
	}
	defer gdE.Release()
	for i, tok := range prompt {
		if _, err := gdE.stepEager(tok, i); err != nil {
			t.Fatal(err)
		}
	}
	must0(t, cuda.GraphSync())
	t1 := time.Now()
	for i, tok := range prompt {
		if _, err := gdE.stepEager(tok, i); err != nil {
			t.Fatal(err)
		}
	}
	must0(t, cuda.GraphSync())
	eager := time.Since(t1)
	t.Logf("prefill M=%d: batched %v vs eager-per-token %v (%.2fx)", M, batched, eager, float64(eager)/float64(batched))
}
