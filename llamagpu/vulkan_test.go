//go:build vulkan && cgo

package llamagpu_test

import (
	"testing"
	"time"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/backend/vulkan"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// §T409: the vulkan Decoder variant generates the SAME greedy token sequence as the library's
// per-op decode — the backend-agnostic core + vulkan adapter is correct end-to-end.
func TestVulkanGenerateMatchesReference(t *testing.T) {
	if !vulkan.Available() {
		t.Skip("vulkan: no compute-capable device (§V4)")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 48, Ctx: 40, Dim: 64, Heads: 8, KVHeads: 2, Layers: 3,
		Hidden: 176, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 3)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := llamagpu.NewVulkan(m)
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
			t.Fatalf("token[%d]: vulkan %d vs ref %d\ngpu=%v\nref=%v", i, gpuOut[i], refOut[i], gpuOut, refOut)
		}
	}
	t.Logf("llamagpu vulkan Decoder.Generate == nlp.Llama.Generate greedy: %d tokens", len(gpuOut))
}

// §T409 (§C3): batched vulkan decode vs the library's per-op DecodeStep on the SAME vulkan backend,
// real model, tokens/s — the vulkan analog of the metal §T404 24× measurement.
func TestVulkanDecodeBatchedVsPerOpThroughput(t *testing.T) {
	if !vulkan.Available() {
		t.Skip("vulkan: no compute-capable device (§V4)")
	}
	vkBE, ok := backend.Get(backend.Vulkan)
	if !ok {
		t.Skip("vulkan backend not registered")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 1024, Ctx: 128, Dim: 512, Heads: 8, KVHeads: 2, Layers: 6,
		Hidden: 1376, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := llamagpu.NewVulkan(m)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Release()

	const steps = 32
	if _, err := dec.Step(0, 0); err != nil { // warmup
		t.Fatal(err)
	}
	sRec := time.Now()
	for pos := range steps {
		if _, err := dec.Step(pos%cfg.Vocab, pos); err != nil {
			t.Fatal(err)
		}
	}
	recTokps := float64(steps) / time.Since(sRec).Seconds()

	vctx := backend.NewContext().WithBackend(vkBE)
	cache := m.NewCache()
	if _, err := m.DecodeStep(vctx, cache, 0, 0); err != nil {
		t.Fatal(err)
	}
	cache = m.NewCache()
	sPer := time.Now()
	for pos := range steps {
		if _, err := m.DecodeStep(vctx, cache, pos%cfg.Vocab, pos); err != nil {
			t.Fatal(err)
		}
	}
	perTokps := float64(steps) / time.Since(sPer).Seconds()

	t.Logf("vulkan Llama decode D=%d GQA%d:%d %d-layer V=%d: batched %.1f tok/s | nlp per-op DecodeStep %.1f tok/s | speedup %.2fx",
		cfg.Dim, cfg.Heads, cfg.KVHeads, cfg.Layers, cfg.Vocab, recTokps, perTokps, recTokps/perTokps)
}

// §T429: vulkan long-context decode after the cooperative subgroup kernel (twin of the metal §T428
// measurement).
func TestVulkanDecodeCostVsContextLength(t *testing.T) {
	if !vulkan.Available() {
		t.Skip("vulkan: no compute-capable device (§V4)")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 1024, Ctx: 2048, Dim: 512, Heads: 8, KVHeads: 2, Layers: 6,
		Hidden: 1376, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := llamagpu.NewVulkan(m)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Release()
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
		const it = 20
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
	t.Logf("vulkan batched decode step D=%d 6-layer: @pos16 %.2f ms | @pos1920 %.2f ms | growth %.0f%%",
		cfg.Dim, short, long, 100*(long-short)/short)
}

// §T431: vulkan prefill-window cost after the generalized cooperative kernel (twin of the metal
// measurement — the two-pass kernel hit the same serial cliff on late windows).
func TestVulkanPrefillWindowCostVsPosition(t *testing.T) {
	if !vulkan.Available() {
		t.Skip("vulkan: no compute-capable device (§V4)")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 1024, Ctx: 2048, Dim: 512, Heads: 8, KVHeads: 2, Layers: 6,
		Hidden: 1376, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := llamagpu.NewVulkan(m)
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
	for p := 0; p+128 <= 1792; p += 128 {
		if _, err := dec.StepN(win, p); err != nil {
			t.Fatal(err)
		}
	}
	early := timeWin(0)
	late := timeWin(1792)
	t.Logf("vulkan StepN(128) window D=%d 6-layer: @pos0 %.2f ms | @pos1792 %.2f ms | growth %.0f%%",
		cfg.Dim, early, late, 100*(late-early)/early)
}
