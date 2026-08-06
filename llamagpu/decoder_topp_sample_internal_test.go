//go:build cuda && cgo && (linux || windows)

package llamagpu

import (
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/nlp"
)

// TestGraphLlamaTopPSampleMatchesFallback proves the on-device PURE-top-p (no top-k) sampling fast-path
// — device TopK(256) + full-vocab softmax stats + nlp.SampleTopPFromCandidates, with a host fallback on
// nucleus overflow — reproduces the full-vocab CPU sampler (GOAI_CUDA_TOPK_SAMPLE=0) token-for-token for
// a penalty-free top-p sampler with the same seed. The device path uses a tree-reduced double Zexp vs the
// CPU's sequential f64 sum, so agreement is expected everywhere except astronomically rare ulp-boundary
// coincidences; this small run must match exactly.
func TestGraphLlamaTopPSampleMatchesFallback(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 2048, Ctx: 128, Dim: 256, Heads: 8, KVHeads: 2, Layers: 4,
		Hidden: 512, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 5)
	if err != nil {
		t.Fatal(err)
	}
	prompt := []int{1, 9, 42, 17}
	const maxNew = 48
	// PURE top-p (no top-k) — this is the config that previously fell to the whole-vocab host path.
	mkSampler := func() nlp.TokenSampler {
		return nlp.NewSampler(1234, nlp.WithTemperature(0.8), nlp.WithTopP(0.92))
	}
	gdF, err := NewLlamaQ4KGraphCUDA(m, cfg.Ctx)
	if err != nil {
		t.Fatal(err)
	}
	fast, err := gdF.Generate(prompt, maxNew, mkSampler())
	gdF.Release()
	if err != nil {
		t.Fatalf("fast Generate: %v", err)
	}
	t.Setenv("GOAI_CUDA_TOPK_SAMPLE", "0")
	gdS, err := NewLlamaQ4KGraphCUDA(m, cfg.Ctx)
	if err != nil {
		t.Fatal(err)
	}
	slow, err := gdS.Generate(prompt, maxNew, mkSampler())
	gdS.Release()
	if err != nil {
		t.Fatalf("fallback Generate: %v", err)
	}
	if len(fast) != len(slow) {
		t.Fatalf("length mismatch: fast %d, fallback %d", len(fast), len(slow))
	}
	for i := range fast {
		if fast[i] != slow[i] {
			t.Fatalf("token %d: fast=%d fallback=%d — on-device pure-top-p sampling diverges from full-vocab", i, fast[i], slow[i])
		}
	}
	t.Logf("on-device pure-top-p sampling == full-vocab fallback over %d tokens (TopP=0.92, temp=0.8)", len(fast))
}

// benchTopPStep isolates the PER-TOKEN sampling cost of pure-top-p decode at large vocab: the on-device
// fast-path (TopK(256)+SoftmaxStatsN+SampleTopPFromCandidates, host fallback on overflow) vs the
// full-vocab fallback (ToHost + host Sample). Peaked logits (small nucleus) resolve on-device; flat
// logits overflow, so the fast path pays TopK+stats THEN falls back (the worst case).
func benchTopPStep(b *testing.B, vocab int, peaked, fast bool) {
	if !cuda.Available() {
		b.Skip("cuda: no CUDA-capable device")
	}
	logits := make([]float32, vocab)
	r := rand.New(rand.NewPCG(42, 42))
	for i := range logits {
		logits[i] = float32(r.NormFloat64()) * 0.05 // near-flat base
	}
	if peaked {
		for k := 0; k < 30; k++ {
			logits[r.IntN(vocab)] = float32(8 + 4*r.Float64()) // a small confident nucleus
		}
	}
	d, _ := cuda.NewDeviceF32(1, vocab)
	if err := d.UploadF32(logits); err != nil {
		b.Fatal(err)
	}
	defer d.Free()
	sp := nlp.NewSampler(1, nlp.WithTemperature(0.8), nlp.WithTopP(0.9))
	buf := make([]float64, vocab)
	doFallback := func() {
		l, _ := d.ToHost()
		for j, x := range l.Storage().F32() {
			buf[j] = float64(x)
		}
		sp.Sample(buf)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if fast {
			maxL, z, _ := d.SoftmaxStatsN(vocab, 0.8)
			resolved := false
			if float64(64)/z >= 0.9 { // overflow guard (mirrors the wiring)
				gi, gv, _ := d.TopK(64)
				cl := make([]float64, len(gv))
				for j, v := range gv {
					cl[j] = float64(v)
				}
				if _, ok := sp.SampleTopPFromCandidates(cl, gi, maxL, z); ok {
					resolved = true
				}
			}
			if !resolved {
				doFallback()
			}
		} else {
			doFallback()
		}
	}
}

func BenchmarkTopPStep_Peaked_Fast(b *testing.B)     { benchTopPStep(b, 128000, true, true) }
func BenchmarkTopPStep_Peaked_Fallback(b *testing.B) { benchTopPStep(b, 128000, true, false) }
func BenchmarkTopPStep_Flat_Fast(b *testing.B)       { benchTopPStep(b, 128000, false, true) }
func BenchmarkTopPStep_Flat_Fallback(b *testing.B)   { benchTopPStep(b, 128000, false, false) }

func benchTopPComponent(b *testing.B, which string) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	const vocab = 128000
	logits := make([]float32, vocab)
	r := rand.New(rand.NewPCG(42, 42))
	for i := range logits {
		logits[i] = float32(r.NormFloat64())
	}
	d, _ := cuda.NewDeviceF32(1, vocab)
	d.UploadF32(logits)
	defer d.Free()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		switch which {
		case "topk256":
			d.TopK(64)
		case "topk40":
			d.TopK(40)
		case "stats":
			d.SoftmaxStatsN(vocab, 0.8)
		case "tohost":
			d.ToHost()
		}
	}
}
func BenchmarkComp_TopK256(b *testing.B) { benchTopPComponent(b, "topk256") }
func BenchmarkComp_TopK40(b *testing.B)  { benchTopPComponent(b, "topk40") }
func BenchmarkComp_Stats(b *testing.B)   { benchTopPComponent(b, "stats") }
func BenchmarkComp_ToHost(b *testing.B)  { benchTopPComponent(b, "tohost") }
