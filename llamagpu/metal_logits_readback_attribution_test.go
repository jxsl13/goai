//go:build darwin && cgo

package llamagpu

import (
	"os"
	"slices"
	"sort"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
)

// TestMetalLogitsReadbackAttribution isolates the host boundary that distinguishes Decoder.Step
// from llama-bench's decode loop. Pinned llama-bench b10450 calls llama_decode and synchronizes but
// never requests logits; Step additionally allocates vocab floats and copies the complete resident
// logits buffer out of Metal shared memory. This gated real-model campaign alternates both arms so
// that a leadership comparison can use matched measurement boundaries instead of attributing the
// host copy to Metal forward execution.
func TestMetalLogitsReadbackAttribution(t *testing.T) {
	if testing.Short() {
		t.Skip("1.1B model; skipped in -short")
	}
	if !metal.Available() {
		t.Skip("Metal unavailable")
	}
	path := os.Getenv("GOAI_TINYLLAMA_GGUF")
	if path == "" {
		t.Skip("set GOAI_TINYLLAMA_GGUF to the pinned TinyLlama GGUF")
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	raw, err := gguf.ReadRaw(f)
	if err != nil {
		t.Fatal(err)
	}
	model, err := nlp.QuantLlamaFromGGUF(raw.Metadata, raw.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	dec, err := NewQuantF16KV(model)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Release()

	const (
		steps  = 64
		rounds = 10
	)
	if dec.maxLen < steps {
		t.Fatalf("model context %d is shorter than %d-step attribution campaign", dec.maxLen, steps)
	}

	// Warm every path, including the shared-memory copy itself.
	if _, err := dec.Step(1, 0); err != nil {
		t.Fatal(err)
	}
	warm := make([]float32, dec.v)
	if err := dec.logits.b.DownloadF32(warm); err != nil {
		t.Fatal(err)
	}

	measure := func(download bool) time.Duration {
		start := time.Now()
		for i := range steps {
			tok := 1 + (i*37)%min(dec.v-1, 30000)
			if download {
				if _, err := dec.Step(tok, i); err != nil {
					t.Fatal(err)
				}
			} else if err := dec.stepInto(tok, i); err != nil {
				t.Fatal(err)
			}
		}
		return time.Since(start)
	}

	withDownload := make([]time.Duration, 0, rounds)
	deviceOnly := make([]time.Duration, 0, rounds)
	for round := range rounds {
		// Reverse the order every round to bracket warm-up and thermal drift.
		if round%2 == 0 {
			withDownload = append(withDownload, measure(true))
			deviceOnly = append(deviceOnly, measure(false))
		} else {
			deviceOnly = append(deviceOnly, measure(false))
			withDownload = append(withDownload, measure(true))
		}
	}

	median := func(v []time.Duration) time.Duration {
		cpy := append([]time.Duration(nil), v...)
		sort.Slice(cpy, func(i, j int) bool { return cpy[i] < cpy[j] })
		return cpy[len(cpy)/2]
	}
	md, mn := median(withDownload), median(deviceOnly)
	if md <= 0 || mn <= 0 {
		t.Fatalf("invalid medians with-download=%s device-only=%s", md, mn)
	}
	t.Logf("TinyLlama f16-KV tg%d matched forward: Step+%d-logit host copy %s (%.2f tok/s), device-only %s (%.2f tok/s), boundary factor %.5fx",
		steps, dec.v, md, float64(steps)/md.Seconds(), mn, float64(steps)/mn.Seconds(), float64(md)/float64(mn))
	t.Logf("with-download samples: %v", withDownload)
	t.Logf("device-only samples: %v", deviceOnly)

	// Measure the already-synchronized shared-memory copy separately. Reuse the destination so this
	// excludes Step's slice allocation and reports the physical host-boundary floor.
	dst := make([]float32, dec.v)
	const copies = 1000
	copyStart := time.Now()
	for range copies {
		if err := dec.logits.b.DownloadF32(dst); err != nil {
			t.Fatal(err)
		}
	}
	copyElapsed := time.Since(copyStart)
	t.Logf("resident %d-logit DownloadF32: %s/copy over %d copies", dec.v, copyElapsed/copies, copies)

	// Generate's host fallback also widens every logit to f64 and runs the sampler over the complete
	// vocabulary. That CPU work, not the UMA memcpy above, is the potential device-TopK lever.
	buf := make([]float64, dec.v)
	measureSampler := func(name string, sampler nlp.TokenSampler) {
		const samples = 200
		start := time.Now()
		for range samples {
			for i, v := range dst {
				buf[i] = float64(v)
			}
			_ = sampler.SampleWithHistory(buf, nil)
		}
		elapsed := time.Since(start)
		t.Logf("host %s over %d logits: %s/sample (%d samples)", name, dec.v, elapsed/samples, samples)
	}
	measureSampler("greedy", nlp.Greedy())
	measureSampler("temperature-0.8/top-k-40/top-p-0.9", nlp.NewSampler(7,
		nlp.WithTemperature(0.8), nlp.WithTopK(40), nlp.WithTopP(0.9),
	))

	if _, ok := dec.logits.b.(deviceTopKer); !ok {
		t.Fatal("Metal logits buffer does not expose deviceTopKer")
	}
	prompt := []int{1, 15043, 29892, 590, 1024, 338}
	newSampler := func() nlp.TokenSampler {
		return nlp.NewSampler(20260818,
			nlp.WithTemperature(0.8), nlp.WithTopK(40), nlp.WithTopP(0.9),
		)
	}
	oldSwitch, hadSwitch := os.LookupEnv("GOAI_DEVICE_TOPK_SAMPLE")
	defer func() {
		if hadSwitch {
			_ = os.Setenv("GOAI_DEVICE_TOPK_SAMPLE", oldSwitch)
		} else {
			_ = os.Unsetenv("GOAI_DEVICE_TOPK_SAMPLE")
		}
	}()
	setFast := func(enabled bool) {
		if enabled {
			_ = os.Unsetenv("GOAI_DEVICE_TOPK_SAMPLE")
		} else {
			_ = os.Setenv("GOAI_DEVICE_TOPK_SAMPLE", "0")
		}
	}
	generate := func(fast bool) ([]int, time.Duration) {
		setFast(fast)
		start := time.Now()
		out, err := dec.Generate(prompt, steps, newSampler())
		if err != nil {
			t.Fatal(err)
		}
		return out, time.Since(start)
	}
	fastTokens, _ := generate(true)
	fallbackTokens, _ := generate(false)
	if !slices.Equal(fastTokens, fallbackTokens) {
		for i := range min(len(fastTokens), len(fallbackTokens)) {
			if fastTokens[i] != fallbackTokens[i] {
				t.Fatalf("Metal resident TopK changed generated token %d: fast=%d fallback=%d", i, fastTokens[i], fallbackTokens[i])
			}
		}
		t.Fatalf("Metal resident TopK changed generated length: fast=%d fallback=%d", len(fastTokens), len(fallbackTokens))
	}

	fastGeneration := make([]time.Duration, 0, rounds)
	fallbackGeneration := make([]time.Duration, 0, rounds)
	for round := range rounds {
		if round%2 == 0 {
			_, d := generate(true)
			fastGeneration = append(fastGeneration, d)
			_, d = generate(false)
			fallbackGeneration = append(fallbackGeneration, d)
		} else {
			_, d := generate(false)
			fallbackGeneration = append(fallbackGeneration, d)
			_, d = generate(true)
			fastGeneration = append(fastGeneration, d)
		}
	}
	mFast, mFallback := median(fastGeneration), median(fallbackGeneration)
	factor := float64(mFallback) / float64(mFast)
	t.Logf("TinyLlama sampled tg%d: resident TopK %s (%.2f tok/s), full-vocab fallback %s (%.2f tok/s), speedup %.5fx; exact %d-token parity",
		steps, mFast, float64(steps)/mFast.Seconds(), mFallback, float64(steps)/mFallback.Seconds(), factor, len(fastTokens))
	t.Logf("resident-TopK samples: %v", fastGeneration)
	t.Logf("full-vocab fallback samples: %v", fallbackGeneration)
	if factor < 1.05 {
		t.Fatalf("Metal resident TopK speedup %.5fx is below the 1.05x promotion gate", factor)
	}
}
