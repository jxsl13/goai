package benchcompare

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"sort"
	"strconv"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// TestProdCPUQuantDecodeGGUF measures the forward-only CPU decode boundary used
// by llama-bench on the same production GGUF. It is gated because the fixture
// is external; CI keeps exercising the underlying model and quantized-kernel
// correctness suites without downloading a 1.1B-parameter checkpoint.
func TestProdCPUQuantDecodeGGUF(t *testing.T) {
	path := os.Getenv("GOAI_CPU_TINYLLAMA_GGUF")
	if path == "" {
		t.Skip("set GOAI_CPU_TINYLLAMA_GGUF to a quantized Llama GGUF")
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	raw, err := gguf.ReadRaw(f)
	if err != nil {
		t.Fatalf("gguf.ReadRaw: %v", err)
	}
	model, err := nlp.QuantLlamaFromGGUF(raw.Metadata, raw.Tensors)
	if err != nil {
		t.Fatalf("nlp.QuantLlamaFromGGUF: %v", err)
	}
	defer model.Close()

	steps := envPositiveInt(t, "GOAI_CPU_DECODE_STEPS", 64)
	reps := envPositiveInt(t, "GOAI_CPU_DECODE_REPS", 7)
	if model.Config.Ctx < steps {
		t.Fatalf("model context %d is shorter than %d decode steps", model.Config.Ctx, steps)
	}

	run := func() (time.Duration, uint64, *nlp.LlamaCache, uint64, uint64) {
		cpu, ok := backend.Get(backend.CPU)
		if !ok {
			t.Fatal("CPU backend is not registered")
		}
		ctx := backend.NewContext().WithBackend(cpu)
		cache := model.NewCache()
		var logits *tensor.Tensor
		var memBefore, memAfter runtime.MemStats
		runtime.ReadMemStats(&memBefore)
		start := time.Now()
		for pos := range steps {
			logits, err = model.DecodeStep(ctx, cache, 1+(pos*131)%min(model.Config.Vocab-1, 30000), pos)
			if err != nil {
				t.Fatalf("DecodeStep(%d): %v", pos, err)
			}
		}
		duration := time.Since(start)
		runtime.ReadMemStats(&memAfter)
		var digest uint64 = 1469598103934665603
		for i := range logits.Numel() {
			value := logits.AtF64(0, i)
			if math.IsNaN(value) || math.IsInf(value, 0) {
				t.Fatalf("final logit %d is non-finite: %g", i, value)
			}
			digest ^= math.Float64bits(value)
			digest *= 1099511628211
		}
		return duration, digest, cache, memAfter.TotalAlloc - memBefore.TotalAlloc, memAfter.Mallocs - memBefore.Mallocs
	}

	_, _, warmCache, _, _ := run() // process-local model and worker-pool warmup; deliberately excluded
	heldCaches := make([]*nlp.LlamaCache, 0, reps+1)
	heldCaches = append(heldCaches, warmCache) // prevent prior-run cache collection from contaminating later samples
	samples := make([]time.Duration, 0, reps)
	byteSamples := make([]uint64, 0, reps)
	allocSamples := make([]uint64, 0, reps)
	var wantDigest uint64
	for i := range reps {
		d, digest, cache, allocBytes, allocs := run()
		heldCaches = append(heldCaches, cache)
		if i == 0 {
			wantDigest = digest
		} else if digest != wantDigest {
			t.Fatalf("forward-only output digest changed: run %d=%016x want %016x", i+1, digest, wantDigest)
		}
		samples = append(samples, d)
		byteSamples = append(byteSamples, allocBytes)
		allocSamples = append(allocSamples, allocs)
	}
	sort.Slice(samples, func(i, j int) bool { return samples[i] < samples[j] })
	sort.Slice(byteSamples, func(i, j int) bool { return byteSamples[i] < byteSamples[j] })
	sort.Slice(allocSamples, func(i, j int) bool { return allocSamples[i] < allocSamples[j] })
	median := samples[len(samples)/2]
	fmt.Printf("GOAI_CPU_PROD threads=%d steps=%d reps=%d median=%s throughput=%.3f tok/s digest=%016x alloc_bytes_median=%d allocs_median=%d samples=%v\n",
		runtime.GOMAXPROCS(0), steps, reps, median, float64(steps)/median.Seconds(), wantDigest,
		byteSamples[len(byteSamples)/2], allocSamples[len(allocSamples)/2], samples)
	runtime.KeepAlive(heldCaches)
}

func envPositiveInt(t *testing.T, name string, fallback int) int {
	t.Helper()
	value := os.Getenv(name)
	if value == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed <= 0 {
		t.Fatalf("%s=%q must be a positive integer", name, value)
	}
	return parsed
}
