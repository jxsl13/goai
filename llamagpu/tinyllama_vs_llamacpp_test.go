//go:build darwin && cgo

package llamagpu_test

import (
	"bufio"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// TestTinyLlamaVsLlamaCpp decodes the SAME GGUF file llama-bench measures, so the two
// numbers are directly comparable rather than related by argument.
//
// The llama.cpp side is MEASURED here whenever llama-bench is on PATH, not read from a constant.
// A hardcoded incumbent rots: the 172.19 t/s recorded for build 48d22e295 was still in this file
// when llama-bench on the same host and the same file measured 201.61 +/- 3.25 t/s, so every ratio
// computed against it flattered GoAI by ~17%. The fallback constant below is only used when
// llama-bench is absent, and it is stamped with the build that produced it precisely because it is
// expected to go stale.
//
// Read the RATIO, not the absolute numbers. Both sides are measured in the same session precisely
// so thermal drift cancels: back-to-back runs on this host gave GoAI 143.17 / llama.cpp 201.61
// (cold, separate runs) and GoAI 112.84 / llama.cpp 155.09 (hot, one run) — absolute values 20%
// apart, ratios 0.710 and 0.728. The ratio is what survives.
//
// Skips when the model is absent so the suite stays hermetic; GOAI_TINYLLAMA_GGUF
// overrides the path.
func TestTinyLlamaVsLlamaCpp(t *testing.T) {
	if testing.Short() {
		t.Skip("1.1B model; skipped in -short")
	}
	if !metal.Available() {
		t.Skip("metal unavailable")
	}
	path := os.Getenv("GOAI_TINYLLAMA_GGUF")
	if path == "" {
		path = "../models/tinyllama-1.1b-q4km.gguf"
	}
	f, err := os.Open(path)
	if err != nil {
		t.Skipf("model not present: %v", err)
	}
	defer f.Close()
	raw, err := gguf.ReadRaw(f)
	if err != nil {
		t.Skipf("ReadRaw: %v", err)
	}
	qm, err := nlp.QuantLlamaFromGGUF(raw.Metadata, raw.Tensors)
	if err != nil {
		t.Skipf("QuantLlamaFromGGUF: %v", err)
	}
	defer qm.Close()

	dec, err := llamagpu.NewQuant(qm)
	if err != nil {
		t.Skipf("NewQuant: %v", err)
	}
	defer dec.Release()

	const genN = 64
	// DECODE ONLY, to match what llama-bench -p 0 -n 64 reports. tg64 is token GENERATION; it does
	// not include prompt processing. Timing dec.Generate instead charges GoAI for a prefill the
	// incumbent is not charged for — on this model that prefill is ~93 ms, which dragged an
	// otherwise-at-parity decode down to an apparent 0.71x.
	sample := func() float64 {
		start := time.Now()
		for i := range genN {
			if _, err := dec.Step(7, 10+i); err != nil {
				t.Fatal(err)
			}
		}
		return float64(genN) / time.Since(start).Seconds()
	}
	sample() // warm
	var tps []float64
	for range 3 {
		tps = append(tps, sample())
	}
	got := coopMedianMetal(tps)
	llamaCpp, src := llamaCppTG64(t, path)
	t.Logf("TinyLlama-1.1B Q4_K_M tg64 (decode only), Metal: GoAI %.2f tok/s vs llama.cpp %.2f tok/s (%s) = %.3fx  (samples %v)",
		got, llamaCpp, src, got/llamaCpp, tps)

	// Prompt processing, separately. This is GoAI's real deficiency and tg64 does not show it:
	// llama-bench pp64 measures 1778.75 tok/s (0.56 ms/prompt token) where GoAI's prefill costs
	// ~15.5 ms/prompt token — roughly 28x. The M>1 path does not reach the cooperative M=1 kernels.
	long := make([]int, 64)
	for i := range long {
		long[i] = 1 + i%2000
	}
	pstart := time.Now()
	if _, err := dec.StepNLast(long, 0); err != nil {
		t.Fatal(err)
	}
	pp := float64(len(long)) / time.Since(pstart).Seconds()
	t.Logf("TinyLlama-1.1B Q4_K_M pp64 (prompt processing): GoAI %.1f tok/s vs llama.cpp 1778.75 tok/s = %.3fx", pp, pp/1778.75)
}

// llamaCppTG64 returns llama.cpp's tg64 throughput on the SAME file, measured live when llama-bench
// is available. Returns the recorded fallback otherwise, naming which one was used so a reader can
// tell a live comparison from a stale one.
func llamaCppTG64(t *testing.T, model string) (float64, string) {
	t.Helper()
	// Recorded 2026-08-15 on an M2 Pro, llama.cpp build 48d22e295 (10360), ggml 0.19.0, Metal+BLAS:
	//   llama-bench -m tinyllama-1.1b-q4km.gguf -p 0 -n 64 -r 3 -> tg64 201.61 +/- 3.25 t/s
	const recorded = 201.61
	bin, err := exec.LookPath("llama-bench")
	if err != nil {
		return recorded, "recorded, build 48d22e295 — llama-bench not on PATH"
	}
	out, err := exec.Command(bin, "-m", model, "-p", "0", "-n", "64", "-r", "3").CombinedOutput()
	if err != nil {
		return recorded, "recorded, build 48d22e295 — llama-bench failed"
	}
	// The result table row ends "| <tps> ± <stddev> |"; take the tg64 row's throughput.
	re := regexp.MustCompile(`\|\s*tg64\s*\|\s*([0-9.]+)\s*±`)
	sc := bufio.NewScanner(strings.NewReader(string(out)))
	for sc.Scan() {
		if m := re.FindStringSubmatch(sc.Text()); m != nil {
			if v, e := strconv.ParseFloat(m[1], 64); e == nil {
				return v, "measured live"
			}
		}
	}
	return recorded, "recorded, build 48d22e295 — could not parse llama-bench output"
}
