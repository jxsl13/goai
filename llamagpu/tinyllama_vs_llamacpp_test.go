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

	// BRACKET the incumbent: measure llama.cpp BEFORE and AFTER GoAI's samples and take the
	// median of the two, so warm-up and thermal drift bias both sides instead of only one.
	// Measuring each side in one contiguous block is not enough, and this file's claim that "the
	// ratio is what survives" does not hold: two consecutive runs put llama.cpp's OWN tg64 at
	// 162.58 and 200.19 t/s — 23% apart — which moved the reported ratio from 1.019x to 0.843x
	// and would have read as GoAI overtaking llama.cpp on decode. It had not.
	llamaCpp, src := llamaCppTG64(t, path)
	if second, s2 := llamaCppTG64(t, path); s2 == "measured live" && src == "measured live" {
		lo, hi := min(llamaCpp, second), max(llamaCpp, second)
		if hi/lo > 1.10 {
			t.Logf("NOTE: llama.cpp tg64 spread %.2f..%.2f (%.0f%%) — the host is not quiet enough "+
				"for a %.3fx-precision claim; read the ratio as a band", lo, hi, 100*(hi/lo-1), got/hi)
		}
		llamaCpp = (llamaCpp + second) / 2
	}
	t.Logf("TinyLlama-1.1B Q4_K_M tg64 (decode only), Metal: GoAI %.2f tok/s vs llama.cpp %.2f tok/s (%s, bracketed) = %.3fx  (samples %v)",
		got, llamaCpp, src, got/llamaCpp, tps)

	// Prompt processing, at SEVERAL lengths. pp64 alone is misleading: GoAI's deficit is
	// concentrated at short prompts, where fixed weight expansion dominates, and shrinks as the
	// prompt grows and attention (now on the matrix-unit flash kernel) amortizes. Measuring only
	// the length llama-bench defaults to hid that for most of this work.
	//
	// Both sides measured in one session, M2 Pro:
	//
	// Both sides measured in one session, M2 Pro, 2026-08-16 — and llama.cpp's side is measured
	// LIVE below rather than read from this table, because the table it replaces had gone stale in
	// GoAI's DISfavour: it recorded pp64 0.46 where a live run now measures 0.675, and tg64 0.97
	// where GoAI is now AHEAD at 1.019. A stale incumbent misleads in whichever direction the code
	// moved; the only fix is to stop writing the ratio down.
	//
	//	          GoAI    llama.cpp   ratio (3 runs)
	//	pp64    ~1177       ~1720      0.664 - 0.695
	//	pp256   ~1910       ~2100      0.904 - 0.909
	//	pp1024  ~1980       ~2090      0.933 - 0.951
	//	tg64     ~160        ~176      0.843 - 0.907   (see the variance note below)
	//
	// Against the table this replaces (pp64 0.46, pp256 0.75, pp1024 0.84) prefill has improved
	// substantially; the stale comment was understating GoAI by a third at pp64.
	//
	// PREFILL ratios are stable across runs and are the trustworthy signal. DECODE is not: it moved
	// 0.843 - 1.019 across three runs because llama.cpp's own tg64 varied 162.58 - 200.19 t/s on
	// this host. One of those runs read as GoAI OVERTAKING llama.cpp on decode; it had not. Quote
	// decode as a band, never a figure, and only from a bracketed run.
	//
	// The remaining gap is CONCENTRATED AT SHORT PROMPTS and essentially closed by pp1024 — the
	// fixed weight-expansion cost, most of which the persistent f16 weight cache already removed.
	lens := []int{64, 256, 1024}
	ref, how := llamaCppPP(t, path, lens)
	for _, n := range lens {
		long := make([]int, n)
		for i := range long {
			long[i] = 1 + i%2000
		}
		if _, err := dec.StepNLast(long, 0); err != nil {
			t.Fatal(err)
		}
		best := 0.0
		for range 3 {
			pstart := time.Now()
			if _, err := dec.StepNLast(long, 0); err != nil {
				t.Fatal(err)
			}
			if v := float64(n) / time.Since(pstart).Seconds(); v > best {
				best = v
			}
		}
		key := "pp" + strconv.Itoa(n)
		if v, ok := ref[key]; ok && v > 0 {
			t.Logf("TinyLlama-1.1B Q4_K_M %s: GoAI %.1f tok/s vs llama.cpp %.1f (%s) = %.3fx",
				key, best, v, how, best/v)
		} else {
			t.Logf("TinyLlama-1.1B Q4_K_M %s: GoAI %.1f tok/s (no llama.cpp reference: %s)", key, best, how)
		}
	}
}

// llamaCppTG64 returns llama.cpp's tg64 throughput on the SAME file, measured live when llama-bench
// is available. Returns the recorded fallback otherwise, naming which one was used so a reader can
// tell a live comparison from a stale one.
func llamaCppTG64(t *testing.T, model string) (float64, string) {
	t.Helper()
	// Recorded 2026-08-15 on an M2 Pro, llama.cpp build 48d22e295 (10360), ggml 0.19.0, Metal+BLAS:
	//   llama-bench -m tinyllama-1.1b-q4km.gguf -p 0 -n 64 -r 3 -> tg64 201.61 +/- 3.25 t/s
	const recorded = 201.61
	v, how := llamaBench(t, model, "-p", "0", "-n", "64", "-r", "3")
	if how != "measured live" {
		return recorded, "recorded, build 48d22e295 — " + how
	}
	return v["tg64"], how
}

// llamaCppPP measures llama.cpp's PROMPT PROCESSING at the given lengths on the same file, in the
// same session. The decode side has been measured live since a hardcoded incumbent was caught
// flattering GoAI by ~17%; prefill was still being compared against a table in a comment, which
// rots the same way and by more — that table read pp64 0.46 while a live run measures 0.675,
// because the library moved and the comment did not.
func llamaCppPP(t *testing.T, model string, lens []int) (map[string]float64, string) {
	t.Helper()
	ps := make([]string, len(lens))
	for i, n := range lens {
		ps[i] = strconv.Itoa(n)
	}
	return llamaBench(t, model, "-p", strings.Join(ps, ","), "-n", "0", "-r", "3")
}

// llamaBench runs llama-bench and returns every "test -> throughput" row it printed, keyed by the
// test name (pp64, tg64, ...). The second result says whether the numbers are live, so a reader can
// tell a real comparison from a stale one.
func llamaBench(t *testing.T, model string, args ...string) (map[string]float64, string) {
	t.Helper()
	bin, err := exec.LookPath("llama-bench")
	if err != nil {
		return nil, "llama-bench not on PATH"
	}
	out, err := exec.Command(bin, append([]string{"-m", model}, args...)...).CombinedOutput()
	if err != nil {
		return nil, "llama-bench failed"
	}
	// Each result row ends "| <test> | <tps> ± <stddev> |".
	re := regexp.MustCompile(`\|\s*(pp\d+|tg\d+)\s*\|\s*([0-9.]+)\s*±`)
	res := map[string]float64{}
	sc := bufio.NewScanner(strings.NewReader(string(out)))
	for sc.Scan() {
		if m := re.FindStringSubmatch(sc.Text()); m != nil {
			if v, e := strconv.ParseFloat(m[2], 64); e == nil {
				res[m[1]] = v
			}
		}
	}
	if len(res) == 0 {
		return nil, "could not parse llama-bench output"
	}
	return res, "measured live"
}
