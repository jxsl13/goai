//go:build darwin && cgo

package llamagpu_test

import (
	"math"
	"os"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// TestSmallBatchPrefillMatchesScalar covers the batch sizes that used to fall through every fast
// path. The expand-then-GEMM gate required M >= 24 because its crossover was measured against the
// COOPERATIVE kernel — and that fallback stopped existing below 24 when the cooperative kernels
// became M==1 only. So M in [2,23] silently landed on the scalar quantized kernel, which re-reads
// the whole weight per output row: prefill of 23 tokens cost 257.0 ms against 47.1 ms for 24, and
// inside the hole the cost still GREW with the batch.
//
// Routing that range onto the cached f16 path is a 2.15x-5.52x win, but it also changes WHICH
// kernel produces the numbers, so this pins that the two agree. The comparison is against the
// scalar path the range used to take, reached by turning the f16 path off — that is the previous
// behaviour, so a disagreement here is a regression in the thing users actually observed.
//
// Tolerance, not bit-equality: the fast path expands the weight to f16 and runs an f16 GEMM. The
// bound is CALIBRATED against the sizes that already took that path before this change, which is
// what makes it a statement about the f16 GEMM rather than about the new routing —
//
//	newly routed   n=2 0.012   n=8 0.024   n=16 0.028   n=23 0.017
//	already f16    n=24 0.028  n=32 0.018  n=64 0.214
//
// — so the range this change moves diverges LESS than the range that was already there, by up to
// 7.6x. 0.5 leaves headroom over the 0.214 the untouched n=64 case measures. The argmax check
// below is the assertion that actually matters: it is what decides the emitted token, and it holds
// exactly at every size including n=64.
func TestSmallBatchPrefillMatchesScalar(t *testing.T) {
	if !metal.Available() {
		t.Skip("no metal")
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

	for _, n := range []int{2, 8, 16, 23, 24, 32, 64} {
		prompt := make([]int, n)
		for i := range prompt {
			prompt[i] = 1 + i%2000
		}
		metal.SetQ4KDequantGemmF16(false) // the scalar path this range used to take
		want, err := dec.StepNLast(prompt, 0)
		if err != nil {
			t.Fatal(err)
		}
		ref := append([]float32(nil), want...)

		metal.SetQ4KDequantGemmF16(true) // the cached f16 path it takes now
		got, err := dec.StepNLast(prompt, 0)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != len(ref) {
			t.Fatalf("n=%d: %d logits, want %d", n, len(got), len(ref))
		}
		worst, at := 0.0, -1
		for i := range got {
			if d := math.Abs(float64(got[i] - ref[i])); d > worst {
				worst, at = d, i
			}
		}
		if worst > 0.5 {
			t.Errorf("n=%d: max logit divergence %.4g at %d — beyond the f16 GEMM's established "+
				"class (n=64 on the untouched path measures 0.214)", n, worst, at)
		}
		// argmax must be identical, which is what actually decides the emitted token.
		am := func(v []float32) int {
			b := 0
			for i := range v {
				if v[i] > v[b] {
					b = i
				}
			}
			return b
		}
		if a, b := am(got), am(ref); a != b {
			t.Errorf("n=%d: argmax moved %d -> %d; the batch path would emit a different token", n, b, a)
		}
	}
}

// TestSmallBatchPrefillHasNoCliff is the tripwire for the dispatch hole itself. Prefill cost must
// not fall off a cliff just below a threshold: 23 tokens cannot cost multiples of what 24 cost,
// because there is no more work in 23 than in 24.
//
// It guards a routing decision, not a speed target, so the bound is deliberately loose — an
// order-of-magnitude tripwire, the shape that survives a noisy host. The defect it exists for was
// 5.45x (257.0 ms against 47.1 ms) and grew with the batch, so 2x catches it with wide margin.
//
// -short-gated: the ratio is between two measurements on the SAME host so it is far steadier than
// an absolute threshold, but shared CI runners have already been shown to invert timing guards in
// this repo, and this one also needs a 636 MB model.
func TestSmallBatchPrefillHasNoCliff(t *testing.T) {
	if testing.Short() {
		t.Skip("needs the 1.1B model and times two prefills")
	}
	if !metal.Available() {
		t.Skip("no metal")
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

	best := func(n int) time.Duration {
		prompt := make([]int, n)
		for i := range prompt {
			prompt[i] = 1 + i%2000
		}
		if _, err := dec.StepNLast(prompt, 0); err != nil { // warm
			t.Fatal(err)
		}
		b := time.Hour
		for range 5 {
			st := time.Now()
			if _, err := dec.StepNLast(prompt, 0); err != nil {
				t.Fatal(err)
			}
			if d := time.Since(st); d < b {
				b = d
			}
		}
		return b
	}
	below, above := best(23), best(24)
	t.Logf("prefill n=23 %v, n=24 %v (%.2fx)", below, above, float64(below)/float64(above))
	if float64(below) > 2*float64(above) {
		t.Errorf("prefill of 23 tokens took %v against %v for 24 (%.2fx) — the batch has fallen off "+
			"a dispatch threshold onto a slower kernel", below, above, float64(below)/float64(above))
	}
}
