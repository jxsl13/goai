package nlp

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/internal/simd"
)

// refSampleMirostat is the pre-optimization full-vocab-sort implementation of
// (*Mirostat).Sample, kept verbatim here as the bit-exactness oracle for the prob-pre-filter
// rewrite. It sorts the whole vocabulary every call and truncates to the surprise-≤-μ prefix.
func refSampleMirostat(m *Mirostat, logits []float64) int {
	n := len(logits)
	if n == 0 {
		return 0
	}
	mx := math.Inf(-1)
	for _, v := range logits {
		if v > mx {
			mx = v
		}
	}
	probs := make([]float64, n)
	sum := simd.ExpSumF64(probs, logits, mx) // identical softmax to Sample, to isolate the sort/filter change
	for i := range probs {
		probs[i] /= sum
	}
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sortIdxDescByProb(idx, probs)
	keep := 1
	for keep < n && surpriseBits(probs[idx[keep]]) <= m.Mu {
		keep++
	}
	var ksum float64
	for _, i := range idx[:keep] {
		ksum += probs[i]
	}
	u := m.rng.Float64() * ksum
	x := idx[keep-1]
	var cum float64
	for _, i := range idx[:keep] {
		cum += probs[i]
		if u <= cum {
			x = i
			break
		}
	}
	observed := surpriseBits(probs[x])
	m.Mu -= m.Eta * (observed - m.Tau)
	return x
}

// TestMirostatSampleBitExactVsReference drives the optimized Sample and the full-sort oracle
// through identical call sequences (same seed ⇒ same rng stream) across peaked, flat, and
// heavily-tied distributions, covering keep=1, small-keep (filter path) and large-keep
// (radix fallback). Every returned token AND the running μ after each call must be bit-equal.
func TestMirostatSampleBitExactVsReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(0xC0FFEE, 0x1234))
	mkLogits := func(n int, kind int) []float64 {
		l := make([]float64, n)
		switch kind {
		case 0: // spread (peaked softmax ⇒ small keep)
			for i := range l {
				l[i] = rng.NormFloat64() * 4
			}
		case 1: // flat-ish (many near-max ⇒ large keep / fallback)
			for i := range l {
				l[i] = rng.Float64() * 0.5
			}
		case 2: // heavy ties: only a few distinct logit values
			vals := []float64{2, 2, 1, 1, 1, 0, 0, -3}
			for i := range l {
				l[i] = vals[rng.IntN(len(vals))]
			}
		case 3: // exact uniform (all tied) ⇒ fallback with all-equal probs
			for i := range l {
				l[i] = 0.7
			}
		}
		return l
	}
	sizes := []int{1, 4, 50, 500, 1100, 3000}
	kinds := []int{0, 1, 2, 3}
	taus := []float64{0, 2.5, 5, 8}
	for _, n := range sizes {
		for _, kind := range kinds {
			for _, tau := range taus {
				seed := rng.Uint64()
				opt := WithMirostatTau(tau)
				got := NewMirostat(seed, opt)
				ref := NewMirostat(seed, opt)
				logits := mkLogits(n, kind)
				for step := 0; step < 12; step++ {
					gt := got.Sample(logits)
					rt := refSampleMirostat(ref, logits)
					if gt != rt {
						t.Fatalf("token mismatch n=%d kind=%d tau=%g step=%d: got %d want %d (μ got=%v ref=%v)",
							n, kind, tau, step, gt, rt, got.Mu, ref.Mu)
					}
					if math.Float64bits(got.Mu) != math.Float64bits(ref.Mu) {
						t.Fatalf("μ mismatch n=%d kind=%d tau=%g step=%d: got %v want %v",
							n, kind, tau, step, got.Mu, ref.Mu)
					}
				}
			}
		}
	}
}
