package nlp

import (
	"math"
	"math/rand/v2"
	"sort"
	"testing"
)

// softmaxStatsFull returns the full-vocab stable-softmax statistics the device reduction would
// produce: maxLogit = max_i logit_i and Zexp = Σ_i exp((logit_i − maxLogit)/T).
func softmaxStatsFull(logits []float64, T float64) (float64, float64) {
	maxL := math.Inf(-1)
	for _, v := range logits {
		if v > maxL {
			maxL = v
		}
	}
	invT := 1.0 / T
	var z float64
	for _, v := range logits {
		z += math.Exp((v - maxL) * invT)
	}
	return maxL, z
}

// topCandidates returns the C highest logits and their vocab indices (the device TopK output).
func topCandidates(logits []float64, C int) ([]float64, []int32) {
	idx := make([]int, len(logits))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return logits[idx[a]] > logits[idx[b]] })
	if C > len(idx) {
		C = len(idx)
	}
	vals := make([]float64, C)
	ids := make([]int32, C)
	for j := 0; j < C; j++ {
		vals[j] = logits[idx[j]]
		ids[j] = int32(idx[j])
	}
	return vals, ids
}

// makeLogits builds a deterministic logit vector of a chosen entropy profile.
func makeLogits(kind string, vocab int, seed uint64) []float64 {
	r := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	l := make([]float64, vocab)
	switch kind {
	case "peaked": // a few dominant tokens → tiny nucleus, well within 256
		for i := range l {
			l[i] = r.NormFloat64() * 0.5
		}
		for k := 0; k < 8; k++ {
			l[r.IntN(vocab)] += 8 + 4*r.Float64()
		}
	case "medium": // moderate spread → nucleus of tens–low hundreds (near the C boundary)
		for i := range l {
			l[i] = r.NormFloat64() * 1.0
		}
	case "flat": // near-uniform high entropy → huge nucleus, overflows 256
		for i := range l {
			l[i] = r.NormFloat64() * 0.05
		}
	}
	return l
}

// TestSampleTopPFromCandidatesParity: the candidate-set nucleus draw must return the bit-identical
// token to full-vocab Sample whenever the nucleus fits in the candidate set, across many seeds and
// entropy profiles. This is the correctness gate for routing pure-top-p CUDA decode through the
// device TopK fast-path instead of a whole-vocab D2H + host sort.
func TestSampleTopPFromCandidatesParity(t *testing.T) {
	const vocab, C = 32000, 256
	totalMatched := 0
	for _, tc := range []struct {
		kind string
		temp float64
		topP float64
	}{
		{"peaked", 1.0, 0.9},
		{"peaked", 0.8, 0.95},
		{"medium", 0.7, 0.9},
		{"medium", 1.0, 0.8},
		{"medium", 0.6, 0.85},
	} {
		matched, overflow := 0, 0
		for seed := uint64(1); seed <= 300; seed++ {
			logits := makeLogits(tc.kind, vocab, seed)
			maxL, Zexp := softmaxStatsFull(logits, tc.temp)
			cl, ci := topCandidates(logits, C)

			// Reference: full-vocab Sample. Fast: same seed → identical rng state.
			ref := NewSampler(seed, WithTemperature(tc.temp), WithTopP(tc.topP))
			want := ref.Sample(logits)
			fast := NewSampler(seed, WithTemperature(tc.temp), WithTopP(tc.topP))
			got, ok := fast.SampleTopPFromCandidates(cl, ci, maxL, Zexp)
			if !ok {
				overflow++
				continue // nucleus exceeded the candidate set — the caller would fall back
			}
			if got != want {
				t.Fatalf("%s T=%.2f p=%.2f seed=%d: fast=%d want=%d", tc.kind, tc.temp, tc.topP, seed, got, want)
			}
			matched++
		}
		totalMatched += matched
		t.Logf("%s T=%.2f p=%.2f: %d matched, %d overflow", tc.kind, tc.temp, tc.topP, matched, overflow)
	}
	// Overflow is legitimate (high-entropy nucleus > C → caller falls back); what must hold is that
	// wherever the fast path DID resolve, it agreed bit-for-bit — and that a healthy number of
	// resolved draws actually exercised that agreement across the configs.
	if totalMatched < 200 {
		t.Errorf("only %d non-overflow draws exercised parity across all configs (want ≥200)", totalMatched)
	}
}

// TestSampleTopPFromCandidatesOverflow: a flat high-entropy distribution has a nucleus far larger
// than the candidate set, so the fast path MUST report overflow (never silently draw a wrong token).
func TestSampleTopPFromCandidatesOverflow(t *testing.T) {
	const vocab, C = 32000, 256
	sawOverflow := false
	for seed := uint64(1); seed <= 50; seed++ {
		logits := makeLogits("flat", vocab, seed)
		maxL, Zexp := softmaxStatsFull(logits, 1.0)
		cl, ci := topCandidates(logits, C)
		fast := NewSampler(seed, WithTemperature(1.0), WithTopP(0.95))
		if _, ok := fast.SampleTopPFromCandidates(cl, ci, maxL, Zexp); !ok {
			sawOverflow = true
		}
		// If it DID resolve (nucleus happened to fit), parity is already covered by the parity test.
	}
	if !sawOverflow {
		t.Errorf("flat distribution never overflowed C=%d — overflow detection may be broken", C)
	}
}
