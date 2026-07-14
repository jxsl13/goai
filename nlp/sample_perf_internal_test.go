package nlp

import (
	"math"
	"math/rand/v2"
	"sort"
	"testing"
)

// realisticLogits builds a V-vocab logit vector with a peaked-but-heavy-tail shape
// (a few high logits, a long low tail) — the distribution the truncation samplers
// actually see. Distinct values (no exact ties) so top-k/top-p sets are unambiguous.
func realisticLogits(v int, seed uint64) []float64 {
	rng := rand.New(rand.NewPCG(seed, 0x9e3779b97f4a7c15))
	out := make([]float64, v)
	for i := range out {
		out[i] = rng.NormFloat64()*2 - float64(i)*1e-6 // tiny ramp guarantees distinctness
	}
	return out
}

// distNaive is the pre-§T626 Dist: the three truncation paths each do a full
// O(n log n) reflection sort.Slice over the whole vocabulary. Kept HERE, in the
// test binary only, as the parity reference for the quickselect/typed-sort rewrite
// (same-session A/B + old-vs-new parity, §V22).
func distNaive(s *Sampler, logits []float64) []float64 {
	n := len(logits)
	if s.Temperature <= 0 {
		d := make([]float64, n)
		if n > 0 {
			d[argmax(logits)] = 1
		}
		return d
	}
	z := make([]float64, n)
	for i, v := range logits {
		z[i] = v / s.Temperature
	}
	if s.TopK > 0 && s.TopK < n {
		idx := make([]int, n)
		for i := range idx {
			idx[i] = i
		}
		sort.Slice(idx, func(a, b int) bool { return z[idx[a]] > z[idx[b]] })
		for _, i := range idx[s.TopK:] {
			z[i] = math.Inf(-1)
		}
	}
	m := math.Inf(-1)
	for _, v := range z {
		if v > m {
			m = v
		}
	}
	probs := make([]float64, n)
	var sum float64
	for i, v := range z {
		probs[i] = math.Exp(v - m)
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}
	if s.TopP > 0 && s.TopP < 1 {
		idx := make([]int, n)
		for i := range idx {
			idx[i] = i
		}
		sort.Slice(idx, func(a, b int) bool { return probs[idx[a]] > probs[idx[b]] })
		var cum float64
		keep := make([]bool, n)
		for _, i := range idx {
			keep[i] = true
			cum += probs[i]
			if cum >= s.TopP {
				break
			}
		}
		var ksum float64
		for i := range probs {
			if !keep[i] {
				probs[i] = 0
			} else {
				ksum += probs[i]
			}
		}
		for i := range probs {
			probs[i] /= ksum
		}
	}
	if s.MinP > 0 {
		var pmax float64
		for _, p := range probs {
			if p > pmax {
				pmax = p
			}
		}
		thresh := s.MinP * pmax
		var ksum float64
		for i := range probs {
			if probs[i] < thresh {
				probs[i] = 0
			} else {
				ksum += probs[i]
			}
		}
		for i := range probs {
			probs[i] /= ksum
		}
	}
	if s.Typical > 0 && s.Typical < 1 {
		var h float64
		for _, p := range probs {
			if p > 0 {
				h -= p * math.Log(p)
			}
		}
		score := make([]float64, n)
		idx := make([]int, n)
		for i, p := range probs {
			idx[i] = i
			if p > 0 {
				score[i] = math.Abs(-math.Log(p) - h)
			} else {
				score[i] = math.Inf(1)
			}
		}
		sort.Slice(idx, func(a, b int) bool { return score[idx[a]] < score[idx[b]] })
		var cum float64
		keep := make([]bool, n)
		for _, i := range idx {
			if probs[i] == 0 {
				break
			}
			keep[i] = true
			cum += probs[i]
			if cum >= s.Typical {
				break
			}
		}
		var ksum float64
		for i := range probs {
			if !keep[i] {
				probs[i] = 0
			} else {
				ksum += probs[i]
			}
		}
		if ksum > 0 {
			for i := range probs {
				probs[i] /= ksum
			}
		}
	}
	if s.Epsilon > 0 {
		truncateAbove(probs, s.Epsilon)
	}
	if s.Eta > 0 {
		var h float64
		for _, p := range probs {
			if p > 0 {
				h -= p * math.Log(p)
			}
		}
		truncateAbove(probs, math.Min(s.Eta, math.Sqrt(s.Eta)*math.Exp(-h)))
	}
	return probs
}

// TestDistQuickselectParity locks the quickselect top-k + typed top-p/typical
// sorts to the old full-sort algorithm bit-for-bit across a spread of sampler
// configs on distinct logits (where the truncation sets are unambiguous).
func TestDistQuickselectParity(t *testing.T) {
	configs := []struct {
		name string
		opts []SamplerOption
	}{
		{"topk1", []SamplerOption{WithTemperature(1), WithTopK(1)}},
		{"topk40", []SamplerOption{WithTemperature(0.8), WithTopK(40)}},
		{"topk1000", []SamplerOption{WithTemperature(1.2), WithTopK(1000)}},
		{"topp90", []SamplerOption{WithTemperature(1), WithTopP(0.9)}},
		{"topp50", []SamplerOption{WithTemperature(0.7), WithTopP(0.5)}},
		{"typical95", []SamplerOption{WithTemperature(1), WithTypical(0.95)}},
		{"topk_topp", []SamplerOption{WithTemperature(1), WithTopK(200), WithTopP(0.85)}},
		{"topk_minp", []SamplerOption{WithTemperature(1), WithTopK(100), WithMinP(0.05)}},
		{"all", []SamplerOption{WithTemperature(0.9), WithTopK(300), WithTopP(0.95), WithTypical(0.98), WithMinP(0.01)}},
	}
	for _, vsz := range []int{64, 1024, 50257} {
		for _, seed := range []uint64{1, 7, 99} {
			logits := realisticLogits(vsz, seed)
			for _, c := range configs {
				s := NewSampler(1, c.opts...)
				got := s.Dist(logits)
				want := distNaive(NewSampler(1, c.opts...), logits)
				if len(got) != len(want) {
					t.Fatalf("%s v=%d seed=%d: len %d != %d", c.name, vsz, seed, len(got), len(want))
				}
				for i := range got {
					if math.Abs(got[i]-want[i]) > 1e-12 {
						t.Fatalf("%s v=%d seed=%d [%d]: %.17g != naive %.17g", c.name, vsz, seed, i, got[i], want[i])
					}
				}
			}
		}
	}
}

// TestKthLargest checks the quickselect helper against a full sort on random and
// adversarial (sorted, reverse, constant-ish) inputs.
func TestKthLargest(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 42))
	for _, n := range []int{1, 2, 5, 100, 5000} {
		base := make([]float64, n)
		for i := range base {
			base[i] = rng.NormFloat64()
		}
		sorted := append([]float64(nil), base...)
		sort.Sort(sort.Reverse(sort.Float64Slice(sorted))) // descending reference
		for _, k := range []int{1, (n + 1) / 2, n} {
			if k < 1 || k > n {
				continue
			}
			scratch := append([]float64(nil), base...)
			if got := kthLargest(scratch, k); got != sorted[k-1] {
				t.Fatalf("n=%d k=%d: kthLargest=%.17g want %.17g", n, k, got, sorted[k-1])
			}
		}
	}
}

func benchDist(b *testing.B, opts ...SamplerOption) {
	s := NewSampler(1, opts...)
	logits := realisticLogits(50257, 7)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = s.Dist(logits)
	}
}

func benchDistNaive(b *testing.B, opts ...SamplerOption) {
	s := NewSampler(1, opts...)
	logits := realisticLogits(50257, 7)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = distNaive(s, logits)
	}
}

func BenchmarkDistTopK40(b *testing.B)  { benchDist(b, WithTemperature(1), WithTopK(40)) }
func BenchmarkDistTopP90(b *testing.B)  { benchDist(b, WithTemperature(1), WithTopP(0.9)) }
func BenchmarkDistTypical(b *testing.B) { benchDist(b, WithTemperature(1), WithTypical(0.95)) }
func BenchmarkDistPlain(b *testing.B)   { benchDist(b, WithTemperature(1)) }

func BenchmarkDistTopK40Naive(b *testing.B) { benchDistNaive(b, WithTemperature(1), WithTopK(40)) }
func BenchmarkDistTopP90Naive(b *testing.B) { benchDistNaive(b, WithTemperature(1), WithTopP(0.9)) }
func BenchmarkDistTypicalNaive(b *testing.B) {
	benchDistNaive(b, WithTemperature(1), WithTypical(0.95))
}

// peakyLogits mimics REAL LLM logits: a sharp head (a handful of dominant tokens)
// with an exponential decay, so the top-p nucleus is small — the realistic case
// where the §T627 bounded selection wins. Distinct by construction.
func peakyLogits(v int, seed uint64) []float64 {
	rng := rand.New(rand.NewPCG(seed, 0x2545f4914f6cdd1d))
	out := make([]float64, v)
	for i := range out {
		out[i] = rng.NormFloat64()*0.3 - float64(i)*0.01 // steep ramp ⟹ concentrated head
	}
	out[0] += 6
	out[1] += 5
	out[2] += 4
	return out
}

func benchDistLogits(b *testing.B, logits []float64, naive bool, opts ...SamplerOption) {
	s := NewSampler(1, opts...)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if naive {
			_ = distNaive(s, logits)
		} else {
			_ = s.Dist(logits)
		}
	}
}

// peaky top-p: the realistic small-nucleus case — bounded selection should win big.
func BenchmarkDistTopP90Peaky(b *testing.B) {
	benchDistLogits(b, peakyLogits(50257, 7), false, WithTemperature(1), WithTopP(0.9))
}
func BenchmarkDistTopP90PeakyNaive(b *testing.B) {
	benchDistLogits(b, peakyLogits(50257, 7), true, WithTemperature(1), WithTopP(0.9))
}

// flat top-p: nucleus exceeds the K candidates ⟹ full-sort fallback — must NOT
// regress versus the old full sort (§C3).
func BenchmarkDistTopP99Flat(b *testing.B) {
	benchDistLogits(b, realisticLogits(50257, 7), false, WithTemperature(1), WithTopP(0.99))
}
func BenchmarkDistTopP99FlatNaive(b *testing.B) {
	benchDistLogits(b, realisticLogits(50257, 7), true, WithTemperature(1), WithTopP(0.99))
}
