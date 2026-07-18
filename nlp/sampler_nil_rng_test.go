package nlp

import "testing"

// A struct-literal Sampler (rng unexported, so nil) must not panic on ANY path.
func TestStructLiteralSamplerAllPaths(t *testing.T) {
	logits := []float64{0.1, 2.0, 0.3, 1.2}
	for _, tc := range []struct {
		name string
		s    *Sampler
	}{
		{"topk+topp", &Sampler{Temperature: 0.8, TopK: 2, TopP: 0.9}},
		{"minp", &Sampler{Temperature: 1, MinP: 0.1}},
		{"xtc", &Sampler{Temperature: 1, XTCProbability: 1.0, XTCThreshold: 0.05}},
		{"typical", &Sampler{Temperature: 1, Typical: 0.9}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("PANIC: %v", r)
				}
			}()
			_ = tc.s.Sample(append([]float64(nil), logits...))
			_ = tc.s.SampleWithHistory(append([]float64(nil), logits...), []int{1, 2})
			_ = tc.s.Dist(append([]float64(nil), logits...))
		})
	}
}
