package classic

import "testing"

// BenchmarkSoftmaxPredictProba covers SoftmaxRegression.PredictProba, which had no benchmark at
// all — the inference path of this model was unmeasured while its Fit path had five.
//
// That gap was not academic: PredictProba is the largest per-sample allocation site left in this
// package (one row per sample) and its copy-out reads the result tensor through a per-element
// accessor (n*k interface dispatches), and neither could be acted on without an instrument. Fit
// computes its own softmax inline rather than calling PredictProba, so no existing benchmark
// reaches this code.
//
// The sizes bracket the two regimes that matter: a wide batch where the copy-out dominates the
// per-call overhead, and a narrow one where it does not.
func BenchmarkSoftmaxPredictProba(b *testing.B) {
	for _, tc := range []struct{ n, d, k int }{{512, 16, 4}, {2048, 32, 8}, {4096, 20, 3}} {
		x, y := softmaxSynthetic(tc.n, tc.d, tc.k, 7)
		m := &SoftmaxRegression{}
		if err := m.Fit(x, y, tc.k, 25, 0.1); err != nil {
			b.Fatal(err)
		}
		b.Run(benchName(tc.n, tc.d, tc.k), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, err := m.PredictProba(x); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func benchName(n, d, k int) string {
	return itoa(n) + "x" + itoa(d) + "x" + itoa(k)
}

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for v > 0 {
		i--
		b[i] = byte('0' + v%10)
		v /= 10
	}
	return string(b[i:])
}
