package classic

import (
	"math"
	"testing"
)

// synthFitData builds a deterministic n×d, k-class dataset for fit-time
// benchmarks (mirrors the perfcompare shape n=4000, d=20, 3-class). It is a
// pure LCG generator so no golden or external data is involved.
func synthFitData(n, d, k int) ([][]float64, []int, []float64) {
	var s uint64 = 0x1234567
	next := func() float64 {
		s ^= s >> 12
		s ^= s << 25
		s ^= s >> 27
		v := (s * 2685821657736338717) >> 11
		return float64(v) / float64(uint64(1)<<53)
	}
	x := make([][]float64, n)
	yi := make([]int, n)
	yf := make([]float64, n)
	for i := 0; i < n; i++ {
		row := make([]float64, d)
		var acc float64
		for j := 0; j < d; j++ {
			row[j] = next()
			acc += row[j] * math.Sin(float64(j+1))
		}
		x[i] = row
		yi[i] = int(math.Abs(acc*7.0)) % k
		yf[i] = acc + 0.1*next()
	}
	return x, yi, yf
}

func BenchmarkTreeFit(b *testing.B) {
	x, y, _ := synthFitData(4000, 20, 3)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m := NewDecisionTreeClassifier()
		if err := m.Fit(x, y); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkForestFit(b *testing.B) {
	x, y, _ := synthFitData(4000, 20, 3)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m := NewRandomForestClassifier(WithNumTrees(100), WithSeed(1))
		if err := m.Fit(x, y); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGBMFit(b *testing.B) {
	x, y, _ := synthFitData(4000, 20, 3)
	yb := make([]int, len(y))
	for i, v := range y {
		if v >= 1 {
			yb[i] = 1
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m := NewGradientBoostingClassifier(WithGBMNEstimators(100))
		if err := m.Fit(x, yb); err != nil {
			b.Fatal(err)
		}
	}
}
