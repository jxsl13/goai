//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// benchQ8SmallM times QMatMulResident at small m — the speculative-decode / small-batch regime.
// origin/main streams the weight M× (GEMV); the branch routes 2<=m<6 to the weight-read-once MT.
func benchQ8SmallM(b *testing.B, m, k, n int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	rec, err := cuda.NewRecorder()
	if err != nil {
		b.Fatal(err)
	}
	defer rec.Free()
	x, _ := cuda.NewDeviceF32(m, k)
	o, _ := cuda.NewDeviceF32(m, n)
	wq8, err := cuda.NewResidentBQ8(bench.RandF32(tensor.Shape{k, n}, 2))
	if err != nil {
		b.Fatal(err)
	}
	defer func() { x.Free(); o.Free(); wq8.Free() }()
	if err := rec.QMatMulResident(x, wq8, o, m); err != nil {
		b.Skipf("QMatMulResident: %v", err)
	}
	rec.Wait()
	b.ResetTimer()
	for range b.N {
		_ = rec.QMatMulResident(x, wq8, o, m)
	}
	rec.Wait()
	b.StopTimer()
	b.ReportMetric(2*float64(m)*float64(k)*float64(n)*float64(b.N)/b.Elapsed().Seconds()/1e12, "TFLOP/s")
}

func BenchmarkQ8SmallM_2x2048x2048(b *testing.B) { benchQ8SmallM(b, 2, 2048, 2048) }
func BenchmarkQ8SmallM_4x2048x2048(b *testing.B) { benchQ8SmallM(b, 4, 2048, 2048) }
func BenchmarkQ8SmallM_5x2048x2048(b *testing.B) { benchQ8SmallM(b, 5, 2048, 2048) }
func BenchmarkQ8SmallM_4x2048x5632(b *testing.B) { benchQ8SmallM(b, 4, 2048, 5632) }
func BenchmarkQ8SmallM_5x5632x2048(b *testing.B) { benchQ8SmallM(b, 5, 5632, 2048) }
