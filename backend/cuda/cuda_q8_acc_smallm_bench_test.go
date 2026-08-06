//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// benchQ8AccSmallM times the residual-fused QMatMulResidentAcc (beta=1) at small m — the down/O proj.
func benchQ8AccSmallM(b *testing.B, m, k, n int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	rec, err := cuda.NewRecorder()
	if err != nil {
		b.Fatal(err)
	}
	defer rec.Free()
	x, _ := cuda.NewDeviceF32(m, k)
	dst, _ := cuda.NewDeviceF32(m, n)
	wq8, err := cuda.NewResidentBQ8(bench.RandF32(tensor.Shape{k, n}, 2))
	if err != nil {
		b.Fatal(err)
	}
	defer func() { x.Free(); dst.Free(); wq8.Free() }()
	if err := rec.QMatMulResidentAcc(x, wq8, dst, m); err != nil {
		b.Skipf("QMatMulResidentAcc: %v", err)
	}
	rec.Wait()
	b.ResetTimer()
	for range b.N {
		_ = rec.QMatMulResidentAcc(x, wq8, dst, m)
	}
	rec.Wait()
	b.StopTimer()
	b.ReportMetric(2*float64(m)*float64(k)*float64(n)*float64(b.N)/b.Elapsed().Seconds()/1e12, "TFLOP/s")
}

func BenchmarkQ8AccSmallM_4x5632x2048(b *testing.B) { benchQ8AccSmallM(b, 4, 5632, 2048) }
func BenchmarkQ8AccSmallM_5x2048x5632(b *testing.B) { benchQ8AccSmallM(b, 5, 2048, 5632) }
func BenchmarkQ8AccSmallM_4x2048x2048(b *testing.B) { benchQ8AccSmallM(b, 4, 2048, 2048) }
