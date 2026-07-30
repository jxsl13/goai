//go:build cuda && cgo && linux

package cuda_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// cu_ssm_step (Mamba selective-scan decode step) does D threads × N iterations, each a
// double-precision exp(Δ·A). Measure whether the FP64 exp makes it compute-bound.
func benchSSMStep(b *testing.B, D, N int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	rec, err := cuda.NewRecorder()
	if err != nil {
		b.Fatal(err)
	}
	defer rec.Free()
	aD := make([]float32, D*N)
	for i := range aD {
		aD[i] = -float32(0.2 + math.Abs(math.Sin(float64(i)*0.05)))
	}
	dskip := make([]float32, D)
	u := make([]float32, D)
	delta := make([]float32, D)
	bb := make([]float32, N)
	cc := make([]float32, N)
	for i := range u {
		u[i] = float32(math.Sin(float64(i) * 0.01))
		delta[i] = float32(0.1 + math.Abs(math.Cos(float64(i)*0.03)))
	}
	for i := range bb {
		bb[i] = float32(math.Sin(float64(i) * 0.02))
		cc[i] = float32(math.Cos(float64(i) * 0.017))
	}
	da, _ := cuda.NewDeviceBufferF32(aD)
	ddk, _ := cuda.NewDeviceBufferF32(dskip)
	dh, _ := cuda.NewDeviceBufferF32(make([]float32, D*N))
	du, _ := cuda.NewDeviceBufferF32(u)
	ddelta, _ := cuda.NewDeviceBufferF32(delta)
	db, _ := cuda.NewDeviceBufferF32(bb)
	dc, _ := cuda.NewDeviceBufferF32(cc)
	dy, _ := cuda.NewDeviceF32(1, D)
	defer da.Free()
	defer ddk.Free()
	defer dh.Free()
	defer du.Free()
	defer ddelta.Free()
	defer db.Free()
	defer dc.Free()
	defer dy.Free()
	if err := rec.SSMStep(du, ddelta, da, db, dc, ddk, dh, dy, D, N); err != nil {
		b.Fatal(err)
	}
	rec.Wait()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = rec.SSMStep(du, ddelta, da, db, dc, ddk, dh, dy, D, N)
	}
	rec.Wait()
	b.StopTimer()
}

func BenchmarkSSMStep_D4096_N16(b *testing.B) { benchSSMStep(b, 4096, 16) }
func BenchmarkSSMStep_D8192_N16(b *testing.B) { benchSSMStep(b, 8192, 16) }
