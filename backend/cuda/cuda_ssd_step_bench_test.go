//go:build cuda && cgo && linux

package cuda_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// cu_ssd_step (Mamba-2 SSD decode) computes a double exp(Δ·A) decay per (head,head-dim) thread.
func benchSSDStep(b *testing.B, H, P, G, N int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	rec, err := cuda.NewRecorder()
	if err != nil {
		b.Fatal(err)
	}
	defer rec.Free()
	HP, gN := H*P, G*N
	f := func(n, seed int, sc, off float64) []float32 {
		r := make([]float32, n)
		for i := range r {
			r[i] = float32(off + sc*math.Sin(float64(i*7+seed)))
		}
		return r
	}
	x := f(HP, 1, 0.05, 0)
	delta := f(H, 2, 0.05, 0.15) // > 0
	A := make([]float32, H)
	for i := range A {
		A[i] = -float32(0.2 + math.Abs(math.Cos(float64(i)*0.1))) // < 0 (stable/contractive)
	}
	bb := f(gN, 3, 0.07, 0)
	cc := f(gN, 4, 0.06, 0)
	dskip := f(H, 6, 0.2, 0)
	dx, _ := cuda.NewDeviceBufferF32(x)
	ddelta, _ := cuda.NewDeviceBufferF32(delta)
	dA, _ := cuda.NewDeviceBufferF32(A)
	dB, _ := cuda.NewDeviceBufferF32(bb)
	dC, _ := cuda.NewDeviceBufferF32(cc)
	dD, _ := cuda.NewDeviceBufferF32(dskip)
	dstate, _ := cuda.NewDeviceBufferF32(make([]float32, H*N*P))
	dy, _ := cuda.NewDeviceF32(1, HP)
	for _, d := range []*cuda.DeviceF32{dx, ddelta, dA, dB, dC, dD, dstate, dy} {
		defer d.Free()
	}
	if err := rec.SSDStep(dx, ddelta, dA, dB, dC, dD, dstate, dy, H, P, G, N); err != nil {
		b.Fatal(err)
	}
	rec.Wait()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = rec.SSDStep(dx, ddelta, dA, dB, dC, dD, dstate, dy, H, P, G, N)
	}
	rec.Wait()
	b.StopTimer()
}

func BenchmarkSSDStep_H64_P64_N128(b *testing.B)  { benchSSDStep(b, 64, 64, 128, 8) }
func BenchmarkSSDStep_H128_P64_N128(b *testing.B) { benchSSDStep(b, 128, 64, 128, 8) }
