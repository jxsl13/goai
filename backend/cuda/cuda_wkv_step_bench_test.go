//go:build cuda && cgo && linux

package cuda_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// cu_wkv_step (RWKV-4 WKV decode) does 4 double exps per channel thread.
func benchWKVStep(b *testing.B, D int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	rec, err := cuda.NewRecorder()
	if err != nil {
		b.Fatal(err)
	}
	defer rec.Free()
	f := func(seed int, sc, off float64) []float32 {
		r := make([]float32, D)
		for i := range r {
			r[i] = float32(off + sc*math.Sin(float64(i*7+seed)))
		}
		return r
	}
	k, v, u := f(1, 0.2, 0), f(2, 0.3, 0), f(4, 0.25, 0)
	w := make([]float32, D)
	for c := range w {
		w[c] = float32(math.Exp(0.15 * math.Sin(float64(c)*0.03))) // > 0
	}
	dk, _ := cuda.NewDeviceBufferF32(k)
	dv, _ := cuda.NewDeviceBufferF32(v)
	dw, _ := cuda.NewDeviceBufferF32(w)
	du, _ := cuda.NewDeviceBufferF32(u)
	daa, _ := cuda.NewDeviceBufferF32(make([]float32, D))
	dbb, _ := cuda.NewDeviceBufferF32(make([]float32, D))
	dpp := func() *cuda.DeviceF32 {
		p := make([]float32, D)
		for i := range p {
			p[i] = -1e38
		}
		d, _ := cuda.NewDeviceBufferF32(p)
		return d
	}()
	dout, _ := cuda.NewDeviceF32(1, D)
	for _, d := range []*cuda.DeviceF32{dk, dv, dw, du, daa, dbb, dpp, dout} {
		defer d.Free()
	}
	if err := rec.WKVStep(dk, dv, dw, du, daa, dbb, dpp, dout, D); err != nil {
		b.Fatal(err)
	}
	rec.Wait()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = rec.WKVStep(dk, dv, dw, du, daa, dbb, dpp, dout, D)
	}
	rec.Wait()
	b.StopTimer()
}

func BenchmarkWKVStep_D4096(b *testing.B) { benchWKVStep(b, 4096) }
func BenchmarkWKVStep_D8192(b *testing.B) { benchWKVStep(b, 8192) }
