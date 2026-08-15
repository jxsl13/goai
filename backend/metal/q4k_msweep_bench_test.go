//go:build darwin && cgo

package metal_test

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
)

// BenchmarkMetalQ4KMSweep measures the M>1 (prefill / batched) resident Q4_K path,
// which the cooperative kernel deliberately does not cover: the bridge selects it
// with `cooperative = M == 1 && enabled`, so every M>1 dispatch runs the scalar
// kernel. The question this answers is whether that matters. The weight bytes are
// INDEPENDENT of M, so if the kernel reads each weight once and reuses it across the
// M rows, time should flatten as M grows; if every row re-walks and re-dequantizes
// the whole weight, time scales with M and the redundant traffic is the headroom.
func BenchmarkMetalQ4KMSweep(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal device unavailable")
	}
	const k, n = 2048, 5632 // TinyLlama SwiGLU gate/up projection
	weight := syntheticQ4K(n, k)
	raw, err := metal.Backend{}.UploadQuant(weight, uint32(gguf.Q4_K), n, k)
	if err != nil {
		b.Fatal(err)
	}
	rw := raw.(*metal.ResidentQWeight)
	defer rw.Close()

	for _, m := range []int{1, 2, 4, 8, 16, 32, 64, 128, 256} {
		b.Run(fmt.Sprintf("M%d", m), func(b *testing.B) {
			x, err := metal.NewDeviceBufferF32(make([]float32, m*k))
			if err != nil {
				b.Fatal(err)
			}
			defer x.Release()
			o, err := metal.NewDeviceBufferF32(make([]float32, m*n))
			if err != nil {
				b.Fatal(err)
			}
			defer o.Release()
			run := func() {
				r, err := metal.NewRecorder()
				if err != nil {
					b.Fatal(err)
				}
				if err := r.QMatMulResident(x, rw, o, m); err != nil {
					r.Free()
					b.Fatal(err)
				}
				if err := r.Finish(); err != nil {
					r.Free()
					b.Fatal(err)
				}
				r.Free()
			}
			for range 20 {
				run()
			}
			// Weight bytes only: constant across M by construction, so ns/op tells us
			// directly whether the weight is re-read per row.
			b.SetBytes(int64(len(weight)))
			b.ResetTimer()
			for range b.N {
				run()
			}
		})
	}
}
