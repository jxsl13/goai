//go:build darwin && cgo

package metal_test

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
)

// BenchmarkMetalQuantIncremental measures the per-op cost of a resident quant matmul
// with the ~149us per-submit floor REMOVED, at TinyLlama-1.1B's exact projection
// shapes. Every other benchmark here records one op per submit, which is right for
// isolating a kernel but leaves the floor in the number; the decoder records a whole
// token into one command buffer, so the incremental cost is what a token actually
// pays.
//
// Method: time a submit carrying 1 op and one carrying 16, then take (t16-t1)/15.
// The floor cancels.
func BenchmarkMetalQuantIncremental(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal device unavailable")
	}
	for _, tc := range []struct {
		name string
		qt   gguf.QuantType
		blk  int
		hdr  int
		k, n int
	}{
		{"Q4K_K2048N2048", gguf.Q4_K, 144, 4, 2048, 2048}, // q, o projections
		{"Q4K_K2048N5632", gguf.Q4_K, 144, 4, 2048, 5632}, // gate, up
		{"Q4K_K5632N2048", gguf.Q4_K, 144, 4, 5632, 2048}, // down
		{"Q6K_K2048N2048", gguf.Q6_K, 210, 2, 2048, 2048},
	} {
		for _, mode := range []struct {
			name string
			coop bool
		}{{"cooperative", true}, {"scalar", false}} {
			b.Run(fmt.Sprintf("%s/%s", tc.name, mode.name), func(b *testing.B) {
				var prev bool
				if tc.qt == gguf.Q4_K {
					prev = metal.SetQ4KCooperative(mode.coop)
					defer metal.SetQ4KCooperative(prev)
				} else {
					prev = metal.SetQ6KCooperative(mode.coop)
					defer metal.SetQ6KCooperative(prev)
				}
				weight := syntheticKQuant(tc.n, tc.k, tc.blk, tc.hdr)
				raw, err := metal.Backend{}.UploadQuant(weight, uint32(tc.qt), tc.n, tc.k)
				if err != nil {
					b.Skipf("upload: %v", err)
				}
				rw := raw.(*metal.ResidentQWeight)
				defer rw.Close()
				x, _ := metal.NewDeviceBufferF32(make([]float32, tc.k))
				defer x.Release()
				o, _ := metal.NewDeviceBufferF32(make([]float32, tc.n))
				defer o.Release()
				run := func(ops int) {
					r, err := metal.NewRecorder()
					if err != nil {
						b.Fatal(err)
					}
					for range ops {
						if err := r.QMatMulResident(x, rw, o, 1); err != nil {
							r.Free()
							b.Fatal(err)
						}
					}
					if err := r.Finish(); err != nil {
						r.Free()
						b.Fatal(err)
					}
					r.Free()
				}
				for range 10 {
					run(1)
				}
				b.Run("1op", func(b *testing.B) {
					for range b.N {
						run(1)
					}
				})
				b.Run("16ops", func(b *testing.B) {
					for range b.N {
						run(16)
					}
				})
			})
		}
	}
}
