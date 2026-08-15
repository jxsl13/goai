//go:build darwin && cgo

package metal_test

import (
	"testing"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
)

// syntheticLegacyQuant builds a weight blob for the 32-weight legacy block formats
// (Q4_0: f16 scale + 16 nibble bytes = 18; Q8_0: f16 scale + 32 int8 = 34). Only the
// timing of the resident matvec is read; these kernels have no data-dependent control
// flow, and the scale is pinned to a finite f16.
func syntheticLegacyQuant(n, k, blockBytes int) []byte {
	blocks := n * (k / 32)
	out := make([]byte, blocks*blockBytes)
	for block := range blocks {
		base := block * blockBytes
		out[base], out[base+1] = 0, 0x3c // f16(1)
		for i := 2; i < blockBytes; i++ {
			out[base+i] = byte((block*17 + i*29) & 0xff)
		}
	}
	return out
}

// BenchmarkMetalLegacyQuantM1Gap asks whether the legacy 32-weight formats carry the
// same M=1 occupancy gap the five K-quant types did before they got cooperative
// kernels. Both still dispatch one thread per output row.
func BenchmarkMetalLegacyQuantM1Gap(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal device unavailable")
	}
	const k, n = 2048, 2048
	for _, tc := range []struct {
		name       string
		qtype      uint32
		blockBytes int
	}{
		{"Q4_0", uint32(gguf.Q4_0), 18},
		{"Q8_0", uint32(gguf.Q8_0), 34},
	} {
		b.Run(tc.name, func(b *testing.B) {
			weight := syntheticLegacyQuant(n, k, tc.blockBytes)
			raw, err := metal.Backend{}.UploadQuant(weight, tc.qtype, n, k)
			if err != nil {
				b.Skipf("UploadQuant %s: %v", tc.name, err)
			}
			rw := raw.(*metal.ResidentQWeight)
			defer rw.Close()
			x, err := metal.NewDeviceBufferF32(make([]float32, k))
			if err != nil {
				b.Fatal(err)
			}
			defer x.Release()
			o, err := metal.NewDeviceBufferF32(make([]float32, n))
			if err != nil {
				b.Fatal(err)
			}
			defer o.Release()
			run := func() {
				r, err := metal.NewRecorder()
				if err != nil {
					b.Fatal(err)
				}
				if err := r.QMatMulResident(x, rw, o, 1); err != nil {
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
			b.SetBytes(int64(len(weight)))
			b.ResetTimer()
			for range b.N {
				run()
			}
		})
	}
}
