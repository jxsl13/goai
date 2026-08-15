//go:build darwin && cgo

package metal_test

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
)

// syntheticKQuant builds a weight blob of the right shape for a K-quant type.
// Only the timing of the resident matvec is read from it; the kernels have no
// data-dependent control flow, so deterministic filler is sufficient. d/dmin are
// pinned to finite f16 values so no NaN/Inf path can be entered.
func syntheticKQuant(n, k, blockBytes, scaleHdr int) []byte {
	blocks := n * (k / 256)
	out := make([]byte, blocks*blockBytes)
	for block := range blocks {
		base := block * blockBytes
		out[base], out[base+1] = 0, 0x3c // f16(1)
		if scaleHdr > 2 {
			out[base+2], out[base+3] = 0, 0x38 // f16(0.5)
		}
		for i := scaleHdr; i < blockBytes; i++ {
			out[base+i] = byte((block*17 + i*29) & 0xff)
		}
	}
	return out
}

// BenchmarkMetalKQuantM1Gap times the M=1 resident decode leaf for every K-quant
// type at one shape. Q4_K and Q6_K have simdgroup-cooperative kernels; Q2_K, Q3_K
// and Q5_K have only the scalar one-thread-per-output-row kernel, which is the
// same occupancy shape the cooperative work was built to fix. This measures
// whether that gap is worth closing for the remaining three.
func BenchmarkMetalKQuantM1Gap(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal device unavailable")
	}
	const k, n = 2048, 2048
	for _, tc := range []struct {
		name       string
		qtype      uint32
		blockBytes int
		scaleHdr   int
	}{
		{"Q2K", uint32(gguf.Q2_K), 84, 4},
		{"Q3K", uint32(gguf.Q3_K), 110, 2},
		{"Q4K_cooperative", uint32(gguf.Q4_K), 144, 4},
		{"Q5K", uint32(gguf.Q5_K), 176, 4},
		{"Q6K_cooperative", uint32(gguf.Q6_K), 210, 2},
	} {
		b.Run(fmt.Sprintf("%s", tc.name), func(b *testing.B) {
			weight := syntheticKQuant(n, k, tc.blockBytes, tc.scaleHdr)
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
