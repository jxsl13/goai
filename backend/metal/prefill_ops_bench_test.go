//go:build darwin && cgo

package metal

import (
	"fmt"
	"testing"
)

// TestPrefillOpCosts reports the non-matmul GPU work in a prefill layer, at PREFILL shapes
// (rows=64) rather than the decode shapes (rows=1) every earlier per-op figure was taken at.
//
// It exists because the prefill budget did not add up twice running, and both times the error was
// an assumed component rather than a measured one. With these numbers it closes:
//
//	expansion 0.99 + GEMM 1.70 + non-matmul 0.54 = 3.23 ms/layer, measured 3.79 (85%)
//
// The distribution is lopsided. Per layer: 2 RMSNorm 12.6us, SwiGLU 9.1us, 2 residual adds 8.4us,
// attention 159.3us — attention is 84% of the non-matmul GPU work, and it runs at 211 GFLOP/s,
// 3.1% of the 6.8 TFLOP/s peak.
//
// That is only ~4% of a layer at a 64-token prompt, so it is not today's bottleneck. It is a
// LATENT one: attention is O(sq*sk), and the measured points already show it (sk=64 159us,
// sk=128 354us). At a 512-token prompt it would dominate the layer. Prefill attention currently
// routes to the decode kernel (mtl_recorder_mha sends causal!=0 there), which streams keys per
// query row rather than tiling them — the right shape for sq=1, the wrong one for sq=512.
//
// Reported, not asserted beyond loose ceilings; absolute timings are machine-dependent.
func TestPrefillOpCosts(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	const rows, dim, hid, heads, kvh, dk = 64, 2048, 5632, 32, 4, 64
	big, _ := NewDeviceBufferF32(make([]float32, rows*hid))
	b2, _ := NewDeviceBufferF32(make([]float32, rows*hid))
	g, _ := NewDeviceBufferF32(make([]float32, hid))
	qkv, _ := NewDeviceBufferF32(make([]float32, rows*(dim+2*kvh*dk)))
	o, _ := NewDeviceBufferF32(make([]float32, rows*dim))
	defer func() {
		for _, b := range []*DeviceBuffer{big, b2, g, qkv, o} {
			b.Release()
		}
	}()
	slope := func(name string, ceiling float64, rec func(r *Recorder) error) {
		meas := func(n int) float64 {
			best := 1e18
			for range 15 {
				r, err := NewRecorder()
				if err != nil {
					t.Fatal(err)
				}
				for range n {
					if err := rec(r); err != nil {
						t.Fatalf("%s: %v", name, err)
					}
				}
				r.Commit()
				r.Wait()
				if d := LastGPUSeconds(); d < best {
					best = d
				}
				r.Free()
			}
			return best
		}
		lo, hi := meas(16), meas(128)
		us := (hi - lo) / (128 - 16) * 1e6
		fmt.Printf("prefill op %-24s %7.2f us\n", name, us)
		if us > ceiling {
			t.Errorf("%s: %.2f us > %.0f us ceiling", name, us, ceiling)
		}
	}
	slope("RMSNorm rows=64 d=2048", 60, func(r *Recorder) error { return r.RMSNorm(big, g, o, rows, dim, 1e-5) })
	slope("SwiGLU 64x5632", 90, func(r *Recorder) error { return r.BinaryN(big, b2, big, 6, rows*hid) })
	slope("residual add 64x2048", 40, func(r *Recorder) error { return r.BinaryN(big, b2, big, 0, rows*dim) })
	for _, sk := range []int{64, 128} {
		kk, _ := NewDeviceBufferF32(make([]float32, sk*kvh*dk))
		vv, _ := NewDeviceBufferF32(make([]float32, sk*kvh*dk))
		ceil := 600.0
		if sk == 128 {
			ceil = 1200
		}
		slope(fmt.Sprintf("attention sq=64 sk=%d", sk), ceil, func(r *Recorder) error {
			return r.MHA(qkv, kk, vv, o, rows, sk, dim, heads, kvh, dk, 1, 0, 0.125)
		})
		kk.Release()
		vv.Release()
	}
}
