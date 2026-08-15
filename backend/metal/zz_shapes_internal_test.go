//go:build darwin && cgo

package metal

import (
	"fmt"
	"testing"
)

// MPS GEMM at the ACTUAL TinyLlama projection shapes, not the one shape the 3286 GFLOP/s figure
// came from.
func TestZZGemmShapes(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	for _, M := range []int{64, 128} {
		var tot float64
		var totFlop float64
		for _, p := range []struct {
			name string
			K, N int
		}{{"qkv", 2048, 2560}, {"o", 2048, 2048}, {"gate|up", 2048, 11264}, {"down", 5632, 2048}} {
			a, _ := NewDeviceBufferF32(make([]float32, M*p.K))
			b, _ := NewDeviceBufferF32(make([]float32, p.K*p.N))
			c, _ := NewDeviceBufferF32(make([]float32, M*p.N))
			best := 1e18
			for range 12 {
				r, _ := NewRecorder()
				for range 4 {
					if err := r.MatMul(a, b, c, M, p.K, p.N); err != nil {
						t.Skip(err)
					}
				}
				r.Commit()
				r.Wait()
				if d := LastGPUSeconds() / 4; d < best {
					best = d
				}
				r.Free()
			}
			fl := float64(2*M*p.K*p.N) / best / 1e9
			fmt.Printf("SHP M=%3d %-8s K=%4d N=%5d  %7.1f us  %6.0f GFLOP/s\n", M, p.name, p.K, p.N, best*1e6, fl)
			tot += best
			totFlop += float64(2 * M * p.K * p.N)
			a.Release()
			b.Release()
			c.Release()
		}
		fmt.Printf("SHP M=%3d TOTAL GEMM per layer: %.2f ms  (%.0f GFLOP/s effective)\n", M, tot*1000, totFlop/tot/1e9)
	}
}
