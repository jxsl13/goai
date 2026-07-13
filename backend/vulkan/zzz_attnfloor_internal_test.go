//go:build vulkan && cgo

package vulkan

import (
	"testing"
	"time"
)

// §T396 (§V22): before porting the metal §T394 MPS-matmul attention (6.9× win) to Vulkan, MEASURE
// whether it even helps here. Vulkan has NO MPS — its tiled matmul.spv is ~594 GFLOP/s vs metal's
// MPS ~1186. The two attention matmuls (8.6 GFLOP at 512×8×64) at 594 GFLOP/s ≈ 14.5ms — possibly
// SLOWER than Vulkan's own flash kernel. This records the matmul-only floor (16 Recorder matmuls,
// no softmax) to decide: if the floor is already ≥ the flash kernel, the reformulation does NOT help
// on Vulkan/MoltenVK and must not be built (§V22 measure-the-floor, like metal §T393).
func TestVulkanAttentionMatmulFloor(t *testing.T) {
	if !Available() {
		t.Skip("vulkan: no compute-capable device (§V4)")
	}
	const seq, heads, dk = 512, 8, 64
	q, _ := NewDeviceBufferF32(make([]float32, seq*dk))
	kt, _ := NewDeviceBufferF32(make([]float32, dk*seq))
	s, _ := NewDeviceBufferF32(make([]float32, seq*seq))
	v, _ := NewDeviceBufferF32(make([]float32, seq*dk))
	o, _ := NewDeviceBufferF32(make([]float32, seq*dk))
	defer q.Release()
	defer kt.Release()
	defer s.Release()
	defer v.Release()
	defer o.Release()

	floor := func() {
		r, err := NewRecorder()
		if err != nil {
			t.Fatal(err)
		}
		for range heads {
			if err := r.MatMul(q, kt, s, seq, dk, seq); err != nil { // S = Q·Kᵀ
				t.Fatal(err)
			}
			if err := r.MatMul(s, v, o, seq, seq, dk); err != nil { // O = S·V
				t.Fatal(err)
			}
		}
		if err := r.Finish(); err != nil {
			t.Fatal(err)
		}
		r.Free()
	}
	floor()
	const N = 20
	start := time.Now()
	for range N {
		floor()
	}
	ms := float64(time.Since(start).Microseconds()) / N / 1000
	gflop := float64(4*seq*seq*dk*heads) / 1e9

	// compare to the current flash kernel: if the matmul floor is already ≥ flash, the metal §T394
	// MPS-matmul reformulation does NOT help on Vulkan and must not be ported (§V22).
	dm := heads * dk
	q2 := make([]float32, seq*dm)
	k2 := make([]float32, seq*heads*dk)
	v2 := make([]float32, seq*heads*dk)
	for i := range q2 {
		q2[i] = float32(i%17-8) * 0.01
	}
	for i := range k2 {
		k2[i] = float32(i%13-6) * 0.01
		v2[i] = float32(i%11-5) * 0.01
	}
	scale := float32(0.125)
	flashMs := func() float64 {
		if _, err := flashAttnHost(q2, k2, v2, seq, dm, heads, dk, 1, heads, scale); err != nil {
			t.Fatal(err)
		}
		s := time.Now()
		for range N {
			if _, err := flashAttnHost(q2, k2, v2, seq, dm, heads, dk, 1, heads, scale); err != nil {
				t.Fatal(err)
			}
		}
		return float64(time.Since(s).Microseconds()) / N / 1000
	}()
	t.Logf("vulkan attn %d×%d×%d: flash kernel %.2f ms | matmul-only floor %.2f ms (%.0f GFLOP/s, +softmax) → reformulation %s",
		seq, heads, dk, flashMs, ms, gflop/(ms/1000),
		map[bool]string{true: "HELPS (build it)", false: "does NOT help (skip)"}[ms*1.5 < flashMs])
}
