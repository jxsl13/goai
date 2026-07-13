//go:build darwin && cgo

package metal

import (
	"testing"
	"time"
)

// §T393 (§V22): the flash attention forward for 512×8×64 measures ~11ms on metal vs torch-mps
// ~0.45ms (~24×). Is that gap closeable by a better hand-written kernel, or is it structural? This
// measures the MATMUL-ONLY FLOOR: attention fundamentally needs, per head, S=Q·Kᵀ ([seq,dk]·[dk,seq])
// and O=P·V ([seq,seq]·[seq,dk]). This records those 2·heads MPS matmuls (the same MPS path the
// recorder uses, near-peak on metal) into ONE command buffer and times them — NO softmax, NO masking.
// If even this pure-matmul floor is far above torch's 0.45ms, then no matmul-based (non-fused) kernel
// can close the gap: torch must use a FUSED attention primitive (MPSGraph scaledDotProductAttention /
// simdgroup ops) that avoids materializing S and runs the reduction in-register — that's the real
// lever, not more hand-tiling of the flash kernel.
func TestAttentionMatmulFloor(t *testing.T) {
	if !Available() {
		t.Skip("metal: no gpu device — skipped")
	}
	const seq, heads, dk = 512, 8, 64
	// per-head device buffers (reused across heads — we measure throughput, not correctness).
	q, _ := NewDeviceBufferF32(make([]float32, seq*dk))
	kt, _ := NewDeviceBufferF32(make([]float32, dk*seq)) // Kᵀ [dk,seq] so plain MatMul gives Q·Kᵀ
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
			// S = Q·Kᵀ : [seq,dk]·[dk,seq] → [seq,seq]
			if err := r.MatMul(q, kt, s, seq, dk, seq); err != nil {
				t.Fatal(err)
			}
			// O = S·V : [seq,seq]·[seq,dk] → [seq,dk]  (S stands in for the softmax weights)
			if err := r.MatMul(s, v, o, seq, seq, dk); err != nil {
				t.Fatal(err)
			}
		}
		if err := r.Finish(); err != nil {
			t.Fatal(err)
		}
		r.Free()
	}
	floor() // warmup
	const N = 30
	start := time.Now()
	for range N {
		floor()
	}
	ms := float64(time.Since(start).Microseconds()) / N / 1000
	// FLOPs: per head 2 matmuls · 2·seq·seq·dk each = 4·seq²·dk; ×heads.
	gflop := float64(4*seq*seq*dk*heads) / 1e9
	t.Logf("attention MATMUL-ONLY floor %d×%d×%d (%d MPS matmuls, no softmax): %.2f ms (%.0f GFLOP/s) — vs flash-kernel ~11ms, torch-mps ~0.45ms",
		seq, heads, dk, 2*heads, ms, gflop/(ms/1000))
}
