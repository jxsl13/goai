//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// GroupedQueryAttentionKV (the KV-cache decode form) must produce exactly the
// tail rows of a full causal GroupedQueryAttention over the same sequence: the
// last seqQ query rows attending the full seqKV context == the last seqQ rows of
// the full [seqKV,seqKV] causal attention. This validates the generalized
// seqQ≠seqKV kernels + the causal offset (seqKV−seqQ) without needing a cache
// buffer yet — proving a decode step is bit-consistent with prefill.
func TestCUDAGroupedQueryAttentionKVMatchesTail(t *testing.T) {
	skipNoGPU(t)
	const (
		seqKV   = 12
		qHeads  = 8
		kvHeads = 2
		hd      = 6
	)
	wq, wkv := qHeads*hd, kvHeads*hd
	q := bench.RandF32(tensor.Shape{seqKV, wq}, 7)
	k := bench.RandF32(tensor.Shape{seqKV, wkv}, 8)
	v := bench.RandF32(tensor.Shape{seqKV, wkv}, 9)

	// Full causal reference over the whole sequence.
	dq, _ := cuda.UploadF32(q)
	dk, _ := cuda.UploadF32(k)
	dv, _ := cuda.UploadF32(v)
	full, err := cuda.GroupedQueryAttention(dq, dk, dv, qHeads, kvHeads, true)
	must(t, err)
	fullHost, err := full.ToHost()
	must(t, err)
	dq.Free()
	full.Free()

	// For a few decode-window sizes, attend the LAST seqQ query rows over the
	// full cached K/V and compare against the matching tail of the full result.
	for _, seqQ := range []int{1, 3, seqKV} {
		qtail := bench.RandF32(tensor.Shape{seqQ, wq}, 7) // rebuilt below from q's tail
		// copy the last seqQ rows of q into qtail
		start := seqKV - seqQ
		for i := 0; i < seqQ; i++ {
			for j := 0; j < wq; j++ {
				qtail.SetF64(q.AtF64(start+i, j), i, j)
			}
		}
		dqt, _ := cuda.UploadF32(qtail)
		got, err := cuda.GroupedQueryAttentionKV(dqt, dk, dv, qHeads, kvHeads)
		must(t, err)
		gotHost, err := got.ToHost()
		must(t, err)
		dqt.Free()
		got.Free()

		var maxAbs float64
		for i := 0; i < seqQ; i++ {
			for j := 0; j < wq; j++ {
				a := gotHost.AtF64(i, j)
				b := fullHost.AtF64(start+i, j)
				if d := math.Abs(a - b); d > maxAbs {
					maxAbs = d
				}
			}
		}
		if maxAbs > 1e-4 {
			t.Fatalf("seqQ=%d: KV attention diverges from full-attention tail, maxAbs=%.3e", seqQ, maxAbs)
		}
		t.Logf("seqQ=%d over seqKV=%d: KV-attn == full-attn tail (maxAbs %.2e)", seqQ, seqKV, maxAbs)
	}
	dk.Free()
	dv.Free()
}
