//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// The paged KV pool must reproduce a plain contiguous append: append tokens in ragged chunks
// that cross block boundaries, gather back, and require an exact match to the reference. Also
// exercises multi-sequence sharing (two seqs interleave block allocations) and Release.
func TestCUDAPagedKV(t *testing.T) {
	skipNoGPU(t)
	const blockSize, wkv, numBlocks = 16, 128, 64
	pool, err := cuda.NewPagedKVPool(numBlocks, blockSize, wkv)
	must(t, err)
	defer pool.Free()

	rng := rand.New(rand.NewSource(41))
	// two sequences appended in interleaved ragged chunks
	seqs := []*cuda.SeqKV{pool.NewSeqKV(), pool.NewSeqKV()}
	refK := [][]float32{{}, {}}
	refV := [][]float32{{}, {}}
	chunks := []int{5, 16, 3, 20, 11, 1, 17} // deliberately cross 16-token blocks
	for _, ct := range chunks {
		for si := range seqs {
			kf := make([]float32, ct*wkv)
			vf := make([]float32, ct*wkv)
			for i := range kf {
				kf[i] = float32(rng.NormFloat64())
				vf[i] = float32(rng.NormFloat64())
			}
			dk, _ := cuda.NewDeviceF32(ct, wkv)
			must(t, dk.UploadF32(kf))
			dv, _ := cuda.NewDeviceF32(ct, wkv)
			must(t, dv.UploadF32(vf))
			must(t, seqs[si].Append(dk, dv))
			dk.Free()
			dv.Free()
			refK[si] = append(refK[si], kf...)
			refV[si] = append(refV[si], vf...)
		}
	}

	for si, s := range seqs {
		gk, err := s.GatherK()
		must(t, err)
		gv, err := s.GatherV()
		must(t, err)
		hk, err := gk.ToHost()
		must(t, err)
		hv, err := gv.ToHost()
		must(t, err)
		gk.Free()
		gv.Free()
		n := s.Len()
		for r := 0; r < n; r++ {
			for c := 0; c < wkv; c++ {
				if got, want := float32(hk.AtF64(r, c)), refK[si][r*wkv+c]; got != want {
					t.Fatalf("seq %d K[%d,%d] = %g want %g", si, r, c, got, want)
				}
				if got, want := float32(hv.AtF64(r, c)), refV[si][r*wkv+c]; got != want {
					t.Fatalf("seq %d V[%d,%d] = %g want %g", si, r, c, got, want)
				}
			}
		}
		t.Logf("seq %d: %d tokens over %d blocks, gathered exact", si, n, len(s.BlockTable()))
	}

	// Release seq 0 -> blocks return to the pool.
	before := pool.FreeBlocks()
	nb := len(seqs[0].BlockTable())
	seqs[0].Release()
	if got := pool.FreeBlocks(); got != before+nb {
		t.Fatalf("Release: free blocks %d, want %d", got, before+nb)
	}
	t.Logf("Release returned %d blocks (pool free %d -> %d)", nb, before, pool.FreeBlocks())
}
