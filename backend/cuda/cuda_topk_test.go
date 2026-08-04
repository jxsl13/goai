//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math/rand"
	"sort"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// TestCUDATopK checks the on-device top-k (cu_topk_f32) against a full CPU sort across a spread of
// vocab sizes and k — the K highest values (and, for the distinct random data here, their indices)
// must match the sorted reference exactly. This is the primitive behind on-device top-k sampling.
func TestCUDATopK(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	rng := rand.New(rand.NewSource(20260805))
	for _, n := range []int{512, 32000, 131072} {
		for _, k := range []int{1, 8, 64, 128} {
			if k > n {
				continue
			}
			x := make([]float32, n)
			for i := range x {
				x[i] = float32(rng.NormFloat64()) * 100 // distinct w.h.p. → no tie ambiguity
			}
			d, err := cuda.NewDeviceF32(1, n)
			must(t, err)
			must(t, d.UploadF32(x))
			gi, gv, err := d.TopK(k)
			d.Free()
			must(t, err)

			// CPU reference: indices sorted by value descending.
			idx := make([]int, n)
			for i := range idx {
				idx[i] = i
			}
			sort.Slice(idx, func(a, b int) bool { return x[idx[a]] > x[idx[b]] })
			for r := 0; r < k; r++ {
				if int(gi[r]) != idx[r] {
					t.Fatalf("n=%d k=%d rank %d: gpu idx %d (val %v) != cpu idx %d (val %v)",
						n, k, r, gi[r], gv[r], idx[r], x[idx[r]])
				}
				if gv[r] != x[idx[r]] {
					t.Fatalf("n=%d k=%d rank %d: gpu val %v != cpu val %v", n, k, r, gv[r], x[idx[r]])
				}
			}
		}
	}
	t.Logf("cu_topk_f32 matches CPU sort across n∈{512,32k,131k}, k∈{1,8,64,128}")
}
