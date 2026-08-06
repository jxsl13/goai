//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math/rand/v2"
	"sort"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// TestTopKMultiBlockCorrect: the multi-block cu_topk_f32 must return exactly the K highest (index,value)
// pairs — validated against a host sort across sizes that exercise both the B==1 single-block path
// (small n) and the multi-block phase-1/merge path (large n), and K values up to the 256 cap.
func TestTopKMultiBlockCorrect(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	for _, n := range []int{600, 2048, 5000, 32000, 128000} {
		for _, K := range []int{1, 40, 64, 256} {
			if K > n {
				continue
			}
			r := rand.New(rand.NewPCG(uint64(n)*131+uint64(K), 0x1234))
			x := make([]float32, n)
			for i := range x {
				x[i] = float32(r.NormFloat64()) * 3
			}
			d, _ := cuda.NewDeviceF32(1, n)
			if err := d.UploadF32(x); err != nil {
				t.Fatal(err)
			}
			gi, gv, err := d.TopKN(n, K)
			d.Free()
			if err != nil {
				t.Fatalf("n=%d K=%d: %v", n, K, err)
			}
			// host reference: indices sorted by value desc, take K.
			idx := make([]int, n)
			for i := range idx {
				idx[i] = i
			}
			sort.Slice(idx, func(a, b int) bool { return x[idx[a]] > x[idx[b]] })
			wantVal := make([]float64, K)
			for j := 0; j < K; j++ {
				wantVal[j] = float64(x[idx[j]])
			}
			// device values sorted desc must equal the host top-K values (multiset), and each returned
			// index must actually carry its reported value.
			gotVal := make([]float64, K)
			for j := 0; j < K; j++ {
				if x[gi[j]] != gv[j] {
					t.Fatalf("n=%d K=%d rank %d: idx %d has logit %v but TopK reported %v", n, K, j, gi[j], x[gi[j]], gv[j])
				}
				gotVal[j] = float64(gv[j])
			}
			sort.Sort(sort.Reverse(sort.Float64Slice(gotVal)))
			for j := 0; j < K; j++ {
				if gotVal[j] != wantVal[j] {
					t.Fatalf("n=%d K=%d rank %d: got value %v want %v (top-K value set mismatch)", n, K, j, gotVal[j], wantVal[j])
				}
			}
			// indices must be distinct
			seen := make(map[int32]bool, K)
			for _, id := range gi {
				if seen[id] {
					t.Fatalf("n=%d K=%d: duplicate index %d in top-K", n, K, id)
				}
				seen[id] = true
			}
		}
	}
}
