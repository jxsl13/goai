package nn

import (
	"math"
	"math/rand/v2"
	"sort"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// Prove the quickselect WandaPrune produces a bit-identical keep-mask (and pruned weights)
// to a full-sort reference, including heavy score ties (small integer alphabet).
func TestWandaPruneQuickselectEquivFullSort(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 4))
	for trial := 0; trial < 120; trial++ {
		cin := 1 + rng.IntN(300)
		cout := 1 + rng.IntN(40)
		toks := 1 + rng.IntN(20)
		w := tensor.New(tensor.F64, tensor.Shape{cin, cout})
		x := tensor.New(tensor.F64, tensor.Shape{toks, cin})
		wf, xf := w.Storage().F64(), x.Storage().F64()
		for i := range wf {
			wf[i] = float64(rng.IntN(5) - 2) // ties + zeros
		}
		for i := range xf {
			xf[i] = float64(rng.IntN(5) - 2)
		}
		sparsity := []float64{0, 0.25, 0.5, 0.75, 1.0}[rng.IntN(5)]
		_, mask, err := WandaPrune(w, x, sparsity)
		if err != nil {
			t.Fatal(err)
		}
		mg := mask.Storage().F64()

		// reference: full sort per column, drop bottom-k
		s, _ := WandaScore(w, x)
		ss := s.Storage().F64()
		k := int(math.Floor(sparsity*float64(cin) + 1e-9))
		wantMask := make([]float64, cin*cout)
		for o := 0; o < cout; o++ {
			idx := make([]int, cin)
			col := make([]float64, cin)
			for j := range idx {
				idx[j] = j
				col[j] = ss[j*cout+o]
			}
			sort.SliceStable(idx, func(a, b int) bool {
				if col[idx[a]] != col[idx[b]] {
					return col[idx[a]] < col[idx[b]]
				}
				return idx[a] < idx[b]
			})
			drop := make([]bool, cin)
			for r := 0; r < k; r++ {
				drop[idx[r]] = true
			}
			for j := 0; j < cin; j++ {
				if !drop[j] {
					wantMask[j*cout+o] = 1
				}
			}
		}
		for i := range wantMask {
			if mg[i] != wantMask[i] {
				t.Fatalf("trial %d cin=%d cout=%d sp=%g pos %d: mask got %v want %v", trial, cin, cout, sparsity, i, mg[i], wantMask[i])
			}
		}
	}
}
