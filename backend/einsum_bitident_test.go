package backend_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// einsumMaps is the ORIGINAL map-driven contraction, kept as the oracle. Written
// before the array rewrite it guards, because a one-ulp perturbation of the
// contraction product passed the entire backend suite (PROC-009): EinsumContract had
// no correctness coverage at the level the rewrite touches.
func einsumMaps(inSubs [][]byte, outSub []byte, order []byte, ops []*tensor.Tensor) []float64 {
	size := map[byte]int{}
	for k, sub := range inSubs {
		for pos := range len(sub) {
			size[sub[pos]] = ops[k].Shape()[pos]
		}
	}
	outShape := make(tensor.Shape, len(outSub))
	for i := range outSub {
		outShape[i] = size[outSub[i]]
	}
	total := 1
	for _, ix := range order {
		total *= size[ix]
	}
	res := make([]float64, outShape.Numel())
	val := map[byte]int{}
	coords := make([][]int, len(inSubs))
	for k := range inSubs {
		coords[k] = make([]int, len(inSubs[k]))
	}
	outCoord := make([]int, len(outSub))
	strides := tensor.RowMajorStrides(outShape)
	for combo := range total {
		rem := combo
		for _, ix := range order {
			val[ix] = rem % size[ix]
			rem /= size[ix]
		}
		prod := 1.0
		for k, sub := range inSubs {
			for pos := range len(sub) {
				coords[k][pos] = val[sub[pos]]
			}
			prod *= ops[k].AtF64(coords[k]...)
		}
		off := 0
		for i := range len(outSub) {
			outCoord[i] = val[outSub[i]]
			off += outCoord[i] * strides[i]
		}
		res[off] += prod
	}
	return res
}

func TestEinsumContractBitIdentical(t *testing.T) {
	cases := []struct {
		in   []string
		out  string
		dims [][]int
	}{
		{[]string{"ij", "jk"}, "ik", [][]int{{3, 4}, {4, 5}}},          // matmul
		{[]string{"ij"}, "ji", [][]int{{3, 5}}},                        // transpose
		{[]string{"ij", "ij"}, "", [][]int{{4, 4}, {4, 4}}},            // full contraction
		{[]string{"bij", "bjk"}, "bik", [][]int{{2, 3, 4}, {2, 4, 3}}}, // batched
		{[]string{"i"}, "i", [][]int{{6}}},                             // identity
	}
	for ci, c := range cases {
		ops := make([]*tensor.Tensor, len(c.in))
		inSubs := make([][]byte, len(c.in))
		for k := range c.in {
			ops[k] = bench.RandF64(tensor.Shape(c.dims[k]), uint64(ci*10+k+1))
			inSubs[k] = []byte(c.in[k])
		}
		got, err := backend.EinsumContract(inSubs, []byte(c.out), ops)
		if err != nil {
			t.Fatal(err)
		}
		// order = every distinct index across inputs, sorted (matches the parser).
		seen := map[byte]bool{}
		var order []byte
		for _, s := range c.in {
			for i := range len(s) {
				if !seen[s[i]] {
					seen[s[i]] = true
					order = append(order, s[i])
				}
			}
		}
		for i := 0; i < len(order); i++ {
			for j := i + 1; j < len(order); j++ {
				if order[j] < order[i] {
					order[i], order[j] = order[j], order[i]
				}
			}
		}
		want := einsumMaps(inSubs, []byte(c.out), order, ops)
		if got.Numel() != len(want) {
			t.Fatalf("case %d: numel %d != %d", ci, got.Numel(), len(want))
		}
		for i := range want {
			g := got.AtF64(tensor.Unravel(i, got.Shape())...)
			if math.Float64bits(g) != math.Float64bits(want[i]) {
				t.Fatalf("case %d elem %d: got %v want %v", ci, i, g, want[i])
			}
		}
	}
}
