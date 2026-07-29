package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// nsaGeom pins NSABranches bit-for-bit across geometries that exercise the branch
// structure: a partial trailing block, topN below and at the block count, a window
// shorter and longer than the sequence, and blockSize 1.
type nsaGeom struct {
	seq, dm, heads, block, topN, window int
	sum                                 uint64 // FNV-1a over the raw bits of cmp|slc|win
}

func nsaChecksum(t *testing.T, g nsaGeom) uint64 {
	t.Helper()
	mk := func(seed float64) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape{g.seq, g.dm})
		s := x.Storage().F64()
		for i := range s {
			s[i] = math.Sin(seed + 1.7*float64(i))
		}
		return x
	}
	cmp, slc, win, err := nn.NSABranches(mk(1), mk(2), mk(3), g.heads, g.block, g.topN, g.window, 0)
	if err != nil {
		t.Fatalf("%+v: %v", g, err)
	}
	h := uint64(14695981039346656037)
	for _, out := range []*tensor.Tensor{cmp, slc, win} {
		for i := range out.Numel() {
			b := math.Float64bits(out.AtF64(tensor.Unravel(i, out.Shape())...))
			for s := 0; s < 64; s += 8 {
				h = (h ^ ((b >> s) & 0xff)) * 1099511628211
			}
		}
	}
	return h
}

// TestNSABranchesBitIdentical guards every optimization to this module. The constants were
// captured from the unoptimized implementation; any change that moves a single bit of any
// of the three branch outputs fails here.
func TestNSABranchesBitIdentical(t *testing.T) {
	for _, g := range nsaGolden {
		if got := nsaChecksum(t, g); got != g.sum {
			t.Fatalf("seq=%d dm=%d heads=%d block=%d topN=%d window=%d: checksum %d, want %d",
				g.seq, g.dm, g.heads, g.block, g.topN, g.window, got, g.sum)
		}
	}
}

// TestNSABranchesDeterministic pins tie handling. Block importances can tie (identical
// pooled rows), and the selection sort's order on equal weights decides which blocks the
// slc branch attends. Repeated runs must agree exactly.
func TestNSABranchesDeterministic(t *testing.T) {
	seq, dm := 24, 16
	// every key row identical => every block importance ties
	flat := tensor.New(tensor.F64, tensor.Shape{seq, dm})
	for i := range flat.Storage().F64() {
		flat.Storage().F64()[i] = 0.25
	}
	q := tensor.New(tensor.F64, tensor.Shape{seq, dm})
	for i := range q.Storage().F64() {
		q.Storage().F64()[i] = 0.5
	}
	var first []float64
	for run := range 4 {
		_, slc, _, err := nn.NSABranches(q, flat, flat, 2, 4, 3, 8, 0)
		if err != nil {
			t.Fatal(err)
		}
		cur := make([]float64, slc.Numel())
		for i := range cur {
			cur[i] = slc.AtF64(tensor.Unravel(i, slc.Shape())...)
		}
		if run == 0 {
			first = cur
			continue
		}
		for i := range cur {
			if math.Float64bits(cur[i]) != math.Float64bits(first[i]) {
				t.Fatalf("run %d differs from run 0 at %d: %g vs %g — tie order is not deterministic",
					run, i, cur[i], first[i])
			}
		}
	}
}
