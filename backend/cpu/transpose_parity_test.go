package cpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// transposeShapes straddles the fan-out gate ON PURPOSE. parallelWork runs the body inline
// below its work threshold, so a shape under it exercises the SERIAL path in both arms and
// cannot fail on a banding mistake however wrong the banding is. The last four clear it, and
// 257 and 129 are not multiples of the tile or of the worker count.
var transposeShapes = [][2]int{
	{1, 1}, {3, 5}, {16, 16}, {17, 33},
	{257, 129}, {129, 257}, {512, 64}, {64, 512},
}

// TestTransposeCPUMatchesRefBitExactly gates the cpu kernel against the reference bit for bit.
// A transpose moves values without arithmetic, so agreement here is a claim about ADDRESSES —
// which cell landed where — and a tolerance comparison would say nothing about it at all.
//
// Run this under -race as well. The two tests catch different mistakes and neither subsumes the
// other: a band that skips or duplicates rows fails the value comparison, while two bands that
// OVERLAP write the same value twice and are invisible to it, showing up only as a data race.
func TestTransposeCPUMatchesRefBitExactly(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		for _, s := range transposeShapes {
			m, n := s[0], s[1]
			x := tensor.New(dt, tensor.Shape{m, n})
			for i := range m {
				for j := range n {
					x.SetF64(math.Sin(float64(i*31+j*7))*3, i, j)
				}
			}
			in := []*tensor.Tensor{x}
			ref, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()),
				backend.OpTranspose, in, nil)
			if err != nil {
				t.Fatalf("%v %dx%d ref: %v", dt, m, n, err)
			}
			got, err := backend.Execute(backend.NewContext(), backend.OpTranspose, in, nil)
			if err != nil {
				t.Fatalf("%v %dx%d cpu: %v", dt, m, n, err)
			}
			if !got[0].Shape().Equal(tensor.Shape{n, m}) {
				t.Fatalf("%v %dx%d: shape %v", dt, m, n, got[0].Shape())
			}
			for i := range n {
				for j := range m {
					w, g := ref[0].AtF64(i, j), got[0].AtF64(i, j)
					if math.Float64bits(w) != math.Float64bits(g) {
						t.Fatalf("%v %dx%d: out[%d,%d] cpu %v != ref %v", dt, m, n, i, j, g, w)
					}
				}
			}
		}
	}
}
