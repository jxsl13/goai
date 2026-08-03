package autograd

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// transposeShapes straddles the fan-out gate ON PURPOSE. parallelBands runs the body in a
// single serial call below d*workPerBand = 1<<15, so a shape under that threshold exercises
// the SERIAL path in both arms and cannot fail on a banding mistake however wrong the
// banding is. The last five shapes clear it; without them a mutation that lets each band
// tile past its end — overwriting its neighbor's rows — passed every case.
//
// The shapes also cover what the band arithmetic can get wrong: a row count that is not a
// multiple of blockT (257, 129, 8001), a row count SMALLER than the worker count (5, which
// takes the nw > d clamp), and a destination row so short that a band is almost all loop
// overhead (n = 5).
var transposeShapes = [][2]int{
	{2, 3}, {33, 31}, {64, 64}, // below the gate: serial in both arms
	{200, 200}, {257, 129}, {129, 257}, {5, 8000}, {8001, 5},
}

// TestTransposeVJPBandsMatchTheDefinition checks the banded copy against the definition of
// the adjoint rather than against a second copy of the loop: gin[i,j] must be g[j,i].
//
// Every source value is non-zero, and gin comes back from a fresh zeroed allocation, so a
// cell no band ever wrote reads as 0 and fails here — the comparison carries its own
// sentinel and needs no prefill. Run it under -race as well; the value check catches a band
// that skips or duplicates rows, and the race detector catches two bands that overlap on
// rows whose values happen to agree.
func TestTransposeVJPBandsMatchTheDefinition(t *testing.T) {
	fn := vjps[backend.OpTranspose]
	ctx := backend.NewContext()
	for _, s := range transposeShapes {
		m, n := s[0], s[1]
		for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
			x := tensor.New(dt, tensor.Shape{m, n})
			g := tensor.New(dt, tensor.Shape{n, m})
			want := make([]float64, n*m)
			for k := range want {
				want[k] = float64(k+1) * 0.5 // never zero, and exact in F32
			}
			switch dt {
			case tensor.F64:
				copy(g.Storage().F64(), want)
			case tensor.F32:
				gs := g.Storage().F32()
				for k, v := range want {
					gs[k] = float32(v)
				}
			}
			out, err := fn(ctx, []*tensor.Tensor{x}, nil, nil, g)
			if err != nil {
				t.Fatalf("%v %dx%d: %v", dt, m, n, err)
			}
			for i := range m {
				for j := range n {
					got := out[0].AtF64(i, j)
					if got != want[j*m+i] {
						t.Fatalf("%v %dx%d: gin[%d,%d] = %v, want g[%d,%d] = %v",
							dt, m, n, i, j, got, j, i, want[j*m+i])
					}
				}
			}
		}
	}
}
