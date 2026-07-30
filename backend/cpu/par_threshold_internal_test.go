package cpu

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestParThresholdArmsAgree forces both arms of parallelWork for a spread of ops and requires
// bit-identical results.
//
// parThreshold is the single gate every op in this package routes through — 28 files call
// parallelWork — and PS6023 reported that no test named it, so nothing forced the serial arm on
// data large enough to fan out, nor the parallel arm on data small enough to stay serial. Each
// kernel documents its partition as value-neutral; this is what checks that claim rather than
// trusting 28 comments.
//
// Bit-for-bit, not a tolerance. A partition either preserves every value or it has reassociated a
// reduction, and a tolerance would hide precisely the reassociation worth catching.
//
// The ops are chosen to span the partition SHAPES rather than to be a long list: an elementwise
// binary (split over a flat range), a row-reduction whose per-row accumulation order must survive
// the split, a bias add over rows, and a matmul (split over row bands).
func TestParThresholdArmsAgree(t *testing.T) {
	saved := parThreshold
	defer func() { parThreshold = saved }()
	be, ok := backend.Get(backend.CPU)
	if !ok {
		t.Fatal("cpu backend not registered")
	}
	ctx := backend.NewContext().WithBackend(be)
	rng := rand.New(rand.NewPCG(17, 41))

	mk := func(shape tensor.Shape) *tensor.Tensor {
		tt := tensor.New(tensor.F64, shape)
		s := tt.Storage().F64()
		for i := range s {
			s[i] = rng.NormFloat64()
		}
		return tt
	}
	const rows, cols = 96, 512 // rows*cols well above the shipped threshold

	for _, tc := range []struct {
		name string
		op   backend.Op
		ins  []*tensor.Tensor
	}{
		{"Add", backend.OpAdd, []*tensor.Tensor{mk(tensor.Shape{rows, cols}), mk(tensor.Shape{rows, cols})}},
		{"Mul", backend.OpMul, []*tensor.Tensor{mk(tensor.Shape{rows, cols}), mk(tensor.Shape{rows, cols})}},
		{"Softmax", backend.OpSoftmax, []*tensor.Tensor{mk(tensor.Shape{rows, cols})}},
		{"MatMul", backend.OpMatMul, []*tensor.Tensor{mk(tensor.Shape{rows, cols}), mk(tensor.Shape{cols, rows})}},
	} {
		run := func(gate int) []float64 {
			parThreshold = gate
			out, err := backend.Execute(ctx, tc.op, tc.ins, nil)
			if err != nil {
				t.Fatalf("%s: %v", tc.name, err)
			}
			return append([]float64(nil), out[0].Contiguous().Storage().F64()...)
		}
		serial := run(1 << 30) // never fans out
		par := run(0)          // always fans out
		if len(serial) != len(par) {
			t.Fatalf("%s: %d values serial, %d parallel", tc.name, len(serial), len(par))
		}
		var nonzero int
		for i := range serial {
			if math.Float64bits(serial[i]) != math.Float64bits(par[i]) {
				t.Fatalf("%s: value %d is %v (%016x) serial but %v (%016x) parallel — the partition "+
					"changed a result", tc.name, i, serial[i], math.Float64bits(serial[i]),
					par[i], math.Float64bits(par[i]))
			}
			if serial[i] != 0 {
				nonzero++
			}
		}
		// Without this the comparison could pass on an all-zero output, which would agree across
		// the arms whatever the partition did.
		if nonzero == 0 {
			t.Fatalf("%s: every output value is zero; the comparison is vacuous", tc.name)
		}
	}
}
