package nlp

import "testing"

import "github.com/jxsl13/goai/tensor"

// §R97: keepSinkRecent keeps the first `sinks` rows (the attention sinks) plus the last
// `window` rows (the rolling window), evicting the middle — and is a no-op under budget.
func TestKeepSinkRecent(t *testing.T) {
	mk := func(rows int) *tensor.Tensor { // row i is labeled by value i in its single column
		x := tensor.New(tensor.F64, tensor.Shape{rows, 1})
		for i := range rows {
			x.SetF64(float64(i), i, 0)
		}
		return x
	}
	if got := keepSinkRecent(mk(5), 2, 4); got.Shape()[0] != 5 {
		t.Errorf("under budget should be a no-op, got %d rows", got.Shape()[0])
	}
	// 8 rows, keep first 2 sinks + last 3 recent → rows [0,1, 5,6,7]
	got := keepSinkRecent(mk(8), 2, 3)
	want := []float64{0, 1, 5, 6, 7}
	if got.Shape()[0] != len(want) {
		t.Fatalf("kept %d rows, want %d", got.Shape()[0], len(want))
	}
	for i, w := range want {
		if got.AtF64(i, 0) != w {
			t.Errorf("kept row %d = %v, want %v (sinks 0,1 + recent 5,6,7)", i, got.AtF64(i, 0), w)
		}
	}
}
