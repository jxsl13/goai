package ops_test

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/ops"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// numpy.broadcast_to: a [3] row broadcasts over rows to [2,3]; a [2,1] column
// broadcasts over columns to [2,3]; the shapes right-align.
func TestBroadcastForward(t *testing.T) {
	row := tensor.FromFloat64(tensor.Shape{3}, []float64{10, 20, 30})
	r, err := ops.BroadcastTo(row, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	if !r.Shape().Equal(tensor.Shape{2, 3}) {
		t.Fatalf("row shape %v", r.Shape())
	}
	for i := range 2 {
		for j := range 3 {
			if r.AtF64(i, j) != float64((j+1)*10) {
				t.Errorf("row[%d,%d]=%g", i, j, r.AtF64(i, j))
			}
		}
	}

	col := tensor.FromFloat64(tensor.Shape{2, 1}, []float64{1, 2})
	c, _ := ops.BroadcastTo(col, 2, 3)
	if !c.Shape().Equal(tensor.Shape{2, 3}) {
		t.Fatalf("col shape %v", c.Shape())
	}
	for i := range 2 {
		for j := range 3 {
			if c.AtF64(i, j) != float64(i+1) {
				t.Errorf("col[%d,%d]=%g, want %d", i, j, c.AtF64(i, j), i+1)
			}
		}
	}

	// scalar → vector
	s, _ := ops.BroadcastTo(tensor.FromFloat64(tensor.Shape{1}, []float64{7}), 4)
	for i := range 4 {
		if s.AtF64(i) != 7 {
			t.Errorf("scalar bcast[%d]=%g", i, s.AtF64(i))
		}
	}
}

func TestBroadcastErrors(t *testing.T) {
	x := tensor.New(tensor.F64, tensor.Shape{2, 3})
	if _, err := ops.BroadcastTo(x, 2, 4); err == nil {
		t.Error("incompatible dim (3→4) must error")
	}
	if _, err := ops.BroadcastTo(x, 3); err == nil {
		t.Error("broadcasting to a smaller rank must error")
	}
	if _, err := ops.BroadcastTo(tensor.New(tensor.F64, tensor.Shape{3}), 2, 5); err == nil {
		t.Error("incompatible (3→5) must error")
	}
}

// BroadcastTo replicates a tensor along size-1/missing axes, like
// numpy.broadcast_to. Here a length-3 row is broadcast to a 2×3 matrix.
func ExampleBroadcastTo() {
	row := tensor.FromFloat64(tensor.Shape{3}, []float64{1, 2, 3})
	out, _ := ops.BroadcastTo(row, 2, 3)
	fmt.Println(out.Shape(), out.AtF64(1, 2))
	// Output: (2, 3) 3
}
