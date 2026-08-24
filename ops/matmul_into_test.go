package ops_test

import (
	"testing"

	"github.com/jxsl13/goai/ops"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/ref"
)

func TestMatMulIntoOverwritesOutput(t *testing.T) {
	a := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	b := tensor.FromFloat64(tensor.Shape{3, 2}, []float64{7, 8, 9, 10, 11, 12})
	out := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{-1, -1, -1, -1})
	if err := ops.MatMulInto(out, a, b); err != nil {
		t.Fatal(err)
	}
	want := []float64{58, 64, 139, 154}
	for i, v := range out.Storage().F64() {
		if v != want[i] {
			t.Fatalf("out[%d] = %v, want %v", i, v, want[i])
		}
	}
}
