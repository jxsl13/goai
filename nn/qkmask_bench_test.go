package nn

import (
	"testing"

	"github.com/jxsl13/goai/tensor"
)

func BenchmarkQKCausalMask(b *testing.B) {
	for range b.N {
		qksink = qkCausalMask(tensor.F64, 512, 512)
	}
}

var qksink *tensor.Tensor

func TestQKCausalMaskByteIdentical(t *testing.T) {
	for _, sh := range [][2]int{{1, 1}, {4, 4}, {5, 8}, {8, 5}, {512, 512}, {7, 3}} {
		sq, sk := sh[0], sh[1]
		got := qkCausalMask(tensor.F64, sq, sk)
		// naive SetF64 reference
		off := sk - sq
		want := tensor.New(tensor.F64, tensor.Shape{sq, sk})
		for i := 0; i < sq; i++ {
			for j := 0; j < sk; j++ {
				if j > i+off {
					want.SetF64(-1e30, i, j)
				}
			}
		}
		for i := 0; i < sq; i++ {
			for j := 0; j < sk; j++ {
				if got.AtF64(i, j) != want.AtF64(i, j) {
					t.Fatalf("sq=%d sk=%d [%d,%d]: got %v want %v", sq, sk, i, j, got.AtF64(i, j), want.AtF64(i, j))
				}
			}
		}
	}
}
