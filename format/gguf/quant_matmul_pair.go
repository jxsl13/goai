package gguf

import (
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// QMatMulPair computes two same-shape Q4_K matrix-vector products under one
// output-row fan-out. It is the eager CPU projection primitive for paired
// gates such as SwiGLU. Each output row still calls the ordinary Q4_K row dot,
// so both returned tensors are bit-identical to independent QMatMul calls.
// Other quant types, activation dtypes, and batch sizes return an error.
func QMatMulPair(x *tensor.Tensor, weight0, weight1 []byte, qt QuantType, n, k int) (*tensor.Tensor, *tensor.Tensor, error) {
	if qt != Q4_K {
		return nil, nil, fmt.Errorf("gguf: QMatMulPair requires Q4_K, got %d", qt)
	}
	if n <= 0 || k <= 0 {
		return nil, nil, fmt.Errorf("gguf: QMatMulPair dimensions must be positive, got n=%d k=%d", n, k)
	}
	maxInt := int(^uint(0) >> 1)
	if k > maxInt/2 {
		return nil, nil, fmt.Errorf("gguf: QMatMulPair input dimension overflows paired work: k=%d", k)
	}
	if x.Ndim() != 2 || x.Shape()[0] != 1 || x.Shape()[1] != k ||
		x.Dtype() != tensor.F32 || !x.IsContiguous() || x.Offset() != 0 {
		return nil, nil, fmt.Errorf("gguf: QMatMulPair x must be contiguous offset-zero F32 [1,%d], got %s %v", k, x.Dtype(), x.Shape())
	}
	rowBytes, err := byteSize(uint32(qt), k)
	if err != nil {
		return nil, nil, err
	}
	if n > maxInt/rowBytes {
		return nil, nil, fmt.Errorf("gguf: QMatMulPair dimensions overflow weight size: n=%d rowBytes=%d", n, rowBytes)
	}
	wantBytes := n * rowBytes
	if len(weight0) != wantBytes || len(weight1) != wantBytes {
		return nil, nil, fmt.Errorf("gguf: QMatMulPair weights must each be %d bytes, got %d and %d", wantBytes, len(weight0), len(weight1))
	}

	out0 := tensor.New(tensor.F32, tensor.Shape{1, n})
	out1 := tensor.New(tensor.F32, tensor.Shape{1, n})
	dst0 := out0.Storage().F32()
	dst1 := out1.Storage().F32()
	row := x.Storage().F32()[:k]
	qmatmulParallelChunks(n, 2*k, func(lo, hi int) {
		for ni := lo; ni < hi; ni++ {
			off := ni * rowBytes
			dst0[ni] = float32(dotQ4KRowFn(row, weight0[off:off+rowBytes], k))
			dst1[ni] = float32(dotQ4KRowFn(row, weight1[off:off+rowBytes], k))
		}
	})
	return out0, out1, nil
}
