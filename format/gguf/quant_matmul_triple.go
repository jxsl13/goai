package gguf

import (
	"fmt"
	"math/bits"

	"github.com/jxsl13/goai/tensor"
)

// QMatMulTriple computes three unequal-row Q4_K or Q6_K matrix-vector
// products under one flattened output-row fan-out. Each output row still
// calls the ordinary quant-specific row dot, so the returned tensors are
// bit-identical to three independent QMatMul calls. Other quant types,
// activation dtypes, and batch sizes return an error.
func QMatMulTriple(x *tensor.Tensor, weights [3][]byte, qts [3]QuantType, ns [3]int, k int) ([3]*tensor.Tensor, error) {
	var outs [3]*tensor.Tensor
	if k <= 0 {
		return outs, fmt.Errorf("gguf: QMatMulTriple input dimension must be positive, got k=%d", k)
	}
	if x.Ndim() != 2 || x.Shape()[0] != 1 || x.Shape()[1] != k ||
		x.Dtype() != tensor.F32 || !x.IsContiguous() || x.Offset() != 0 {
		return outs, fmt.Errorf("gguf: QMatMulTriple x must be contiguous offset-zero F32 [1,%d], got %s %v", k, x.Dtype(), x.Shape())
	}

	maxInt := int(^uint(0) >> 1)
	var rowBytes [3]int
	totalRows := 0
	for i := range weights {
		if qts[i] != Q4_K && qts[i] != Q6_K {
			return outs, fmt.Errorf("gguf: QMatMulTriple quant %d must be Q4_K or Q6_K, got %d", i, qts[i])
		}
		if ns[i] <= 0 {
			return outs, fmt.Errorf("gguf: QMatMulTriple output dimension %d must be positive, got n=%d", i, ns[i])
		}
		if ns[i] > maxInt-totalRows {
			return outs, fmt.Errorf("gguf: QMatMulTriple output dimensions overflow: %v", ns)
		}
		totalRows += ns[i]
		var err error
		rowBytes[i], err = byteSize(uint32(qts[i]), k)
		if err != nil {
			return outs, err
		}
		if ns[i] > maxInt/rowBytes[i] {
			return outs, fmt.Errorf("gguf: QMatMulTriple dimensions overflow weight %d size: n=%d rowBytes=%d", i, ns[i], rowBytes[i])
		}
		wantBytes := ns[i] * rowBytes[i]
		if len(weights[i]) != wantBytes {
			return outs, fmt.Errorf("gguf: QMatMulTriple weight %d has %d bytes, want %d rows x %d = %d", i, len(weights[i]), ns[i], rowBytes[i], wantBytes)
		}
	}
	if totalRows > maxInt/k {
		return [3]*tensor.Tensor{}, fmt.Errorf("gguf: QMatMulTriple work overflows: rows=%d k=%d", totalRows, k)
	}
	for i := range outs {
		outs[i] = tensor.New(tensor.F32, tensor.Shape{1, ns[i]})
	}

	row := x.Storage().F32()[:k]
	qmatmulParallelChunks(totalRows, k, func(lo, hi int) {
		for matrix := range weights {
			// Partition every matrix proportionally across the scheduler's
			// virtual row range. A contiguous concatenation would put the
			// final (often Q6_K) matrix on only the last workers and turn its
			// slower rows into a barrier tail. Proportional ranges remain
			// disjoint and cover [0, ns[matrix]) exactly.
			start := qmatmulTriplePartition(lo, totalRows, ns[matrix])
			end := qmatmulTriplePartition(hi, totalRows, ns[matrix])
			if start >= end {
				continue
			}
			dst := outs[matrix].Storage().F32()
			weight := weights[matrix]
			rb := rowBytes[matrix]
			switch qts[matrix] {
			case Q4_K:
				for ni := start; ni < end; ni++ {
					off := ni * rb
					dst[ni] = float32(dotQ4KRowFn(row, weight[off:off+rb], k))
				}
			case Q6_K:
				for ni := start; ni < end; ni++ {
					off := ni * rb
					dst[ni] = float32(dotQ6KRowFn(row, weight[off:off+rb], k))
				}
			}
		}
	})
	return outs, nil
}

func qmatmulTriplePartition(index, total, rows int) int {
	hi, lo := bits.Mul64(uint64(index), uint64(rows))
	q, _ := bits.Div64(hi, lo, uint64(total))
	return int(q)
}
