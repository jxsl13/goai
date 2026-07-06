package gguf

import (
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// QuantType identifies a supported quantized weight format for QMatMul.
type QuantType uint32

const (
	Q8_0 QuantType = tQ8_0
	Q4_0 QuantType = tQ4_0
)

// QMatMul computes y[M,N] = x[M,K] · dequant(W[N,K])ᵀ where W is stored quantized
// (row-major, K/32 blocks per row) — a quantized linear layer (weight [out,in]).
// The weight is dequantized ONE ROW at a time (not the whole matrix), so a
// quantized model runs without materializing full-precision weights — the point
// of quantized inference (§T39). Accumulation is f64 (§V10); the dequant per
// block is the ggml-verified path (§R19/§R21).
func QMatMul(x *tensor.Tensor, weight []byte, qt QuantType, n, k int) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != k {
		return nil, fmt.Errorf("gguf: QMatMul x must be [M,%d], got %v", k, x.Shape())
	}
	m := x.Shape()[0]
	rowBytes, err := byteSize(uint32(qt), k)
	if err != nil {
		return nil, err
	}
	if len(weight) != n*rowBytes {
		return nil, fmt.Errorf("gguf: QMatMul weight %d bytes != %d rows × %d", len(weight), n, rowBytes)
	}

	out := tensor.New(tensor.F32, tensor.Shape{m, n})
	for ni := range n {
		rowBits := weight[ni*rowBytes : (ni+1)*rowBytes]
		var wrow *tensor.Tensor
		switch qt {
		case Q8_0:
			wrow, err = dequantQ8_0(tensor.Shape{k}, rowBits)
		case Q4_0:
			wrow, err = dequantQ4_0(tensor.Shape{k}, rowBits)
		default:
			return nil, fmt.Errorf("gguf: QMatMul unsupported quant type %d", qt)
		}
		if err != nil {
			return nil, err
		}
		wf := wrow.Storage().F32()
		for mi := range m {
			var acc float64
			for ki := range k {
				acc += x.AtF64(mi, ki) * float64(wf[ki])
			}
			out.SetF64(acc, mi, ni)
		}
	}
	return out, nil
}
