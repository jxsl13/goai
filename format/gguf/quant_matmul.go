package gguf

import (
	"encoding/binary"
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// QuantType identifies a supported quantized weight format for QMatMul.
type QuantType uint32

const (
	Q8_0    QuantType = tQ8_0    // 8-bit: f16 block scale + 32 int8 quants
	Q4_0    QuantType = tQ4_0    // 4-bit: f16 block scale + 32 nibbles, offset −8
	Q2_K    QuantType = tQ2_K    // 2-bit k-quant: asymmetric affine 256-element super-block (§R104)
	Q3_K    QuantType = tQ3_K    // 3-bit k-quant: symmetric 256-element super-block (§R103)
	Q4_K    QuantType = tQ4_K    // 4-bit k-quant: 256-element affine super-block (§R100)
	Q5_K    QuantType = tQ5_K    // 5-bit k-quant: affine super-block + high-bit plane (§R102)
	Q6_K    QuantType = tQ6_K    // 6-bit k-quant: 256-element super-block (§R99)
	IQ2_XXS QuantType = tIQ2_XXS // 2.06-bit i-quant: E8-lattice grid codebook, READ path (§T554)
	IQ2_XS  QuantType = tIQ2_XS  // 2.31-bit i-quant: 512-entry grid + explicit 4-bit scales, READ path (§T554)
	IQ3_XXS QuantType = tIQ3_XXS // 3.06-bit i-quant: 256×4 grid over an 8-value codebook, READ path (§T554)
	IQ4_NL  QuantType = tIQ4_NL  // 4-bit i-quant: nonlinear 16-value codebook, 32-element blocks, READ path (§T554)
	IQ4_XS  QuantType = tIQ4_XS  // 4-bit i-quant: nonlinear codebook + 6-bit sub-scales, 256-element super-block, READ path (§T554)
	IQ3_S   QuantType = tIQ3_S   // 3.44-bit i-quant: 512×4 odd-value grid, 9-bit indices, direct signs, READ path (§T554)
	IQ2_S   QuantType = tIQ2_S   // 2.5-bit i-quant: 1024×8 grid, 10-bit indices, direct signs, READ path (§T554)
	IQ1_S   QuantType = tIQ1_S   // 1.56-bit ternary i-quant: 2048×8 {−1,0,+1} grid + ±δ, READ path (§T554)
	IQ1_M   QuantType = tIQ1_M   // 1.75-bit ternary i-quant: split-f16 super-scale, READ path (§T554)
	MXFP4   QuantType = tMXFP4   // OCP microscaling FP4 (gpt-oss): E2M1 elements + E8M0 block scale (§T555)
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

	// Read x through contiguous storage once instead of the per-element AtF64
	// dispatch in the K-loop (§base-perf: the AtF64 anti-pattern; measured
	// 2.4–3.3× depending on M, docs/perf-notes-lowlevel.md).
	xc := x.Contiguous()
	var xf32 []float32
	var xf64 []float64
	switch xc.Dtype() {
	case tensor.F32:
		xf32 = xc.Storage().F32()
	case tensor.F64:
		xf64 = xc.Storage().F64()
	}

	out := tensor.New(tensor.F32, tensor.Shape{m, n})
	outf := out.Storage().F32()

	// Fused single-token (decode) path for Q8_0: the general path below dequantizes
	// every weight row into a freshly-allocated [k] tensor before dotting it — n such
	// allocations per matmul, which dominate decode (dequant + GC churn measured at
	// ~48% + 6.76 GB/500 tokens). With m==1 there is exactly one activation row, so
	// fold the per-block dequant (wv = d·int8(q)) straight into the dot product: no
	// per-row allocation, one pass over the quantized bytes. The scalar wv, the
	// ascending-k accumulation order and the float64 accumulator are identical to the
	// general path, so the result is bit-for-bit unchanged. (m>1 keeps the general
	// path — fusing there would re-dequantize the row for every activation row.)
	if qt == Q8_0 && m == 1 && xf32 != nil {
		row := xf32[:k]
		for ni := range n {
			rowBits := weight[ni*rowBytes : (ni+1)*rowBytes]
			var acc float64
			for b := 0; b*blockElems < k; b++ {
				blk := rowBits[b*34 : b*34+34]
				d := f16ToF32(binary.LittleEndian.Uint16(blk))
				q := blk[2:34]
				base := b * blockElems
				for i := 0; i < blockElems; i++ {
					wv := d * float32(int8(q[i]))
					acc += float64(row[base+i]) * float64(wv)
				}
			}
			outf[ni] = float32(acc)
		}
		return out, nil
	}

	// Reused row buffer for the quant types with a fill-into-slice variant (Q4_K/Q6_K —
	// llama.cpp's common deployment formats): dequant each weight row into one buffer
	// rather than allocating a [k] tensor per row, the same n-allocs-per-matmul cost the
	// Q8_0 decode path above avoids. The fill and the dot are byte-for-byte identical to
	// the per-row-tensor path, and this covers prefill (m>1) too. Other types keep the
	// per-row path below.
	var scratch []float32
	if qt == Q4_K || qt == Q6_K {
		scratch = make([]float32, k)
	}
	for ni := range n {
		rowBits := weight[ni*rowBytes : (ni+1)*rowBytes]
		var wf []float32
		switch qt {
		case Q4_K:
			dequantQ4_KInto(scratch, rowBits)
			wf = scratch
		case Q6_K:
			dequantQ6_KInto(scratch, rowBits)
			wf = scratch
		default:
			var wrow *tensor.Tensor
			switch qt {
			case Q8_0:
				wrow, err = dequantQ8_0(tensor.Shape{k}, rowBits)
			case Q4_0:
				wrow, err = dequantQ4_0(tensor.Shape{k}, rowBits)
			case Q2_K:
				wrow, err = dequantQ2_K(tensor.Shape{k}, rowBits)
			case Q3_K:
				wrow, err = dequantQ3_K(tensor.Shape{k}, rowBits)
			case Q5_K:
				wrow, err = dequantQ5_K(tensor.Shape{k}, rowBits)
			default:
				return nil, fmt.Errorf("gguf: QMatMul unsupported quant type %d", qt)
			}
			if err != nil {
				return nil, err
			}
			wf = wrow.Storage().F32()[:k]
		}
		for mi := range m {
			var acc float64
			switch {
			case xf32 != nil:
				row := xf32[mi*k : mi*k+k]
				for ki, wv := range wf {
					acc += float64(row[ki]) * float64(wv)
				}
			case xf64 != nil:
				row := xf64[mi*k : mi*k+k]
				for ki, wv := range wf {
					acc += row[ki] * float64(wv)
				}
			default:
				for ki := range k {
					acc += x.AtF64(mi, ki) * float64(wf[ki])
				}
			}
			outf[mi*n+ni] = float32(acc)
		}
	}
	return out, nil
}
