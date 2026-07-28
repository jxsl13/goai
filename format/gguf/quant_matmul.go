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
		// Register-block the OUTPUT (weight-row) loop by 4: m==1 so the single activation
		// row is shared across all n outputs — reuse each row element's load+f64-convert
		// across 4 weight rows (and the fixed-length block subslices drop the per-element
		// bounds check). This is the dual of the m>1 unroll-and-jam. Bit-exact: each output
		// keeps its own accumulator, wv=d·int8(q) is computed as before, ascending-k order
		// preserved. Scalar tail for n%4.
		ni := 0
		for ; ni+4 <= n; ni += 4 {
			rb0 := weight[(ni+0)*rowBytes:]
			rb1 := weight[(ni+1)*rowBytes:]
			rb2 := weight[(ni+2)*rowBytes:]
			rb3 := weight[(ni+3)*rowBytes:]
			var a0, a1, a2, a3 float64
			for b := 0; b*blockElems < k; b++ {
				o := b * 34
				d0 := f16ToF32(binary.LittleEndian.Uint16(rb0[o : o+2]))
				d1 := f16ToF32(binary.LittleEndian.Uint16(rb1[o : o+2]))
				d2 := f16ToF32(binary.LittleEndian.Uint16(rb2[o : o+2]))
				d3 := f16ToF32(binary.LittleEndian.Uint16(rb3[o : o+2]))
				q0 := rb0[o+2 : o+34]
				q1 := rb1[o+2 : o+34]
				q2 := rb2[o+2 : o+34]
				q3 := rb3[o+2 : o+34]
				rrow := row[b*blockElems : b*blockElems+blockElems]
				for i := 0; i < blockElems; i++ {
					xv := float64(rrow[i])
					a0 += xv * float64(d0*float32(int8(q0[i])))
					a1 += xv * float64(d1*float32(int8(q1[i])))
					a2 += xv * float64(d2*float32(int8(q2[i])))
					a3 += xv * float64(d3*float32(int8(q3[i])))
				}
			}
			outf[ni+0] = float32(a0)
			outf[ni+1] = float32(a1)
			outf[ni+2] = float32(a2)
			outf[ni+3] = float32(a3)
		}
		for ; ni < n; ni++ {
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
	// Every supported quant type dequantizes each weight row into ONE reused buffer
	// rather than a per-row tensor — the n-allocs-per-matmul anti-pattern is gone for all
	// of them (Q8_0/Q4_0/Q4_K/Q6_K landed first as the common formats; Q2_K/Q3_K/Q5_K —
	// aggressive quants — complete the set). Fill + dot are byte-for-byte the per-row form.
	scratch := make([]float32, k)
	for ni := range n {
		rowBits := weight[ni*rowBytes : (ni+1)*rowBytes]
		switch qt {
		case Q8_0:
			dequantQ8_0Into(scratch, rowBits)
		case Q4_0:
			dequantQ4_0Into(scratch, rowBits)
		case Q2_K:
			dequantQ2_KInto(scratch, rowBits)
		case Q3_K:
			dequantQ3_KInto(scratch, rowBits)
		case Q4_K:
			dequantQ4_KInto(scratch, rowBits)
		case Q5_K:
			dequantQ5_KInto(scratch, rowBits)
		case Q6_K:
			dequantQ6_KInto(scratch, rowBits)
		default:
			return nil, fmt.Errorf("gguf: QMatMul unsupported quant type %d", qt)
		}
		wf := scratch
		// Unroll-and-jam the activation-row (mi) loop by 4: the dequantized weight row wf is
		// invariant across mi, so 4 independent f64 accumulators break the single-acc FADD
		// dependency chain (latency-bound → throughput-bound) AND load+convert each wf[ki]
		// once for 4 rows. Bit-exact: each output keeps its OWN accumulator summing ascending
		// ki — no reassociation of any individual dot (the §V10 ascending-k order holds). The
		// dtype switch is hoisted out of the mi loop (loop-invariant). M1 decode is unaffected
		// (the tail handles m<4).
		switch {
		case xf32 != nil:
			mi := 0
			for ; mi+4 <= m; mi += 4 {
				r0 := xf32[(mi+0)*k : (mi+0)*k+k]
				r1 := xf32[(mi+1)*k : (mi+1)*k+k]
				r2 := xf32[(mi+2)*k : (mi+2)*k+k]
				r3 := xf32[(mi+3)*k : (mi+3)*k+k]
				var a0, a1, a2, a3 float64
				for ki, wv := range wf {
					w := float64(wv)
					a0 += float64(r0[ki]) * w
					a1 += float64(r1[ki]) * w
					a2 += float64(r2[ki]) * w
					a3 += float64(r3[ki]) * w
				}
				outf[(mi+0)*n+ni] = float32(a0)
				outf[(mi+1)*n+ni] = float32(a1)
				outf[(mi+2)*n+ni] = float32(a2)
				outf[(mi+3)*n+ni] = float32(a3)
			}
			for ; mi < m; mi++ {
				row := xf32[mi*k : mi*k+k]
				var acc float64
				for ki, wv := range wf {
					acc += float64(row[ki]) * float64(wv)
				}
				outf[mi*n+ni] = float32(acc)
			}
		case xf64 != nil:
			mi := 0
			for ; mi+4 <= m; mi += 4 {
				r0 := xf64[(mi+0)*k : (mi+0)*k+k]
				r1 := xf64[(mi+1)*k : (mi+1)*k+k]
				r2 := xf64[(mi+2)*k : (mi+2)*k+k]
				r3 := xf64[(mi+3)*k : (mi+3)*k+k]
				var a0, a1, a2, a3 float64
				for ki, wv := range wf {
					w := float64(wv)
					a0 += r0[ki] * w
					a1 += r1[ki] * w
					a2 += r2[ki] * w
					a3 += r3[ki] * w
				}
				outf[(mi+0)*n+ni] = float32(a0)
				outf[(mi+1)*n+ni] = float32(a1)
				outf[(mi+2)*n+ni] = float32(a2)
				outf[(mi+3)*n+ni] = float32(a3)
			}
			for ; mi < m; mi++ {
				row := xf64[mi*k : mi*k+k]
				var acc float64
				for ki, wv := range wf {
					acc += row[ki] * float64(wv)
				}
				outf[mi*n+ni] = float32(acc)
			}
		default:
			for mi := range m {
				var acc float64
				for ki := range k {
					acc += x.AtF64(mi, ki) * float64(wf[ki])
				}
				outf[mi*n+ni] = float32(acc)
			}
		}
	}
	return out, nil
}
