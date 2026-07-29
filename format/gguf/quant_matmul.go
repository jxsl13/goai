package gguf

import (
	"encoding/binary"
	"fmt"

	"github.com/jxsl13/goai/internal/parallel"
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
// dotQ4KRowFn is dotQ4_KRow (scalar) by default; the amd64+simd asm kernel overrides
// it in init() with the 2.6x VPMOVZXBD/FMA row dot (tolerance-gated).
var dotQ4KRowFn = dotQ4_KRow

// dotQ4KRowSIMD reports whether the asm kernel above actually replaced the scalar dot. It
// cannot be derived by comparing function values — Go does not permit that — so the asm
// init sets it. Where the override is absent (every non-amd64 build, and amd64 without the
// simd experiment) Q4_K should take the 4-row blocked scalar path like the other K-quants.
var dotQ4KRowSIMD bool

// q8FusedDecodeM1, when non-nil (amd64+simd build), computes the Q8_0 m==1 decode
// matmul with the SIMD dequant-dot kernel (tolerance-gated). Nil → scalar fused path.
var q8FusedDecodeM1 func(row []float32, weight []byte, n, k, rowBytes int, outf []float32)

// qmatDecodeParThreshold is the decode path's own crossover — higher than the general
// path's because a decode chunk carries one dot per output row, not m of them.
const qmatDecodeParThreshold = 1 << 17

// qmatmulParallelChunks splits the n output rows of the m==1 decode paths across the
// shared bounded pool. Spawning GOMAXPROCS goroutines per call instead would multiply
// against any caller that is ALREADY parallel — the nlp Mamba2 mixers now call QMatMul
// from inside their own parallel regions, so a per-call spawn would put GOMAXPROCS²
// goroutines on a 12-core host. Same partition, same bit-identical row independence as
// parallelRows; only the execution mechanism is shared (ADR-01KYMWJ76AFA2).
func qmatmulParallelChunks(n, workPerRow int, body func(lo, hi int)) {
	if n*workPerRow < qmatDecodeParThreshold {
		body(0, n)
		return
	}
	parallel.Rows(n, body)
}

// QMatMul multiplies x by a quantized weight matrix held in its packed GGUF form,
// dequantizing on the fly rather than materializing the [n,k] f32 matrix. x is [m,k], the
// result is [m,n], and weight carries n rows of k quantized values in qt's block layout.
//
// Single-row inputs (m == 1, the decode step) take fused per-type kernels that dequantize
// straight into the dot product; wider inputs use the blocked path. Both parallelize across
// output rows.
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
		// Decode (m==1) is chunk-parallel over the independent output rows. The SIMD kernel
		// is called per weight-offset/output-slice chunk; the scalar 4-way path runs its
		// register-block + tail within each chunk. Each output row is an independent dot
		// (own accumulator, ascending-k order, disjoint outf) → bit-identical.
		qmatmulParallelChunks(n, k, func(lo, hi int) {
			if q8FusedDecodeM1 != nil {
				q8FusedDecodeM1(row, weight[lo*rowBytes:], hi-lo, k, rowBytes, outf[lo:hi])
				return
			}
			ni := lo
			for ; ni+4 <= hi; ni += 4 {
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
			for ; ni < hi; ni++ {
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
		})
		return out, nil
	}

	// The same fusion for Q4_0, which had none and so paid a full materialize-and-reread
	// of the weight row per output — measurably SLOWER at decode than Q8_0 despite half
	// the memory traffic, which is backwards for the smaller format and was the tell.
	//
	// Q4_0 packs two elements per byte, and NOT adjacently: the low nibbles are elements
	// 0..15 of the block and the high nibbles are 16..31. Dotting them in byte order
	// would pair every weight with the wrong activation, so the two nibble halves are
	// walked as two sequential passes, which also keeps the ascending-k order the
	// general path uses.
	if qt == Q4_0 && m == 1 && xf32 != nil {
		row := xf32[:k]
		qmatmulParallelChunks(n, k, func(nlo, nhi int) {
			// Register-block the weight-row loop by 4, as the Q8_0 path above does: m==1
			// means ONE activation row shared by every output, so each element's load and
			// float64 convert is reused across 4 rows instead of repeated per row. Blocks
			// within [lo,hi) — the pool owns the partition, and blocking across the whole
			// n would write outside this chunk. Bit-exact: each output keeps its own
			// accumulator and still sees block b's 16 low-nibble terms then its 16
			// high-nibble terms, ascending. Scalar tail for the chunk's n%4.
			ni := nlo
			for ; ni+4 <= nhi; ni += 4 {
				rb0 := weight[(ni+0)*rowBytes:]
				rb1 := weight[(ni+1)*rowBytes:]
				rb2 := weight[(ni+2)*rowBytes:]
				rb3 := weight[(ni+3)*rowBytes:]
				var a0, a1, a2, a3 float64
				for b := 0; b*blockElems < k; b++ {
					o := b * 18
					d0 := f16ToF32(binary.LittleEndian.Uint16(rb0[o : o+2]))
					d1 := f16ToF32(binary.LittleEndian.Uint16(rb1[o : o+2]))
					d2 := f16ToF32(binary.LittleEndian.Uint16(rb2[o : o+2]))
					d3 := f16ToF32(binary.LittleEndian.Uint16(rb3[o : o+2]))
					q0, q1 := rb0[o+2:o+18], rb1[o+2:o+18]
					q2, q3 := rb2[o+2:o+18], rb3[o+2:o+18]
					base := b * blockElems
					xlo, xhi := row[base:base+16], row[base+16:base+32]
					for i := range 16 {
						xv := float64(xlo[i])
						a0 += xv * float64(d0*float32(int(q0[i]&0x0F)-8))
						a1 += xv * float64(d1*float32(int(q1[i]&0x0F)-8))
						a2 += xv * float64(d2*float32(int(q2[i]&0x0F)-8))
						a3 += xv * float64(d3*float32(int(q3[i]&0x0F)-8))
					}
					for i := range 16 {
						xv := float64(xhi[i])
						a0 += xv * float64(d0*float32(int(q0[i]>>4)-8))
						a1 += xv * float64(d1*float32(int(q1[i]>>4)-8))
						a2 += xv * float64(d2*float32(int(q2[i]>>4)-8))
						a3 += xv * float64(d3*float32(int(q3[i]>>4)-8))
					}
				}
				outf[ni+0], outf[ni+1] = float32(a0), float32(a1)
				outf[ni+2], outf[ni+3] = float32(a2), float32(a3)
			}
			for ; ni < nhi; ni++ {
				rowBits := weight[ni*rowBytes : (ni+1)*rowBytes]
				var acc float64
				for b := 0; b*blockElems < k; b++ {
					blk := rowBits[b*18 : b*18+18]
					d := f16ToF32(binary.LittleEndian.Uint16(blk))
					qs := blk[2:18]
					base := b * blockElems
					lo, hi := row[base:base+16], row[base+16:base+32]
					for i, q := range qs {
						acc += float64(lo[i]) * float64(d*float32(int(q&0x0F)-8))
					}
					for i, q := range qs {
						acc += float64(hi[i]) * float64(d*float32(int(q>>4)-8))
					}
				}
				outf[ni] = float32(acc)
			}
		})
		return out, nil
	}

	// The K-quants carry the same gap, and they are llama.cpp's common deployment
	// formats. Their per-row dot lives in a helper each: the superblock unpacking is
	// long enough that inlining four copies of it here would bury the dispatch.
	// The aggressive quants (Q2_K/Q3_K/Q5_K) were measured before being fused rather
	// than assumed to behave like the deployment formats — they carried the same tell,
	// 107 allocs per decode step against the fused paths' 102, and were the three
	// slowest of the seven types.
	//
	// The covered types are named in the GUARD rather than left implicit in the switch
	// below it. Spelling it the other way — one `if m == 1` around a dispatch switch —
	// hides the coverage from PS6003, which reads guards and cannot follow a function
	// value into a later return. That would have made this function report a gap it no
	// longer has, and silenced the check for whoever adds the eighth quant type.
	if m == 1 && xf32 != nil &&
		(qt == Q2_K || qt == Q3_K || qt == Q4_K || qt == Q5_K || qt == Q6_K) {
		var dot func([]float32, []byte, int) float64
		// dot4 computes FOUR output rows at once so a single dequantized activation load
		// feeds four accumulators — register blocking, orthogonal to the chunk-parallelism
		// below (that splits ACROSS row groups, this works WITHIN one). Bit-identical:
		// blocking the OUTPUT rows leaves each row's accumulation order untouched.
		var dot4 func(row []float32, r0, r1, r2, r3 []byte, k int) (float64, float64, float64, float64)
		switch qt {
		case Q2_K:
			dot, dot4 = dotQ2_KRow, dotQ2_K4Rows
		case Q3_K:
			dot, dot4 = dotQ3_KRow, dotQ3_K4Rows
		case Q4_K:
			// Blocked only when the asm kernel is NOT in play. With the override active
			// the asm row dot is expected to beat 4-row scalar blocking; without it Q4_K
			// was the only K-quant left unblocked, and measured 2525 MB/s against
			// 7185-7585 for its blocked peers on this host.
			dot = dotQ4KRowFn
			if !dotQ4KRowSIMD {
				dot4 = dotQ4_K4Rows
			}
		case Q5_K:
			dot, dot4 = dotQ5_KRow, dotQ5_K4Rows
		case Q6_K:
			dot, dot4 = dotQ6_KRow, dotQ6_K4Rows
		}
		row := xf32[:k]
		// Decode (m==1) K-quant row dots are independent per output row → chunk-parallel.
		qmatmulParallelChunks(n, k, func(lo, hi int) {
			ni := lo
			if dot4 != nil { // blocks within [lo,hi): the pool owns the partition
				for ; ni+4 <= hi; ni += 4 {
					a0, a1, a2, a3 := dot4(row,
						weight[(ni+0)*rowBytes:], weight[(ni+1)*rowBytes:],
						weight[(ni+2)*rowBytes:], weight[(ni+3)*rowBytes:], k)
					outf[ni+0], outf[ni+1] = float32(a0), float32(a1)
					outf[ni+2], outf[ni+3] = float32(a2), float32(a3)
				}
			}
			for ; ni < hi; ni++ {
				outf[ni] = float32(dot(row, weight[ni*rowBytes:(ni+1)*rowBytes], k))
			}
		})
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
	//
	// PARALLEL over the weight rows. Prefill measured NO speedup at all across
	// GOMAXPROCS 1..12 (22.1ms vs 22.6ms) because this loop was the serial spine of it,
	// and a profile put barely one core's worth of samples on a 12-core host. Each ni
	// produces its own column of outputs from its own weight row, so the rows are
	// independent; the only shared mutable state was the dequant scratch, which now
	// belongs to the chunk rather than the call.
	//
	// The unsupported-type rejection is hoisted ABOVE the loop: it must stay an error
	// return, and a chunk body running on a pool worker has nowhere to return one to.
	// Checking it once up front is also strictly cheaper than once per row.
	switch qt {
	case Q8_0, Q4_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K:
	default:
		return nil, fmt.Errorf("gguf: QMatMul unsupported quant type %d", qt)
	}
	parallelRows2D(n, m*k, func(lo, hi int) {
		scratch := make([]float32, k)
		for ni := lo; ni < hi; ni++ {
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
	})
	return out, nil
}

// qmatParThreshold is the total work (output rows x k) below which splitting the row
// loop costs more than it saves. Taken from backend/cpu's parThreshold, the measured
// M-series crossover; at this model scale a tied head gemv falls under it and stays
// serial while the projections cross it.
const qmatParThreshold = 1 << 15

// parallelRows splits the n output rows across the shared bounded pool when there is
// enough work, and runs them inline otherwise.
//
// BIT-IDENTICAL BY CONSTRUCTION, not by measurement: with m==1 every output row is an
// independent dot writing its own index, reading a shared read-only activation row and
// a disjoint slice of the weight bytes. No accumulation crosses a row, so partitioning
// the range cannot change any value — only the order in which distinct destinations are
// written. Per ADR-01KYMWJ76AFA2 this routes through internal/parallel's bounded pool
// rather than spawning per call, so a caller already serving requests concurrently does
// not multiply the process's goroutine count, and nested calls degrade to inline.
func parallelRows(n, k int, body func(lo, hi int)) {
	if n*k < qmatParThreshold {
		body(0, n)
		return
	}
	parallel.Rows(n, body)
}

// parallelRows2D splits the n weight rows of the general (m>1) path across the shared
// bounded pool. work is the per-row cost (m*k) so the threshold compares total work, the
// same quantity qmatParThreshold is calibrated against.
//
// Each chunk allocates its own dequant scratch — that buffer was the ONLY shared mutable
// state in the loop, and per-chunk it costs a handful of allocations against a matmul
// that already dominates prefill. Every output element outf[mi*n+ni] is written by
// exactly one chunk, so the partition changes no value.
func parallelRows2D(n, work int, body func(lo, hi int)) {
	if n*work < qmatParThreshold {
		body(0, n)
		return
	}
	parallel.Rows(n, body)
}
