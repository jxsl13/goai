package gguf

import (
	"encoding/binary"
	"fmt"
	"runtime"
	"sync"

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

// dotQ4KRowFn is dotQ4_KRow (scalar) by default; the amd64+simd asm kernel overrides
// it in init() with the 2.6x VPMOVZXBD/FMA row dot (tolerance-gated).
var dotQ4KRowFn = dotQ4_KRow

// q8FusedDecodeM1, when non-nil (amd64+simd build), computes the Q8_0 m==1 decode
// matmul with the SIMD dequant-dot kernel (tolerance-gated). Nil → scalar fused path.
var q8FusedDecodeM1 func(row []float32, weight []byte, n, k, rowBytes int, outf []float32)

// qmatmulParallelChunks runs body over disjoint output-row chunks of [0,n) across
// GOMAXPROCS. Each output row is an independent dequant-dot (disjoint outf[...ni], no
// cross-ni reduction), so chunking is bit-identical to the serial loop. Serial below a
// small total-work threshold.
// qmatmulGrain is the element-work one worker must be given before another is added. Measured
// on BenchmarkQuantLlamaGenerate500 against the fixed GOMAXPROCS split: 1<<15 is 552.9 to 517.6
// ms with user CPU 18.99 to 16.51 s and system 2.91 to 2.13 s — better on all three. A coarser
// 1<<16 saves more CPU (16.10 s user, 1.19 s system) and gives back wall clock at 536.1 ms, and
// 1<<17 and above collapse to serial at about 604 ms.
const qmatmulGrain = 1 << 15

func qmatmulParallelChunks(n, workPerRow int, body func(lo, hi int)) {
	nw := runtime.GOMAXPROCS(0)
	if nw > n {
		nw = n
	}
	// SIZE THE FAN-OUT TO THE WORK, do not just gate it on/off. Decode calls this thousands of
	// times per generation with one activation row, and splitting a small matmul twelve ways
	// costs more in wakeups than the split saves: a profile of a 500-token quantized Llama
	// generation spent 88%% of its samples in pthread_cond_signal and pthread_cond_wait and
	// 1.5%% in this kernel. One worker per qmatmulGrain of element-work, capped by GOMAXPROCS,
	// beats both the fixed twelve and no fan-out at all — on wall clock AND on CPU burned.
	if w := n * workPerRow / qmatmulGrain; w < nw {
		nw = w
	}
	if nw <= 1 || n*workPerRow < 1<<17 {
		body(0, n)
		return
	}
	csz := (n + nw - 1) / nw
	var wg sync.WaitGroup
	for c := 0; c < nw; c++ {
		lo := c * csz
		if lo >= n {
			break
		}
		hi := lo + csz
		if hi > n {
			hi = n
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			body(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}

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
		// Decode (m==1) row dots are independent per output row → chunk-parallel (nlo/nhi are
		// the chunk's output-row bounds; the inner lo/hi are the block's nibble halves).
		qmatmulParallelChunks(n, k, func(nlo, nhi int) {
			for ni := nlo; ni < nhi; ni++ {
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
		switch qt {
		case Q2_K:
			dot = dotQ2_KRow
		case Q3_K:
			dot = dotQ3_KRow
		case Q4_K:
			dot = dotQ4KRowFn
		case Q5_K:
			dot = dotQ5_KRow
		case Q6_K:
			dot = dotQ6_KRow
		}
		row := xf32[:k]
		// Decode (m==1) K-quant row dots are independent per output row → chunk-parallel.
		qmatmulParallelChunks(n, k, func(lo, hi int) {
			for ni := lo; ni < hi; ni++ {
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
	// qt is validated once here (loop-invariant), so the per-ni body below cannot error.
	switch qt {
	case Q8_0, Q4_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K:
	default:
		return nil, fmt.Errorf("gguf: QMatMul unsupported quant type %d", qt)
	}
	// process dequantizes weight row ni into its OWN scratch and writes the disjoint output
	// column outf[*n+ni] for all m activation rows. Rows ni are independent (per-ni scratch,
	// disjoint output column, no cross-ni reduction), so chunk-parallel across GOMAXPROCS is
	// byte-for-byte identical to the serial loop; the default dtype branch reads x read-only.
	dequantInto := func(scratch []float32, ni int) {
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
	}
	process := func(scratch []float32, ni int) {
		dequantInto(scratch, ni)
		wf := scratch
		// Unroll-and-jam the activation-row (mi) loop by 8 — swept, not argued: at 4, 6, 8 and 10
		// BenchmarkQuantMamba2Prefill_512 read 307.0, 247.4, 239.9 and 242.5 ms, so eight is the
		// optimum and ten is already past it. Decode is unaffected either way (m=1 takes the
		// tail), which the QuantLlamaGenerate500 cell confirms at 505.8 vs 504.4 ms.
		//
		// Unroll-and-jam the activation-row (mi) loop: the dequantized weight row wf is
		// invariant across mi, so 8 independent f64 accumulators break the single-acc FADD
		// dependency chain (latency-bound → throughput-bound) AND load+convert each wf[ki]
		// once for all of them. Bit-exact: each output keeps its OWN accumulator summing ascending
		// ki — no reassociation of any individual dot (the §V10 ascending-k order holds). The
		// dtype switch is hoisted out of the mi loop (loop-invariant). M1 decode is unaffected
		// (the tail handles m<8).
		//
		// THE f32->f64 CONVERSIONS ARE NOT A COST — MEASURED, AND THE ATTEMPT LOST. This loop
		// converts eight activation elements and one weight element per step, and it runs once
		// per OUTPUT COLUMN, so an f32 activation matrix is converted n times over. Pre-widening
		// it once into an m*k f64 buffer removes all of that and measured WORSE:
		// BenchmarkQuantMamba2Prefill_512 273.5 to 284.1 ms, +3.9%, with decode flat. The loop
		// is LOAD-bound, so doubling the activation bytes costs more than the conversions saved;
		// a widening instruction rides in the load pipeline and is close to free.
		//
		// The lever that follows from that, and is NOT taken here: blocking over the OUTPUT
		// COLUMN. Each step reads eight activation elements and one weight element to do eight
		// FMAs; dequantizing two weight rows per call and computing two outputs per activation
		// group would give sixteen FMAs for the same eight activation loads. It restructures the
		// per-column fan-out unit rather than the loop body, so it wants its own round against
		// the digest this file already carries.
		switch {
		case xf32 != nil:
			mi := 0
			for ; mi+8 <= m; mi += 8 {
				r0 := xf32[(mi+0)*k : (mi+0)*k+k]
				r1 := xf32[(mi+1)*k : (mi+1)*k+k]
				r2 := xf32[(mi+2)*k : (mi+2)*k+k]
				r3 := xf32[(mi+3)*k : (mi+3)*k+k]
				r4 := xf32[(mi+4)*k : (mi+4)*k+k]
				r5 := xf32[(mi+5)*k : (mi+5)*k+k]
				r6 := xf32[(mi+6)*k : (mi+6)*k+k]
				r7 := xf32[(mi+7)*k : (mi+7)*k+k]
				var a0, a1, a2, a3, a4, a5, a6, a7 float64
				for ki, wv := range wf {
					w := float64(wv)
					a0 += float64(r0[ki]) * w
					a1 += float64(r1[ki]) * w
					a2 += float64(r2[ki]) * w
					a3 += float64(r3[ki]) * w
					a4 += float64(r4[ki]) * w
					a5 += float64(r5[ki]) * w
					a6 += float64(r6[ki]) * w
					a7 += float64(r7[ki]) * w
				}
				outf[(mi+0)*n+ni] = float32(a0)
				outf[(mi+1)*n+ni] = float32(a1)
				outf[(mi+2)*n+ni] = float32(a2)
				outf[(mi+3)*n+ni] = float32(a3)
				outf[(mi+4)*n+ni] = float32(a4)
				outf[(mi+5)*n+ni] = float32(a5)
				outf[(mi+6)*n+ni] = float32(a6)
				outf[(mi+7)*n+ni] = float32(a7)
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
			for ; mi+8 <= m; mi += 8 {
				r0 := xf64[(mi+0)*k : (mi+0)*k+k]
				r1 := xf64[(mi+1)*k : (mi+1)*k+k]
				r2 := xf64[(mi+2)*k : (mi+2)*k+k]
				r3 := xf64[(mi+3)*k : (mi+3)*k+k]
				r4 := xf64[(mi+4)*k : (mi+4)*k+k]
				r5 := xf64[(mi+5)*k : (mi+5)*k+k]
				r6 := xf64[(mi+6)*k : (mi+6)*k+k]
				r7 := xf64[(mi+7)*k : (mi+7)*k+k]
				var a0, a1, a2, a3, a4, a5, a6, a7 float64
				for ki, wv := range wf {
					w := float64(wv)
					a0 += r0[ki] * w
					a1 += r1[ki] * w
					a2 += r2[ki] * w
					a3 += r3[ki] * w
					a4 += r4[ki] * w
					a5 += r5[ki] * w
					a6 += r6[ki] * w
					a7 += r7[ki] * w
				}
				outf[(mi+0)*n+ni] = float32(a0)
				outf[(mi+1)*n+ni] = float32(a1)
				outf[(mi+2)*n+ni] = float32(a2)
				outf[(mi+3)*n+ni] = float32(a3)
				outf[(mi+4)*n+ni] = float32(a4)
				outf[(mi+5)*n+ni] = float32(a5)
				outf[(mi+6)*n+ni] = float32(a6)
				outf[(mi+7)*n+ni] = float32(a7)
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
	// THREE OUTPUT COLUMNS PER PASS. The eight-row jam reads eight activation elements and
	// one weight element to do eight FMAs, and it runs once per output column — the
	// activation matrix is streamed n times and the loop is LOAD-bound, which is why
	// pre-widening it to f64 measured WORSE (see the note above). Three weight rows per
	// pass give twenty-four FMAs for the same eight activation loads.
	//
	// THREE BECAUSE IT WAS SWEPT: at 2, 3 and 4 columns BenchmarkQuantMamba2Prefill_512
	// read 237.2, 220.3 and 261.3 ms against 270.8 serial-per-column, so four already
	// spills — twenty-four accumulators fit and thirty-two do not. Passing the scratch
	// rows as NAMED parameters rather than a slice of slices is worth another 4%: the
	// swept forms used sc[c][ki] and measured 237.2 at two columns where the named form
	// measures 222.5.
	//
	// BIT-IDENTICAL: every output keeps its OWN accumulator summing ascending ki over the
	// same operands, exactly as three separate calls produced.
	process3 := func(sa, sb, scc []float32, ni int) {
		if xf32 == nil || m < 8 {
			process(sa, ni)
			process(sb, ni+1)
			process(scc, ni+2)
			return
		}
		dequantInto(sa, ni+0)
		dequantInto(sb, ni+1)
		dequantInto(scc, ni+2)
		mi := 0
		for ; mi+8 <= m; mi += 8 {
			r0 := xf32[(mi+0)*k : (mi+0)*k+k]
			r1 := xf32[(mi+1)*k : (mi+1)*k+k]
			r2 := xf32[(mi+2)*k : (mi+2)*k+k]
			r3 := xf32[(mi+3)*k : (mi+3)*k+k]
			r4 := xf32[(mi+4)*k : (mi+4)*k+k]
			r5 := xf32[(mi+5)*k : (mi+5)*k+k]
			r6 := xf32[(mi+6)*k : (mi+6)*k+k]
			r7 := xf32[(mi+7)*k : (mi+7)*k+k]
			var a0, a1, a2, a3, a4, a5, a6, a7 float64
			var b0, b1, b2, b3, b4, b5, b6, b7 float64
			var c0, c1, c2, c3, c4, c5, c6, c7 float64
			for ki := 0; ki < k; ki++ {
				w0, w1, w2 := float64(sa[ki]), float64(sb[ki]), float64(scc[ki])
				v0 := float64(r0[ki])
				a0 += v0 * w0
				b0 += v0 * w1
				c0 += v0 * w2
				v1 := float64(r1[ki])
				a1 += v1 * w0
				b1 += v1 * w1
				c1 += v1 * w2
				v2 := float64(r2[ki])
				a2 += v2 * w0
				b2 += v2 * w1
				c2 += v2 * w2
				v3 := float64(r3[ki])
				a3 += v3 * w0
				b3 += v3 * w1
				c3 += v3 * w2
				v4 := float64(r4[ki])
				a4 += v4 * w0
				b4 += v4 * w1
				c4 += v4 * w2
				v5 := float64(r5[ki])
				a5 += v5 * w0
				b5 += v5 * w1
				c5 += v5 * w2
				v6 := float64(r6[ki])
				a6 += v6 * w0
				b6 += v6 * w1
				c6 += v6 * w2
				v7 := float64(r7[ki])
				a7 += v7 * w0
				b7 += v7 * w1
				c7 += v7 * w2
			}
			outf[(mi+0)*n+ni+0] = float32(a0)
			outf[(mi+0)*n+ni+1] = float32(b0)
			outf[(mi+0)*n+ni+2] = float32(c0)
			outf[(mi+1)*n+ni+0] = float32(a1)
			outf[(mi+1)*n+ni+1] = float32(b1)
			outf[(mi+1)*n+ni+2] = float32(c1)
			outf[(mi+2)*n+ni+0] = float32(a2)
			outf[(mi+2)*n+ni+1] = float32(b2)
			outf[(mi+2)*n+ni+2] = float32(c2)
			outf[(mi+3)*n+ni+0] = float32(a3)
			outf[(mi+3)*n+ni+1] = float32(b3)
			outf[(mi+3)*n+ni+2] = float32(c3)
			outf[(mi+4)*n+ni+0] = float32(a4)
			outf[(mi+4)*n+ni+1] = float32(b4)
			outf[(mi+4)*n+ni+2] = float32(c4)
			outf[(mi+5)*n+ni+0] = float32(a5)
			outf[(mi+5)*n+ni+1] = float32(b5)
			outf[(mi+5)*n+ni+2] = float32(c5)
			outf[(mi+6)*n+ni+0] = float32(a6)
			outf[(mi+6)*n+ni+1] = float32(b6)
			outf[(mi+6)*n+ni+2] = float32(c6)
			outf[(mi+7)*n+ni+0] = float32(a7)
			outf[(mi+7)*n+ni+1] = float32(b7)
			outf[(mi+7)*n+ni+2] = float32(c7)
		}
		for ; mi < m; mi++ {
			row := xf32[mi*k : mi*k+k]
			var t0, t1, t2 float64
			for ki := 0; ki < k; ki++ {
				v := float64(row[ki])
				t0 += v * float64(sa[ki])
				t1 += v * float64(sb[ki])
				t2 += v * float64(scc[ki])
			}
			outf[mi*n+ni] = float32(t0)
			outf[mi*n+ni+1] = float32(t1)
			outf[mi*n+ni+2] = float32(t2)
		}
	}
	nw := runtime.GOMAXPROCS(0)
	if nw > n {
		nw = n
	}
	if nw <= 1 || m*n*k < 1<<20 {
		scratch := make([]float32, k)
		sb := make([]float32, k)
		sc := make([]float32, k)
		ni := 0
		for ; ni+3 <= n; ni += 3 {
			process3(scratch, sb, sc, ni)
		}
		for ; ni < n; ni++ {
			process(scratch, ni)
		}
		return out, nil
	}
	csz := (n + nw - 1) / nw
	var wg sync.WaitGroup
	for c := 0; c < nw; c++ {
		lo := c * csz
		if lo >= n {
			break
		}
		hi := lo + csz
		if hi > n {
			hi = n
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			scratch := make([]float32, k)
			sb := make([]float32, k)
			sc := make([]float32, k)
			ni := lo
			for ; ni+3 <= hi; ni += 3 {
				process3(scratch, sb, sc, ni)
			}
			for ; ni < hi; ni++ {
				process(scratch, ni)
			}
		}(lo, hi)
	}
	wg.Wait()
	return out, nil
}
