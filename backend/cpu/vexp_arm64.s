//go:build goexperiment.simd

#include "textflag.h"

// func vexpQuadsNeonF32(p *float32, quads int, m float32) float32
//
// 4-wide f32 exp for the MHA softmax (vexp.go): p[i] = exp(p[i]-m) in place
// for i in [0, 4*quads), returns the sum of the results. Cephes expf range
// reduction per lane — clamp to [-87.336, 88], n = FRINTN(x·log2e),
// r = x − n·ln2hi − n·ln2lo, degree-5 Horner poly in r, scaled by 2^n via
// integer exponent-bit insertion ((n+127)<<23 reinterpreted as f32). The main
// loop processes TWO quads per pass with disjoint registers so the two
// serial FMLA poly chains overlap (single-chain latency ~halved); odd quad
// counts run one single-quad pass. The row sum accumulates 4-lane f32 in two
// registers (one per chain) and reduces once at the end — reassociated vs the
// scalar path's serial f64 sum, inside the vexp path's tolerance budget
// (vexp.go). The Go assembler has no mnemonics for most vector FP ops
// (FSUB/FMUL/FMAX/FMIN/FRINTN/FCVTZS/ORR/FADDP), so those are WORD-encoded
// like gemm_neon_arm64.s; encodings generated + cross-checked by disassembly
// (go tool objdump decodes every WORD to the intended instruction).
//
// Register map:
//   R0 load ptr, R1 store ptr, R2 quads, R3 consts, R4 pairs, R5 odd
//   V0 m (broadcast), V29/V30 sum accumulators
//   V16..V28 constants: log2e ln2hi ln2lo p0..p5 one lo hi 127
//   quad A: V1 x, V2 n, V3 r, V4/V5 poly ping-pong, V6 r²
//   quad B: V8 x, V9 n, V10 r, V11/V12 poly ping-pong, V13 r²

DATA vexpConsts<>+0(SB)/4, $0x3FB8AA3B   // log2e
DATA vexpConsts<>+4(SB)/4, $0x3FB8AA3B   // log2e
DATA vexpConsts<>+8(SB)/4, $0x3FB8AA3B   // log2e
DATA vexpConsts<>+12(SB)/4, $0x3FB8AA3B  // log2e
DATA vexpConsts<>+16(SB)/4, $0x3F318000  // ln2hi = 0.693359375
DATA vexpConsts<>+20(SB)/4, $0x3F318000  // ln2hi
DATA vexpConsts<>+24(SB)/4, $0x3F318000  // ln2hi
DATA vexpConsts<>+28(SB)/4, $0x3F318000  // ln2hi
DATA vexpConsts<>+32(SB)/4, $0xB95E8083  // ln2lo = -2.12194440e-4
DATA vexpConsts<>+36(SB)/4, $0xB95E8083  // ln2lo
DATA vexpConsts<>+40(SB)/4, $0xB95E8083  // ln2lo
DATA vexpConsts<>+44(SB)/4, $0xB95E8083  // ln2lo
DATA vexpConsts<>+48(SB)/4, $0x39506967  // p0 = 1.9875691500e-4
DATA vexpConsts<>+52(SB)/4, $0x39506967  // p0
DATA vexpConsts<>+56(SB)/4, $0x39506967  // p0
DATA vexpConsts<>+60(SB)/4, $0x39506967  // p0
DATA vexpConsts<>+64(SB)/4, $0x3AB743CE  // p1 = 1.3981999507e-3
DATA vexpConsts<>+68(SB)/4, $0x3AB743CE  // p1
DATA vexpConsts<>+72(SB)/4, $0x3AB743CE  // p1
DATA vexpConsts<>+76(SB)/4, $0x3AB743CE  // p1
DATA vexpConsts<>+80(SB)/4, $0x3C088908  // p2 = 8.3334519073e-3
DATA vexpConsts<>+84(SB)/4, $0x3C088908  // p2
DATA vexpConsts<>+88(SB)/4, $0x3C088908  // p2
DATA vexpConsts<>+92(SB)/4, $0x3C088908  // p2
DATA vexpConsts<>+96(SB)/4, $0x3D2AA9C1  // p3 = 4.1665795894e-2
DATA vexpConsts<>+100(SB)/4, $0x3D2AA9C1 // p3
DATA vexpConsts<>+104(SB)/4, $0x3D2AA9C1 // p3
DATA vexpConsts<>+108(SB)/4, $0x3D2AA9C1 // p3
DATA vexpConsts<>+112(SB)/4, $0x3E2AAAAA // p4 = 1.6666665459e-1
DATA vexpConsts<>+116(SB)/4, $0x3E2AAAAA // p4
DATA vexpConsts<>+120(SB)/4, $0x3E2AAAAA // p4
DATA vexpConsts<>+124(SB)/4, $0x3E2AAAAA // p4
DATA vexpConsts<>+128(SB)/4, $0x3F000000 // p5 = 0.5
DATA vexpConsts<>+132(SB)/4, $0x3F000000 // p5
DATA vexpConsts<>+136(SB)/4, $0x3F000000 // p5
DATA vexpConsts<>+140(SB)/4, $0x3F000000 // p5
DATA vexpConsts<>+144(SB)/4, $0x3F800000 // one
DATA vexpConsts<>+148(SB)/4, $0x3F800000 // one
DATA vexpConsts<>+152(SB)/4, $0x3F800000 // one
DATA vexpConsts<>+156(SB)/4, $0x3F800000 // one
DATA vexpConsts<>+160(SB)/4, $0xC2AEAC4F // lo clamp = -87.33654
DATA vexpConsts<>+164(SB)/4, $0xC2AEAC4F // lo
DATA vexpConsts<>+168(SB)/4, $0xC2AEAC4F // lo
DATA vexpConsts<>+172(SB)/4, $0xC2AEAC4F // lo
DATA vexpConsts<>+176(SB)/4, $0x42B00000 // hi clamp = 88.0
DATA vexpConsts<>+180(SB)/4, $0x42B00000 // hi
DATA vexpConsts<>+184(SB)/4, $0x42B00000 // hi
DATA vexpConsts<>+188(SB)/4, $0x42B00000 // hi
DATA vexpConsts<>+192(SB)/4, $0x0000007F // int32 127
DATA vexpConsts<>+196(SB)/4, $0x0000007F // int32 127
DATA vexpConsts<>+200(SB)/4, $0x0000007F // int32 127
DATA vexpConsts<>+204(SB)/4, $0x0000007F // int32 127
// Tail entries: the FULL-DOMAIN standalone exp only (vexpFullQuadsNeonF32
// below, §T666) — the softmax kernel's init loads stop at +208.
DATA vexpConsts<>+208(SB)/4, $0x42B20000 // hiU = 89.0 (unary hi clamp, past the Inf cutoff)
DATA vexpConsts<>+212(SB)/4, $0x42B20000 // hiU
DATA vexpConsts<>+216(SB)/4, $0x42B20000 // hiU
DATA vexpConsts<>+220(SB)/4, $0x42B20000 // hiU
DATA vexpConsts<>+224(SB)/4, $0x42B17217 // infCut = 88.72283172607422 (largest finite-exp f32)
DATA vexpConsts<>+228(SB)/4, $0x42B17217 // infCut
DATA vexpConsts<>+232(SB)/4, $0x42B17217 // infCut
DATA vexpConsts<>+236(SB)/4, $0x42B17217 // infCut
DATA vexpConsts<>+240(SB)/4, $0x7F7FFFFF // maxFinite f32
DATA vexpConsts<>+244(SB)/4, $0x7F7FFFFF // maxFinite
DATA vexpConsts<>+248(SB)/4, $0x7F7FFFFF // maxFinite
DATA vexpConsts<>+252(SB)/4, $0x7F7FFFFF // maxFinite
DATA vexpConsts<>+256(SB)/4, $0x7F800000 // +Inf
DATA vexpConsts<>+260(SB)/4, $0x7F800000 // +Inf
DATA vexpConsts<>+264(SB)/4, $0x7F800000 // +Inf
DATA vexpConsts<>+268(SB)/4, $0x7F800000 // +Inf
GLOBL vexpConsts<>(SB), RODATA|NOPTR, $272

TEXT ·vexpQuadsNeonF32(SB), NOSPLIT, $0-28
	MOVD  p+0(FP), R0
	MOVD  quads+8(FP), R2
	FMOVS m+16(FP), F0
	VDUP  V0.S[0], V0.S4
	MOVD  R0, R1
	MOVD  $vexpConsts<>(SB), R3
	VLD1.P 64(R3), [V16.S4, V17.S4, V18.S4, V19.S4]
	VLD1.P 64(R3), [V20.S4, V21.S4, V22.S4, V23.S4]
	VLD1.P 64(R3), [V24.S4, V25.S4, V26.S4, V27.S4]
	VLD1  (R3), [V28.S4]
	VMOVI $0, V29.B16
	VMOVI $0, V30.B16
	LSR   $1, R2, R4 // pairs of quads
	AND   $1, R2, R5 // odd quad
	CBZ   R4, odd

loop2:
	VLD1.P 16(R0), [V1.S4]
	VLD1.P 16(R0), [V8.S4]

	// clamp(x-m) and z = x·log2e, n = rint(z), both quads.
	WORD $0x4EA0D421 // FSUB V1.4S, V1.4S, V0.4S    (A: x -= m)
	WORD $0x4EA0D508 // FSUB V8.4S, V8.4S, V0.4S    (B: x -= m)
	WORD $0x4E3AF421 // FMAX V1.4S, V1.4S, V26.4S   (A: clamp lo)
	WORD $0x4E3AF508 // FMAX V8.4S, V8.4S, V26.4S   (B: clamp lo)
	WORD $0x4EBBF421 // FMIN V1.4S, V1.4S, V27.4S   (A: clamp hi)
	WORD $0x4EBBF508 // FMIN V8.4S, V8.4S, V27.4S   (B: clamp hi)
	WORD $0x6E30DC22 // FMUL V2.4S, V1.4S, V16.4S   (A: z = x·log2e)
	WORD $0x6E30DD09 // FMUL V9.4S, V8.4S, V16.4S   (B: z = x·log2e)
	WORD $0x4E218842 // FRINTN V2.4S, V2.4S         (A: n = rint(z))
	WORD $0x4E218929 // FRINTN V9.4S, V9.4S         (B: n = rint(z))

	// r = x − n·ln2hi − n·ln2lo
	WORD  $0x4EA11C23 // ORR V3.16B, V1.16B, V1.16B  (A: r = x)
	WORD  $0x4EA81D0A // ORR V10.16B, V8.16B, V8.16B (B: r = x)
	VFMLS V2.S4, V17.S4, V3.S4  // A: r -= n·ln2hi
	VFMLS V9.S4, V17.S4, V10.S4 // B: r -= n·ln2hi
	VFMLS V2.S4, V18.S4, V3.S4  // A: r -= n·ln2lo
	VFMLS V9.S4, V18.S4, V10.S4 // B: r -= n·ln2lo

	// Horner: p = ((((p0·r+p1)·r+p2)·r+p3)·r+p4)·r+p5, chains interleaved.
	WORD  $0x4EB31E65 // ORR V5 = p0                 (A)
	WORD  $0x4EB31E6C // ORR V12 = p0                (B)
	WORD  $0x4EB41E84 // ORR V4 = p1                 (A)
	WORD  $0x4EB41E8B // ORR V11 = p1                (B)
	VFMLA V5.S4, V3.S4, V4.S4    // A: p = p1 + p·r
	VFMLA V12.S4, V10.S4, V11.S4 // B: p = p1 + p·r
	WORD  $0x4EB51EA5 // ORR V5 = p2                 (A)
	WORD  $0x4EB51EAC // ORR V12 = p2                (B)
	VFMLA V4.S4, V3.S4, V5.S4    // A: p = p2 + p·r
	VFMLA V11.S4, V10.S4, V12.S4 // B: p = p2 + p·r
	WORD  $0x4EB61EC4 // ORR V4 = p3                 (A)
	WORD  $0x4EB61ECB // ORR V11 = p3                (B)
	VFMLA V5.S4, V3.S4, V4.S4    // A: p = p3 + p·r
	VFMLA V12.S4, V10.S4, V11.S4 // B: p = p3 + p·r
	WORD  $0x4EB71EE5 // ORR V5 = p4                 (A)
	WORD  $0x4EB71EEC // ORR V12 = p4                (B)
	VFMLA V4.S4, V3.S4, V5.S4    // A: p = p4 + p·r
	VFMLA V11.S4, V10.S4, V12.S4 // B: p = p4 + p·r
	WORD  $0x4EB81F04 // ORR V4 = p5                 (A)
	WORD  $0x4EB81F0B // ORR V11 = p5                (B)
	VFMLA V5.S4, V3.S4, V4.S4    // A: p = p5 + p·r
	VFMLA V12.S4, V10.S4, V11.S4 // B: p = p5 + p·r

	// res = (1 + r) + p·r², scaled by 2^n.
	WORD  $0x6E23DC66 // FMUL V6.4S, V3.4S, V3.4S    (A: r² = r·r)
	WORD  $0x6E2ADD4D // FMUL V13.4S, V10.4S, V10.4S (B: r² = r·r)
	WORD  $0x4EB91F25 // ORR V5 = one                (A)
	WORD  $0x4EB91F2C // ORR V12 = one               (B)
	WORD  $0x4E23D4A5 // FADD V5.4S, V5.4S, V3.4S    (A: q = 1 + r)
	WORD  $0x4E2AD58C // FADD V12.4S, V12.4S, V10.4S (B: q = 1 + r)
	VFMLA V4.S4, V6.S4, V5.S4    // A: res = q + p·r²
	VFMLA V11.S4, V13.S4, V12.S4 // B: res = q + p·r²
	WORD  $0x4EA1B842 // FCVTZS V2.4S, V2.4S         (A: ni = int(n))
	WORD  $0x4EA1B929 // FCVTZS V9.4S, V9.4S         (B: ni = int(n))
	VADD  V28.S4, V2.S4, V2.S4   // A: ni += 127
	VADD  V28.S4, V9.S4, V9.S4   // B: ni += 127
	VSHL  $23, V2.S4, V2.S4      // A: 2^n bits = ni<<23
	VSHL  $23, V9.S4, V9.S4      // B: 2^n bits = ni<<23
	WORD  $0x6E22DCA5 // FMUL V5.4S, V5.4S, V2.4S    (A: res *= 2^n)
	WORD  $0x6E29DD8C // FMUL V12.4S, V12.4S, V9.4S  (B: res *= 2^n)

	// accumulate + store.
	WORD   $0x4E25D7BD // FADD V29.4S, V29.4S, V5.4S  (A: sum += res)
	WORD   $0x4E2CD7DE // FADD V30.4S, V30.4S, V12.4S (B: sum += res)
	VST1.P [V5.S4], 16(R1)
	VST1.P [V12.S4], 16(R1)

	SUBS $1, R4, R4
	BNE  loop2

odd:
	CBZ R5, reduce

	VLD1.P 16(R0), [V1.S4]
	WORD   $0x4EA0D421 // FSUB V1.4S, V1.4S, V0.4S
	WORD   $0x4E3AF421 // FMAX V1.4S, V1.4S, V26.4S
	WORD   $0x4EBBF421 // FMIN V1.4S, V1.4S, V27.4S
	WORD   $0x6E30DC22 // FMUL V2.4S, V1.4S, V16.4S
	WORD   $0x4E218842 // FRINTN V2.4S, V2.4S
	WORD   $0x4EA11C23 // ORR V3.16B, V1.16B, V1.16B
	VFMLS  V2.S4, V17.S4, V3.S4
	VFMLS  V2.S4, V18.S4, V3.S4
	WORD   $0x4EB31E65 // ORR V5 = p0
	WORD   $0x4EB41E84 // ORR V4 = p1
	VFMLA  V5.S4, V3.S4, V4.S4
	WORD   $0x4EB51EA5 // ORR V5 = p2
	VFMLA  V4.S4, V3.S4, V5.S4
	WORD   $0x4EB61EC4 // ORR V4 = p3
	VFMLA  V5.S4, V3.S4, V4.S4
	WORD   $0x4EB71EE5 // ORR V5 = p4
	VFMLA  V4.S4, V3.S4, V5.S4
	WORD   $0x4EB81F04 // ORR V4 = p5
	VFMLA  V5.S4, V3.S4, V4.S4
	WORD   $0x6E23DC66 // FMUL V6.4S, V3.4S, V3.4S
	WORD   $0x4EB91F25 // ORR V5 = one
	WORD   $0x4E23D4A5 // FADD V5.4S, V5.4S, V3.4S
	VFMLA  V4.S4, V6.S4, V5.S4
	WORD   $0x4EA1B842 // FCVTZS V2.4S, V2.4S
	VADD   V28.S4, V2.S4, V2.S4
	VSHL   $23, V2.S4, V2.S4
	WORD   $0x6E22DCA5 // FMUL V5.4S, V5.4S, V2.4S
	WORD   $0x4E25D7BD // FADD V29.4S, V29.4S, V5.4S
	VST1.P [V5.S4], 16(R1)

reduce:
	WORD  $0x4E3ED7BD // FADD V29.4S, V29.4S, V30.4S
	WORD  $0x6E3DD7BD // FADDP V29.4S, V29.4S, V29.4S
	WORD  $0x6E3DD7BD // FADDP V29.4S, V29.4S, V29.4S
	FMOVS F29, ret+24(FP)
	RET

// func vgeluQuadsNeonF32(dst, src *float32, quads int)
//
// 4-wide f32 GELU (vexp.go): dst[i] = 0.5·x·(1+erf(x/√2)) for x = src[i],
// i in [0, 4*quads). erf via Abramowitz-Stegun 7.1.26 — t = 1/(1+p·|u|),
// erf(|u|) = 1 − t·(a1 + t·(a2 + t·(a3 + t·(a4 + t·a5))))·e^(−u²) — assembled
// on the SAME Cephes exp reduction as vexpQuadsNeonF32 above (only the lo
// clamp is needed: −u² ≤ 0 never overflows 2^n). The sign is re-applied
// bitwise (erf bits = E bits | x's sign bit; E ∈ [0,1]), then
// gelu = x·(0.5 + 0.5·erf). Like the vexp kernel, the main loop runs TWO
// quads per pass with disjoint registers (A: V0..V5, B: V6..V11) so the two
// serial FMLA chains overlap; V12..V31 hold the constants. WORD encodings
// generated + objdump-verified like the vexp kernel above.
DATA geluConsts<>+0(SB)/4, $0x0000007F   // V12 int32 127
DATA geluConsts<>+4(SB)/4, $0x0000007F   // V12
DATA geluConsts<>+8(SB)/4, $0x0000007F   // V12
DATA geluConsts<>+12(SB)/4, $0x0000007F  // V12
DATA geluConsts<>+16(SB)/4, $0x3F3504F3  // V13 invSqrt2 = 0.7071067811865476
DATA geluConsts<>+20(SB)/4, $0x3F3504F3  // V13
DATA geluConsts<>+24(SB)/4, $0x3F3504F3  // V13
DATA geluConsts<>+28(SB)/4, $0x3F3504F3  // V13
DATA geluConsts<>+32(SB)/4, $0x3EA7BA05  // V14 pAS = 0.3275911
DATA geluConsts<>+36(SB)/4, $0x3EA7BA05  // V14
DATA geluConsts<>+40(SB)/4, $0x3EA7BA05  // V14
DATA geluConsts<>+44(SB)/4, $0x3EA7BA05  // V14
DATA geluConsts<>+48(SB)/4, $0x3E827906  // V15 a1 = 0.254829592
DATA geluConsts<>+52(SB)/4, $0x3E827906  // V15
DATA geluConsts<>+56(SB)/4, $0x3E827906  // V15
DATA geluConsts<>+60(SB)/4, $0x3E827906  // V15
DATA geluConsts<>+64(SB)/4, $0xBE91A98E  // V16 a2 = -0.284496736
DATA geluConsts<>+68(SB)/4, $0xBE91A98E  // V16
DATA geluConsts<>+72(SB)/4, $0xBE91A98E  // V16
DATA geluConsts<>+76(SB)/4, $0xBE91A98E  // V16
DATA geluConsts<>+80(SB)/4, $0x3FB5F0E3  // V17 a3 = 1.421413741
DATA geluConsts<>+84(SB)/4, $0x3FB5F0E3  // V17
DATA geluConsts<>+88(SB)/4, $0x3FB5F0E3  // V17
DATA geluConsts<>+92(SB)/4, $0x3FB5F0E3  // V17
DATA geluConsts<>+96(SB)/4, $0xBFBA00E3  // V18 a4 = -1.453152027
DATA geluConsts<>+100(SB)/4, $0xBFBA00E3 // V18
DATA geluConsts<>+104(SB)/4, $0xBFBA00E3 // V18
DATA geluConsts<>+108(SB)/4, $0xBFBA00E3 // V18
DATA geluConsts<>+112(SB)/4, $0x3F87DC22 // V19 a5 = 1.061405429
DATA geluConsts<>+116(SB)/4, $0x3F87DC22 // V19
DATA geluConsts<>+120(SB)/4, $0x3F87DC22 // V19
DATA geluConsts<>+124(SB)/4, $0x3F87DC22 // V19
DATA geluConsts<>+128(SB)/4, $0x3F800000 // V20 one
DATA geluConsts<>+132(SB)/4, $0x3F800000 // V20
DATA geluConsts<>+136(SB)/4, $0x3F800000 // V20
DATA geluConsts<>+140(SB)/4, $0x3F800000 // V20
DATA geluConsts<>+144(SB)/4, $0x3F000000 // V21 half (also exp p5)
DATA geluConsts<>+148(SB)/4, $0x3F000000 // V21
DATA geluConsts<>+152(SB)/4, $0x3F000000 // V21
DATA geluConsts<>+156(SB)/4, $0x3F000000 // V21
DATA geluConsts<>+160(SB)/4, $0x80000000 // V22 sign mask
DATA geluConsts<>+164(SB)/4, $0x80000000 // V22
DATA geluConsts<>+168(SB)/4, $0x80000000 // V22
DATA geluConsts<>+172(SB)/4, $0x80000000 // V22
DATA geluConsts<>+176(SB)/4, $0x3FB8AA3B // V23 log2e
DATA geluConsts<>+180(SB)/4, $0x3FB8AA3B // V23
DATA geluConsts<>+184(SB)/4, $0x3FB8AA3B // V23
DATA geluConsts<>+188(SB)/4, $0x3FB8AA3B // V23
DATA geluConsts<>+192(SB)/4, $0x3F318000 // V24 ln2hi = 0.693359375
DATA geluConsts<>+196(SB)/4, $0x3F318000 // V24
DATA geluConsts<>+200(SB)/4, $0x3F318000 // V24
DATA geluConsts<>+204(SB)/4, $0x3F318000 // V24
DATA geluConsts<>+208(SB)/4, $0xB95E8083 // V25 ln2lo = -2.12194440e-4
DATA geluConsts<>+212(SB)/4, $0xB95E8083 // V25
DATA geluConsts<>+216(SB)/4, $0xB95E8083 // V25
DATA geluConsts<>+220(SB)/4, $0xB95E8083 // V25
DATA geluConsts<>+224(SB)/4, $0x39506967 // V26 p0 = 1.9875691500e-4
DATA geluConsts<>+228(SB)/4, $0x39506967 // V26
DATA geluConsts<>+232(SB)/4, $0x39506967 // V26
DATA geluConsts<>+236(SB)/4, $0x39506967 // V26
DATA geluConsts<>+240(SB)/4, $0x3AB743CE // V27 p1 = 1.3981999507e-3
DATA geluConsts<>+244(SB)/4, $0x3AB743CE // V27
DATA geluConsts<>+248(SB)/4, $0x3AB743CE // V27
DATA geluConsts<>+252(SB)/4, $0x3AB743CE // V27
DATA geluConsts<>+256(SB)/4, $0x3C088908 // V28 p2 = 8.3334519073e-3
DATA geluConsts<>+260(SB)/4, $0x3C088908 // V28
DATA geluConsts<>+264(SB)/4, $0x3C088908 // V28
DATA geluConsts<>+268(SB)/4, $0x3C088908 // V28
DATA geluConsts<>+272(SB)/4, $0x3D2AA9C1 // V29 p3 = 4.1665795894e-2
DATA geluConsts<>+276(SB)/4, $0x3D2AA9C1 // V29
DATA geluConsts<>+280(SB)/4, $0x3D2AA9C1 // V29
DATA geluConsts<>+284(SB)/4, $0x3D2AA9C1 // V29
DATA geluConsts<>+288(SB)/4, $0x3E2AAAAA // V30 p4 = 1.6666665459e-1
DATA geluConsts<>+292(SB)/4, $0x3E2AAAAA // V30
DATA geluConsts<>+296(SB)/4, $0x3E2AAAAA // V30
DATA geluConsts<>+300(SB)/4, $0x3E2AAAAA // V30
DATA geluConsts<>+304(SB)/4, $0xC2AEAC4F // V31 lo clamp = -87.33654
DATA geluConsts<>+308(SB)/4, $0xC2AEAC4F // V31
DATA geluConsts<>+312(SB)/4, $0xC2AEAC4F // V31
DATA geluConsts<>+316(SB)/4, $0xC2AEAC4F // V31
// Tail entries: the GRAD kernel only (vgeluGradQuadsNeonF32 below) — loaded
// per pass into free scratch (no register spare for 22 resident constants).
// The forward kernel's init loads stop at +320.
DATA geluConsts<>+320(SB)/4, $0x008D8EB6 // pdfMin = 1.3e-38 (geluGradPdfMin, vexp.go)
DATA geluConsts<>+324(SB)/4, $0x008D8EB6 // pdfMin
DATA geluConsts<>+328(SB)/4, $0x008D8EB6 // pdfMin
DATA geluConsts<>+332(SB)/4, $0x008D8EB6 // pdfMin
DATA geluConsts<>+336(SB)/4, $0x3ECC422A // invSqrt2Pi = 0.3989422804014327
DATA geluConsts<>+340(SB)/4, $0x3ECC422A // invSqrt2Pi
DATA geluConsts<>+344(SB)/4, $0x3ECC422A // invSqrt2Pi
DATA geluConsts<>+348(SB)/4, $0x3ECC422A // invSqrt2Pi
GLOBL geluConsts<>(SB), RODATA|NOPTR, $352

TEXT ·vgeluQuadsNeonF32(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R1
	MOVD src+8(FP), R0
	MOVD quads+16(FP), R2
	MOVD $geluConsts<>(SB), R3
	VLD1.P 64(R3), [V12.S4, V13.S4, V14.S4, V15.S4]
	VLD1.P 64(R3), [V16.S4, V17.S4, V18.S4, V19.S4]
	VLD1.P 64(R3), [V20.S4, V21.S4, V22.S4, V23.S4]
	VLD1.P 64(R3), [V24.S4, V25.S4, V26.S4, V27.S4]
	VLD1   (R3), [V28.S4, V29.S4, V30.S4, V31.S4]
	LSR $1, R2, R4 // pairs of quads
	AND $1, R2, R5 // odd quad
	CBZ R4, geluOdd

geluLoop2:
	VLD1.P 16(R0), [V0.S4]
	VLD1.P 16(R0), [V6.S4]

	// t = 1/(1 + pAS·|x·invSqrt2|), both quads.
	WORD  $0x6E2DDC01 // FMUL V1.4S, V0.4S, V13.4S   (A: u = x·invSqrt2)
	WORD  $0x6E2DDCC7 // FMUL V7.4S, V6.4S, V13.4S   (B: u = x·invSqrt2)
	WORD  $0x4EA0F821 // FABS V1.4S, V1.4S           (A: a = |u|)
	WORD  $0x4EA0F8E7 // FABS V7.4S, V7.4S           (B: a = |u|)
	WORD  $0x4EB41E82 // ORR V2 = one                (A)
	WORD  $0x4EB41E88 // ORR V8 = one                (B)
	VFMLA V1.S4, V14.S4, V2.S4 // A: denom = 1 + pAS·a
	VFMLA V7.S4, V14.S4, V8.S4 // B: denom = 1 + pAS·a
	WORD  $0x6E22FE83 // FDIV V3.4S, V20.4S, V2.4S   (A: t = 1/denom)
	WORD  $0x6E28FE89 // FDIV V9.4S, V20.4S, V8.4S   (B: t = 1/denom)

	// P = t·(a1 + t·(a2 + t·(a3 + t·(a4 + t·a5)))), Horner ping-pong.
	WORD  $0x4EB21E42 // ORR V2 = a4                 (A)
	WORD  $0x4EB21E48 // ORR V8 = a4                 (B)
	VFMLA V19.S4, V3.S4, V2.S4 // A: s = a4 + a5·t
	VFMLA V19.S4, V9.S4, V8.S4 // B: s = a4 + a5·t
	WORD  $0x4EB11E24 // ORR V4 = a3                 (A)
	WORD  $0x4EB11E2A // ORR V10 = a3                (B)
	VFMLA V2.S4, V3.S4, V4.S4  // A: s = a3 + s·t
	VFMLA V8.S4, V9.S4, V10.S4 // B: s = a3 + s·t
	WORD  $0x4EB01E02 // ORR V2 = a2                 (A)
	WORD  $0x4EB01E08 // ORR V8 = a2                 (B)
	VFMLA V4.S4, V3.S4, V2.S4  // A: s = a2 + s·t
	VFMLA V10.S4, V9.S4, V8.S4 // B: s = a2 + s·t
	WORD  $0x4EAF1DE4 // ORR V4 = a1                 (A)
	WORD  $0x4EAF1DEA // ORR V10 = a1                (B)
	VFMLA V2.S4, V3.S4, V4.S4  // A: s = a1 + s·t
	VFMLA V8.S4, V9.S4, V10.S4 // B: s = a1 + s·t
	WORD  $0x6E23DC85 // FMUL V5.4S, V4.4S, V3.4S    (A: P = s·t)
	WORD  $0x6E29DD4B // FMUL V11.4S, V10.4S, V9.4S  (B: P = s·t)

	// e = exp(clamp(−a²)) — the vexp reduction (lo clamp only).
	WORD  $0x6E21DC21 // FMUL V1.4S, V1.4S, V1.4S    (A: a² = a·a)
	WORD  $0x6E27DCE7 // FMUL V7.4S, V7.4S, V7.4S    (B: a² = a·a)
	WORD  $0x6EA0F821 // FNEG V1.4S, V1.4S           (A: w = −a²)
	WORD  $0x6EA0F8E7 // FNEG V7.4S, V7.4S           (B: w = −a²)
	WORD  $0x4E3FF421 // FMAX V1.4S, V1.4S, V31.4S   (A: clamp lo)
	WORD  $0x4E3FF4E7 // FMAX V7.4S, V7.4S, V31.4S   (B: clamp lo)
	WORD  $0x6E37DC22 // FMUL V2.4S, V1.4S, V23.4S   (A: z = w·log2e)
	WORD  $0x6E37DCE8 // FMUL V8.4S, V7.4S, V23.4S   (B: z = w·log2e)
	WORD  $0x4E218842 // FRINTN V2.4S, V2.4S         (A: n = rint(z))
	WORD  $0x4E218908 // FRINTN V8.4S, V8.4S         (B: n = rint(z))
	VFMLS V2.S4, V24.S4, V1.S4 // A: r -= n·ln2hi
	VFMLS V8.S4, V24.S4, V7.S4 // B: r -= n·ln2hi
	VFMLS V2.S4, V25.S4, V1.S4 // A: r -= n·ln2lo
	VFMLS V8.S4, V25.S4, V7.S4 // B: r -= n·ln2lo
	WORD  $0x4EBB1F63 // ORR V3 = p1                 (A)
	WORD  $0x4EBB1F69 // ORR V9 = p1                 (B)
	VFMLA V26.S4, V1.S4, V3.S4 // A: p = p1 + p0·r
	VFMLA V26.S4, V7.S4, V9.S4 // B: p = p1 + p0·r
	WORD  $0x4EBC1F84 // ORR V4 = p2                 (A)
	WORD  $0x4EBC1F8A // ORR V10 = p2                (B)
	VFMLA V3.S4, V1.S4, V4.S4  // A: p = p2 + p·r
	VFMLA V9.S4, V7.S4, V10.S4 // B: p = p2 + p·r
	WORD  $0x4EBD1FA3 // ORR V3 = p3                 (A)
	WORD  $0x4EBD1FA9 // ORR V9 = p3                 (B)
	VFMLA V4.S4, V1.S4, V3.S4  // A: p = p3 + p·r
	VFMLA V10.S4, V7.S4, V9.S4 // B: p = p3 + p·r
	WORD  $0x4EBE1FC4 // ORR V4 = p4                 (A)
	WORD  $0x4EBE1FCA // ORR V10 = p4                (B)
	VFMLA V3.S4, V1.S4, V4.S4  // A: p = p4 + p·r
	VFMLA V9.S4, V7.S4, V10.S4 // B: p = p4 + p·r
	WORD  $0x4EB51EA3 // ORR V3 = p5(=half)          (A)
	WORD  $0x4EB51EA9 // ORR V9 = p5(=half)          (B)
	VFMLA V4.S4, V1.S4, V3.S4  // A: p = p5 + p·r
	VFMLA V10.S4, V7.S4, V9.S4 // B: p = p5 + p·r
	WORD  $0x6E21DC24 // FMUL V4.4S, V1.4S, V1.4S    (A: r² = r·r)
	WORD  $0x6E27DCEA // FMUL V10.4S, V7.4S, V7.4S   (B: r² = r·r)
	WORD  $0x4E34D421 // FADD V1.4S, V1.4S, V20.4S   (A: q = 1 + r)
	WORD  $0x4E34D4E7 // FADD V7.4S, V7.4S, V20.4S   (B: q = 1 + r)
	VFMLA V3.S4, V4.S4, V1.S4  // A: e = q + p·r²
	VFMLA V9.S4, V10.S4, V7.S4 // B: e = q + p·r²
	WORD  $0x4EA1B842 // FCVTZS V2.4S, V2.4S         (A: ni = int(n))
	WORD  $0x4EA1B908 // FCVTZS V8.4S, V8.4S         (B: ni = int(n))
	VADD  V12.S4, V2.S4, V2.S4 // A: ni += 127
	VADD  V12.S4, V8.S4, V8.S4 // B: ni += 127
	VSHL  $23, V2.S4, V2.S4    // A: 2^n bits
	VSHL  $23, V8.S4, V8.S4    // B: 2^n bits
	WORD  $0x6E22DC21 // FMUL V1.4S, V1.4S, V2.4S    (A: e *= 2^n)
	WORD  $0x6E28DCE7 // FMUL V7.4S, V7.4S, V8.4S    (B: e *= 2^n)

	// erf = sign(x) | (1 − e·P); gelu = x·(0.5 + 0.5·erf).
	WORD  $0x4EB41E82 // ORR V2 = one                (A)
	WORD  $0x4EB41E88 // ORR V8 = one                (B)
	VFMLS V1.S4, V5.S4, V2.S4  // A: E = 1 − e·P
	VFMLS V7.S4, V11.S4, V8.S4 // B: E = 1 − e·P
	WORD  $0x4E361C04 // AND V4.16B, V0.16B, V22.16B (A: sign bit of x)
	WORD  $0x4E361CCA // AND V10.16B, V6.16B, V22.16B (B: sign bit of x)
	WORD  $0x4EA41C42 // ORR V2.16B, V2.16B, V4.16B  (A: erf = sign|E)
	WORD  $0x4EAA1D08 // ORR V8.16B, V8.16B, V10.16B (B: erf = sign|E)
	WORD  $0x4EB51EA3 // ORR V3 = half               (A)
	WORD  $0x4EB51EA9 // ORR V9 = half               (B)
	VFMLA V2.S4, V21.S4, V3.S4 // A: h = 0.5 + 0.5·erf
	VFMLA V8.S4, V21.S4, V9.S4 // B: h = 0.5 + 0.5·erf
	WORD  $0x6E23DC00 // FMUL V0.4S, V0.4S, V3.4S    (A: out = x·h)
	WORD  $0x6E29DCC6 // FMUL V6.4S, V6.4S, V9.4S    (B: out = x·h)

	VST1.P [V0.S4], 16(R1)
	VST1.P [V6.S4], 16(R1)

	SUBS $1, R4, R4
	BNE  geluLoop2

geluOdd:
	CBZ R5, geluDone

	VLD1.P 16(R0), [V0.S4]
	WORD   $0x6E2DDC01 // FMUL V1.4S, V0.4S, V13.4S  (u = x·invSqrt2)
	WORD   $0x4EA0F821 // FABS V1.4S, V1.4S          (a = |u|)
	WORD   $0x4EB41E82 // ORR V2 = one
	VFMLA  V1.S4, V14.S4, V2.S4 // denom = 1 + pAS·a
	WORD   $0x6E22FE83 // FDIV V3.4S, V20.4S, V2.4S  (t = 1/denom)
	WORD   $0x4EB21E42 // ORR V2 = a4
	VFMLA  V19.S4, V3.S4, V2.S4 // s = a4 + a5·t
	WORD   $0x4EB11E24 // ORR V4 = a3
	VFMLA  V2.S4, V3.S4, V4.S4 // s = a3 + s·t
	WORD   $0x4EB01E02 // ORR V2 = a2
	VFMLA  V4.S4, V3.S4, V2.S4 // s = a2 + s·t
	WORD   $0x4EAF1DE4 // ORR V4 = a1
	VFMLA  V2.S4, V3.S4, V4.S4 // s = a1 + s·t
	WORD   $0x6E23DC85 // FMUL V5.4S, V4.4S, V3.4S   (P = s·t)
	WORD   $0x6E21DC21 // FMUL V1.4S, V1.4S, V1.4S   (a² = a·a)
	WORD   $0x6EA0F821 // FNEG V1.4S, V1.4S          (w = −a²)
	WORD   $0x4E3FF421 // FMAX V1.4S, V1.4S, V31.4S  (clamp lo)
	WORD   $0x6E37DC22 // FMUL V2.4S, V1.4S, V23.4S  (z = w·log2e)
	WORD   $0x4E218842 // FRINTN V2.4S, V2.4S        (n = rint(z))
	VFMLS  V2.S4, V24.S4, V1.S4 // r -= n·ln2hi
	VFMLS  V2.S4, V25.S4, V1.S4 // r -= n·ln2lo
	WORD   $0x4EBB1F63 // ORR V3 = p1
	VFMLA  V26.S4, V1.S4, V3.S4 // p = p1 + p0·r
	WORD   $0x4EBC1F84 // ORR V4 = p2
	VFMLA  V3.S4, V1.S4, V4.S4 // p = p2 + p·r
	WORD   $0x4EBD1FA3 // ORR V3 = p3
	VFMLA  V4.S4, V1.S4, V3.S4 // p = p3 + p·r
	WORD   $0x4EBE1FC4 // ORR V4 = p4
	VFMLA  V3.S4, V1.S4, V4.S4 // p = p4 + p·r
	WORD   $0x4EB51EA3 // ORR V3 = p5(=half)
	VFMLA  V4.S4, V1.S4, V3.S4 // p = p5 + p·r
	WORD   $0x6E21DC24 // FMUL V4.4S, V1.4S, V1.4S   (r² = r·r)
	WORD   $0x4E34D421 // FADD V1.4S, V1.4S, V20.4S  (q = 1 + r)
	VFMLA  V3.S4, V4.S4, V1.S4 // e = q + p·r²
	WORD   $0x4EA1B842 // FCVTZS V2.4S, V2.4S        (ni = int(n))
	VADD   V12.S4, V2.S4, V2.S4 // ni += 127
	VSHL   $23, V2.S4, V2.S4    // 2^n bits
	WORD   $0x6E22DC21 // FMUL V1.4S, V1.4S, V2.4S   (e *= 2^n)
	WORD   $0x4EB41E82 // ORR V2 = one
	VFMLS  V1.S4, V5.S4, V2.S4 // E = 1 − e·P
	WORD   $0x4E361C04 // AND V4.16B, V0.16B, V22.16B (sign bit of x)
	WORD   $0x4EA41C42 // ORR V2.16B, V2.16B, V4.16B  (erf = sign|E)
	WORD   $0x4EB51EA3 // ORR V3 = half
	VFMLA  V2.S4, V21.S4, V3.S4 // h = 0.5 + 0.5·erf
	WORD   $0x6E23DC00 // FMUL V0.4S, V0.4S, V3.4S   (out = x·h)
	VST1.P [V0.S4], 16(R1)

geluDone:
	RET

// func vgeluGradQuadsNeonF32(dst, x, grad *float32, quads int)
//
// 4-wide f32 GELU BACKWARD (vexp.go, §T664): dst[i] = g[i]·(Φ(x) + x·φ(x)),
// Φ(x) = 0.5·(1+erf(x/√2)), φ(x) = e^(−x²/2)/√(2π), for x = x[i],
// i in [0, 4*quads). Same erf-on-exp pipeline as vgeluQuadsNeonF32 above —
// with a = |x|/√2 the AS-7.1.26 erf already computes e^(−a²) = e^(−x²/2), so
// ONE exp feeds both erf and the pdf. Two extras vs the forward: (1) e is
// masked to a true 0 when ≤ pdfMin (FCMGT+AND) so the exp underflow clamp
// can't leak into x·φ (x=±Inf must give Inf·0 = NaN like ref — see vexp.go);
// (2) the tail multiplies by the pdf and by g. The two per-pass tail
// constants (pdfMin, invSqrt2Pi) live past the 20 resident constants
// (geluConsts+320) and load into scratch each pass — no spare registers.
// Same two-quads-per-pass structure (A: V0..V5, B: V6..V11); g loads reuse
// V0/V6 after x dies. WORD encodings objdump-verified like the kernels above.
//
// Register map: R0 x ptr, R1 dst ptr, R6 g ptr, R2 quads, R3 consts tail,
// R4 pairs, R5 odd; V12..V31 constants as in vgeluQuadsNeonF32.
TEXT ·vgeluGradQuadsNeonF32(SB), NOSPLIT, $0-32
	MOVD dst+0(FP), R1
	MOVD x+8(FP), R0
	MOVD grad+16(FP), R6
	MOVD quads+24(FP), R2
	MOVD $geluConsts<>(SB), R3
	VLD1.P 64(R3), [V12.S4, V13.S4, V14.S4, V15.S4]
	VLD1.P 64(R3), [V16.S4, V17.S4, V18.S4, V19.S4]
	VLD1.P 64(R3), [V20.S4, V21.S4, V22.S4, V23.S4]
	VLD1.P 64(R3), [V24.S4, V25.S4, V26.S4, V27.S4]
	VLD1.P 64(R3), [V28.S4, V29.S4, V30.S4, V31.S4]
	// R3 now points at the tail: +320 pdfMin, +336 invSqrt2Pi.
	LSR $1, R2, R4 // pairs of quads
	AND $1, R2, R5 // odd quad
	CBZ R4, gradOdd

gradLoop2:
	VLD1.P 16(R0), [V0.S4]
	VLD1.P 16(R0), [V6.S4]

	// t = 1/(1 + pAS·|x·invSqrt2|), both quads (as the forward).
	WORD  $0x6E2DDC01 // FMUL V1.4S, V0.4S, V13.4S   (A: u = x·invSqrt2)
	WORD  $0x6E2DDCC7 // FMUL V7.4S, V6.4S, V13.4S   (B: u = x·invSqrt2)
	WORD  $0x4EA0F821 // FABS V1.4S, V1.4S           (A: a = |u|)
	WORD  $0x4EA0F8E7 // FABS V7.4S, V7.4S           (B: a = |u|)
	WORD  $0x4EB41E82 // ORR V2 = one                (A)
	WORD  $0x4EB41E88 // ORR V8 = one                (B)
	VFMLA V1.S4, V14.S4, V2.S4 // A: denom = 1 + pAS·a
	VFMLA V7.S4, V14.S4, V8.S4 // B: denom = 1 + pAS·a
	WORD  $0x6E22FE83 // FDIV V3.4S, V20.4S, V2.4S   (A: t = 1/denom)
	WORD  $0x6E28FE89 // FDIV V9.4S, V20.4S, V8.4S   (B: t = 1/denom)

	// P = t·(a1 + t·(a2 + t·(a3 + t·(a4 + t·a5)))), Horner ping-pong.
	WORD  $0x4EB21E42 // ORR V2 = a4                 (A)
	WORD  $0x4EB21E48 // ORR V8 = a4                 (B)
	VFMLA V19.S4, V3.S4, V2.S4 // A: s = a4 + a5·t
	VFMLA V19.S4, V9.S4, V8.S4 // B: s = a4 + a5·t
	WORD  $0x4EB11E24 // ORR V4 = a3                 (A)
	WORD  $0x4EB11E2A // ORR V10 = a3                (B)
	VFMLA V2.S4, V3.S4, V4.S4  // A: s = a3 + s·t
	VFMLA V8.S4, V9.S4, V10.S4 // B: s = a3 + s·t
	WORD  $0x4EB01E02 // ORR V2 = a2                 (A)
	WORD  $0x4EB01E08 // ORR V8 = a2                 (B)
	VFMLA V4.S4, V3.S4, V2.S4  // A: s = a2 + s·t
	VFMLA V10.S4, V9.S4, V8.S4 // B: s = a2 + s·t
	WORD  $0x4EAF1DE4 // ORR V4 = a1                 (A)
	WORD  $0x4EAF1DEA // ORR V10 = a1                (B)
	VFMLA V2.S4, V3.S4, V4.S4  // A: s = a1 + s·t
	VFMLA V8.S4, V9.S4, V10.S4 // B: s = a1 + s·t
	WORD  $0x6E23DC85 // FMUL V5.4S, V4.4S, V3.4S    (A: P = s·t)
	WORD  $0x6E29DD4B // FMUL V11.4S, V10.4S, V9.4S  (B: P = s·t)

	// e = exp(clamp(−a²)) — the vexp reduction (lo clamp only).
	WORD  $0x6E21DC21 // FMUL V1.4S, V1.4S, V1.4S    (A: a² = a·a)
	WORD  $0x6E27DCE7 // FMUL V7.4S, V7.4S, V7.4S    (B: a² = a·a)
	WORD  $0x6EA0F821 // FNEG V1.4S, V1.4S           (A: w = −a²)
	WORD  $0x6EA0F8E7 // FNEG V7.4S, V7.4S           (B: w = −a²)
	WORD  $0x4E3FF421 // FMAX V1.4S, V1.4S, V31.4S   (A: clamp lo)
	WORD  $0x4E3FF4E7 // FMAX V7.4S, V7.4S, V31.4S   (B: clamp lo)
	WORD  $0x6E37DC22 // FMUL V2.4S, V1.4S, V23.4S   (A: z = w·log2e)
	WORD  $0x6E37DCE8 // FMUL V8.4S, V7.4S, V23.4S   (B: z = w·log2e)
	WORD  $0x4E218842 // FRINTN V2.4S, V2.4S         (A: n = rint(z))
	WORD  $0x4E218908 // FRINTN V8.4S, V8.4S         (B: n = rint(z))
	VFMLS V2.S4, V24.S4, V1.S4 // A: r -= n·ln2hi
	VFMLS V8.S4, V24.S4, V7.S4 // B: r -= n·ln2hi
	VFMLS V2.S4, V25.S4, V1.S4 // A: r -= n·ln2lo
	VFMLS V8.S4, V25.S4, V7.S4 // B: r -= n·ln2lo
	WORD  $0x4EBB1F63 // ORR V3 = p1                 (A)
	WORD  $0x4EBB1F69 // ORR V9 = p1                 (B)
	VFMLA V26.S4, V1.S4, V3.S4 // A: p = p1 + p0·r
	VFMLA V26.S4, V7.S4, V9.S4 // B: p = p1 + p0·r
	WORD  $0x4EBC1F84 // ORR V4 = p2                 (A)
	WORD  $0x4EBC1F8A // ORR V10 = p2                (B)
	VFMLA V3.S4, V1.S4, V4.S4  // A: p = p2 + p·r
	VFMLA V9.S4, V7.S4, V10.S4 // B: p = p2 + p·r
	WORD  $0x4EBD1FA3 // ORR V3 = p3                 (A)
	WORD  $0x4EBD1FA9 // ORR V9 = p3                 (B)
	VFMLA V4.S4, V1.S4, V3.S4  // A: p = p3 + p·r
	VFMLA V10.S4, V7.S4, V9.S4 // B: p = p3 + p·r
	WORD  $0x4EBE1FC4 // ORR V4 = p4                 (A)
	WORD  $0x4EBE1FCA // ORR V10 = p4                (B)
	VFMLA V3.S4, V1.S4, V4.S4  // A: p = p4 + p·r
	VFMLA V9.S4, V7.S4, V10.S4 // B: p = p4 + p·r
	WORD  $0x4EB51EA3 // ORR V3 = p5(=half)          (A)
	WORD  $0x4EB51EA9 // ORR V9 = p5(=half)          (B)
	VFMLA V4.S4, V1.S4, V3.S4  // A: p = p5 + p·r
	VFMLA V10.S4, V7.S4, V9.S4 // B: p = p5 + p·r
	WORD  $0x6E21DC24 // FMUL V4.4S, V1.4S, V1.4S    (A: r² = r·r)
	WORD  $0x6E27DCEA // FMUL V10.4S, V7.4S, V7.4S   (B: r² = r·r)
	WORD  $0x4E34D421 // FADD V1.4S, V1.4S, V20.4S   (A: q = 1 + r)
	WORD  $0x4E34D4E7 // FADD V7.4S, V7.4S, V20.4S   (B: q = 1 + r)
	VFMLA V3.S4, V4.S4, V1.S4  // A: e = q + p·r²
	VFMLA V9.S4, V10.S4, V7.S4 // B: e = q + p·r²
	WORD  $0x4EA1B842 // FCVTZS V2.4S, V2.4S         (A: ni = int(n))
	WORD  $0x4EA1B908 // FCVTZS V8.4S, V8.4S         (B: ni = int(n))
	VADD  V12.S4, V2.S4, V2.S4 // A: ni += 127
	VADD  V12.S4, V8.S4, V8.S4 // B: ni += 127
	VSHL  $23, V2.S4, V2.S4    // A: 2^n bits
	VSHL  $23, V8.S4, V8.S4    // B: 2^n bits
	WORD  $0x6E22DC21 // FMUL V1.4S, V1.4S, V2.4S    (A: e *= 2^n)
	WORD  $0x6E28DCE7 // FMUL V7.4S, V7.4S, V8.4S    (B: e *= 2^n)

	// Grad tail: zero sub-pdfMin e, E = 1 − e·P, erf = sign(x)|E,
	// Φ = 0.5 + 0.5·erf, d = Φ + x·(e·invSqrt2Pi), out = g·d.
	VLD1  (R3), [V2.S4, V3.S4] // V2 = pdfMin, V3 = invSqrt2Pi
	WORD  $0x6EA2E424 // FCMGT V4.4S, V1.4S, V2.4S   (A: mask e > pdfMin)
	WORD  $0x6EA2E4EA // FCMGT V10.4S, V7.4S, V2.4S  (B: mask e > pdfMin)
	WORD  $0x4E241C21 // AND V1.16B, V1.16B, V4.16B  (A: e &= mask)
	WORD  $0x4E2A1CE7 // AND V7.16B, V7.16B, V10.16B (B: e &= mask)
	WORD  $0x4EB41E82 // ORR V2 = one                (A)
	WORD  $0x4EB41E88 // ORR V8 = one                (B)
	VFMLS V1.S4, V5.S4, V2.S4  // A: E = 1 − e·P
	VFMLS V7.S4, V11.S4, V8.S4 // B: E = 1 − e·P
	WORD  $0x4E361C04 // AND V4.16B, V0.16B, V22.16B  (A: sign bit of x)
	WORD  $0x4E361CCA // AND V10.16B, V6.16B, V22.16B (B: sign bit of x)
	WORD  $0x4EA41C42 // ORR V2.16B, V2.16B, V4.16B  (A: erf = sign|E)
	WORD  $0x4EAA1D08 // ORR V8.16B, V8.16B, V10.16B (B: erf = sign|E)
	WORD  $0x4EB51EA4 // ORR V4 = half               (A)
	WORD  $0x4EB51EAA // ORR V10 = half              (B)
	VFMLA V2.S4, V21.S4, V4.S4  // A: Φ = 0.5 + 0.5·erf
	VFMLA V8.S4, V21.S4, V10.S4 // B: Φ = 0.5 + 0.5·erf
	WORD  $0x6E23DC21 // FMUL V1.4S, V1.4S, V3.4S    (A: pdf = e·invSqrt2Pi)
	WORD  $0x6E23DCE7 // FMUL V7.4S, V7.4S, V3.4S    (B: pdf = e·invSqrt2Pi)
	VFMLA V1.S4, V0.S4, V4.S4   // A: d = Φ + pdf·x
	VFMLA V7.S4, V6.S4, V10.S4  // B: d = Φ + pdf·x
	VLD1.P 16(R6), [V0.S4] // A: g (x dead)
	VLD1.P 16(R6), [V6.S4] // B: g (x dead)
	WORD  $0x6E24DC00 // FMUL V0.4S, V0.4S, V4.4S    (A: out = g·d)
	WORD  $0x6E2ADCC6 // FMUL V6.4S, V6.4S, V10.4S   (B: out = g·d)

	VST1.P [V0.S4], 16(R1)
	VST1.P [V6.S4], 16(R1)

	SUBS $1, R4, R4
	BNE  gradLoop2

gradOdd:
	CBZ R5, gradDone

	VLD1.P 16(R0), [V0.S4]
	WORD   $0x6E2DDC01 // FMUL V1.4S, V0.4S, V13.4S  (u = x·invSqrt2)
	WORD   $0x4EA0F821 // FABS V1.4S, V1.4S          (a = |u|)
	WORD   $0x4EB41E82 // ORR V2 = one
	VFMLA  V1.S4, V14.S4, V2.S4 // denom = 1 + pAS·a
	WORD   $0x6E22FE83 // FDIV V3.4S, V20.4S, V2.4S  (t = 1/denom)
	WORD   $0x4EB21E42 // ORR V2 = a4
	VFMLA  V19.S4, V3.S4, V2.S4 // s = a4 + a5·t
	WORD   $0x4EB11E24 // ORR V4 = a3
	VFMLA  V2.S4, V3.S4, V4.S4 // s = a3 + s·t
	WORD   $0x4EB01E02 // ORR V2 = a2
	VFMLA  V4.S4, V3.S4, V2.S4 // s = a2 + s·t
	WORD   $0x4EAF1DE4 // ORR V4 = a1
	VFMLA  V2.S4, V3.S4, V4.S4 // s = a1 + s·t
	WORD   $0x6E23DC85 // FMUL V5.4S, V4.4S, V3.4S   (P = s·t)
	WORD   $0x6E21DC21 // FMUL V1.4S, V1.4S, V1.4S   (a² = a·a)
	WORD   $0x6EA0F821 // FNEG V1.4S, V1.4S          (w = −a²)
	WORD   $0x4E3FF421 // FMAX V1.4S, V1.4S, V31.4S  (clamp lo)
	WORD   $0x6E37DC22 // FMUL V2.4S, V1.4S, V23.4S  (z = w·log2e)
	WORD   $0x4E218842 // FRINTN V2.4S, V2.4S        (n = rint(z))
	VFMLS  V2.S4, V24.S4, V1.S4 // r -= n·ln2hi
	VFMLS  V2.S4, V25.S4, V1.S4 // r -= n·ln2lo
	WORD   $0x4EBB1F63 // ORR V3 = p1
	VFMLA  V26.S4, V1.S4, V3.S4 // p = p1 + p0·r
	WORD   $0x4EBC1F84 // ORR V4 = p2
	VFMLA  V3.S4, V1.S4, V4.S4 // p = p2 + p·r
	WORD   $0x4EBD1FA3 // ORR V3 = p3
	VFMLA  V4.S4, V1.S4, V3.S4 // p = p3 + p·r
	WORD   $0x4EBE1FC4 // ORR V4 = p4
	VFMLA  V3.S4, V1.S4, V4.S4 // p = p4 + p·r
	WORD   $0x4EB51EA3 // ORR V3 = p5(=half)
	VFMLA  V4.S4, V1.S4, V3.S4 // p = p5 + p·r
	WORD   $0x6E21DC24 // FMUL V4.4S, V1.4S, V1.4S   (r² = r·r)
	WORD   $0x4E34D421 // FADD V1.4S, V1.4S, V20.4S  (q = 1 + r)
	VFMLA  V3.S4, V4.S4, V1.S4 // e = q + p·r²
	WORD   $0x4EA1B842 // FCVTZS V2.4S, V2.4S        (ni = int(n))
	VADD   V12.S4, V2.S4, V2.S4 // ni += 127
	VSHL   $23, V2.S4, V2.S4    // 2^n bits
	WORD   $0x6E22DC21 // FMUL V1.4S, V1.4S, V2.4S   (e *= 2^n)
	VLD1   (R3), [V2.S4, V3.S4] // V2 = pdfMin, V3 = invSqrt2Pi
	WORD   $0x6EA2E424 // FCMGT V4.4S, V1.4S, V2.4S  (mask e > pdfMin)
	WORD   $0x4E241C21 // AND V1.16B, V1.16B, V4.16B (e &= mask)
	WORD   $0x4EB41E82 // ORR V2 = one
	VFMLS  V1.S4, V5.S4, V2.S4 // E = 1 − e·P
	WORD   $0x4E361C04 // AND V4.16B, V0.16B, V22.16B (sign bit of x)
	WORD   $0x4EA41C42 // ORR V2.16B, V2.16B, V4.16B  (erf = sign|E)
	WORD   $0x4EB51EA4 // ORR V4 = half
	VFMLA  V2.S4, V21.S4, V4.S4 // Φ = 0.5 + 0.5·erf
	WORD   $0x6E23DC21 // FMUL V1.4S, V1.4S, V3.4S   (pdf = e·invSqrt2Pi)
	VFMLA  V1.S4, V0.S4, V4.S4 // d = Φ + pdf·x
	VLD1.P 16(R6), [V0.S4]     // g (x dead)
	WORD   $0x6E24DC00 // FMUL V0.4S, V0.4S, V4.4S   (out = g·d)
	VST1.P [V0.S4], 16(R1)

gradDone:
	RET

// F64 SiLU constants, each duplicated for one 2-lane NEON vector. The exp
// polynomial is the same degree-13 FMA chain as expF64poly/vexp_amd64.go.
DATA siluF64Consts<>+0(SB)/8, $0x3FF71547652B82FE
DATA siluF64Consts<>+8(SB)/8, $0x3FF71547652B82FE  // V12 log2(e)
DATA siluF64Consts<>+16(SB)/8, $0x3FE62E42FEE00000
DATA siluF64Consts<>+24(SB)/8, $0x3FE62E42FEE00000 // V13 ln2 high
DATA siluF64Consts<>+32(SB)/8, $0x3DEA39EF35793C76
DATA siluF64Consts<>+40(SB)/8, $0x3DEA39EF35793C76 // V14 ln2 low
DATA siluF64Consts<>+48(SB)/8, $0x3FF0000000000000
DATA siluF64Consts<>+56(SB)/8, $0x3FF0000000000000 // V15 one
DATA siluF64Consts<>+64(SB)/8, $0xC086200000000000
DATA siluF64Consts<>+72(SB)/8, $0xC086200000000000 // V16 -708 clamp
DATA siluF64Consts<>+80(SB)/8, $0x00000000000003FF
DATA siluF64Consts<>+88(SB)/8, $0x00000000000003FF // V17 exponent bias
DATA siluF64Consts<>+96(SB)/8, $0x3DE6124613A86D09
DATA siluF64Consts<>+104(SB)/8, $0x3DE6124613A86D09 // V18 c13
DATA siluF64Consts<>+112(SB)/8, $0x3E21EED8EFF8D898
DATA siluF64Consts<>+120(SB)/8, $0x3E21EED8EFF8D898 // V19 c12
DATA siluF64Consts<>+128(SB)/8, $0x3E5AE64567F544E4
DATA siluF64Consts<>+136(SB)/8, $0x3E5AE64567F544E4 // V20 c11
DATA siluF64Consts<>+144(SB)/8, $0x3E927E4FB7789F5C
DATA siluF64Consts<>+152(SB)/8, $0x3E927E4FB7789F5C // V21 c10
DATA siluF64Consts<>+160(SB)/8, $0x3EC71DE3A556C734
DATA siluF64Consts<>+168(SB)/8, $0x3EC71DE3A556C734 // V22 c9
DATA siluF64Consts<>+176(SB)/8, $0x3EFA01A01A01A01A
DATA siluF64Consts<>+184(SB)/8, $0x3EFA01A01A01A01A // V23 c8
DATA siluF64Consts<>+192(SB)/8, $0x3F2A01A01A01A01A
DATA siluF64Consts<>+200(SB)/8, $0x3F2A01A01A01A01A // V24 c7
DATA siluF64Consts<>+208(SB)/8, $0x3F56C16C16C16C17
DATA siluF64Consts<>+216(SB)/8, $0x3F56C16C16C16C17 // V25 c6
DATA siluF64Consts<>+224(SB)/8, $0x3F81111111111111
DATA siluF64Consts<>+232(SB)/8, $0x3F81111111111111 // V26 c5
DATA siluF64Consts<>+240(SB)/8, $0x3FA5555555555554
DATA siluF64Consts<>+248(SB)/8, $0x3FA5555555555554 // V27 c4
DATA siluF64Consts<>+256(SB)/8, $0x3FC5555555555555
DATA siluF64Consts<>+264(SB)/8, $0x3FC5555555555555 // V28 c3
DATA siluF64Consts<>+272(SB)/8, $0x3FE0000000000000
DATA siluF64Consts<>+280(SB)/8, $0x3FE0000000000000 // V29 c2
GLOBL siluF64Consts<>(SB), RODATA|NOPTR, $288

// func vsiluPairsNeonF64(dst, src *float64, pairs int)
//
// Two f64 lanes per vector, with two independent vectors interleaved so the
// degree-13 FMA chains overlap. WORD encodings were generated with Apple clang
// and verified through go tool objdump.
TEXT ·vsiluPairsNeonF64(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R1
	MOVD src+8(FP), R0
	MOVD pairs+16(FP), R2
	MOVD $siluF64Consts<>(SB), R3
	VLD1.P 64(R3), [V12.D2, V13.D2, V14.D2, V15.D2]
	VLD1.P 64(R3), [V16.D2, V17.D2, V18.D2, V19.D2]
	VLD1.P 64(R3), [V20.D2, V21.D2, V22.D2, V23.D2]
	VLD1.P 64(R3), [V24.D2, V25.D2, V26.D2, V27.D2]
	VLD1 (R3), [V28.D2, V29.D2]
	WORD $0x6EE0FA1E // FNEG V30.2D, V16.2D: +708 cutoff
	LSR $1, R2, R4
	AND $1, R2, R5
	CBZ R4, siluF64Odd

siluF64Loop2:
	VLD1.P 16(R0), [V0.D2]
	VLD1.P 16(R0), [V6.D2]
	WORD $0x4EE0F801 // FABS V1.2D, V0.2D
	WORD $0x4EE0F8C7 // FABS V7.2D, V6.2D
	WORD $0x6E61E7C5 // FCMGE V5.2D, V30.2D, V1.2D: |x| <= 708
	WORD $0x6E67E7CB // FCMGE V11.2D, V30.2D, V7.2D
	WORD $0x6EE0F821 // FNEG V1.2D, V1.2D
	WORD $0x6EE0F8E7 // FNEG V7.2D, V7.2D
	WORD $0x4E70F421 // FMAX V1.2D, V1.2D, V16.2D
	WORD $0x4E70F4E7 // FMAX V7.2D, V7.2D, V16.2D
	WORD $0x6E6CDC22 // FMUL V2.2D, V1.2D, V12.2D
	WORD $0x6E6CDCE8 // FMUL V8.2D, V7.2D, V12.2D
	WORD $0x4E618842 // FRINTN V2.2D, V2.2D
	WORD $0x4E618908 // FRINTN V8.2D, V8.2D
	WORD $0x4EEDCC41 // FMLS V1.2D, V2.2D, V13.2D
	WORD $0x4EEDCD07 // FMLS V7.2D, V8.2D, V13.2D
	WORD $0x4EEECC41 // FMLS V1.2D, V2.2D, V14.2D
	WORD $0x4EEECD07 // FMLS V7.2D, V8.2D, V14.2D

	WORD $0x4EB21E43 // MOV V3.16B, V18.16B
	WORD $0x4EB21E49 // MOV V9.16B, V18.16B
	WORD $0x4EB31E64 // MOV V4.16B, V19.16B
	WORD $0x4EB31E6A // MOV V10.16B, V19.16B
	WORD $0x4E61CC64 // FMLA V4.2D, V3.2D, V1.2D
	WORD $0x4E67CD2A // FMLA V10.2D, V9.2D, V7.2D
	WORD $0x4EB41E83 // MOV V3.16B, V20.16B
	WORD $0x4EB41E89 // MOV V9.16B, V20.16B
	WORD $0x4E61CC83 // FMLA V3.2D, V4.2D, V1.2D
	WORD $0x4E67CD49 // FMLA V9.2D, V10.2D, V7.2D
	WORD $0x4EB51EA4 // MOV V4.16B, V21.16B
	WORD $0x4EB51EAA // MOV V10.16B, V21.16B
	WORD $0x4E61CC64
	WORD $0x4E67CD2A
	WORD $0x4EB61EC3 // MOV V3.16B, V22.16B
	WORD $0x4EB61EC9 // MOV V9.16B, V22.16B
	WORD $0x4E61CC83
	WORD $0x4E67CD49
	WORD $0x4EB71EE4 // MOV V4.16B, V23.16B
	WORD $0x4EB71EEA // MOV V10.16B, V23.16B
	WORD $0x4E61CC64
	WORD $0x4E67CD2A
	WORD $0x4EB81F03 // MOV V3.16B, V24.16B
	WORD $0x4EB81F09 // MOV V9.16B, V24.16B
	WORD $0x4E61CC83
	WORD $0x4E67CD49
	WORD $0x4EB91F24 // MOV V4.16B, V25.16B
	WORD $0x4EB91F2A // MOV V10.16B, V25.16B
	WORD $0x4E61CC64
	WORD $0x4E67CD2A
	WORD $0x4EBA1F43 // MOV V3.16B, V26.16B
	WORD $0x4EBA1F49 // MOV V9.16B, V26.16B
	WORD $0x4E61CC83
	WORD $0x4E67CD49
	WORD $0x4EBB1F64 // MOV V4.16B, V27.16B
	WORD $0x4EBB1F6A // MOV V10.16B, V27.16B
	WORD $0x4E61CC64
	WORD $0x4E67CD2A
	WORD $0x4EBC1F83 // MOV V3.16B, V28.16B
	WORD $0x4EBC1F89 // MOV V9.16B, V28.16B
	WORD $0x4E61CC83
	WORD $0x4E67CD49
	WORD $0x4EBD1FA4 // MOV V4.16B, V29.16B
	WORD $0x4EBD1FAA // MOV V10.16B, V29.16B
	WORD $0x4E61CC64
	WORD $0x4E67CD2A
	WORD $0x4EAF1DE3 // MOV V3.16B, V15.16B
	WORD $0x4EAF1DE9 // MOV V9.16B, V15.16B
	WORD $0x4E61CC83
	WORD $0x4E67CD49
	WORD $0x4EAF1DE4 // MOV V4.16B, V15.16B
	WORD $0x4EAF1DEA // MOV V10.16B, V15.16B
	WORD $0x4E61CC64
	WORD $0x4E67CD2A

	WORD $0x4EE1B842 // FCVTZS V2.2D, V2.2D
	WORD $0x4EE1B908 // FCVTZS V8.2D, V8.2D
	WORD $0x4EF18442 // ADD V2.2D, V2.2D, V17.2D
	WORD $0x4EF18508 // ADD V8.2D, V8.2D, V17.2D
	WORD $0x4F745442 // SHL V2.2D, V2.2D, 52
	WORD $0x4F745508 // SHL V8.2D, V8.2D, 52
	WORD $0x6E62DC81 // FMUL V1.2D, V4.2D, V2.2D
	WORD $0x6E68DD47 // FMUL V7.2D, V10.2D, V8.2D
	WORD $0x4E251C21 // AND V1.16B, V1.16B, V5.16B: true underflow zero
	WORD $0x4E2B1CE7 // AND V7.16B, V7.16B, V11.16B
	WORD $0x6EE0C802 // FCMGE V2.2D, V0.2D, 0
	WORD $0x6EE0C8C8 // FCMGE V8.2D, V6.2D, 0
	WORD $0x6E611DE2 // BSL V2.16B, V15.16B, V1.16B
	WORD $0x6E671DE8 // BSL V8.16B, V15.16B, V7.16B
	WORD $0x4E6FD423 // FADD V3.2D, V1.2D, V15.2D
	WORD $0x4E6FD4E9 // FADD V9.2D, V7.2D, V15.2D
	WORD $0x6E62DC00 // FMUL V0.2D, V0.2D, V2.2D
	WORD $0x6E68DCC6 // FMUL V6.2D, V6.2D, V8.2D
	WORD $0x6E63FC00 // FDIV V0.2D, V0.2D, V3.2D
	WORD $0x6E69FCC6 // FDIV V6.2D, V6.2D, V9.2D
	VST1.P [V0.D2], 16(R1)
	VST1.P [V6.D2], 16(R1)
	SUBS $1, R4, R4
	BNE siluF64Loop2

siluF64Odd:
	CBZ R5, siluF64Done
	VLD1.P 16(R0), [V0.D2]
	WORD $0x4EE0F801
	WORD $0x6E61E7C5
	WORD $0x6EE0F821
	WORD $0x4E70F421
	WORD $0x6E6CDC22
	WORD $0x4E618842
	WORD $0x4EEDCC41
	WORD $0x4EEECC41
	WORD $0x4EB21E43
	WORD $0x4EB31E64
	WORD $0x4E61CC64
	WORD $0x4EB41E83
	WORD $0x4E61CC83
	WORD $0x4EB51EA4
	WORD $0x4E61CC64
	WORD $0x4EB61EC3
	WORD $0x4E61CC83
	WORD $0x4EB71EE4
	WORD $0x4E61CC64
	WORD $0x4EB81F03
	WORD $0x4E61CC83
	WORD $0x4EB91F24
	WORD $0x4E61CC64
	WORD $0x4EBA1F43
	WORD $0x4E61CC83
	WORD $0x4EBB1F64
	WORD $0x4E61CC64
	WORD $0x4EBC1F83
	WORD $0x4E61CC83
	WORD $0x4EBD1FA4
	WORD $0x4E61CC64
	WORD $0x4EAF1DE3
	WORD $0x4E61CC83
	WORD $0x4EAF1DE4
	WORD $0x4E61CC64
	WORD $0x4EE1B842
	WORD $0x4EF18442
	WORD $0x4F745442
	WORD $0x6E62DC81
	WORD $0x4E251C21
	WORD $0x6EE0C802
	WORD $0x6E611DE2
	WORD $0x4E6FD423
	WORD $0x6E62DC00
	WORD $0x6E63FC00
	VST1.P [V0.D2], 16(R1)

siluF64Done:
	RET

// func vsiluQuadsNeonF32(dst, src *float32, quads int)
//
// 4-wide f32 SiLU (vexp.go, §T665): dst[i] = x·σ(x) for x = src[i],
// i in [0, 4*quads), via the branch-free stable split — z = exp(−|x|) through
// the SAME Cephes exp reduction as the kernels above (lo clamp only: −|x| ≤ 0
// never overflows 2^n), z masked to a true 0 at/below the underflow clamp
// (FCMGT+AND on pdfMin, exactly the vgeluGrad trick — restores silu(−Inf) =
// −Inf·0 = NaN), num = (x ≥ 0 ? 1 : z) selected with FCMGE-zero + BSL, then
// out = x·num/(1+z). Constants are the geluConsts block above — the exp
// registers (V12, V20/V21, V23..V31) are shared; the erf-only entries
// (V13/V14, V16..V19, V22) are loaded but unused, and V15 (erf a1) is
// overwritten once with pdfMin from the tail, so the mask constant stays
// resident (unlike vgeluGrad there are registers to spare). Same
// two-quads-per-pass structure (A: V0..V4, B: V6..V10); WORD encodings
// objdump-verified like the kernels above.
//
// Register map: R0 src, R1 dst, R2 quads, R3 consts, R4 pairs, R5 odd;
// A: V0 x, V1 z, V2 n/sel/num, V3 r/den, V4 poly/mask; B: V6..V10 mirrored.
TEXT ·vsiluQuadsNeonF32(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R1
	MOVD src+8(FP), R0
	MOVD quads+16(FP), R2
	MOVD $geluConsts<>(SB), R3
	VLD1.P 64(R3), [V12.S4, V13.S4, V14.S4, V15.S4]
	VLD1.P 64(R3), [V16.S4, V17.S4, V18.S4, V19.S4]
	VLD1.P 64(R3), [V20.S4, V21.S4, V22.S4, V23.S4]
	VLD1.P 64(R3), [V24.S4, V25.S4, V26.S4, V27.S4]
	VLD1.P 64(R3), [V28.S4, V29.S4, V30.S4, V31.S4]
	VLD1   (R3), [V15.S4] // V15 = pdfMin (overwrites unused erf a1)
	LSR $1, R2, R4 // pairs of quads
	AND $1, R2, R5 // odd quad
	CBZ R4, siluOdd

siluLoop2:
	VLD1.P 16(R0), [V0.S4]
	VLD1.P 16(R0), [V6.S4]

	// w = clamp(−|x|), both quads.
	WORD $0x4EA0F801 // FABS V1.4S, V0.4S           (A: a = |x|)
	WORD $0x4EA0F8C7 // FABS V7.4S, V6.4S           (B: a = |x|)
	WORD $0x6EA0F821 // FNEG V1.4S, V1.4S           (A: w = −a)
	WORD $0x6EA0F8E7 // FNEG V7.4S, V7.4S           (B: w = −a)
	WORD $0x4E3FF421 // FMAX V1.4S, V1.4S, V31.4S   (A: clamp lo)
	WORD $0x4E3FF4E7 // FMAX V7.4S, V7.4S, V31.4S   (B: clamp lo)

	// z = exp(w) — the vexp reduction (identical to the kernels above).
	WORD  $0x6E37DC22 // FMUL V2.4S, V1.4S, V23.4S  (A: z = w·log2e)
	WORD  $0x6E37DCE8 // FMUL V8.4S, V7.4S, V23.4S  (B: z = w·log2e)
	WORD  $0x4E218842 // FRINTN V2.4S, V2.4S        (A: n = rint(z))
	WORD  $0x4E218908 // FRINTN V8.4S, V8.4S        (B: n = rint(z))
	VFMLS V2.S4, V24.S4, V1.S4 // A: r -= n·ln2hi
	VFMLS V8.S4, V24.S4, V7.S4 // B: r -= n·ln2hi
	VFMLS V2.S4, V25.S4, V1.S4 // A: r -= n·ln2lo
	VFMLS V8.S4, V25.S4, V7.S4 // B: r -= n·ln2lo
	WORD  $0x4EBB1F63 // ORR V3 = p1                (A)
	WORD  $0x4EBB1F69 // ORR V9 = p1                (B)
	VFMLA V26.S4, V1.S4, V3.S4 // A: p = p1 + p0·r
	VFMLA V26.S4, V7.S4, V9.S4 // B: p = p1 + p0·r
	WORD  $0x4EBC1F84 // ORR V4 = p2                (A)
	WORD  $0x4EBC1F8A // ORR V10 = p2               (B)
	VFMLA V3.S4, V1.S4, V4.S4  // A: p = p2 + p·r
	VFMLA V9.S4, V7.S4, V10.S4 // B: p = p2 + p·r
	WORD  $0x4EBD1FA3 // ORR V3 = p3                (A)
	WORD  $0x4EBD1FA9 // ORR V9 = p3                (B)
	VFMLA V4.S4, V1.S4, V3.S4  // A: p = p3 + p·r
	VFMLA V10.S4, V7.S4, V9.S4 // B: p = p3 + p·r
	WORD  $0x4EBE1FC4 // ORR V4 = p4                (A)
	WORD  $0x4EBE1FCA // ORR V10 = p4               (B)
	VFMLA V3.S4, V1.S4, V4.S4  // A: p = p4 + p·r
	VFMLA V9.S4, V7.S4, V10.S4 // B: p = p4 + p·r
	WORD  $0x4EB51EA3 // ORR V3 = p5(=half)         (A)
	WORD  $0x4EB51EA9 // ORR V9 = p5(=half)         (B)
	VFMLA V4.S4, V1.S4, V3.S4  // A: p = p5 + p·r
	VFMLA V10.S4, V7.S4, V9.S4 // B: p = p5 + p·r
	WORD  $0x6E21DC24 // FMUL V4.4S, V1.4S, V1.4S   (A: r² = r·r)
	WORD  $0x6E27DCEA // FMUL V10.4S, V7.4S, V7.4S  (B: r² = r·r)
	WORD  $0x4E34D421 // FADD V1.4S, V1.4S, V20.4S  (A: q = 1 + r)
	WORD  $0x4E34D4E7 // FADD V7.4S, V7.4S, V20.4S  (B: q = 1 + r)
	VFMLA V3.S4, V4.S4, V1.S4  // A: z = q + p·r²
	VFMLA V9.S4, V10.S4, V7.S4 // B: z = q + p·r²
	WORD  $0x4EA1B842 // FCVTZS V2.4S, V2.4S        (A: ni = int(n))
	WORD  $0x4EA1B908 // FCVTZS V8.4S, V8.4S        (B: ni = int(n))
	VADD  V12.S4, V2.S4, V2.S4 // A: ni += 127
	VADD  V12.S4, V8.S4, V8.S4 // B: ni += 127
	VSHL  $23, V2.S4, V2.S4    // A: 2^n bits
	VSHL  $23, V8.S4, V8.S4    // B: 2^n bits
	WORD  $0x6E22DC21 // FMUL V1.4S, V1.4S, V2.4S   (A: z *= 2^n)
	WORD  $0x6E28DCE7 // FMUL V7.4S, V7.4S, V8.4S   (B: z *= 2^n)

	// SiLU tail: zero sub-pdfMin z, num = (x≥0 ? 1 : z), out = x·num/(1+z).
	WORD $0x6EAFE424 // FCMGT V4.4S, V1.4S, V15.4S  (A: mask z > pdfMin)
	WORD $0x6EAFE4EA // FCMGT V10.4S, V7.4S, V15.4S (B: mask z > pdfMin)
	WORD $0x4E241C21 // AND V1.16B, V1.16B, V4.16B  (A: z &= mask)
	WORD $0x4E2A1CE7 // AND V7.16B, V7.16B, V10.16B (B: z &= mask)
	WORD $0x6EA0C802 // FCMGE V2.4S, V0.4S, #0.0    (A: sel = x ≥ 0)
	WORD $0x6EA0C8C8 // FCMGE V8.4S, V6.4S, #0.0    (B: sel = x ≥ 0)
	WORD $0x6E611E82 // BSL V2.16B, V20.16B, V1.16B (A: num = sel ? 1 : z)
	WORD $0x6E671E88 // BSL V8.16B, V20.16B, V7.16B (B: num = sel ? 1 : z)
	WORD $0x4E34D423 // FADD V3.4S, V1.4S, V20.4S   (A: den = 1 + z)
	WORD $0x4E34D4E9 // FADD V9.4S, V7.4S, V20.4S   (B: den = 1 + z)
	WORD $0x6E22DC00 // FMUL V0.4S, V0.4S, V2.4S    (A: x·num)
	WORD $0x6E28DCC6 // FMUL V6.4S, V6.4S, V8.4S    (B: x·num)
	WORD $0x6E23FC00 // FDIV V0.4S, V0.4S, V3.4S    (A: out = x·num/den)
	WORD $0x6E29FCC6 // FDIV V6.4S, V6.4S, V9.4S    (B: out = x·num/den)

	VST1.P [V0.S4], 16(R1)
	VST1.P [V6.S4], 16(R1)

	SUBS $1, R4, R4
	BNE  siluLoop2

siluOdd:
	CBZ R5, siluDone

	VLD1.P 16(R0), [V0.S4]
	WORD   $0x4EA0F801 // FABS V1.4S, V0.4S         (a = |x|)
	WORD   $0x6EA0F821 // FNEG V1.4S, V1.4S         (w = −a)
	WORD   $0x4E3FF421 // FMAX V1.4S, V1.4S, V31.4S (clamp lo)
	WORD   $0x6E37DC22 // FMUL V2.4S, V1.4S, V23.4S (z = w·log2e)
	WORD   $0x4E218842 // FRINTN V2.4S, V2.4S       (n = rint(z))
	VFMLS  V2.S4, V24.S4, V1.S4 // r -= n·ln2hi
	VFMLS  V2.S4, V25.S4, V1.S4 // r -= n·ln2lo
	WORD   $0x4EBB1F63 // ORR V3 = p1
	VFMLA  V26.S4, V1.S4, V3.S4 // p = p1 + p0·r
	WORD   $0x4EBC1F84 // ORR V4 = p2
	VFMLA  V3.S4, V1.S4, V4.S4 // p = p2 + p·r
	WORD   $0x4EBD1FA3 // ORR V3 = p3
	VFMLA  V4.S4, V1.S4, V3.S4 // p = p3 + p·r
	WORD   $0x4EBE1FC4 // ORR V4 = p4
	VFMLA  V3.S4, V1.S4, V4.S4 // p = p4 + p·r
	WORD   $0x4EB51EA3 // ORR V3 = p5(=half)
	VFMLA  V4.S4, V1.S4, V3.S4 // p = p5 + p·r
	WORD   $0x6E21DC24 // FMUL V4.4S, V1.4S, V1.4S  (r² = r·r)
	WORD   $0x4E34D421 // FADD V1.4S, V1.4S, V20.4S (q = 1 + r)
	VFMLA  V3.S4, V4.S4, V1.S4 // z = q + p·r²
	WORD   $0x4EA1B842 // FCVTZS V2.4S, V2.4S       (ni = int(n))
	VADD   V12.S4, V2.S4, V2.S4 // ni += 127
	VSHL   $23, V2.S4, V2.S4    // 2^n bits
	WORD   $0x6E22DC21 // FMUL V1.4S, V1.4S, V2.4S  (z *= 2^n)
	WORD   $0x6EAFE424 // FCMGT V4.4S, V1.4S, V15.4S (mask z > pdfMin)
	WORD   $0x4E241C21 // AND V1.16B, V1.16B, V4.16B (z &= mask)
	WORD   $0x6EA0C802 // FCMGE V2.4S, V0.4S, #0.0   (sel = x ≥ 0)
	WORD   $0x6E611E82 // BSL V2.16B, V20.16B, V1.16B (num = sel ? 1 : z)
	WORD   $0x4E34D423 // FADD V3.4S, V1.4S, V20.4S  (den = 1 + z)
	WORD   $0x6E22DC00 // FMUL V0.4S, V0.4S, V2.4S   (x·num)
	WORD   $0x6E23FC00 // FDIV V0.4S, V0.4S, V3.4S   (out = x·num/den)
	VST1.P [V0.S4], 16(R1)

siluDone:
	RET

// func vsiluGradQuadsNeonF32(dst, x, grad *float32, quads int)
//
// 4-wide f32 SiLU BACKWARD (vexp.go, §T665): dst[i] = g[i]·silu'(x[i]),
// silu'(x) = (num·(1+z) + x·z)/(1+z)², z = exp(−|x|), num = (x ≥ 0 ? 1 : z)
// — the single-division fold of ref's σ(x)·(1+x·(1−σ(x))) (§T362; derivation
// in vexp.go). Same pipeline as vsiluQuadsNeonF32 above (one exp, pdfMin
// mask for exact ±Inf semantics, FCMGE-zero + BSL select), plus the g
// multiply; g loads reuse V0/V6 after x dies. Constants: geluConsts as in
// vsilu (V15 = pdfMin). Register map: R0 x, R1 dst, R6 g, R2 quads, R3
// consts, R4 pairs, R5 odd; A: V0..V4, B: V6..V10.
TEXT ·vsiluGradQuadsNeonF32(SB), NOSPLIT, $0-32
	MOVD dst+0(FP), R1
	MOVD x+8(FP), R0
	MOVD grad+16(FP), R6
	MOVD quads+24(FP), R2
	MOVD $geluConsts<>(SB), R3
	VLD1.P 64(R3), [V12.S4, V13.S4, V14.S4, V15.S4]
	VLD1.P 64(R3), [V16.S4, V17.S4, V18.S4, V19.S4]
	VLD1.P 64(R3), [V20.S4, V21.S4, V22.S4, V23.S4]
	VLD1.P 64(R3), [V24.S4, V25.S4, V26.S4, V27.S4]
	VLD1.P 64(R3), [V28.S4, V29.S4, V30.S4, V31.S4]
	VLD1   (R3), [V15.S4] // V15 = pdfMin (overwrites unused erf a1)
	LSR $1, R2, R4 // pairs of quads
	AND $1, R2, R5 // odd quad
	CBZ R4, siluGradOdd

siluGradLoop2:
	VLD1.P 16(R0), [V0.S4]
	VLD1.P 16(R0), [V6.S4]

	// w = clamp(−|x|), both quads.
	WORD $0x4EA0F801 // FABS V1.4S, V0.4S           (A: a = |x|)
	WORD $0x4EA0F8C7 // FABS V7.4S, V6.4S           (B: a = |x|)
	WORD $0x6EA0F821 // FNEG V1.4S, V1.4S           (A: w = −a)
	WORD $0x6EA0F8E7 // FNEG V7.4S, V7.4S           (B: w = −a)
	WORD $0x4E3FF421 // FMAX V1.4S, V1.4S, V31.4S   (A: clamp lo)
	WORD $0x4E3FF4E7 // FMAX V7.4S, V7.4S, V31.4S   (B: clamp lo)

	// z = exp(w) — the vexp reduction (identical to the kernels above).
	WORD  $0x6E37DC22 // FMUL V2.4S, V1.4S, V23.4S  (A: z = w·log2e)
	WORD  $0x6E37DCE8 // FMUL V8.4S, V7.4S, V23.4S  (B: z = w·log2e)
	WORD  $0x4E218842 // FRINTN V2.4S, V2.4S        (A: n = rint(z))
	WORD  $0x4E218908 // FRINTN V8.4S, V8.4S        (B: n = rint(z))
	VFMLS V2.S4, V24.S4, V1.S4 // A: r -= n·ln2hi
	VFMLS V8.S4, V24.S4, V7.S4 // B: r -= n·ln2hi
	VFMLS V2.S4, V25.S4, V1.S4 // A: r -= n·ln2lo
	VFMLS V8.S4, V25.S4, V7.S4 // B: r -= n·ln2lo
	WORD  $0x4EBB1F63 // ORR V3 = p1                (A)
	WORD  $0x4EBB1F69 // ORR V9 = p1                (B)
	VFMLA V26.S4, V1.S4, V3.S4 // A: p = p1 + p0·r
	VFMLA V26.S4, V7.S4, V9.S4 // B: p = p1 + p0·r
	WORD  $0x4EBC1F84 // ORR V4 = p2                (A)
	WORD  $0x4EBC1F8A // ORR V10 = p2               (B)
	VFMLA V3.S4, V1.S4, V4.S4  // A: p = p2 + p·r
	VFMLA V9.S4, V7.S4, V10.S4 // B: p = p2 + p·r
	WORD  $0x4EBD1FA3 // ORR V3 = p3                (A)
	WORD  $0x4EBD1FA9 // ORR V9 = p3                (B)
	VFMLA V4.S4, V1.S4, V3.S4  // A: p = p3 + p·r
	VFMLA V10.S4, V7.S4, V9.S4 // B: p = p3 + p·r
	WORD  $0x4EBE1FC4 // ORR V4 = p4                (A)
	WORD  $0x4EBE1FCA // ORR V10 = p4               (B)
	VFMLA V3.S4, V1.S4, V4.S4  // A: p = p4 + p·r
	VFMLA V9.S4, V7.S4, V10.S4 // B: p = p4 + p·r
	WORD  $0x4EB51EA3 // ORR V3 = p5(=half)         (A)
	WORD  $0x4EB51EA9 // ORR V9 = p5(=half)         (B)
	VFMLA V4.S4, V1.S4, V3.S4  // A: p = p5 + p·r
	VFMLA V10.S4, V7.S4, V9.S4 // B: p = p5 + p·r
	WORD  $0x6E21DC24 // FMUL V4.4S, V1.4S, V1.4S   (A: r² = r·r)
	WORD  $0x6E27DCEA // FMUL V10.4S, V7.4S, V7.4S  (B: r² = r·r)
	WORD  $0x4E34D421 // FADD V1.4S, V1.4S, V20.4S  (A: q = 1 + r)
	WORD  $0x4E34D4E7 // FADD V7.4S, V7.4S, V20.4S  (B: q = 1 + r)
	VFMLA V3.S4, V4.S4, V1.S4  // A: z = q + p·r²
	VFMLA V9.S4, V10.S4, V7.S4 // B: z = q + p·r²
	WORD  $0x4EA1B842 // FCVTZS V2.4S, V2.4S        (A: ni = int(n))
	WORD  $0x4EA1B908 // FCVTZS V8.4S, V8.4S        (B: ni = int(n))
	VADD  V12.S4, V2.S4, V2.S4 // A: ni += 127
	VADD  V12.S4, V8.S4, V8.S4 // B: ni += 127
	VSHL  $23, V2.S4, V2.S4    // A: 2^n bits
	VSHL  $23, V8.S4, V8.S4    // B: 2^n bits
	WORD  $0x6E22DC21 // FMUL V1.4S, V1.4S, V2.4S   (A: z *= 2^n)
	WORD  $0x6E28DCE7 // FMUL V7.4S, V7.4S, V8.4S   (B: z *= 2^n)

	// Grad tail: zero sub-pdfMin z, num = (x≥0 ? 1 : z), den = 1+z,
	// d = (num·den + x·z)/den², out = g·d.
	WORD  $0x6EAFE424 // FCMGT V4.4S, V1.4S, V15.4S  (A: mask z > pdfMin)
	WORD  $0x6EAFE4EA // FCMGT V10.4S, V7.4S, V15.4S (B: mask z > pdfMin)
	WORD  $0x4E241C21 // AND V1.16B, V1.16B, V4.16B  (A: z &= mask)
	WORD  $0x4E2A1CE7 // AND V7.16B, V7.16B, V10.16B (B: z &= mask)
	WORD  $0x6EA0C802 // FCMGE V2.4S, V0.4S, #0.0    (A: sel = x ≥ 0)
	WORD  $0x6EA0C8C8 // FCMGE V8.4S, V6.4S, #0.0    (B: sel = x ≥ 0)
	WORD  $0x6E611E82 // BSL V2.16B, V20.16B, V1.16B (A: num = sel ? 1 : z)
	WORD  $0x6E671E88 // BSL V8.16B, V20.16B, V7.16B (B: num = sel ? 1 : z)
	WORD  $0x4E34D423 // FADD V3.4S, V1.4S, V20.4S   (A: den = 1 + z)
	WORD  $0x4E34D4E9 // FADD V9.4S, V7.4S, V20.4S   (B: den = 1 + z)
	WORD  $0x6E23DC42 // FMUL V2.4S, V2.4S, V3.4S    (A: num·den)
	WORD  $0x6E29DD08 // FMUL V8.4S, V8.4S, V9.4S    (B: num·den)
	VFMLA V0.S4, V1.S4, V2.S4  // A: += x·z
	VFMLA V6.S4, V7.S4, V8.S4  // B: += x·z
	WORD  $0x6E23DC63 // FMUL V3.4S, V3.4S, V3.4S    (A: den²)
	WORD  $0x6E29DD29 // FMUL V9.4S, V9.4S, V9.4S    (B: den²)
	WORD  $0x6E23FC42 // FDIV V2.4S, V2.4S, V3.4S    (A: d = ·/den²)
	WORD  $0x6E29FD08 // FDIV V8.4S, V8.4S, V9.4S    (B: d = ·/den²)
	VLD1.P 16(R6), [V0.S4] // A: g (x dead)
	VLD1.P 16(R6), [V6.S4] // B: g (x dead)
	WORD  $0x6E22DC00 // FMUL V0.4S, V0.4S, V2.4S    (A: out = g·d)
	WORD  $0x6E28DCC6 // FMUL V6.4S, V6.4S, V8.4S    (B: out = g·d)

	VST1.P [V0.S4], 16(R1)
	VST1.P [V6.S4], 16(R1)

	SUBS $1, R4, R4
	BNE  siluGradLoop2

siluGradOdd:
	CBZ R5, siluGradDone

	VLD1.P 16(R0), [V0.S4]
	WORD   $0x4EA0F801 // FABS V1.4S, V0.4S         (a = |x|)
	WORD   $0x6EA0F821 // FNEG V1.4S, V1.4S         (w = −a)
	WORD   $0x4E3FF421 // FMAX V1.4S, V1.4S, V31.4S (clamp lo)
	WORD   $0x6E37DC22 // FMUL V2.4S, V1.4S, V23.4S (z = w·log2e)
	WORD   $0x4E218842 // FRINTN V2.4S, V2.4S       (n = rint(z))
	VFMLS  V2.S4, V24.S4, V1.S4 // r -= n·ln2hi
	VFMLS  V2.S4, V25.S4, V1.S4 // r -= n·ln2lo
	WORD   $0x4EBB1F63 // ORR V3 = p1
	VFMLA  V26.S4, V1.S4, V3.S4 // p = p1 + p0·r
	WORD   $0x4EBC1F84 // ORR V4 = p2
	VFMLA  V3.S4, V1.S4, V4.S4 // p = p2 + p·r
	WORD   $0x4EBD1FA3 // ORR V3 = p3
	VFMLA  V4.S4, V1.S4, V3.S4 // p = p3 + p·r
	WORD   $0x4EBE1FC4 // ORR V4 = p4
	VFMLA  V3.S4, V1.S4, V4.S4 // p = p4 + p·r
	WORD   $0x4EB51EA3 // ORR V3 = p5(=half)
	VFMLA  V4.S4, V1.S4, V3.S4 // p = p5 + p·r
	WORD   $0x6E21DC24 // FMUL V4.4S, V1.4S, V1.4S  (r² = r·r)
	WORD   $0x4E34D421 // FADD V1.4S, V1.4S, V20.4S (q = 1 + r)
	VFMLA  V3.S4, V4.S4, V1.S4 // z = q + p·r²
	WORD   $0x4EA1B842 // FCVTZS V2.4S, V2.4S       (ni = int(n))
	VADD   V12.S4, V2.S4, V2.S4 // ni += 127
	VSHL   $23, V2.S4, V2.S4    // 2^n bits
	WORD   $0x6E22DC21 // FMUL V1.4S, V1.4S, V2.4S  (z *= 2^n)
	WORD   $0x6EAFE424 // FCMGT V4.4S, V1.4S, V15.4S (mask z > pdfMin)
	WORD   $0x4E241C21 // AND V1.16B, V1.16B, V4.16B (z &= mask)
	WORD   $0x6EA0C802 // FCMGE V2.4S, V0.4S, #0.0   (sel = x ≥ 0)
	WORD   $0x6E611E82 // BSL V2.16B, V20.16B, V1.16B (num = sel ? 1 : z)
	WORD   $0x4E34D423 // FADD V3.4S, V1.4S, V20.4S  (den = 1 + z)
	WORD   $0x6E23DC42 // FMUL V2.4S, V2.4S, V3.4S   (num·den)
	VFMLA  V0.S4, V1.S4, V2.S4 // += x·z
	WORD   $0x6E23DC63 // FMUL V3.4S, V3.4S, V3.4S   (den²)
	WORD   $0x6E23FC42 // FDIV V2.4S, V2.4S, V3.4S   (d = ·/den²)
	VLD1.P 16(R6), [V0.S4]     // g (x dead)
	WORD   $0x6E22DC00 // FMUL V0.4S, V0.4S, V2.4S   (out = g·d)
	VST1.P [V0.S4], 16(R1)

siluGradDone:
	RET

// func vsigmoidQuadsNeonF32(dst, src *float32, quads int)
//
// 4-wide f32 sigmoid (vexp.go, §T665): dst[i] = σ(src[i]) = num/(1+z),
// z = exp(−|x|), num = (x ≥ 0 ? 1 : z) — the same branch-free stable split as
// vsiluQuadsNeonF32 above, WITHOUT the ·x product and WITHOUT the pdfMin mask
// (no x multiply rescues NaN here, so NaN must flow through z itself; the
// underflow-clamp residue at deep-negative x is ≈1.18e-38 vs ref's exact 0 —
// inside the tolerant budget; see vexp.go). Constants: geluConsts, exp
// registers only. Register map as vsilu (V15 stays erf a1, unused).
TEXT ·vsigmoidQuadsNeonF32(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R1
	MOVD src+8(FP), R0
	MOVD quads+16(FP), R2
	MOVD $geluConsts<>(SB), R3
	VLD1.P 64(R3), [V12.S4, V13.S4, V14.S4, V15.S4]
	VLD1.P 64(R3), [V16.S4, V17.S4, V18.S4, V19.S4]
	VLD1.P 64(R3), [V20.S4, V21.S4, V22.S4, V23.S4]
	VLD1.P 64(R3), [V24.S4, V25.S4, V26.S4, V27.S4]
	VLD1   (R3), [V28.S4, V29.S4, V30.S4, V31.S4]
	LSR $1, R2, R4 // pairs of quads
	AND $1, R2, R5 // odd quad
	CBZ R4, sigmoidOdd

sigmoidLoop2:
	VLD1.P 16(R0), [V0.S4]
	VLD1.P 16(R0), [V6.S4]

	// w = clamp(−|x|), both quads.
	WORD $0x4EA0F801 // FABS V1.4S, V0.4S           (A: a = |x|)
	WORD $0x4EA0F8C7 // FABS V7.4S, V6.4S           (B: a = |x|)
	WORD $0x6EA0F821 // FNEG V1.4S, V1.4S           (A: w = −a)
	WORD $0x6EA0F8E7 // FNEG V7.4S, V7.4S           (B: w = −a)
	WORD $0x4E3FF421 // FMAX V1.4S, V1.4S, V31.4S   (A: clamp lo)
	WORD $0x4E3FF4E7 // FMAX V7.4S, V7.4S, V31.4S   (B: clamp lo)

	// z = exp(w) — the vexp reduction (identical to the kernels above).
	WORD  $0x6E37DC22 // FMUL V2.4S, V1.4S, V23.4S  (A: z = w·log2e)
	WORD  $0x6E37DCE8 // FMUL V8.4S, V7.4S, V23.4S  (B: z = w·log2e)
	WORD  $0x4E218842 // FRINTN V2.4S, V2.4S        (A: n = rint(z))
	WORD  $0x4E218908 // FRINTN V8.4S, V8.4S        (B: n = rint(z))
	VFMLS V2.S4, V24.S4, V1.S4 // A: r -= n·ln2hi
	VFMLS V8.S4, V24.S4, V7.S4 // B: r -= n·ln2hi
	VFMLS V2.S4, V25.S4, V1.S4 // A: r -= n·ln2lo
	VFMLS V8.S4, V25.S4, V7.S4 // B: r -= n·ln2lo
	WORD  $0x4EBB1F63 // ORR V3 = p1                (A)
	WORD  $0x4EBB1F69 // ORR V9 = p1                (B)
	VFMLA V26.S4, V1.S4, V3.S4 // A: p = p1 + p0·r
	VFMLA V26.S4, V7.S4, V9.S4 // B: p = p1 + p0·r
	WORD  $0x4EBC1F84 // ORR V4 = p2                (A)
	WORD  $0x4EBC1F8A // ORR V10 = p2               (B)
	VFMLA V3.S4, V1.S4, V4.S4  // A: p = p2 + p·r
	VFMLA V9.S4, V7.S4, V10.S4 // B: p = p2 + p·r
	WORD  $0x4EBD1FA3 // ORR V3 = p3                (A)
	WORD  $0x4EBD1FA9 // ORR V9 = p3                (B)
	VFMLA V4.S4, V1.S4, V3.S4  // A: p = p3 + p·r
	VFMLA V10.S4, V7.S4, V9.S4 // B: p = p3 + p·r
	WORD  $0x4EBE1FC4 // ORR V4 = p4                (A)
	WORD  $0x4EBE1FCA // ORR V10 = p4               (B)
	VFMLA V3.S4, V1.S4, V4.S4  // A: p = p4 + p·r
	VFMLA V9.S4, V7.S4, V10.S4 // B: p = p4 + p·r
	WORD  $0x4EB51EA3 // ORR V3 = p5(=half)         (A)
	WORD  $0x4EB51EA9 // ORR V9 = p5(=half)         (B)
	VFMLA V4.S4, V1.S4, V3.S4  // A: p = p5 + p·r
	VFMLA V10.S4, V7.S4, V9.S4 // B: p = p5 + p·r
	WORD  $0x6E21DC24 // FMUL V4.4S, V1.4S, V1.4S   (A: r² = r·r)
	WORD  $0x6E27DCEA // FMUL V10.4S, V7.4S, V7.4S  (B: r² = r·r)
	WORD  $0x4E34D421 // FADD V1.4S, V1.4S, V20.4S  (A: q = 1 + r)
	WORD  $0x4E34D4E7 // FADD V7.4S, V7.4S, V20.4S  (B: q = 1 + r)
	VFMLA V3.S4, V4.S4, V1.S4  // A: z = q + p·r²
	VFMLA V9.S4, V10.S4, V7.S4 // B: z = q + p·r²
	WORD  $0x4EA1B842 // FCVTZS V2.4S, V2.4S        (A: ni = int(n))
	WORD  $0x4EA1B908 // FCVTZS V8.4S, V8.4S        (B: ni = int(n))
	VADD  V12.S4, V2.S4, V2.S4 // A: ni += 127
	VADD  V12.S4, V8.S4, V8.S4 // B: ni += 127
	VSHL  $23, V2.S4, V2.S4    // A: 2^n bits
	VSHL  $23, V8.S4, V8.S4    // B: 2^n bits
	WORD  $0x6E22DC21 // FMUL V1.4S, V1.4S, V2.4S   (A: z *= 2^n)
	WORD  $0x6E28DCE7 // FMUL V7.4S, V7.4S, V8.4S   (B: z *= 2^n)

	// Sigmoid tail: num = (x≥0 ? 1 : z), out = num/(1+z).
	WORD $0x6EA0C802 // FCMGE V2.4S, V0.4S, #0.0    (A: sel = x ≥ 0)
	WORD $0x6EA0C8C8 // FCMGE V8.4S, V6.4S, #0.0    (B: sel = x ≥ 0)
	WORD $0x6E611E82 // BSL V2.16B, V20.16B, V1.16B (A: num = sel ? 1 : z)
	WORD $0x6E671E88 // BSL V8.16B, V20.16B, V7.16B (B: num = sel ? 1 : z)
	WORD $0x4E34D423 // FADD V3.4S, V1.4S, V20.4S   (A: den = 1 + z)
	WORD $0x4E34D4E9 // FADD V9.4S, V7.4S, V20.4S   (B: den = 1 + z)
	WORD $0x6E23FC40 // FDIV V0.4S, V2.4S, V3.4S    (A: out = num/den)
	WORD $0x6E29FD06 // FDIV V6.4S, V8.4S, V9.4S    (B: out = num/den)

	VST1.P [V0.S4], 16(R1)
	VST1.P [V6.S4], 16(R1)

	SUBS $1, R4, R4
	BNE  sigmoidLoop2

sigmoidOdd:
	CBZ R5, sigmoidDone

	VLD1.P 16(R0), [V0.S4]
	WORD   $0x4EA0F801 // FABS V1.4S, V0.4S         (a = |x|)
	WORD   $0x6EA0F821 // FNEG V1.4S, V1.4S         (w = −a)
	WORD   $0x4E3FF421 // FMAX V1.4S, V1.4S, V31.4S (clamp lo)
	WORD   $0x6E37DC22 // FMUL V2.4S, V1.4S, V23.4S (z = w·log2e)
	WORD   $0x4E218842 // FRINTN V2.4S, V2.4S       (n = rint(z))
	VFMLS  V2.S4, V24.S4, V1.S4 // r -= n·ln2hi
	VFMLS  V2.S4, V25.S4, V1.S4 // r -= n·ln2lo
	WORD   $0x4EBB1F63 // ORR V3 = p1
	VFMLA  V26.S4, V1.S4, V3.S4 // p = p1 + p0·r
	WORD   $0x4EBC1F84 // ORR V4 = p2
	VFMLA  V3.S4, V1.S4, V4.S4 // p = p2 + p·r
	WORD   $0x4EBD1FA3 // ORR V3 = p3
	VFMLA  V4.S4, V1.S4, V3.S4 // p = p3 + p·r
	WORD   $0x4EBE1FC4 // ORR V4 = p4
	VFMLA  V3.S4, V1.S4, V4.S4 // p = p4 + p·r
	WORD   $0x4EB51EA3 // ORR V3 = p5(=half)
	VFMLA  V4.S4, V1.S4, V3.S4 // p = p5 + p·r
	WORD   $0x6E21DC24 // FMUL V4.4S, V1.4S, V1.4S  (r² = r·r)
	WORD   $0x4E34D421 // FADD V1.4S, V1.4S, V20.4S (q = 1 + r)
	VFMLA  V3.S4, V4.S4, V1.S4 // z = q + p·r²
	WORD   $0x4EA1B842 // FCVTZS V2.4S, V2.4S       (ni = int(n))
	VADD   V12.S4, V2.S4, V2.S4 // ni += 127
	VSHL   $23, V2.S4, V2.S4    // 2^n bits
	WORD   $0x6E22DC21 // FMUL V1.4S, V1.4S, V2.4S  (z *= 2^n)
	WORD   $0x6EA0C802 // FCMGE V2.4S, V0.4S, #0.0   (sel = x ≥ 0)
	WORD   $0x6E611E82 // BSL V2.16B, V20.16B, V1.16B (num = sel ? 1 : z)
	WORD   $0x4E34D423 // FADD V3.4S, V1.4S, V20.4S  (den = 1 + z)
	WORD   $0x6E23FC40 // FDIV V0.4S, V2.4S, V3.4S   (out = num/den)
	VST1.P [V0.S4], 16(R1)

sigmoidDone:
	RET

// func vexpFullQuadsNeonF32(dst, src *float32, quads int)
//
// 4-wide f32 FULL-DOMAIN exp (vexp.go, §T666): dst[i] = exp(src[i]) for
// i in [0, 4*quads) — the standalone OpExp kernel. Same Cephes reduction as
// vexpQuadsNeonF32 above, with three additions for the unrestricted domain
// (the softmax kernel only ever sees x−max ≤ 0):
//  1. hi clamp raised to 89 (V27 reloaded from the vexpConsts tail) and the
//     2^n scale SPLIT into 2^(n>>1)·2^(n−(n>>1)) — n reaches 128 at the f32
//     overflow cutoff, where the single (n+127)<<23 insertion would wrap the
//     exponent field into the Inf encoding for still-finite results.
//  2. the scaled result is pinned to maxFinite (FMIN) so a poly overshoot at
//     the last finite ulp cannot round to a spurious +Inf.
//  3. exact special masks on the ORIGINAL x (NaN-safe: FCMGT with a NaN lane
//     is false, so NaN keeps the poly's NaN): x < loClamp → true 0 (BIC;
//     covers −Inf), x > infCut = 88.72283172607422 → +Inf (BSL; covers +Inf;
//     the cutoff is the largest f32 whose float32(math.Exp(x)) is finite, so
//     the mask matches the f64 ref's overflow set exactly).
// Same two-quads-per-pass structure as the kernels above. WORD encodings
// generated + objdump-verified like the kernels above.
//
// Register map: R0 src, R1 dst, R2 quads, R3 consts, R4 pairs, R5 odd;
// consts V16..V25 as vexpQuadsNeonF32, V26 lo, V27 hiU(89), V28 int127,
// V29 infCut, V30 maxFinite, V31 +Inf.
// A: V0 x (live to the masks), V1 xc/n1/mask, V2 n, V3 r, V4/V5 poly, V6 r²;
// B: V7 x, V8 xc/n1/mask, V9 n, V10 r, V11/V12 poly, V13 r².
TEXT ·vexpFullQuadsNeonF32(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R1
	MOVD src+8(FP), R0
	MOVD quads+16(FP), R2
	MOVD $vexpConsts<>(SB), R3
	VLD1.P 64(R3), [V16.S4, V17.S4, V18.S4, V19.S4]
	VLD1.P 64(R3), [V20.S4, V21.S4, V22.S4, V23.S4]
	VLD1.P 64(R3), [V24.S4, V25.S4, V26.S4, V27.S4]
	VLD1.P 16(R3), [V28.S4]
	VLD1.P 16(R3), [V27.S4] // hiU = 89 overwrites the softmax hi clamp (88)
	VLD1   (R3), [V29.S4, V30.S4, V31.S4]
	LSR $1, R2, R4 // pairs of quads
	AND $1, R2, R5 // odd quad
	CBZ R4, expfOdd

expfLoop2:
	VLD1.P 16(R0), [V0.S4]
	VLD1.P 16(R0), [V7.S4]

	// xc = clamp(x), z = xc·log2e, n = rint(z), both quads.
	WORD $0x4EA01C01 // ORR V1.16B, V0.16B, V0.16B  (A: xc = x)
	WORD $0x4EA71CE8 // ORR V8.16B, V7.16B, V7.16B  (B: xc = x)
	WORD $0x4E3AF421 // FMAX V1.4S, V1.4S, V26.4S   (A: clamp lo)
	WORD $0x4E3AF508 // FMAX V8.4S, V8.4S, V26.4S   (B: clamp lo)
	WORD $0x4EBBF421 // FMIN V1.4S, V1.4S, V27.4S   (A: clamp hi=89)
	WORD $0x4EBBF508 // FMIN V8.4S, V8.4S, V27.4S   (B: clamp hi=89)
	WORD $0x6E30DC22 // FMUL V2.4S, V1.4S, V16.4S   (A: z = xc·log2e)
	WORD $0x6E30DD09 // FMUL V9.4S, V8.4S, V16.4S   (B: z = xc·log2e)
	WORD $0x4E218842 // FRINTN V2.4S, V2.4S         (A: n = rint(z))
	WORD $0x4E218929 // FRINTN V9.4S, V9.4S         (B: n = rint(z))

	// r = xc − n·ln2hi − n·ln2lo
	WORD  $0x4EA11C23 // ORR V3.16B, V1.16B, V1.16B  (A: r = xc)
	WORD  $0x4EA81D0A // ORR V10.16B, V8.16B, V8.16B (B: r = xc)
	VFMLS V2.S4, V17.S4, V3.S4  // A: r -= n·ln2hi
	VFMLS V9.S4, V17.S4, V10.S4 // B: r -= n·ln2hi
	VFMLS V2.S4, V18.S4, V3.S4  // A: r -= n·ln2lo
	VFMLS V9.S4, V18.S4, V10.S4 // B: r -= n·ln2lo

	// Horner: p = ((((p0·r+p1)·r+p2)·r+p3)·r+p4)·r+p5, chains interleaved.
	WORD  $0x4EB31E65 // ORR V5 = p0                 (A)
	WORD  $0x4EB31E6C // ORR V12 = p0                (B)
	WORD  $0x4EB41E84 // ORR V4 = p1                 (A)
	WORD  $0x4EB41E8B // ORR V11 = p1                (B)
	VFMLA V5.S4, V3.S4, V4.S4    // A: p = p1 + p·r
	VFMLA V12.S4, V10.S4, V11.S4 // B: p = p1 + p·r
	WORD  $0x4EB51EA5 // ORR V5 = p2                 (A)
	WORD  $0x4EB51EAC // ORR V12 = p2                (B)
	VFMLA V4.S4, V3.S4, V5.S4    // A: p = p2 + p·r
	VFMLA V11.S4, V10.S4, V12.S4 // B: p = p2 + p·r
	WORD  $0x4EB61EC4 // ORR V4 = p3                 (A)
	WORD  $0x4EB61ECB // ORR V11 = p3                (B)
	VFMLA V5.S4, V3.S4, V4.S4    // A: p = p3 + p·r
	VFMLA V12.S4, V10.S4, V11.S4 // B: p = p3 + p·r
	WORD  $0x4EB71EE5 // ORR V5 = p4                 (A)
	WORD  $0x4EB71EEC // ORR V12 = p4                (B)
	VFMLA V4.S4, V3.S4, V5.S4    // A: p = p4 + p·r
	VFMLA V11.S4, V10.S4, V12.S4 // B: p = p4 + p·r
	WORD  $0x4EB81F04 // ORR V4 = p5                 (A)
	WORD  $0x4EB81F0B // ORR V11 = p5                (B)
	VFMLA V5.S4, V3.S4, V4.S4    // A: p = p5 + p·r
	VFMLA V12.S4, V10.S4, V11.S4 // B: p = p5 + p·r

	// res = (1 + r) + p·r², then the SPLIT 2^n scale.
	WORD  $0x6E23DC66 // FMUL V6.4S, V3.4S, V3.4S    (A: r² = r·r)
	WORD  $0x6E2ADD4D // FMUL V13.4S, V10.4S, V10.4S (B: r² = r·r)
	WORD  $0x4EB91F25 // ORR V5 = one                (A)
	WORD  $0x4EB91F2C // ORR V12 = one               (B)
	WORD  $0x4E23D4A5 // FADD V5.4S, V5.4S, V3.4S    (A: q = 1 + r)
	WORD  $0x4E2AD58C // FADD V12.4S, V12.4S, V10.4S (B: q = 1 + r)
	VFMLA V4.S4, V6.S4, V5.S4    // A: res = q + p·r²
	VFMLA V11.S4, V13.S4, V12.S4 // B: res = q + p·r²
	WORD  $0x4EA1B842 // FCVTZS V2.4S, V2.4S         (A: ni = int(n))
	WORD  $0x4EA1B929 // FCVTZS V9.4S, V9.4S         (B: ni = int(n))
	WORD  $0x4F3F0441 // SSHR V1.4S, V2.4S, #1       (A: n1 = ni>>1)
	WORD  $0x4F3F0528 // SSHR V8.4S, V9.4S, #1       (B: n1 = ni>>1)
	WORD  $0x6EA18442 // SUB V2.4S, V2.4S, V1.4S     (A: n2 = ni − n1)
	WORD  $0x6EA88529 // SUB V9.4S, V9.4S, V8.4S     (B: n2 = ni − n1)
	VADD  V28.S4, V1.S4, V1.S4   // A: n1 += 127
	VADD  V28.S4, V8.S4, V8.S4   // B: n1 += 127
	VADD  V28.S4, V2.S4, V2.S4   // A: n2 += 127
	VADD  V28.S4, V9.S4, V9.S4   // B: n2 += 127
	VSHL  $23, V1.S4, V1.S4      // A: 2^n1 bits
	VSHL  $23, V8.S4, V8.S4      // B: 2^n1 bits
	VSHL  $23, V2.S4, V2.S4      // A: 2^n2 bits
	VSHL  $23, V9.S4, V9.S4      // B: 2^n2 bits
	WORD  $0x6E21DCA5 // FMUL V5.4S, V5.4S, V1.4S    (A: res *= 2^n1)
	WORD  $0x6E28DD8C // FMUL V12.4S, V12.4S, V8.4S  (B: res *= 2^n1)
	WORD  $0x6E22DCA5 // FMUL V5.4S, V5.4S, V2.4S    (A: res *= 2^n2)
	WORD  $0x6E29DD8C // FMUL V12.4S, V12.4S, V9.4S  (B: res *= 2^n2)
	WORD  $0x4EBEF4A5 // FMIN V5.4S, V5.4S, V30.4S   (A: pin to maxFinite)
	WORD  $0x4EBEF58C // FMIN V12.4S, V12.4S, V30.4S (B: pin to maxFinite)

	// Exact special masks on the original x (NaN lanes match nothing).
	WORD $0x6EA0E741 // FCMGT V1.4S, V26.4S, V0.4S  (A: x < lo)
	WORD $0x6EA7E748 // FCMGT V8.4S, V26.4S, V7.4S  (B: x < lo)
	WORD $0x4E611CA5 // BIC V5.16B, V5.16B, V1.16B  (A: underflow → 0)
	WORD $0x4E681D8C // BIC V12.16B, V12.16B, V8.16B (B: underflow → 0)
	WORD $0x6EBDE401 // FCMGT V1.4S, V0.4S, V29.4S  (A: x > infCut)
	WORD $0x6EBDE4E8 // FCMGT V8.4S, V7.4S, V29.4S  (B: x > infCut)
	WORD $0x6E651FE1 // BSL V1.16B, V31.16B, V5.16B (A: overflow → +Inf)
	WORD $0x6E6C1FE8 // BSL V8.16B, V31.16B, V12.16B (B: overflow → +Inf)

	VST1.P [V1.S4], 16(R1)
	VST1.P [V8.S4], 16(R1)

	SUBS $1, R4, R4
	BNE  expfLoop2

expfOdd:
	CBZ R5, expfDone

	VLD1.P 16(R0), [V0.S4]
	WORD   $0x4EA01C01 // ORR V1.16B, V0.16B, V0.16B (xc = x)
	WORD   $0x4E3AF421 // FMAX V1.4S, V1.4S, V26.4S  (clamp lo)
	WORD   $0x4EBBF421 // FMIN V1.4S, V1.4S, V27.4S  (clamp hi=89)
	WORD   $0x6E30DC22 // FMUL V2.4S, V1.4S, V16.4S  (z = xc·log2e)
	WORD   $0x4E218842 // FRINTN V2.4S, V2.4S        (n = rint(z))
	WORD   $0x4EA11C23 // ORR V3.16B, V1.16B, V1.16B (r = xc)
	VFMLS  V2.S4, V17.S4, V3.S4 // r -= n·ln2hi
	VFMLS  V2.S4, V18.S4, V3.S4 // r -= n·ln2lo
	WORD   $0x4EB31E65 // ORR V5 = p0
	WORD   $0x4EB41E84 // ORR V4 = p1
	VFMLA  V5.S4, V3.S4, V4.S4
	WORD   $0x4EB51EA5 // ORR V5 = p2
	VFMLA  V4.S4, V3.S4, V5.S4
	WORD   $0x4EB61EC4 // ORR V4 = p3
	VFMLA  V5.S4, V3.S4, V4.S4
	WORD   $0x4EB71EE5 // ORR V5 = p4
	VFMLA  V4.S4, V3.S4, V5.S4
	WORD   $0x4EB81F04 // ORR V4 = p5
	VFMLA  V5.S4, V3.S4, V4.S4
	WORD   $0x6E23DC66 // FMUL V6.4S, V3.4S, V3.4S   (r²)
	WORD   $0x4EB91F25 // ORR V5 = one
	WORD   $0x4E23D4A5 // FADD V5.4S, V5.4S, V3.4S   (q = 1 + r)
	VFMLA  V4.S4, V6.S4, V5.S4 // res = q + p·r²
	WORD   $0x4EA1B842 // FCVTZS V2.4S, V2.4S        (ni = int(n))
	WORD   $0x4F3F0441 // SSHR V1.4S, V2.4S, #1      (n1 = ni>>1)
	WORD   $0x6EA18442 // SUB V2.4S, V2.4S, V1.4S    (n2 = ni − n1)
	VADD   V28.S4, V1.S4, V1.S4 // n1 += 127
	VADD   V28.S4, V2.S4, V2.S4 // n2 += 127
	VSHL   $23, V1.S4, V1.S4    // 2^n1 bits
	VSHL   $23, V2.S4, V2.S4    // 2^n2 bits
	WORD   $0x6E21DCA5 // FMUL V5.4S, V5.4S, V1.4S   (res *= 2^n1)
	WORD   $0x6E22DCA5 // FMUL V5.4S, V5.4S, V2.4S   (res *= 2^n2)
	WORD   $0x4EBEF4A5 // FMIN V5.4S, V5.4S, V30.4S  (pin to maxFinite)
	WORD   $0x6EA0E741 // FCMGT V1.4S, V26.4S, V0.4S (x < lo)
	WORD   $0x4E611CA5 // BIC V5.16B, V5.16B, V1.16B (underflow → 0)
	WORD   $0x6EBDE401 // FCMGT V1.4S, V0.4S, V29.4S (x > infCut)
	WORD   $0x6E651FE1 // BSL V1.16B, V31.16B, V5.16B (overflow → +Inf)
	VST1.P [V1.S4], 16(R1)

expfDone:
	RET

// func vtanhQuadsNeonF32(dst, src *float32, quads int)
//
// 4-wide f32 tanh (vexp.go, §T666): dst[i] = tanh(src[i]) via the stable
// sign-split — z = exp(−2|x|) through the SAME Cephes exp reduction as the
// kernels above (only the lo clamp is live: −2|x| ≤ 0), magnitude
// t = (1−z)/(1+z), sign re-applied bitwise from x (t ∈ [0,1], the OR is
// exact; NaN payloads stay NaN — same trick as vgelu). No pdfMin mask needed:
// the underflow-clamp residue z ≈ 1.18e-38 still rounds (1−z)/(1+z) to an
// exact 1, so tanh(±Inf) = ±1 and the deep tails saturate exactly like the
// f64 ref. Constants: geluConsts, exp registers + signMask only (as
// vsigmoid; V15 stays erf a1, unused). Same two-quads-per-pass structure
// (A: V0..V4, B: V6..V10); WORD encodings objdump-verified like the kernels
// above.
TEXT ·vtanhQuadsNeonF32(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R1
	MOVD src+8(FP), R0
	MOVD quads+16(FP), R2
	MOVD $geluConsts<>(SB), R3
	VLD1.P 64(R3), [V12.S4, V13.S4, V14.S4, V15.S4]
	VLD1.P 64(R3), [V16.S4, V17.S4, V18.S4, V19.S4]
	VLD1.P 64(R3), [V20.S4, V21.S4, V22.S4, V23.S4]
	VLD1.P 64(R3), [V24.S4, V25.S4, V26.S4, V27.S4]
	VLD1   (R3), [V28.S4, V29.S4, V30.S4, V31.S4]
	LSR $1, R2, R4 // pairs of quads
	AND $1, R2, R5 // odd quad
	CBZ R4, tanhOdd

tanhLoop2:
	VLD1.P 16(R0), [V0.S4]
	VLD1.P 16(R0), [V6.S4]

	// w = clamp(−2|x|), both quads.
	WORD $0x4EA0F801 // FABS V1.4S, V0.4S           (A: a = |x|)
	WORD $0x4EA0F8C7 // FABS V7.4S, V6.4S           (B: a = |x|)
	WORD $0x4E21D421 // FADD V1.4S, V1.4S, V1.4S    (A: 2a)
	WORD $0x4E27D4E7 // FADD V7.4S, V7.4S, V7.4S    (B: 2a)
	WORD $0x6EA0F821 // FNEG V1.4S, V1.4S           (A: w = −2a)
	WORD $0x6EA0F8E7 // FNEG V7.4S, V7.4S           (B: w = −2a)
	WORD $0x4E3FF421 // FMAX V1.4S, V1.4S, V31.4S   (A: clamp lo)
	WORD $0x4E3FF4E7 // FMAX V7.4S, V7.4S, V31.4S   (B: clamp lo)

	// z = exp(w) — the vexp reduction (identical to the kernels above).
	WORD  $0x6E37DC22 // FMUL V2.4S, V1.4S, V23.4S  (A: z = w·log2e)
	WORD  $0x6E37DCE8 // FMUL V8.4S, V7.4S, V23.4S  (B: z = w·log2e)
	WORD  $0x4E218842 // FRINTN V2.4S, V2.4S        (A: n = rint(z))
	WORD  $0x4E218908 // FRINTN V8.4S, V8.4S        (B: n = rint(z))
	VFMLS V2.S4, V24.S4, V1.S4 // A: r -= n·ln2hi
	VFMLS V8.S4, V24.S4, V7.S4 // B: r -= n·ln2hi
	VFMLS V2.S4, V25.S4, V1.S4 // A: r -= n·ln2lo
	VFMLS V8.S4, V25.S4, V7.S4 // B: r -= n·ln2lo
	WORD  $0x4EBB1F63 // ORR V3 = p1                (A)
	WORD  $0x4EBB1F69 // ORR V9 = p1                (B)
	VFMLA V26.S4, V1.S4, V3.S4 // A: p = p1 + p0·r
	VFMLA V26.S4, V7.S4, V9.S4 // B: p = p1 + p0·r
	WORD  $0x4EBC1F84 // ORR V4 = p2                (A)
	WORD  $0x4EBC1F8A // ORR V10 = p2               (B)
	VFMLA V3.S4, V1.S4, V4.S4  // A: p = p2 + p·r
	VFMLA V9.S4, V7.S4, V10.S4 // B: p = p2 + p·r
	WORD  $0x4EBD1FA3 // ORR V3 = p3                (A)
	WORD  $0x4EBD1FA9 // ORR V9 = p3                (B)
	VFMLA V4.S4, V1.S4, V3.S4  // A: p = p3 + p·r
	VFMLA V10.S4, V7.S4, V9.S4 // B: p = p3 + p·r
	WORD  $0x4EBE1FC4 // ORR V4 = p4                (A)
	WORD  $0x4EBE1FCA // ORR V10 = p4               (B)
	VFMLA V3.S4, V1.S4, V4.S4  // A: p = p4 + p·r
	VFMLA V9.S4, V7.S4, V10.S4 // B: p = p4 + p·r
	WORD  $0x4EB51EA3 // ORR V3 = p5(=half)         (A)
	WORD  $0x4EB51EA9 // ORR V9 = p5(=half)         (B)
	VFMLA V4.S4, V1.S4, V3.S4  // A: p = p5 + p·r
	VFMLA V10.S4, V7.S4, V9.S4 // B: p = p5 + p·r
	WORD  $0x6E21DC24 // FMUL V4.4S, V1.4S, V1.4S   (A: r² = r·r)
	WORD  $0x6E27DCEA // FMUL V10.4S, V7.4S, V7.4S  (B: r² = r·r)
	WORD  $0x4E34D421 // FADD V1.4S, V1.4S, V20.4S  (A: q = 1 + r)
	WORD  $0x4E34D4E7 // FADD V7.4S, V7.4S, V20.4S  (B: q = 1 + r)
	VFMLA V3.S4, V4.S4, V1.S4  // A: z = q + p·r²
	VFMLA V9.S4, V10.S4, V7.S4 // B: z = q + p·r²
	WORD  $0x4EA1B842 // FCVTZS V2.4S, V2.4S        (A: ni = int(n))
	WORD  $0x4EA1B908 // FCVTZS V8.4S, V8.4S        (B: ni = int(n))
	VADD  V12.S4, V2.S4, V2.S4 // A: ni += 127
	VADD  V12.S4, V8.S4, V8.S4 // B: ni += 127
	VSHL  $23, V2.S4, V2.S4    // A: 2^n bits
	VSHL  $23, V8.S4, V8.S4    // B: 2^n bits
	WORD  $0x6E22DC21 // FMUL V1.4S, V1.4S, V2.4S   (A: z *= 2^n)
	WORD  $0x6E28DCE7 // FMUL V7.4S, V7.4S, V8.4S   (B: z *= 2^n)

	// Tanh tail: t = (1−z)/(1+z), sign(x) re-applied bitwise.
	WORD $0x4EA1D682 // FSUB V2.4S, V20.4S, V1.4S   (A: num = 1 − z)
	WORD $0x4EA7D688 // FSUB V8.4S, V20.4S, V7.4S   (B: num = 1 − z)
	WORD $0x4E34D423 // FADD V3.4S, V1.4S, V20.4S   (A: den = 1 + z)
	WORD $0x4E34D4E9 // FADD V9.4S, V7.4S, V20.4S   (B: den = 1 + z)
	WORD $0x6E23FC42 // FDIV V2.4S, V2.4S, V3.4S    (A: t = num/den)
	WORD $0x6E29FD08 // FDIV V8.4S, V8.4S, V9.4S    (B: t = num/den)
	WORD $0x4E361C04 // AND V4.16B, V0.16B, V22.16B (A: sign bit of x)
	WORD $0x4E361CCA // AND V10.16B, V6.16B, V22.16B (B: sign bit of x)
	WORD $0x4EA41C42 // ORR V2.16B, V2.16B, V4.16B  (A: out = sign|t)
	WORD $0x4EAA1D08 // ORR V8.16B, V8.16B, V10.16B (B: out = sign|t)

	VST1.P [V2.S4], 16(R1)
	VST1.P [V8.S4], 16(R1)

	SUBS $1, R4, R4
	BNE  tanhLoop2

tanhOdd:
	CBZ R5, tanhDone

	VLD1.P 16(R0), [V0.S4]
	WORD   $0x4EA0F801 // FABS V1.4S, V0.4S         (a = |x|)
	WORD   $0x4E21D421 // FADD V1.4S, V1.4S, V1.4S  (2a)
	WORD   $0x6EA0F821 // FNEG V1.4S, V1.4S         (w = −2a)
	WORD   $0x4E3FF421 // FMAX V1.4S, V1.4S, V31.4S (clamp lo)
	WORD   $0x6E37DC22 // FMUL V2.4S, V1.4S, V23.4S (z = w·log2e)
	WORD   $0x4E218842 // FRINTN V2.4S, V2.4S       (n = rint(z))
	VFMLS  V2.S4, V24.S4, V1.S4 // r -= n·ln2hi
	VFMLS  V2.S4, V25.S4, V1.S4 // r -= n·ln2lo
	WORD   $0x4EBB1F63 // ORR V3 = p1
	VFMLA  V26.S4, V1.S4, V3.S4 // p = p1 + p0·r
	WORD   $0x4EBC1F84 // ORR V4 = p2
	VFMLA  V3.S4, V1.S4, V4.S4 // p = p2 + p·r
	WORD   $0x4EBD1FA3 // ORR V3 = p3
	VFMLA  V4.S4, V1.S4, V3.S4 // p = p3 + p·r
	WORD   $0x4EBE1FC4 // ORR V4 = p4
	VFMLA  V3.S4, V1.S4, V4.S4 // p = p4 + p·r
	WORD   $0x4EB51EA3 // ORR V3 = p5(=half)
	VFMLA  V4.S4, V1.S4, V3.S4 // p = p5 + p·r
	WORD   $0x6E21DC24 // FMUL V4.4S, V1.4S, V1.4S  (r² = r·r)
	WORD   $0x4E34D421 // FADD V1.4S, V1.4S, V20.4S (q = 1 + r)
	VFMLA  V3.S4, V4.S4, V1.S4 // z = q + p·r²
	WORD   $0x4EA1B842 // FCVTZS V2.4S, V2.4S       (ni = int(n))
	VADD   V12.S4, V2.S4, V2.S4 // ni += 127
	VSHL   $23, V2.S4, V2.S4    // 2^n bits
	WORD   $0x6E22DC21 // FMUL V1.4S, V1.4S, V2.4S  (z *= 2^n)
	WORD   $0x4EA1D682 // FSUB V2.4S, V20.4S, V1.4S (num = 1 − z)
	WORD   $0x4E34D423 // FADD V3.4S, V1.4S, V20.4S (den = 1 + z)
	WORD   $0x6E23FC42 // FDIV V2.4S, V2.4S, V3.4S  (t = num/den)
	WORD   $0x4E361C04 // AND V4.16B, V0.16B, V22.16B (sign bit of x)
	WORD   $0x4EA41C42 // ORR V2.16B, V2.16B, V4.16B  (out = sign|t)
	VST1.P [V2.S4], 16(R1)

tanhDone:
	RET

// vlog constants (vlogQuadsNeonF32 below): 16 resident (V16..V31) + two
// 3-entry per-pass tail groups (V12..V14 — see the kernel comment).
DATA logConsts<>+0(SB)/4, $0x007FFFFF    // V16 mantissa mask
DATA logConsts<>+4(SB)/4, $0x007FFFFF    // V16
DATA logConsts<>+8(SB)/4, $0x007FFFFF    // V16
DATA logConsts<>+12(SB)/4, $0x007FFFFF   // V16
DATA logConsts<>+16(SB)/4, $0x3F800000   // V17 one (float 1.0 AND the implicit-1 OR bits)
DATA logConsts<>+20(SB)/4, $0x3F800000   // V17
DATA logConsts<>+24(SB)/4, $0x3F800000   // V17
DATA logConsts<>+28(SB)/4, $0x3F800000   // V17
DATA logConsts<>+32(SB)/4, $0x3FB504F3   // V18 √2 = 1.41421354
DATA logConsts<>+36(SB)/4, $0x3FB504F3   // V18
DATA logConsts<>+40(SB)/4, $0x3FB504F3   // V18
DATA logConsts<>+44(SB)/4, $0x3FB504F3   // V18
DATA logConsts<>+48(SB)/4, $0x3F000000   // V19 half
DATA logConsts<>+52(SB)/4, $0x3F000000   // V19
DATA logConsts<>+56(SB)/4, $0x3F000000   // V19
DATA logConsts<>+60(SB)/4, $0x3F000000   // V19
DATA logConsts<>+64(SB)/4, $0x0000007F   // V20 int32 127
DATA logConsts<>+68(SB)/4, $0x0000007F   // V20
DATA logConsts<>+72(SB)/4, $0x0000007F   // V20
DATA logConsts<>+76(SB)/4, $0x0000007F   // V20
DATA logConsts<>+80(SB)/4, $0x3F318000   // V21 ln2hi = 0.693359375
DATA logConsts<>+84(SB)/4, $0x3F318000   // V21
DATA logConsts<>+88(SB)/4, $0x3F318000   // V21
DATA logConsts<>+92(SB)/4, $0x3F318000   // V21
DATA logConsts<>+96(SB)/4, $0xB95E8083   // V22 ln2lo = -2.12194440e-4
DATA logConsts<>+100(SB)/4, $0xB95E8083  // V22
DATA logConsts<>+104(SB)/4, $0xB95E8083  // V22
DATA logConsts<>+108(SB)/4, $0xB95E8083  // V22
DATA logConsts<>+112(SB)/4, $0x3D9021BB  // V23 L0 = 7.0376836292e-2
DATA logConsts<>+116(SB)/4, $0x3D9021BB  // V23
DATA logConsts<>+120(SB)/4, $0x3D9021BB  // V23
DATA logConsts<>+124(SB)/4, $0x3D9021BB  // V23
DATA logConsts<>+128(SB)/4, $0xBDEBD1B8  // V24 L1 = -1.1514610310e-1
DATA logConsts<>+132(SB)/4, $0xBDEBD1B8  // V24
DATA logConsts<>+136(SB)/4, $0xBDEBD1B8  // V24
DATA logConsts<>+140(SB)/4, $0xBDEBD1B8  // V24
DATA logConsts<>+144(SB)/4, $0x3DEF251A  // V25 L2 = 1.1676998740e-1
DATA logConsts<>+148(SB)/4, $0x3DEF251A  // V25
DATA logConsts<>+152(SB)/4, $0x3DEF251A  // V25
DATA logConsts<>+156(SB)/4, $0x3DEF251A  // V25
DATA logConsts<>+160(SB)/4, $0xBDFE5D4F  // V26 L3 = -1.2420140846e-1
DATA logConsts<>+164(SB)/4, $0xBDFE5D4F  // V26
DATA logConsts<>+168(SB)/4, $0xBDFE5D4F  // V26
DATA logConsts<>+172(SB)/4, $0xBDFE5D4F  // V26
DATA logConsts<>+176(SB)/4, $0x3E11E9BF  // V27 L4 = 1.4249322787e-1
DATA logConsts<>+180(SB)/4, $0x3E11E9BF  // V27
DATA logConsts<>+184(SB)/4, $0x3E11E9BF  // V27
DATA logConsts<>+188(SB)/4, $0x3E11E9BF  // V27
DATA logConsts<>+192(SB)/4, $0xBE2AAE50  // V28 L5 = -1.6668057665e-1
DATA logConsts<>+196(SB)/4, $0xBE2AAE50  // V28
DATA logConsts<>+200(SB)/4, $0xBE2AAE50  // V28
DATA logConsts<>+204(SB)/4, $0xBE2AAE50  // V28
DATA logConsts<>+208(SB)/4, $0x3E4CCEAC  // V29 L6 = 2.0000714765e-1
DATA logConsts<>+212(SB)/4, $0x3E4CCEAC  // V29
DATA logConsts<>+216(SB)/4, $0x3E4CCEAC  // V29
DATA logConsts<>+220(SB)/4, $0x3E4CCEAC  // V29
DATA logConsts<>+224(SB)/4, $0xBE7FFFFC  // V30 L7 = -2.4999993993e-1
DATA logConsts<>+228(SB)/4, $0xBE7FFFFC  // V30
DATA logConsts<>+232(SB)/4, $0xBE7FFFFC  // V30
DATA logConsts<>+236(SB)/4, $0xBE7FFFFC  // V30
DATA logConsts<>+240(SB)/4, $0x3EAAAAAA  // V31 L8 = 3.3333331174e-1
DATA logConsts<>+244(SB)/4, $0x3EAAAAAA  // V31
DATA logConsts<>+248(SB)/4, $0x3EAAAAAA  // V31
DATA logConsts<>+252(SB)/4, $0x3EAAAAAA  // V31
// Tail 1 (R7, per-pass → V12..V14): the subnormal pre-scale group.
DATA logConsts<>+256(SB)/4, $0x00800000  // minNormal = 1.1754944e-38
DATA logConsts<>+260(SB)/4, $0x00800000  // minNormal
DATA logConsts<>+264(SB)/4, $0x00800000  // minNormal
DATA logConsts<>+268(SB)/4, $0x00800000  // minNormal
DATA logConsts<>+272(SB)/4, $0x4C000000  // 2^25
DATA logConsts<>+276(SB)/4, $0x4C000000  // 2^25
DATA logConsts<>+280(SB)/4, $0x4C000000  // 2^25
DATA logConsts<>+284(SB)/4, $0x4C000000  // 2^25
DATA logConsts<>+288(SB)/4, $0x00000019  // int32 25
DATA logConsts<>+292(SB)/4, $0x00000019  // int32 25
DATA logConsts<>+296(SB)/4, $0x00000019  // int32 25
DATA logConsts<>+300(SB)/4, $0x00000019  // int32 25
// Tail 2 (R8, per-pass → V12..V14): the special-select group.
DATA logConsts<>+304(SB)/4, $0x7F800000  // +Inf
DATA logConsts<>+308(SB)/4, $0x7F800000  // +Inf
DATA logConsts<>+312(SB)/4, $0x7F800000  // +Inf
DATA logConsts<>+316(SB)/4, $0x7F800000  // +Inf
DATA logConsts<>+320(SB)/4, $0xFF800000  // −Inf
DATA logConsts<>+324(SB)/4, $0xFF800000  // −Inf
DATA logConsts<>+328(SB)/4, $0xFF800000  // −Inf
DATA logConsts<>+332(SB)/4, $0xFF800000  // −Inf
DATA logConsts<>+336(SB)/4, $0x7FC00000  // NaN (quiet, Go's f32 NaN pattern)
DATA logConsts<>+340(SB)/4, $0x7FC00000  // NaN
DATA logConsts<>+344(SB)/4, $0x7FC00000  // NaN
DATA logConsts<>+348(SB)/4, $0x7FC00000  // NaN
GLOBL logConsts<>(SB), RODATA|NOPTR, $352

// func vlogQuadsNeonF32(dst, src *float32, quads int)
//
// 4-wide f32 natural log (vexp.go, §T666): dst[i] = log(src[i]) for i in
// [0, 4*quads) — a NEW NEON primitive (the Cephes logf reduction, not the exp
// leaf). Pipeline per lane: subnormal pre-scale (x < minNormal → x·2^25,
// e −= 25 — FCMGT+BSL select, NaN-safe), exponent-field extraction by integer
// USHR/SUB, mantissa → m ∈ [1,2) by bit masking, fold to [√½, √2) (m ≥ √2 →
// m·0.5, e += 1 — the mask is integer −1, so a SUB adds 1), f = m − 1 (exact,
// Sterbenz), degree-8 Horner in f, then log = e·ln2hi + (f + (f·f²·P(f) +
// e·ln2lo − f²/2)) with the same hi/lo ln2 split as exp. Exact special
// selects on the ORIGINAL x close the domain: unordered → x (NaN propagates),
// x == ±0 → −Inf, x < 0 → NaN, x == +Inf → +Inf.
//
// The 16 resident constants (V16..V31) fill the register file, so the two
// 3-constant tail groups load per pass into scratch: V12..V14 = {minNormal,
// 2^25, int 25} from R7 at the top (used only in the pre-scale), then
// V12..V14 = {+Inf, −Inf, NaN} from R8 before the special selects. Same
// two-quads-per-pass structure (A: V0..V5, B: V6..V11); WORD encodings
// generated + objdump-verified like the kernels above.
//
// Register map: R0 src, R1 dst, R2 quads, R3 consts, R7 tail1, R8 tail2,
// R4 pairs, R5 odd; consts V16 mantMask, V17 one(=1.0 bits), V18 √2, V19 0.5,
// V20 int127, V21 ln2hi, V22 ln2lo, V23..V31 L0..L8.
// A: V0 x (live to the selects), V1..V5 work, result in V4; B: V6 x, V7..V11
// work, result in V10.
TEXT ·vlogQuadsNeonF32(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R1
	MOVD src+8(FP), R0
	MOVD quads+16(FP), R2
	MOVD $logConsts<>(SB), R3
	VLD1.P 64(R3), [V16.S4, V17.S4, V18.S4, V19.S4]
	VLD1.P 64(R3), [V20.S4, V21.S4, V22.S4, V23.S4]
	VLD1.P 64(R3), [V24.S4, V25.S4, V26.S4, V27.S4]
	VLD1.P 64(R3), [V28.S4, V29.S4, V30.S4, V31.S4]
	MOVD R3, R7     // tail 1: minNormal, 2^25, int 25
	ADD  $48, R3, R8 // tail 2: +Inf, −Inf, NaN
	LSR $1, R2, R4 // pairs of quads
	AND $1, R2, R5 // odd quad
	CBZ R4, logOdd

logLoop2:
	VLD1   (R7), [V12.S4, V13.S4, V14.S4]
	VLD1.P 16(R0), [V0.S4]
	VLD1.P 16(R0), [V6.S4]

	// Subnormal pre-scale + exponent extraction, both quads.
	WORD $0x6EA0E581 // FCMGT V1.4S, V12.4S, V0.4S  (A: tiny = minNormal > x)
	WORD $0x6EA6E587 // FCMGT V7.4S, V12.4S, V6.4S  (B: tiny)
	WORD $0x6E2DDC02 // FMUL V2.4S, V0.4S, V13.4S   (A: xa = x·2^25)
	WORD $0x6E2DDCC8 // FMUL V8.4S, V6.4S, V13.4S   (B: xa = x·2^25)
	WORD $0x4E2E1C23 // AND V3.16B, V1.16B, V14.16B (A: eAdj = tiny & 25)
	WORD $0x4E2E1CE9 // AND V9.16B, V7.16B, V14.16B (B: eAdj = tiny & 25)
	WORD $0x6E601C41 // BSL V1.16B, V2.16B, V0.16B  (A: xs = tiny ? xa : x)
	WORD $0x6E661D07 // BSL V7.16B, V8.16B, V6.16B  (B: xs = tiny ? xa : x)
	WORD $0x6F290422 // USHR V2.4S, V1.4S, #23      (A: exponent field)
	WORD $0x6F2904E8 // USHR V8.4S, V7.4S, #23      (B: exponent field)
	WORD $0x6EB48442 // SUB V2.4S, V2.4S, V20.4S    (A: e −= 127)
	WORD $0x6EB48508 // SUB V8.4S, V8.4S, V20.4S    (B: e −= 127)
	WORD $0x6EA38442 // SUB V2.4S, V2.4S, V3.4S     (A: e −= eAdj)
	WORD $0x6EA98508 // SUB V8.4S, V8.4S, V9.4S     (B: e −= eAdj)

	// m ∈ [1,2) by bit ops, folded to [√½, √2); f = m − 1 (exact).
	WORD $0x4E301C21 // AND V1.16B, V1.16B, V16.16B (A: mantissa bits)
	WORD $0x4E301CE7 // AND V7.16B, V7.16B, V16.16B (B: mantissa bits)
	WORD $0x4EB11C21 // ORR V1.16B, V1.16B, V17.16B (A: m = 1.mant)
	WORD $0x4EB11CE7 // ORR V7.16B, V7.16B, V17.16B (B: m = 1.mant)
	WORD $0x6E32E423 // FCMGE V3.4S, V1.4S, V18.4S  (A: fold = m ≥ √2)
	WORD $0x6E32E4E9 // FCMGE V9.4S, V7.4S, V18.4S  (B: fold = m ≥ √2)
	WORD $0x6E33DC24 // FMUL V4.4S, V1.4S, V19.4S   (A: mHalf = m·0.5)
	WORD $0x6E33DCEA // FMUL V10.4S, V7.4S, V19.4S  (B: mHalf = m·0.5)
	WORD $0x6EA38442 // SUB V2.4S, V2.4S, V3.4S     (A: e += 1 where fold)
	WORD $0x6EA98508 // SUB V8.4S, V8.4S, V9.4S     (B: e += 1 where fold)
	WORD $0x6E611C83 // BSL V3.16B, V4.16B, V1.16B  (A: m = fold ? mHalf : m)
	WORD $0x6E671D49 // BSL V9.16B, V10.16B, V7.16B (B: m = fold ? mHalf : m)
	WORD $0x4EB1D463 // FSUB V3.4S, V3.4S, V17.4S   (A: f = m − 1)
	WORD $0x4EB1D529 // FSUB V9.4S, V9.4S, V17.4S   (B: f = m − 1)
	WORD $0x4E21D842 // SCVTF V2.4S, V2.4S          (A: ef = float(e))
	WORD $0x4E21D908 // SCVTF V8.4S, V8.4S          (B: ef = float(e))
	WORD $0x6E23DC64 // FMUL V4.4S, V3.4S, V3.4S    (A: z = f²)
	WORD $0x6E29DD2A // FMUL V10.4S, V9.4S, V9.4S   (B: z = f²)

	// Horner: p = ((...((L0·f+L1)·f+L2)...)·f+L8), chains interleaved.
	WORD  $0x4EB71EE1 // ORR V1 = L0                (A)
	WORD  $0x4EB71EE7 // ORR V7 = L0                (B)
	WORD  $0x4EB81F05 // ORR V5 = L1                (A)
	WORD  $0x4EB81F0B // ORR V11 = L1               (B)
	VFMLA V1.S4, V3.S4, V5.S4   // A: p = L1 + p·f
	VFMLA V7.S4, V9.S4, V11.S4  // B: p = L1 + p·f
	WORD  $0x4EB91F21 // ORR V1 = L2                (A)
	WORD  $0x4EB91F27 // ORR V7 = L2                (B)
	VFMLA V5.S4, V3.S4, V1.S4   // A: p = L2 + p·f
	VFMLA V11.S4, V9.S4, V7.S4  // B: p = L2 + p·f
	WORD  $0x4EBA1F45 // ORR V5 = L3                (A)
	WORD  $0x4EBA1F4B // ORR V11 = L3               (B)
	VFMLA V1.S4, V3.S4, V5.S4   // A: p = L3 + p·f
	VFMLA V7.S4, V9.S4, V11.S4  // B: p = L3 + p·f
	WORD  $0x4EBB1F61 // ORR V1 = L4                (A)
	WORD  $0x4EBB1F67 // ORR V7 = L4                (B)
	VFMLA V5.S4, V3.S4, V1.S4   // A: p = L4 + p·f
	VFMLA V11.S4, V9.S4, V7.S4  // B: p = L4 + p·f
	WORD  $0x4EBC1F85 // ORR V5 = L5                (A)
	WORD  $0x4EBC1F8B // ORR V11 = L5               (B)
	VFMLA V1.S4, V3.S4, V5.S4   // A: p = L5 + p·f
	VFMLA V7.S4, V9.S4, V11.S4  // B: p = L5 + p·f
	WORD  $0x4EBD1FA1 // ORR V1 = L6                (A)
	WORD  $0x4EBD1FA7 // ORR V7 = L6                (B)
	VFMLA V5.S4, V3.S4, V1.S4   // A: p = L6 + p·f
	VFMLA V11.S4, V9.S4, V7.S4  // B: p = L6 + p·f
	WORD  $0x4EBE1FC5 // ORR V5 = L7                (A)
	WORD  $0x4EBE1FCB // ORR V11 = L7               (B)
	VFMLA V1.S4, V3.S4, V5.S4   // A: p = L7 + p·f
	VFMLA V7.S4, V9.S4, V11.S4  // B: p = L7 + p·f
	WORD  $0x4EBF1FE1 // ORR V1 = L8                (A)
	WORD  $0x4EBF1FE7 // ORR V7 = L8                (B)
	VFMLA V5.S4, V3.S4, V1.S4   // A: p = L8 + p·f
	VFMLA V11.S4, V9.S4, V7.S4  // B: p = L8 + p·f

	// y = f·z·p + e·ln2lo − z/2; out = f + y + e·ln2hi.
	WORD  $0x6E21DC85 // FMUL V5.4S, V4.4S, V1.4S   (A: z·p)
	WORD  $0x6E27DD4B // FMUL V11.4S, V10.4S, V7.4S (B: z·p)
	WORD  $0x6E23DCA5 // FMUL V5.4S, V5.4S, V3.4S   (A: y = f·z·p)
	WORD  $0x6E29DD6B // FMUL V11.4S, V11.4S, V9.4S (B: y = f·z·p)
	VFMLA V22.S4, V2.S4, V5.S4  // A: y += ef·ln2lo
	VFMLA V22.S4, V8.S4, V11.S4 // B: y += ef·ln2lo
	VFMLS V4.S4, V19.S4, V5.S4  // A: y −= 0.5·z
	VFMLS V10.S4, V19.S4, V11.S4 // B: y −= 0.5·z
	WORD  $0x4E23D4A5 // FADD V5.4S, V5.4S, V3.4S   (A: out = f + y)
	WORD  $0x4E29D56B // FADD V11.4S, V11.4S, V9.4S (B: out = f + y)
	VFMLA V21.S4, V2.S4, V5.S4  // A: out += ef·ln2hi
	VFMLA V21.S4, V8.S4, V11.S4 // B: out += ef·ln2hi

	// Exact special selects on the original x (later select wins).
	VLD1 (R8), [V12.S4, V13.S4, V14.S4]
	WORD $0x4E20E401 // FCMEQ V1.4S, V0.4S, V0.4S   (A: ordered)
	WORD $0x4E26E4C7 // FCMEQ V7.4S, V6.4S, V6.4S   (B: ordered)
	WORD $0x6E601CA1 // BSL V1.16B, V5.16B, V0.16B  (A: NaN → x)
	WORD $0x6E661D67 // BSL V7.16B, V11.16B, V6.16B (B: NaN → x)
	WORD $0x4EA0D802 // FCMEQ V2.4S, V0.4S, #0.0    (A: x == ±0)
	WORD $0x4EA0D8C8 // FCMEQ V8.4S, V6.4S, #0.0    (B: x == ±0)
	WORD $0x6E611DA2 // BSL V2.16B, V13.16B, V1.16B (A: ±0 → −Inf)
	WORD $0x6E671DA8 // BSL V8.16B, V13.16B, V7.16B (B: ±0 → −Inf)
	WORD $0x4EA0E803 // FCMLT V3.4S, V0.4S, #0.0    (A: x < 0)
	WORD $0x4EA0E8C9 // FCMLT V9.4S, V6.4S, #0.0    (B: x < 0)
	WORD $0x6E621DC3 // BSL V3.16B, V14.16B, V2.16B (A: neg → NaN)
	WORD $0x6E681DC9 // BSL V9.16B, V14.16B, V8.16B (B: neg → NaN)
	WORD $0x4E2CE404 // FCMEQ V4.4S, V0.4S, V12.4S  (A: x == +Inf)
	WORD $0x4E2CE4CA // FCMEQ V10.4S, V6.4S, V12.4S (B: x == +Inf)
	WORD $0x6E631D84 // BSL V4.16B, V12.16B, V3.16B (A: +Inf → +Inf)
	WORD $0x6E691D8A // BSL V10.16B, V12.16B, V9.16B (B: +Inf → +Inf)

	VST1.P [V4.S4], 16(R1)
	VST1.P [V10.S4], 16(R1)

	SUBS $1, R4, R4
	BNE  logLoop2

logOdd:
	CBZ R5, logDone

	VLD1   (R7), [V12.S4, V13.S4, V14.S4]
	VLD1.P 16(R0), [V0.S4]
	WORD   $0x6EA0E581 // FCMGT V1.4S, V12.4S, V0.4S  (tiny = minNormal > x)
	WORD   $0x6E2DDC02 // FMUL V2.4S, V0.4S, V13.4S   (xa = x·2^25)
	WORD   $0x4E2E1C23 // AND V3.16B, V1.16B, V14.16B (eAdj = tiny & 25)
	WORD   $0x6E601C41 // BSL V1.16B, V2.16B, V0.16B  (xs = tiny ? xa : x)
	WORD   $0x6F290422 // USHR V2.4S, V1.4S, #23      (exponent field)
	WORD   $0x6EB48442 // SUB V2.4S, V2.4S, V20.4S    (e −= 127)
	WORD   $0x6EA38442 // SUB V2.4S, V2.4S, V3.4S     (e −= eAdj)
	WORD   $0x4E301C21 // AND V1.16B, V1.16B, V16.16B (mantissa bits)
	WORD   $0x4EB11C21 // ORR V1.16B, V1.16B, V17.16B (m = 1.mant)
	WORD   $0x6E32E423 // FCMGE V3.4S, V1.4S, V18.4S  (fold = m ≥ √2)
	WORD   $0x6E33DC24 // FMUL V4.4S, V1.4S, V19.4S   (mHalf = m·0.5)
	WORD   $0x6EA38442 // SUB V2.4S, V2.4S, V3.4S     (e += 1 where fold)
	WORD   $0x6E611C83 // BSL V3.16B, V4.16B, V1.16B  (m = fold ? mHalf : m)
	WORD   $0x4EB1D463 // FSUB V3.4S, V3.4S, V17.4S   (f = m − 1)
	WORD   $0x4E21D842 // SCVTF V2.4S, V2.4S          (ef = float(e))
	WORD   $0x6E23DC64 // FMUL V4.4S, V3.4S, V3.4S    (z = f²)
	WORD   $0x4EB71EE1 // ORR V1 = L0
	WORD   $0x4EB81F05 // ORR V5 = L1
	VFMLA  V1.S4, V3.S4, V5.S4 // p = L1 + p·f
	WORD   $0x4EB91F21 // ORR V1 = L2
	VFMLA  V5.S4, V3.S4, V1.S4 // p = L2 + p·f
	WORD   $0x4EBA1F45 // ORR V5 = L3
	VFMLA  V1.S4, V3.S4, V5.S4 // p = L3 + p·f
	WORD   $0x4EBB1F61 // ORR V1 = L4
	VFMLA  V5.S4, V3.S4, V1.S4 // p = L4 + p·f
	WORD   $0x4EBC1F85 // ORR V5 = L5
	VFMLA  V1.S4, V3.S4, V5.S4 // p = L5 + p·f
	WORD   $0x4EBD1FA1 // ORR V1 = L6
	VFMLA  V5.S4, V3.S4, V1.S4 // p = L6 + p·f
	WORD   $0x4EBE1FC5 // ORR V5 = L7
	VFMLA  V1.S4, V3.S4, V5.S4 // p = L7 + p·f
	WORD   $0x4EBF1FE1 // ORR V1 = L8
	VFMLA  V5.S4, V3.S4, V1.S4 // p = L8 + p·f
	WORD   $0x6E21DC85 // FMUL V5.4S, V4.4S, V1.4S   (z·p)
	WORD   $0x6E23DCA5 // FMUL V5.4S, V5.4S, V3.4S   (y = f·z·p)
	VFMLA  V22.S4, V2.S4, V5.S4 // y += ef·ln2lo
	VFMLS  V4.S4, V19.S4, V5.S4 // y −= 0.5·z
	WORD   $0x4E23D4A5 // FADD V5.4S, V5.4S, V3.4S   (out = f + y)
	VFMLA  V21.S4, V2.S4, V5.S4 // out += ef·ln2hi
	VLD1   (R8), [V12.S4, V13.S4, V14.S4]
	WORD   $0x4E20E401 // FCMEQ V1.4S, V0.4S, V0.4S  (ordered)
	WORD   $0x6E601CA1 // BSL V1.16B, V5.16B, V0.16B (NaN → x)
	WORD   $0x4EA0D802 // FCMEQ V2.4S, V0.4S, #0.0   (x == ±0)
	WORD   $0x6E611DA2 // BSL V2.16B, V13.16B, V1.16B (±0 → −Inf)
	WORD   $0x4EA0E803 // FCMLT V3.4S, V0.4S, #0.0   (x < 0)
	WORD   $0x6E621DC3 // BSL V3.16B, V14.16B, V2.16B (neg → NaN)
	WORD   $0x4E2CE404 // FCMEQ V4.4S, V0.4S, V12.4S (x == +Inf)
	WORD   $0x6E631D84 // BSL V4.16B, V12.16B, V3.16B (+Inf → +Inf)
	VST1.P [V4.S4], 16(R1)

logDone:
	RET
