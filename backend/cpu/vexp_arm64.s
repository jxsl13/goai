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
GLOBL vexpConsts<>(SB), RODATA|NOPTR, $208

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
