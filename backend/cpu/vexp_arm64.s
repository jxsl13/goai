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
