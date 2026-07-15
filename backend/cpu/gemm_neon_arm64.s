//go:build goexperiment.simd

#include "textflag.h"

// func gemmF32Tile4x16Neon(a, b, c *float32, k, lda, ldb, ldc int)
//
// f32-NATIVE 4×16 GEMM register tile (ADR-0026): C[4][16] = Σ_p A[r][p]·B[p][j],
// store-once. Sixteen 4-lane f32 accumulators (V0–V15) live in NEON registers
// across the whole k-loop. The k-loop is unrolled ×4: one 128-bit VLD1.P per A
// row pulls FOUR p-steps of the A column (V20–V23), each p-step loads its
// 16-wide B row slice with one 64-byte VLD1 (V16–V19) and issues 16
// BY-ELEMENT FMLAs — FMLA Vd.4S, Vn.4S, Vm.S[e] — which read the A scalar
// straight out of the quad's lane, so no VDUP broadcasts compete with the
// FMLAs for the 4 FP/SIMD pipes (the VDUP variant measured 87 GFLOP/s
// single-core; this one 104.6 ≈ 93% of the 4-pipe FMA peak on M2). The Go
// assembler has no by-element VFMLA form, so those 64 instructions are
// WORD-encoded (encodings cross-checked against clang -arch arm64; the
// generator lives in the ADR). Each C element still accumulates its k products
// in ascending p order in one fused-FMA chain → the K·u_f32 error bound the
// ADR-0021 tolerance test gates. k may be any value ≥ 1 (k%4 remainder runs
// the VDUP tail loop); the Go caller guards k==0.
//
// a: &A[i0*lda] (4 rows, row stride lda ELEMENTS)
// b: &B[jLo]    (k rows, row stride ldb ELEMENTS; 16 for packed panels)
// c: &C[i0*ldc+jLo] (4 rows written once, row stride ldc ELEMENTS)
TEXT ·gemmF32Tile4x16Neon(SB), NOSPLIT, $0-56
	MOVD a+0(FP), R0
	MOVD b+8(FP), R1
	MOVD c+16(FP), R2
	MOVD k+24(FP), R3
	MOVD lda+32(FP), R4
	MOVD ldb+40(FP), R5
	MOVD ldc+48(FP), R6

	LSL $2, R4, R4 // lda elements -> bytes
	LSL $2, R5, R5 // ldb elements -> bytes
	LSL $2, R6, R6 // ldc elements -> bytes

	// A row pointers R7..R10, each walked 16 bytes per unrolled step.
	MOVD R0, R7
	ADD  R4, R7, R8
	ADD  R4, R8, R9
	ADD  R4, R9, R10

	// Zero the 16 accumulators.
	VMOVI $0, V0.B16
	VMOVI $0, V1.B16
	VMOVI $0, V2.B16
	VMOVI $0, V3.B16
	VMOVI $0, V4.B16
	VMOVI $0, V5.B16
	VMOVI $0, V6.B16
	VMOVI $0, V7.B16
	VMOVI $0, V8.B16
	VMOVI $0, V9.B16
	VMOVI $0, V10.B16
	VMOVI $0, V11.B16
	VMOVI $0, V12.B16
	VMOVI $0, V13.B16
	VMOVI $0, V14.B16
	VMOVI $0, V15.B16

	LSR $2, R3, R11 // R11 = k/4 unrolled iterations
	AND $3, R3, R12 // R12 = k%4 remainder
	CBZ R11, tail

loop4:
	// Four p-steps of the A column per row, one 128-bit load each.
	VLD1.P 16(R7), [V20.S4]
	VLD1.P 16(R8), [V21.S4]
	VLD1.P 16(R9), [V22.S4]
	VLD1.P 16(R10), [V23.S4]
	VLD1 (R1), [V16.S4, V17.S4, V18.S4, V19.S4]
	ADD  R5, R1, R1
	// e=0: acc[r][v] += B(p+0)[v] * Ar[0]
	WORD $0x4F941200 // FMLA V0.4S, V16.4S, V20.S[0]
	WORD $0x4F941221 // FMLA V1.4S, V17.4S, V20.S[0]
	WORD $0x4F941242 // FMLA V2.4S, V18.4S, V20.S[0]
	WORD $0x4F941263 // FMLA V3.4S, V19.4S, V20.S[0]
	WORD $0x4F951204 // FMLA V4.4S, V16.4S, V21.S[0]
	WORD $0x4F951225 // FMLA V5.4S, V17.4S, V21.S[0]
	WORD $0x4F951246 // FMLA V6.4S, V18.4S, V21.S[0]
	WORD $0x4F951267 // FMLA V7.4S, V19.4S, V21.S[0]
	WORD $0x4F961208 // FMLA V8.4S, V16.4S, V22.S[0]
	WORD $0x4F961229 // FMLA V9.4S, V17.4S, V22.S[0]
	WORD $0x4F96124A // FMLA V10.4S, V18.4S, V22.S[0]
	WORD $0x4F96126B // FMLA V11.4S, V19.4S, V22.S[0]
	WORD $0x4F97120C // FMLA V12.4S, V16.4S, V23.S[0]
	WORD $0x4F97122D // FMLA V13.4S, V17.4S, V23.S[0]
	WORD $0x4F97124E // FMLA V14.4S, V18.4S, V23.S[0]
	WORD $0x4F97126F // FMLA V15.4S, V19.4S, V23.S[0]
	VLD1 (R1), [V16.S4, V17.S4, V18.S4, V19.S4]
	ADD  R5, R1, R1
	// e=1: acc[r][v] += B(p+1)[v] * Ar[1]
	WORD $0x4FB41200 // FMLA V0.4S, V16.4S, V20.S[1]
	WORD $0x4FB41221 // FMLA V1.4S, V17.4S, V20.S[1]
	WORD $0x4FB41242 // FMLA V2.4S, V18.4S, V20.S[1]
	WORD $0x4FB41263 // FMLA V3.4S, V19.4S, V20.S[1]
	WORD $0x4FB51204 // FMLA V4.4S, V16.4S, V21.S[1]
	WORD $0x4FB51225 // FMLA V5.4S, V17.4S, V21.S[1]
	WORD $0x4FB51246 // FMLA V6.4S, V18.4S, V21.S[1]
	WORD $0x4FB51267 // FMLA V7.4S, V19.4S, V21.S[1]
	WORD $0x4FB61208 // FMLA V8.4S, V16.4S, V22.S[1]
	WORD $0x4FB61229 // FMLA V9.4S, V17.4S, V22.S[1]
	WORD $0x4FB6124A // FMLA V10.4S, V18.4S, V22.S[1]
	WORD $0x4FB6126B // FMLA V11.4S, V19.4S, V22.S[1]
	WORD $0x4FB7120C // FMLA V12.4S, V16.4S, V23.S[1]
	WORD $0x4FB7122D // FMLA V13.4S, V17.4S, V23.S[1]
	WORD $0x4FB7124E // FMLA V14.4S, V18.4S, V23.S[1]
	WORD $0x4FB7126F // FMLA V15.4S, V19.4S, V23.S[1]
	VLD1 (R1), [V16.S4, V17.S4, V18.S4, V19.S4]
	ADD  R5, R1, R1
	// e=2: acc[r][v] += B(p+2)[v] * Ar[2]
	WORD $0x4F941A00 // FMLA V0.4S, V16.4S, V20.S[2]
	WORD $0x4F941A21 // FMLA V1.4S, V17.4S, V20.S[2]
	WORD $0x4F941A42 // FMLA V2.4S, V18.4S, V20.S[2]
	WORD $0x4F941A63 // FMLA V3.4S, V19.4S, V20.S[2]
	WORD $0x4F951A04 // FMLA V4.4S, V16.4S, V21.S[2]
	WORD $0x4F951A25 // FMLA V5.4S, V17.4S, V21.S[2]
	WORD $0x4F951A46 // FMLA V6.4S, V18.4S, V21.S[2]
	WORD $0x4F951A67 // FMLA V7.4S, V19.4S, V21.S[2]
	WORD $0x4F961A08 // FMLA V8.4S, V16.4S, V22.S[2]
	WORD $0x4F961A29 // FMLA V9.4S, V17.4S, V22.S[2]
	WORD $0x4F961A4A // FMLA V10.4S, V18.4S, V22.S[2]
	WORD $0x4F961A6B // FMLA V11.4S, V19.4S, V22.S[2]
	WORD $0x4F971A0C // FMLA V12.4S, V16.4S, V23.S[2]
	WORD $0x4F971A2D // FMLA V13.4S, V17.4S, V23.S[2]
	WORD $0x4F971A4E // FMLA V14.4S, V18.4S, V23.S[2]
	WORD $0x4F971A6F // FMLA V15.4S, V19.4S, V23.S[2]
	VLD1 (R1), [V16.S4, V17.S4, V18.S4, V19.S4]
	ADD  R5, R1, R1
	// e=3: acc[r][v] += B(p+3)[v] * Ar[3]
	WORD $0x4FB41A00 // FMLA V0.4S, V16.4S, V20.S[3]
	WORD $0x4FB41A21 // FMLA V1.4S, V17.4S, V20.S[3]
	WORD $0x4FB41A42 // FMLA V2.4S, V18.4S, V20.S[3]
	WORD $0x4FB41A63 // FMLA V3.4S, V19.4S, V20.S[3]
	WORD $0x4FB51A04 // FMLA V4.4S, V16.4S, V21.S[3]
	WORD $0x4FB51A25 // FMLA V5.4S, V17.4S, V21.S[3]
	WORD $0x4FB51A46 // FMLA V6.4S, V18.4S, V21.S[3]
	WORD $0x4FB51A67 // FMLA V7.4S, V19.4S, V21.S[3]
	WORD $0x4FB61A08 // FMLA V8.4S, V16.4S, V22.S[3]
	WORD $0x4FB61A29 // FMLA V9.4S, V17.4S, V22.S[3]
	WORD $0x4FB61A4A // FMLA V10.4S, V18.4S, V22.S[3]
	WORD $0x4FB61A6B // FMLA V11.4S, V19.4S, V22.S[3]
	WORD $0x4FB71A0C // FMLA V12.4S, V16.4S, V23.S[3]
	WORD $0x4FB71A2D // FMLA V13.4S, V17.4S, V23.S[3]
	WORD $0x4FB71A4E // FMLA V14.4S, V18.4S, V23.S[3]
	WORD $0x4FB71A6F // FMLA V15.4S, V19.4S, V23.S[3]
	SUBS $1, R11, R11
	BNE  loop4

tail:
	CBZ R12, store

	// k%4 remainder: one p-step per pass, scalar A load + VDUP broadcast.
tailloop:
	VLD1 (R1), [V16.S4, V17.S4, V18.S4, V19.S4]
	ADD  R5, R1, R1

	FMOVS (R7), F20
	ADD   $4, R7, R7
	VDUP  V20.S[0], V20.S4
	FMOVS (R8), F21
	ADD   $4, R8, R8
	VDUP  V21.S[0], V21.S4
	FMOVS (R9), F22
	ADD   $4, R9, R9
	VDUP  V22.S[0], V22.S4
	FMOVS (R10), F23
	ADD   $4, R10, R10
	VDUP  V23.S[0], V23.S4

	VFMLA V16.S4, V20.S4, V0.S4
	VFMLA V17.S4, V20.S4, V1.S4
	VFMLA V18.S4, V20.S4, V2.S4
	VFMLA V19.S4, V20.S4, V3.S4
	VFMLA V16.S4, V21.S4, V4.S4
	VFMLA V17.S4, V21.S4, V5.S4
	VFMLA V18.S4, V21.S4, V6.S4
	VFMLA V19.S4, V21.S4, V7.S4
	VFMLA V16.S4, V22.S4, V8.S4
	VFMLA V17.S4, V22.S4, V9.S4
	VFMLA V18.S4, V22.S4, V10.S4
	VFMLA V19.S4, V22.S4, V11.S4
	VFMLA V16.S4, V23.S4, V12.S4
	VFMLA V17.S4, V23.S4, V13.S4
	VFMLA V18.S4, V23.S4, V14.S4
	VFMLA V19.S4, V23.S4, V15.S4

	SUBS $1, R12, R12
	BNE  tailloop

store:
	// Store the 4 C rows once.
	VST1 [V0.S4, V1.S4, V2.S4, V3.S4], (R2)
	ADD  R6, R2, R2
	VST1 [V4.S4, V5.S4, V6.S4, V7.S4], (R2)
	ADD  R6, R2, R2
	VST1 [V8.S4, V9.S4, V10.S4, V11.S4], (R2)
	ADD  R6, R2, R2
	VST1 [V12.S4, V13.S4, V14.S4, V15.S4], (R2)
	RET
