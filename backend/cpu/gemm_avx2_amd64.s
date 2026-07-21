//go:build amd64 && goexperiment.simd

#include "textflag.h"

// func gemmF32Tile6x16AVX2(a, b, c *float32, k, lda, ldb, ldc int)
//
// C[6][16] = Σ_{p<k} A[r][p]·B[p][j], store-once. a=&A[i0*lda] (6 rows, row stride lda
// ELEMENTS), b=&B[jLo] (k rows, row stride ldb ELEMENTS), c=&C[i0*ldc+jLo] (6 rows, row
// stride ldc ELEMENTS). Twelve YMM accumulators Y0..Y11 (row r → Y{2r} cols 0:8, Y{2r+1}
// cols 8:16) live across the whole k-loop — the 6×16 register tile the archsimd allocator
// cannot hold (golang/go#76969); hand-managed here it fits (12 accs + 2 B + 1 broadcast =
// 15 YMM). Each C element accumulates its k products in ascending-p order through one fused
// VFMADD231PS chain, so the result is within the ADR-0021 f32 tolerance of the scalar kernel.
// k must be ≥ 1 (the Go caller guards k==0 and the m%6 / n%16 remainders).
TEXT ·gemmF32Tile6x16AVX2(SB), NOSPLIT, $0-56
	MOVQ a+0(FP), AX
	MOVQ b+8(FP), BX
	MOVQ k+24(FP), DX
	MOVQ lda+32(FP), SI
	SHLQ $2, SI          // lda bytes
	MOVQ ldb+40(FP), DI
	SHLQ $2, DI          // ldb bytes
	// six A-row base pointers R8..R13 = a + r*lda
	MOVQ AX, R8
	LEAQ (AX)(SI*1), R9
	LEAQ (R9)(SI*1), R10
	LEAQ (R10)(SI*1), R11
	LEAQ (R11)(SI*1), R12
	LEAQ (R12)(SI*1), R13
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	VXORPS Y6, Y6, Y6
	VXORPS Y7, Y7, Y7
	VXORPS Y8, Y8, Y8
	VXORPS Y9, Y9, Y9
	VXORPS Y10, Y10, Y10
	VXORPS Y11, Y11, Y11
	TESTQ DX, DX
	JLE   store
kloop:
	VMOVUPS (BX), Y12
	VMOVUPS 32(BX), Y13
	VBROADCASTSS (R8), Y14
	VFMADD231PS  Y12, Y14, Y0
	VFMADD231PS  Y13, Y14, Y1
	VBROADCASTSS (R9), Y14
	VFMADD231PS  Y12, Y14, Y2
	VFMADD231PS  Y13, Y14, Y3
	VBROADCASTSS (R10), Y14
	VFMADD231PS  Y12, Y14, Y4
	VFMADD231PS  Y13, Y14, Y5
	VBROADCASTSS (R11), Y14
	VFMADD231PS  Y12, Y14, Y6
	VFMADD231PS  Y13, Y14, Y7
	VBROADCASTSS (R12), Y14
	VFMADD231PS  Y12, Y14, Y8
	VFMADD231PS  Y13, Y14, Y9
	VBROADCASTSS (R13), Y14
	VFMADD231PS  Y12, Y14, Y10
	VFMADD231PS  Y13, Y14, Y11
	ADDQ $4, R8
	ADDQ $4, R9
	ADDQ $4, R10
	ADDQ $4, R11
	ADDQ $4, R12
	ADDQ $4, R13
	ADDQ DI, BX
	DECQ DX
	JNZ  kloop
store:
	MOVQ c+16(FP), CX
	MOVQ ldc+48(FP), R14
	SHLQ $2, R14         // ldc bytes
	VMOVUPS Y0, (CX)
	VMOVUPS Y1, 32(CX)
	LEAQ (CX)(R14*1), CX
	VMOVUPS Y2, (CX)
	VMOVUPS Y3, 32(CX)
	LEAQ (CX)(R14*1), CX
	VMOVUPS Y4, (CX)
	VMOVUPS Y5, 32(CX)
	LEAQ (CX)(R14*1), CX
	VMOVUPS Y6, (CX)
	VMOVUPS Y7, 32(CX)
	LEAQ (CX)(R14*1), CX
	VMOVUPS Y8, (CX)
	VMOVUPS Y9, 32(CX)
	LEAQ (CX)(R14*1), CX
	VMOVUPS Y10, (CX)
	VMOVUPS Y11, 32(CX)
	VZEROUPPER
	RET

// func gemmF32Tile6x16AVX2Acc(a, b, c *float32, k, lda, ldb, ldc int)
// C[6][16] += Σ_p A[r][p]·B[p][j]  (loads C, accumulates, stores — for kc-blocking).
TEXT ·gemmF32Tile6x16AVX2Acc(SB), NOSPLIT, $0-56
	MOVQ a+0(FP), AX
	MOVQ b+8(FP), BX
	MOVQ c+16(FP), CX
	MOVQ k+24(FP), DX
	MOVQ lda+32(FP), SI
	SHLQ $2, SI
	MOVQ ldb+40(FP), DI
	SHLQ $2, DI
	MOVQ ldc+48(FP), R14
	SHLQ $2, R14
	MOVQ AX, R8
	LEAQ (AX)(SI*1), R9
	LEAQ (R9)(SI*1), R10
	LEAQ (R10)(SI*1), R11
	LEAQ (R11)(SI*1), R12
	LEAQ (R12)(SI*1), R13
	// load C accumulators
	MOVQ CX, R15
	VMOVUPS (R15), Y0
	VMOVUPS 32(R15), Y1
	LEAQ (R15)(R14*1), R15
	VMOVUPS (R15), Y2
	VMOVUPS 32(R15), Y3
	LEAQ (R15)(R14*1), R15
	VMOVUPS (R15), Y4
	VMOVUPS 32(R15), Y5
	LEAQ (R15)(R14*1), R15
	VMOVUPS (R15), Y6
	VMOVUPS 32(R15), Y7
	LEAQ (R15)(R14*1), R15
	VMOVUPS (R15), Y8
	VMOVUPS 32(R15), Y9
	LEAQ (R15)(R14*1), R15
	VMOVUPS (R15), Y10
	VMOVUPS 32(R15), Y11
	TESTQ DX, DX
	JLE   accstore
accloop:
	VMOVUPS (BX), Y12
	VMOVUPS 32(BX), Y13
	VBROADCASTSS (R8), Y14
	VFMADD231PS  Y12, Y14, Y0
	VFMADD231PS  Y13, Y14, Y1
	VBROADCASTSS (R9), Y14
	VFMADD231PS  Y12, Y14, Y2
	VFMADD231PS  Y13, Y14, Y3
	VBROADCASTSS (R10), Y14
	VFMADD231PS  Y12, Y14, Y4
	VFMADD231PS  Y13, Y14, Y5
	VBROADCASTSS (R11), Y14
	VFMADD231PS  Y12, Y14, Y6
	VFMADD231PS  Y13, Y14, Y7
	VBROADCASTSS (R12), Y14
	VFMADD231PS  Y12, Y14, Y8
	VFMADD231PS  Y13, Y14, Y9
	VBROADCASTSS (R13), Y14
	VFMADD231PS  Y12, Y14, Y10
	VFMADD231PS  Y13, Y14, Y11
	ADDQ $4, R8
	ADDQ $4, R9
	ADDQ $4, R10
	ADDQ $4, R11
	ADDQ $4, R12
	ADDQ $4, R13
	ADDQ DI, BX
	DECQ DX
	JNZ  accloop
accstore:
	VMOVUPS Y0, (CX)
	VMOVUPS Y1, 32(CX)
	LEAQ (CX)(R14*1), CX
	VMOVUPS Y2, (CX)
	VMOVUPS Y3, 32(CX)
	LEAQ (CX)(R14*1), CX
	VMOVUPS Y4, (CX)
	VMOVUPS Y5, 32(CX)
	LEAQ (CX)(R14*1), CX
	VMOVUPS Y6, (CX)
	VMOVUPS Y7, 32(CX)
	LEAQ (CX)(R14*1), CX
	VMOVUPS Y8, (CX)
	VMOVUPS Y9, 32(CX)
	LEAQ (CX)(R14*1), CX
	VMOVUPS Y10, (CX)
	VMOVUPS Y11, 32(CX)
	VZEROUPPER
	RET
