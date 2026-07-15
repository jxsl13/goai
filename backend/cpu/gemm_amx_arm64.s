//go:build darwin && goexperiment.simd

#include "textflag.h"

// Raw Apple-AMX 32×32 GEMM tile (ADR-0027, Path B — pure Go, no cgo).
//
// AMX is the undocumented per-cluster matrix coprocessor of the M1–M3
// generations (distinct from ARM SME, which replaced it on M4). Its ops live
// in the A64 reserved-instruction space and are emitted here with the same
// WORD trick ADR-0026 used for by-element FMLA:
//
//	WORD $(0x00201000 | (op << 5) | Rn)
//
// where op selects the AMX instruction (0 ldx, 1 ldy, 4 ldz, 5 stz, 12 fma32,
// 17 set/clr) and Rn is the GPR holding the 64-bit operand descriptor —
// encodings per the corsix/amx reference (cross-checked against its emulator):
//
//	ldx/ldy: ptr[55:0] | reg[58:56]           | pair<<62 | quad<<60 (M2+)
//	stz:     ptr[55:0] | zrow[61:56]          | pair<<62
//	fma32:   yoff[8:0] | xoff[18:10] | zrow[25:20] | skips[29:27] | vec<<63
//	set/clr: three NOPs then imm5 = 0 (set) / 1 (clr)
//
// fma32 in matrix mode (vec=0, no skips) is a 16×16 f32 outer product:
// z[4j + (zrow&3)][i] += x[i]·y[j] — 512 FLOPs per instruction. The 64-row Z
// grid holds four such 16×16 banks (zrow 0..3), so a 32×32 C tile uses the
// whole Z file: X = 32 B-columns (two 16-lane X regs), Y = 32 A-rows (two Y
// regs), 4 fma32 per k-step (one per bank). The k-loop is unrolled ×2 with
// M2 quad loads: ONE ldx pulls 256 B-panel bytes into X0–X3 (two k-steps) and
// one ldy mirrors it for A — 10 AMX ops per 2 k-steps ≈ 4096 FLOPs.
//
// PREEMPTION SAFETY: AMX state is per-thread (XNU context-switches it once
// AMX_SET enables it), but a goroutine that migrated OS threads mid-tile
// would lose the Z accumulators. The whole AMX_SET → load/fma loop → stz →
// AMX_CLR sequence therefore lives inside this ONE non-CALLing NOSPLIT
// function: Go's async preemption never fires inside an assembly frame, and
// with no CALL there is no synchronous preemption point either.
//
// Layout contracts (the Go driver packs both operands):
//   apack: k rows × 32 f32 — row p = A[i0..i0+32][p] (the tile's A column),
//          256-byte aligned (quad loads need it; rows are 128 B).
//   bpack: k rows × 32 f32 — row p = B[p][j0..j0+32], 256-byte aligned.
//   c:     &C[i0*ldc + j0], row stride ldc ELEMENTS; the 32×32 tile is
//          written once (store semantics). k must be ≥ 1.

#define AMX_NOP3 WORD $0xD503201F; WORD $0xD503201F; WORD $0xD503201F
#define AMX_SET  AMX_NOP3; WORD $0x00201220
#define AMX_CLR  AMX_NOP3; WORD $0x00201221
#define AMX_LDY_R5   WORD $0x00201025
#define AMX_LDX_R6   WORD $0x00201006
#define AMX_LDY_R0   WORD $0x00201020
#define AMX_LDX_R1   WORD $0x00201001
#define AMX_FMA32_R7  WORD $0x00201187
#define AMX_FMA32_R8  WORD $0x00201188
#define AMX_FMA32_R9  WORD $0x00201189
#define AMX_FMA32_R10 WORD $0x0020118A
#define AMX_FMA32_R11 WORD $0x0020118B
#define AMX_FMA32_R12 WORD $0x0020118C
#define AMX_FMA32_R13 WORD $0x0020118D
#define AMX_FMA32_R14 WORD $0x0020118E
#define AMX_FMA32_R19 WORD $0x00201193
#define AMX_FMA32_R20 WORD $0x00201194
#define AMX_FMA32_R21 WORD $0x00201195
#define AMX_FMA32_R22 WORD $0x00201196
#define AMX_FMA32_R23 WORD $0x00201197
#define AMX_FMA32_R24 WORD $0x00201198
#define AMX_FMA32_R25 WORD $0x00201199
#define AMX_FMA32_R26 WORD $0x0020119A
#define AMX_STZ_R23   WORD $0x002010B7

// 8 fmas of ping buffer A (X0..X3 × Y0..Y3, k-steps p and p+1)
#define AMX_FMA_BUFA \
	AMX_FMA32_R7;  \
	AMX_FMA32_R8;  \
	AMX_FMA32_R9;  \
	AMX_FMA32_R10; \
	AMX_FMA32_R11; \
	AMX_FMA32_R12; \
	AMX_FMA32_R13; \
	AMX_FMA32_R14

// 8 fmas of pong buffer B (X4..X7 × Y4..Y7, k-steps p+2 and p+3)
#define AMX_FMA_BUFB \
	AMX_FMA32_R19; \
	AMX_FMA32_R20; \
	AMX_FMA32_R21; \
	AMX_FMA32_R22; \
	AMX_FMA32_R23; \
	AMX_FMA32_R24; \
	AMX_FMA32_R25; \
	AMX_FMA32_R26

// func gemmF32TileAMX32x32(apack, bpack, c *float32, k, ldc int)
TEXT ·gemmF32TileAMX32x32(SB), NOSPLIT, $0-40
	MOVD apack+0(FP), R0
	MOVD bpack+8(FP), R1
	MOVD c+16(FP), R2
	MOVD k+24(FP), R3
	MOVD ldc+32(FP), R4
	LSL  $2, R4, R4 // ldc elements -> bytes

	// Load-operand descriptors: quad-load (bits 62|60) four consecutive
	// registers = 256 bytes = TWO packed k-rows per issue. Two descriptor
	// sets ping-pong between register halves 0..3 and 4..7 so each load has
	// a full 8-fma block of independent work to hide its latency behind
	// (software pipelining; measured +23% at 1024³ over the naive
	// load-then-fma loop).
	MOVD $0x5000000000000000, R16
	ORR  R16, R0, R5 // ldy ping: Y0..Y3 <- apack rows p, p+1
	ORR  R16, R1, R6 // ldx ping: X0..X3 <- bpack rows p, p+1
	MOVD $0x5400000000000000, R16
	ADD  $256, R0, R0
	ADD  $256, R1, R1
	ORR  R16, R0, R0 // ldy pong: Y4..Y7 <- apack rows p+2, p+3
	ORR  R16, R1, R1 // ldx pong: X4..X7 <- bpack rows p+2, p+3

	// fma32 matrix-mode operand descriptors (xoff<<10 | yoff | zrow<<20).
	// Ping k-step even: X0/X1 (xoff 0/64) × Y0/Y1 (yoff 0/64) -> banks 0..3;
	// ping k-step odd: X2/X3 (128/192) × Y2/Y3 (128/192) -> the same banks.
	MOVD $0x00000000, R7  // X0·Y0 -> bank 0: C rows 0-15,  cols 0-15
	MOVD $0x00110000, R8  // X1·Y0 -> bank 1: C rows 0-15,  cols 16-31
	MOVD $0x00200040, R9  // X0·Y1 -> bank 2: C rows 16-31, cols 0-15
	MOVD $0x00310040, R10 // X1·Y1 -> bank 3: C rows 16-31, cols 16-31
	MOVD $0x00020080, R11 // X2·Y2 -> bank 0
	MOVD $0x00130080, R12 // X3·Y2 -> bank 1
	MOVD $0x002200C0, R13 // X2·Y3 -> bank 2
	MOVD $0x003300C0, R14 // X3·Y3 -> bank 3
	// Pong: the same grid one register-half up (X/Y offsets +256).
	MOVD $0x00040100, R19 // X4·Y4 -> bank 0
	MOVD $0x00150100, R20 // X5·Y4 -> bank 1
	MOVD $0x00240140, R21 // X4·Y5 -> bank 2
	MOVD $0x00350140, R22 // X5·Y5 -> bank 3
	MOVD $0x00060180, R23 // X6·Y6 -> bank 0
	MOVD $0x00170180, R24 // X7·Y6 -> bank 1
	MOVD $0x002601C0, R25 // X6·Y7 -> bank 2
	MOVD $0x003701C0, R26 // X7·Y7 -> bank 3

	AMX_SET // enables AMX for the thread and zeroes X/Y/Z

	LSR $2, R3, R15 // R15 = k/4 pipelined groups
	CBZ R15, tail

	// Rotated pipeline: ping is loaded before the loop; each iteration loads
	// pong, runs ping's fmas, loads the NEXT ping, runs pong's fmas.
	AMX_LDY_R5 // Y0..Y3 = A columns p, p+1
	AMX_LDX_R6 // X0..X3 = B rows    p, p+1
	ADD  $512, R5, R5
	ADD  $512, R6, R6
	SUBS $1, R15, R15
	BEQ  fin

loop4:
	AMX_LDY_R0 // pong <- k-steps p+2, p+3
	AMX_LDX_R1
	ADD $512, R0, R0
	ADD $512, R1, R1
	AMX_FMA_BUFA
	AMX_LDY_R5 // next ping <- k-steps p+4, p+5
	AMX_LDX_R6
	ADD $512, R5, R5
	ADD $512, R6, R6
	AMX_FMA_BUFB
	SUBS $1, R15, R15
	BNE  loop4

fin:
	AMX_LDY_R0 // final pong
	AMX_LDX_R1
	AMX_FMA_BUFA
	AMX_FMA_BUFB
	// Ping's descriptors were advanced one group past the k/4 interior and
	// now point at the k%4 remainder rows — exactly where tail wants them.

tail:
	AND  $3, R3, R15
	CBZ  R15, store
	// Remainder k-steps one at a time: drop the quad bit (60) -> 128-byte
	// PAIR loads of one packed row (always 128-byte aligned) into X0/X1 and
	// Y0/Y1, then the ping even-step fmas.
	EOR $0x1000000000000000, R5, R5
	EOR $0x1000000000000000, R6, R6

tailloop:
	AMX_LDY_R5
	AMX_LDX_R6
	ADD $128, R5, R5
	ADD $128, R6, R6
	AMX_FMA32_R7
	AMX_FMA32_R8
	AMX_FMA32_R9
	AMX_FMA32_R10
	SUBS $1, R15, R15
	BNE  tailloop

store:
	// C tile row r (0..15):  cols 0-15 in Z[4r+0], cols 16-31 in Z[4r+1];
	// row 16+r: cols 0-15 in Z[4r+2], cols 16-31 in Z[4r+3]. stz stores one
	// 64-byte Z row (no alignment requirement) straight into C.
	ADD  R4<<4, R2, R17 // R17 = c + 16 rows
	MOVD $0, R19        // z-row bits (4r)<<56, += 4<<56 per pass
	MOVD $0x0100000000000000, R20
	MOVD $0x0200000000000000, R21
	MOVD $0x0300000000000000, R22
	MOVD $16, R15

storeloop:
	ORR  R19, R2, R23 // Z[4r+0] -> C[r][0:16]
	AMX_STZ_R23
	ADD  $64, R2, R23 // Z[4r+1] -> C[r][16:32]
	ORR  R19, R23, R23
	ADD  R20, R23, R23
	AMX_STZ_R23
	ORR  R19, R17, R23 // Z[4r+2] -> C[16+r][0:16]
	ADD  R21, R23, R23
	AMX_STZ_R23
	ADD  $64, R17, R23 // Z[4r+3] -> C[16+r][16:32]
	ORR  R19, R23, R23
	ADD  R22, R23, R23
	AMX_STZ_R23
	ADD  R4, R2, R2
	ADD  R4, R17, R17
	ADD  R20<<2, R19, R19
	SUBS $1, R15, R15
	BNE  storeloop

	AMX_CLR // release the AMX unit before returning to Go code
	RET
