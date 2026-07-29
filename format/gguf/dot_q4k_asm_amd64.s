//go:build amd64 && goexperiment.simd

#include "textflag.h"

DATA q4dmask<>+0(SB)/4, $0x0000000F
GLOBL q4dmask<>(SB), RODATA, $4

// func dotQ4KRowAsm(x *float32, raw *byte, scales *float32, nsb int) float32
// AX=x, BX=raw(super-block base), DX=scales, SI=nsb. Y15=acc, Y14=nibble mask.
TEXT ·dotQ4KRowAsm(SB), NOSPLIT, $0-36
	MOVQ         x+0(FP), AX
	MOVQ         raw+8(FP), BX
	MOVQ         scales+16(FP), DX
	MOVQ         nsb+24(FP), SI
	VXORPS       Y15, Y15, Y15
	VPBROADCASTD q4dmask<>(SB), Y14

sbloop:
	// R8 = qs base = raw + 16 ; process 4 pairs (32 bytes each) of this super-block
	LEAQ 16(BX), R8
	MOVQ $4, R9   // pair counter
pairloop:
	// scales for this pair: DX[0]=d1, DX[4]=-off1, DX[8]=d2, DX[12]=-off2
	VBROADCASTSS 0(DX), Y10
	VBROADCASTSS 4(DX), Y11
	VBROADCASTSS 8(DX), Y12
	VBROADCASTSS 12(DX), Y13
	MOVQ $4, R10  // 4 groups of 8
grouploop:
	// widen 8 qs bytes -> int32 (0-255)
	VPMOVZXBD (R8), Y0
	VPAND     Y14, Y0, Y1     // low nibble
	VPSRLD    $4, Y0, Y2      // high nibble
	VCVTDQ2PS Y1, Y1
	VCVTDQ2PS Y2, Y2
	// low: acc += (xlo*nib)*d1 + xlo*(-off1)
	VMOVUPS   (AX), Y3        // xlo group
	VMULPS    Y3, Y1, Y1      // xlo*nib
	VFMADD231PS Y10, Y1, Y15  // acc += Y1*d1
	VFMADD231PS Y11, Y3, Y15  // acc += xlo*(-off1)
	// high: xhi is at AX + 32 floats = +128 bytes
	VMOVUPS   128(AX), Y4     // xhi group
	VMULPS    Y4, Y2, Y2      // xhi*nib
	VFMADD231PS Y12, Y2, Y15  // acc += Y2*d2
	VFMADD231PS Y13, Y4, Y15  // acc += xhi*(-off2)
	ADDQ  $8, R8              // next 8 qs bytes
	ADDQ  $32, AX             // next 8 activations (low half)
	DECQ  R10
	JNZ   grouploop
	// after 4 groups: AX advanced by 128 (low half done); skip the high half (already used at +128)
	ADDQ  $128, AX           // advance past the high half → next pair base
	ADDQ  $16, DX            // next pair scales
	DECQ  R9
	JNZ   pairloop
	ADDQ  $144, BX           // next super-block
	DECQ  SI
	JNZ   sbloop

	// horizontal sum Y15 -> ret
	VEXTRACTF128 $1, Y15, X0
	VADDPS       X0, X15, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0
	MOVSS        X0, ret+32(FP)
	VZEROUPPER
	RET
