//go:build goexperiment.simd

#include "textflag.h"

// func rowMaxF32BlocksNeon(lanes, x *float32, blocks int, negInf float32)
//
// Four independent 4-lane FMAXNM chains hide reduction latency. FMAXNM has
// the scalar loop's NaN-skipping behavior; the Go driver repairs its signed
// zero tie rule and reduces the 16 lane maxima in scalar order.
TEXT ·rowMaxF32BlocksNeon(SB), NOSPLIT, $0-28
	MOVD  lanes+0(FP), R0
	MOVD  x+8(FP), R1
	MOVD  blocks+16(FP), R2
	FMOVS negInf+24(FP), F28
	VDUP  V28.S[0], V28.S4
	VORR  V28.B16, V28.B16, V29.B16
	VORR  V28.B16, V28.B16, V30.B16
	VORR  V28.B16, V28.B16, V31.B16

row_max_loop:
	VLD1.P 64(R1), [V0.S4, V1.S4, V2.S4, V3.S4]
	WORD $0x4E20C79C // FMAXNM V28.4S, V28.4S, V0.4S
	WORD $0x4E21C7BD // FMAXNM V29.4S, V29.4S, V1.4S
	WORD $0x4E22C7DE // FMAXNM V30.4S, V30.4S, V2.4S
	WORD $0x4E23C7FF // FMAXNM V31.4S, V31.4S, V3.4S
	SUBS $1, R2, R2
	BNE  row_max_loop
	VST1 [V28.S4, V29.S4, V30.S4, V31.S4], (R0)
	RET

// func scaleRowF32BlocksNeon(x *float32, blocks int, scale float32)
TEXT ·scaleRowF32BlocksNeon(SB), NOSPLIT, $0-20
	MOVD  x+0(FP), R0
	MOVD  blocks+8(FP), R1
	FMOVS scale+16(FP), F30
	VDUP  V30.S[0], V30.S4

scale_loop:
	VLD1  (R0), [V0.S4, V1.S4, V2.S4, V3.S4]
	WORD $0x6E3EDC00 // FMUL V0.4S, V0.4S, V30.4S
	WORD $0x6E3EDC21 // FMUL V1.4S, V1.4S, V30.4S
	WORD $0x6E3EDC42 // FMUL V2.4S, V2.4S, V30.4S
	WORD $0x6E3EDC63 // FMUL V3.4S, V3.4S, V30.4S
	VST1.P [V0.S4, V1.S4, V2.S4, V3.S4], 64(R0)
	SUBS $1, R1, R1
	BNE  scale_loop
	RET

// func axpbRowF32BlocksNeon(x *float32, blocks int, a, b float32)
TEXT ·axpbRowF32BlocksNeon(SB), NOSPLIT, $0-24
	MOVD  x+0(FP), R0
	MOVD  blocks+8(FP), R1
	FMOVS a+16(FP), F30
	FMOVS b+20(FP), F31
	VDUP  V30.S[0], V30.S4
	VDUP  V31.S[0], V31.S4

axpb_loop:
	VLD1 (R0), [V0.S4, V1.S4, V2.S4, V3.S4]
	VORR V31.B16, V31.B16, V4.B16
	VORR V31.B16, V31.B16, V5.B16
	VORR V31.B16, V31.B16, V6.B16
	VORR V31.B16, V31.B16, V7.B16
	VFMLA V0.S4, V30.S4, V4.S4
	VFMLA V1.S4, V30.S4, V5.S4
	VFMLA V2.S4, V30.S4, V6.S4
	VFMLA V3.S4, V30.S4, V7.S4
	VST1.P [V4.S4, V5.S4, V6.S4, V7.S4], 64(R0)
	SUBS $1, R1, R1
	BNE  axpb_loop
	RET
