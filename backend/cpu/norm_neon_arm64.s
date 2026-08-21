//go:build goexperiment.simd && arm64

#include "textflag.h"

// func rmsNormNormalizeF32BlocksNeon(out, x, gamma *float32, blocks int, inv float32)
TEXT ·rmsNormNormalizeF32BlocksNeon(SB), NOSPLIT, $0-36
	MOVD  out+0(FP), R0
	MOVD  x+8(FP), R1
	MOVD  gamma+16(FP), R2
	MOVD  blocks+24(FP), R3
	FMOVS inv+32(FP), F31
	VDUP  V31.S[0], V31.S4

rms_loop:
	VLD1.P 64(R1), [V0.S4, V1.S4, V2.S4, V3.S4]
	VLD1.P 64(R2), [V4.S4, V5.S4, V6.S4, V7.S4]
	WORD $0x6E3FDC00 // FMUL V0.4S, V0.4S, V31.4S
	WORD $0x6E3FDC21 // FMUL V1.4S, V1.4S, V31.4S
	WORD $0x6E3FDC42 // FMUL V2.4S, V2.4S, V31.4S
	WORD $0x6E3FDC63 // FMUL V3.4S, V3.4S, V31.4S
	WORD $0x6E24DC00 // FMUL V0.4S, V0.4S, V4.4S
	WORD $0x6E25DC21 // FMUL V1.4S, V1.4S, V5.4S
	WORD $0x6E26DC42 // FMUL V2.4S, V2.4S, V6.4S
	WORD $0x6E27DC63 // FMUL V3.4S, V3.4S, V7.4S
	VST1.P [V0.S4, V1.S4, V2.S4, V3.S4], 64(R0)
	SUBS $1, R3, R3
	BNE rms_loop
	RET

// func layerNormNormalizeF32BlocksNeon(out, x, gamma, beta *float32, blocks int, mean, inv float32)
TEXT ·layerNormNormalizeF32BlocksNeon(SB), NOSPLIT, $0-48
	MOVD  out+0(FP), R0
	MOVD  x+8(FP), R1
	MOVD  gamma+16(FP), R2
	MOVD  beta+24(FP), R3
	MOVD  blocks+32(FP), R4
	FMOVS mean+40(FP), F30
	FMOVS inv+44(FP), F31
	VDUP  V30.S[0], V30.S4
	VDUP  V31.S[0], V31.S4

layer_loop:
	VLD1.P 64(R1), [V0.S4, V1.S4, V2.S4, V3.S4]
	VLD1.P 64(R2), [V4.S4, V5.S4, V6.S4, V7.S4]
	VLD1.P 64(R3), [V8.S4, V9.S4, V10.S4, V11.S4]
	WORD $0x4EBED400 // FSUB V0.4S, V0.4S, V30.4S
	WORD $0x4EBED421 // FSUB V1.4S, V1.4S, V30.4S
	WORD $0x4EBED442 // FSUB V2.4S, V2.4S, V30.4S
	WORD $0x4EBED463 // FSUB V3.4S, V3.4S, V30.4S
	WORD $0x6E3FDC00 // FMUL V0.4S, V0.4S, V31.4S
	WORD $0x6E3FDC21 // FMUL V1.4S, V1.4S, V31.4S
	WORD $0x6E3FDC42 // FMUL V2.4S, V2.4S, V31.4S
	WORD $0x6E3FDC63 // FMUL V3.4S, V3.4S, V31.4S
	VFMLA V0.S4, V4.S4, V8.S4
	VFMLA V1.S4, V5.S4, V9.S4
	VFMLA V2.S4, V6.S4, V10.S4
	VFMLA V3.S4, V7.S4, V11.S4
	VST1.P [V8.S4, V9.S4, V10.S4, V11.S4], 64(R0)
	SUBS $1, R4, R4
	BNE layer_loop
	RET
