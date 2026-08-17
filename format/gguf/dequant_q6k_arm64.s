//go:build arm64

#include "textflag.h"

// BASE identifies one lane of V12..V15 for ARM64's by-element FMUL encoding.
// The four destination vectors retain four consecutive q values each.
#define MUL_Q6(BASE) \
	WORD $(BASE + 0x108); \
	WORD $(BASE + 0x129); \
	WORD $(BASE + 0x14A); \
	WORD $(BASE + 0x16B)

// Convert sixteen signed q6 bytes in SRC to four f32 vectors, multiply by one
// lane of the already-vectorized d*scale products, and store 64 bytes at R4.
#define CONVERT_Q6(SRC, BASE) \
	VTBL V24.B16, [SRC.B16], V8.B16; \
	VTBL V25.B16, [SRC.B16], V9.B16; \
	VTBL V26.B16, [SRC.B16], V10.B16; \
	VTBL V27.B16, [SRC.B16], V11.B16; \
	WORD $0x4F28E508; /* SCVTF V8.4S, V8.4S, #24 */ \
	WORD $0x4F28E529; /* SCVTF V9.4S, V9.4S, #24 */ \
	WORD $0x4F28E54A; /* SCVTF V10.4S, V10.4S, #24 */ \
	WORD $0x4F28E56B; /* SCVTF V11.4S, V11.4S, #24 */ \
	MUL_Q6(BASE); \
	VST1 [V8.S4, V9.S4, V10.S4, V11.S4], (R4)

// Assemble the four q6 streams for one sixteen-byte lane group. Adding 0xe0
// performs the required q-32 in byte arithmetic before signed conversion.
#define EXTRACT_Q6() \
	VAND V21.B16, V0.B16, V3.B16; \
	VSHL $4, V2.B16, V7.B16; \
	VAND V22.B16, V7.B16, V7.B16; \
	VORR V7.B16, V3.B16, V3.B16; \
	VADD V23.B16, V3.B16, V3.B16; \
	VAND V21.B16, V1.B16, V4.B16; \
	VSHL $2, V2.B16, V7.B16; \
	VAND V22.B16, V7.B16, V7.B16; \
	VORR V7.B16, V4.B16, V4.B16; \
	VADD V23.B16, V4.B16, V4.B16; \
	VUSHR $4, V0.B16, V5.B16; \
	VAND V22.B16, V2.B16, V7.B16; \
	VORR V7.B16, V5.B16, V5.B16; \
	VADD V23.B16, V5.B16, V5.B16; \
	VUSHR $4, V1.B16, V6.B16; \
	VUSHR $2, V2.B16, V7.B16; \
	VAND V22.B16, V7.B16, V7.B16; \
	VORR V7.B16, V6.B16, V6.B16; \
	VADD V23.B16, V6.B16, V6.B16

// func dequantQ6KBlockNeon(dst *float32, raw *byte, d float32, indexes *byte)
TEXT ·dequantQ6KBlockNeon(SB), NOSPLIT, $0-32
	MOVD dst+0(FP), R0
	MOVD raw+8(FP), R1
	FMOVS d+16(FP), F28
	VDUP V28.S[0], V28.S4
	MOVD indexes+24(FP), R3

	VLD1 (R3), [V24.B16, V25.B16, V26.B16, V27.B16]
	VMOVI $0x0f, V21.B16
	VMOVI $0x30, V22.B16
	VMOVI $0xe0, V23.B16
	// Turn all sixteen int8 scales into four f32 vectors and preserve the
	// scalar operation order d*scale before multiplying by each q6 value.
	ADD $192, R1, R2
	VLD1 (R2), [V19.B16]
	VTBL V24.B16, [V19.B16], V12.B16
	VTBL V25.B16, [V19.B16], V13.B16
	VTBL V26.B16, [V19.B16], V14.B16
	VTBL V27.B16, [V19.B16], V15.B16
	WORD $0x4F28E58C // SCVTF V12.4S, V12.4S, #24
	WORD $0x4F28E5AD // SCVTF V13.4S, V13.4S, #24
	WORD $0x4F28E5CE // SCVTF V14.4S, V14.4S, #24
	WORD $0x4F28E5EF // SCVTF V15.4S, V15.4S, #24
	WORD $0x6E3CDD8C // FMUL V12.4S, V12.4S, V28.4S
	WORD $0x6E3CDDAD // FMUL V13.4S, V13.4S, V28.4S
	WORD $0x6E3CDDCE // FMUL V14.4S, V14.4S, V28.4S
	WORD $0x6E3CDDEF // FMUL V15.4S, V15.4S, V28.4S

	// ql[0:16], ql[32:48], qh[0:16].
	VLD1 (R1), [V0.B16]
	ADD $32, R1, R5
	VLD1 (R5), [V1.B16]
	ADD $128, R1, R5
	VLD1 (R5), [V2.B16]
	EXTRACT_Q6()
	MOVD R0, R4
	CONVERT_Q6(V3, 0x4F8C9000) // scale 0: V12.S[0]
	ADD $128, R0, R4
	CONVERT_Q6(V4, 0x4F8C9800) // scale 2: V12.S[2]
	ADD $256, R0, R4
	CONVERT_Q6(V5, 0x4F8D9000) // scale 4: V13.S[0]
	ADD $384, R0, R4
	CONVERT_Q6(V6, 0x4F8D9800) // scale 6: V13.S[2]

	// ql[16:32], ql[48:64], qh[16:32].
	ADD $16, R1, R6
	VLD1 (R6), [V0.B16]
	ADD $48, R1, R5
	VLD1 (R5), [V1.B16]
	ADD $144, R1, R5
	VLD1 (R5), [V2.B16]
	EXTRACT_Q6()
	ADD $64, R0, R4
	CONVERT_Q6(V3, 0x4FAC9000) // scale 1: V12.S[1]
	ADD $192, R0, R4
	CONVERT_Q6(V4, 0x4FAC9800) // scale 3: V12.S[3]
	ADD $320, R0, R4
	CONVERT_Q6(V5, 0x4FAD9000) // scale 5: V13.S[1]
	ADD $448, R0, R4
	CONVERT_Q6(V6, 0x4FAD9800) // scale 7: V13.S[3]

	// Second 128-value group: ql[64:], qh[32:], scales[8:].
	ADD $64, R1, R6
	VLD1 (R6), [V0.B16]
	ADD $96, R1, R5
	VLD1 (R5), [V1.B16]
	ADD $160, R1, R5
	VLD1 (R5), [V2.B16]
	EXTRACT_Q6()
	ADD $512, R0, R4
	CONVERT_Q6(V3, 0x4F8E9000) // scale 8: V14.S[0]
	ADD $640, R0, R4
	CONVERT_Q6(V4, 0x4F8E9800) // scale 10: V14.S[2]
	ADD $768, R0, R4
	CONVERT_Q6(V5, 0x4F8F9000) // scale 12: V15.S[0]
	ADD $896, R0, R4
	CONVERT_Q6(V6, 0x4F8F9800) // scale 14: V15.S[2]

	ADD $80, R1, R6
	VLD1 (R6), [V0.B16]
	ADD $112, R1, R5
	VLD1 (R5), [V1.B16]
	ADD $176, R1, R5
	VLD1 (R5), [V2.B16]
	EXTRACT_Q6()
	ADD $576, R0, R4
	CONVERT_Q6(V3, 0x4FAE9000) // scale 9: V14.S[1]
	ADD $704, R0, R4
	CONVERT_Q6(V4, 0x4FAE9800) // scale 11: V14.S[3]
	ADD $832, R0, R4
	CONVERT_Q6(V5, 0x4FAF9000) // scale 13: V15.S[1]
	ADD $960, R0, R4
	CONVERT_Q6(V6, 0x4FAF9800) // scale 15: V15.S[3]
	RET
