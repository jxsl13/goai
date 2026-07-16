//go:build cuda && cgo && (linux || windows)

package cuda

import (
	"fmt"

	"github.com/jxsl13/goai/format/gguf"
)

// ResidentQuant is the common interface of every GPU-resident quantized weight the backend builds
// natively (Q4_0, Q2_K–Q6_K, MXFP4, and the IQ1–IQ4 i-quant family). It is the decode-GEMV contract:
// out = a·dequant(W) (QMatMulInto, beta=0) or c += a·dequant(W) (QMatMulAccInto, beta=1), plus Free.
type ResidentQuant interface {
	QMatMulInto(a, out *DeviceF32) error
	QMatMulAccInto(a, c *DeviceF32) error
	Free()
}

// wrapQuant lifts a concrete (*ResidentB*, error) constructor result to (ResidentQuant, error),
// returning a genuine nil interface on error (not a non-nil interface wrapping a nil pointer).
func wrapQuant[T ResidentQuant](r T, err error) (ResidentQuant, error) {
	if err != nil {
		return nil, err
	}
	return r, nil
}

// NewResidentQuant builds the NATIVE GPU-resident weight for a GGUF quant tensor, dispatching on its
// ggml type — the public, one-call form of the load path the tests exercise via quantDirect. It
// uploads the raw ggml blocks bit-native (no dequantize→requantize), so a Q4_K_M / IQ-quant GGUF
// loads with zero requantization loss. qt.Shape is [out, in]; qt.Data is the raw block bytes.
//
// Formats WITHOUT a native kernel (F32/F16, Q4_1, Q5_0/Q5_1, Q8_0, …) return an error; the caller
// can dequantize (gguf.QuantTensor.Dequantize) and use NewResidentBQ8 for those. This is the
// backend/cuda primitive; wiring it into the llamagpu decoder is a separate (coordinated) step.
func NewResidentQuant(qt gguf.QuantTensor) (ResidentQuant, error) {
	if qt.Shape.Ndim() != 2 {
		return nil, fmt.Errorf("cuda: NewResidentQuant needs a rank-2 weight, got %dD", qt.Shape.Ndim())
	}
	n, k := qt.Shape[0], qt.Shape[1]
	switch qt.GGType {
	case 2: // Q4_0
		return wrapQuant(NewResidentBQ40FromBlocks(qt.Data, k, n))
	case 10: // Q2_K
		return wrapQuant(NewResidentBQ2KFromBlocks(qt.Data, k, n))
	case 11: // Q3_K
		return wrapQuant(NewResidentBQ3KFromBlocks(qt.Data, k, n))
	case 12: // Q4_K
		return wrapQuant(NewResidentBQ4KFromBlocks(qt.Data, k, n))
	case 13: // Q5_K
		return wrapQuant(NewResidentBQ5KFromBlocks(qt.Data, k, n))
	case 14: // Q6_K
		return wrapQuant(NewResidentBQ6KFromBlocks(qt.Data, k, n))
	case 16: // IQ2_XXS
		return wrapQuant(NewResidentBIQ2XXSFromBlocks(qt.Data, k, n))
	case 17: // IQ2_XS
		return wrapQuant(NewResidentBIQ2XSFromBlocks(qt.Data, k, n))
	case 18: // IQ3_XXS
		return wrapQuant(NewResidentBIQ3XXSFromBlocks(qt.Data, k, n))
	case 21: // IQ3_S
		return wrapQuant(NewResidentBIQ3SFromBlocks(qt.Data, k, n))
	case 20: // IQ4_NL
		return wrapQuant(NewResidentBIQ4NLFromBlocks(qt.Data, k, n))
	case 23: // IQ4_XS
		return wrapQuant(NewResidentBIQ4XSFromBlocks(qt.Data, k, n))
	case 24: // IQ1_S
		return wrapQuant(NewResidentBIQ1SFromBlocks(qt.Data, k, n))
	case 29: // IQ1_M
		return wrapQuant(NewResidentBIQ1MFromBlocks(qt.Data, k, n))
	case 39: // MXFP4
		return wrapQuant(NewResidentBMXFP4FromBlocks(qt.Data, k, n))
	}
	return nil, fmt.Errorf("cuda: NewResidentQuant has no native kernel for ggml type %d (dequantize + use NewResidentBQ8)", qt.GGType)
}
