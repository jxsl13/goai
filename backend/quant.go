package backend

import (
	"errors"

	"github.com/jxsl13/goai/tensor"
)

// ErrQuantUnsupported is returned by a QuantMatMuler for a ggml quant type it does not
// accelerate, signalling the caller to fall back to a CPU quantized matmul rather than treating
// it as a hard failure. Any OTHER error from QMatMul is a real fault and must propagate.
var ErrQuantUnsupported = errors.New("backend: quant type not accelerated")

// QuantMatMuler is an OPTIONAL capability a Backend may implement to run a quantized linear
// layer on its accelerator: y[M,N] = x[M,K] · dequant(weight)ᵀ with the weight kept in its
// ggml block-quantized byte form and dequantized IN-KERNEL, instead of materializing a
// full-precision weight matrix (§T142). quantType is the ggml type code (8=Q8_0, 12=Q4_K,
// 14=Q6_K, …); a backend returns ErrQuantUnsupported for a code it does not accelerate so
// callers fall back to the CPU path. This is a capability probe, not part of the core Backend
// interface — higher layers type-assert Default() to it (like Recorder), so a backend that
// lacks it, or lacks a given quant type, simply degrades to CPU with identical results.
type QuantMatMuler interface {
	QMatMul(x *tensor.Tensor, weight []byte, quantType uint32, n, k int) (*tensor.Tensor, error)
}

// ResidentWeight is a quantized weight uploaded ONCE to a device-resident accelerator buffer, run
// against many activations via QMatMul without re-uploading — the perf lever for the decode loop,
// which reuses every weight for every token (§T154). Close frees the device buffer.
type ResidentWeight interface {
	QMatMul(x *tensor.Tensor) (*tensor.Tensor, error)
	Close() error
}

// ResidentQuantMatMuler is an OPTIONAL capability (extends QuantMatMuler): a backend that can hold
// a quantized weight resident on the device. Higher layers (nn.QuantLinear) type-assert Default()
// to it and, when present, upload each weight once and reuse the handle. UploadQuant returns
// ErrQuantUnsupported for a quant type it does not accelerate resident, so the caller falls back
// to the per-call QuantMatMuler path.
type ResidentQuantMatMuler interface {
	UploadQuant(weight []byte, quantType uint32, n, k int) (ResidentWeight, error)
}
