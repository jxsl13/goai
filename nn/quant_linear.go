package nn

import (
	"errors"
	"fmt"
	"sync"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/tensor"
)

// QuantLinear is a linear layer y = x·dequant(W)ᵀ whose weight W [Out, In] is kept in its ggml
// block-quantized byte form and NEVER materialized as a full-precision matrix — the memory
// saving that is the whole point of a quantized model. Forward dequantizes on the fly: it runs
// on the active accelerator when that backend implements backend.QuantMatMuler for this quant
// type (GPU quantized inference, §T142), and otherwise on the CPU reference (gguf.QMatMul),
// with matching results. There is no bias (ggml's linear layers carry none).
type QuantLinear struct {
	Weight []byte         // W [Out, In] in the ggml block layout for QT (row-major, In innermost)
	QT     gguf.QuantType // ggml quant type (Q8_0 / Q4_K / Q6_K / …)
	In     int            // input features (the weight's inner dimension K)
	Out    int            // output features (the weight's outer dimension N)

	// lazy device-resident weight cache: on the first Forward, if the active backend can hold the
	// weight resident (backend.ResidentQuantMatMuler), it is uploaded once and reused across every
	// subsequent call — the decode-loop perf lever (§T154). Never populated when there is no such
	// backend; freed by Close.
	resOnce sync.Once
	res     backend.ResidentWeight
}

// Forward computes y[M,Out] = x[M,In] · dequant(W)ᵀ. It prefers the active backend's accelerated
// quantized matmul, falling back to the CPU path when the backend does not implement one or does
// not accelerate this quant type — so the same call transparently runs on GPU or CPU. A real
// (non-"unsupported") accelerator error propagates rather than being masked by the fallback.
func (q *QuantLinear) Forward(_ *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != q.In {
		return nil, fmt.Errorf("nn: QuantLinear x must be [M,%d], got %v", q.In, x.Shape())
	}
	// Once: try to make the weight device-resident (uploaded a single time). Reused every call after.
	q.resOnce.Do(func() {
		if rb, ok := backend.Default().(backend.ResidentQuantMatMuler); ok {
			if rw, err := rb.UploadQuant(q.Weight, uint32(q.QT), q.Out, q.In); err == nil {
				q.res = rw
			}
		}
	})
	if q.res != nil {
		return q.res.QMatMul(asF32(x))
	}
	if qm, ok := backend.Default().(backend.QuantMatMuler); ok {
		y, err := qm.QMatMul(asF32(x), q.Weight, uint32(q.QT), q.Out, q.In)
		if err == nil {
			return y, nil
		}
		if !errors.Is(err, backend.ErrQuantUnsupported) {
			return nil, err // a genuine accelerator failure, not a missing capability
		}
		// quant type not accelerated on this backend → fall through to the CPU reference
	}
	return gguf.QMatMul(x, q.Weight, q.QT, q.Out, q.In)
}

// Close frees the device-resident weight buffer if one was uploaded. Idempotent; after Close the
// next Forward runs the per-call path again (no re-upload attempt, the once already fired).
func (q *QuantLinear) Close() error {
	if q.res != nil {
		err := q.res.Close()
		q.res = nil
		return err
	}
	return nil
}

// asF32 returns x as a contiguous F32 tensor (the accelerator quantized-matmul kernels are
// f32-only), copying only when x is not already F32.
func asF32(x *tensor.Tensor) *tensor.Tensor {
	if x.Dtype() == tensor.F32 {
		return x.Contiguous()
	}
	out := tensor.New(tensor.F32, x.Shape())
	dst := out.Storage().F32()
	for i := range x.Numel() {
		dst[i] = float32(x.AtF64(tensor.Unravel(i, x.Shape())...))
	}
	return out
}
