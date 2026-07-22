package nn

import (
	"sync"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Recorder-guarded input-slice pools for nn modules on the inference hot path (T963,
// the nlp T960 pattern applied in nn). backend.Execute retains its inputs slice only via
// ctx.Recorder.Record, so with a nil Recorder — every inference DecodeStep/Prefill — the
// slice is dead the instant Execute returns and can go back to the pool. Under a recorder
// (training) the helpers pass a fresh slice the tape node can keep. RMSNorm/LayerNorm run
// twice per layer in every model, so pooling their input slice speeds every decode.
var (
	nnIns1Pool = sync.Pool{New: func() any { s := make([]*tensor.Tensor, 1); return &s }}
	nnIns2Pool = sync.Pool{New: func() any { s := make([]*tensor.Tensor, 2); return &s }}
	nnIns3Pool = sync.Pool{New: func() any { s := make([]*tensor.Tensor, 3); return &s }}
)

// execPool1 runs a 1-input op, reusing a pooled input slice when not recording.
func execPool1(ctx *backend.Context, op backend.Op, attrs backend.Attrs, a *tensor.Tensor) (*tensor.Tensor, error) {
	if ctx.Recorder != nil {
		out, err := backend.Execute(ctx, op, []*tensor.Tensor{a}, attrs)
		if err != nil {
			return nil, err
		}
		return out[0], nil
	}
	sp := nnIns1Pool.Get().(*[]*tensor.Tensor)
	s := *sp
	s[0] = a
	out, err := backend.Execute(ctx, op, s, attrs)
	s[0] = nil
	nnIns1Pool.Put(sp)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// execPool2 runs a 2-input op, reusing a pooled input slice when not recording.
func execPool2(ctx *backend.Context, op backend.Op, attrs backend.Attrs, a, b *tensor.Tensor) (*tensor.Tensor, error) {
	if ctx.Recorder != nil {
		out, err := backend.Execute(ctx, op, []*tensor.Tensor{a, b}, attrs)
		if err != nil {
			return nil, err
		}
		return out[0], nil
	}
	sp := nnIns2Pool.Get().(*[]*tensor.Tensor)
	s := *sp
	s[0], s[1] = a, b
	out, err := backend.Execute(ctx, op, s, attrs)
	s[0], s[1] = nil, nil // don't pin the input tensors alive in the pool
	nnIns2Pool.Put(sp)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// execPool3 runs a 3-input op, reusing a pooled input slice when not recording.
func execPool3(ctx *backend.Context, op backend.Op, attrs backend.Attrs, a, b, c *tensor.Tensor) (*tensor.Tensor, error) {
	if ctx.Recorder != nil {
		out, err := backend.Execute(ctx, op, []*tensor.Tensor{a, b, c}, attrs)
		if err != nil {
			return nil, err
		}
		return out[0], nil
	}
	sp := nnIns3Pool.Get().(*[]*tensor.Tensor)
	s := *sp
	s[0], s[1], s[2] = a, b, c
	out, err := backend.Execute(ctx, op, s, attrs)
	s[0], s[1], s[2] = nil, nil, nil
	nnIns3Pool.Put(sp)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}
