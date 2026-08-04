package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Conv2D is the 2-D convolution layer y = conv(x, W) + b over NCHW tensors
// (§T456): x[N,C,H,W], W[F,C,KH,KW], b[F], output [N,F,H',W'] with
// H' = (H+2·Pad−KH)/Stride + 1 (cross-correlation with zero padding, the deep-
// learning convention — identical to torch.nn.Conv2d). Both forward and the
// fused backward run through the backend dispatch (OpConv2D/OpConv2DBackward),
// so gradients flow with zero layer-specific autograd code and the layer runs
// on whichever backend is active.
type Conv2D struct {
	W      *tensor.Tensor // [F, C, KH, KW]
	B      *tensor.Tensor // [F]; nil → no bias
	Stride int            // window step in both spatial dims; 0 → 1
	Pad    int            // zero padding on every spatial edge
}

// NewConv2D builds a Conv2D layer with Kaiming-normal W (He 2015 — the ReLU-net
// default, fan-in = C·KH·KW) and zero bias. Deterministic via seed.
func NewConv2D(dtype tensor.Dtype, inC, outF, kernel, stride, pad int, seed uint64) (*Conv2D, error) {
	if inC <= 0 || outF <= 0 || kernel <= 0 {
		return nil, fmt.Errorf("nn: NewConv2D needs inC, outF, kernel > 0 (got %d, %d, %d)", inC, outF, kernel)
	}
	if stride < 0 || pad < 0 {
		return nil, fmt.Errorf("nn: NewConv2D stride/pad must be ≥ 0 (got %d, %d)", stride, pad)
	}
	w := tensor.New(dtype, tensor.Shape{outF, inC, kernel, kernel})
	KaimingNormal(w, inC*kernel*kernel, seed)
	b := tensor.New(dtype, tensor.Shape{outF})
	Zeros(b)
	return &Conv2D{W: w, B: b, Stride: stride, Pad: pad}, nil
}

// Forward convolves x[N,C,H,W] through ctx.
func (l *Conv2D) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 4 {
		return nil, fmt.Errorf("nn: Conv2D expects x rank-4 [N,C,H,W], got %v", x.Shape())
	}
	if x.Shape()[1] != l.W.Shape()[1] {
		return nil, fmt.Errorf("nn: Conv2D in-channels %d != x channels %d", l.W.Shape()[1], x.Shape()[1])
	}
	ins := []*tensor.Tensor{x, l.W}
	if l.B != nil {
		ins = append(ins, l.B)
	}
	out, err := backend.Execute(ctx, backend.OpConv2D, ins, backend.ConvAttrs{Stride: l.Stride, Pad: l.Pad})
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Params returns the trainable tensors (W, and B if present).
func (l *Conv2D) Params() []*tensor.Tensor {
	if l.B == nil {
		return []*tensor.Tensor{l.W}
	}
	return []*tensor.Tensor{l.W, l.B}
}

// MaxPool2D is the parameter-free window-max downsampling layer over NCHW
// tensors: each Kernel×Kernel window (stepped by Stride; 0 → Kernel,
// non-overlapping) is reduced to its maximum. The standard CNN downsampler
// between convolution stages.
type MaxPool2D struct {
	Kernel int // pooling window size
	Stride int // window step; 0 → Kernel
}

// Forward pools x[N,C,H,W] through ctx.
func (l *MaxPool2D) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 4 {
		return nil, fmt.Errorf("nn: MaxPool2D expects x rank-4 [N,C,H,W], got %v", x.Shape())
	}
	if l.Kernel <= 0 {
		return nil, fmt.Errorf("nn: MaxPool2D kernel must be > 0, got %d", l.Kernel)
	}
	//perfscan:ignore PS3038 resource-only, time flat (alloc-only)
	out, err := backend.Execute(ctx, backend.OpMaxPool2D, []*tensor.Tensor{x}, backend.PoolAttrs{Kernel: l.Kernel, Stride: l.Stride})
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Params returns nil — pooling has no trainable tensors.
func (l *MaxPool2D) Params() []*tensor.Tensor { return nil }
