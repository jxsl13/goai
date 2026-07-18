package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// AvgPool2D is the parameter-free window-mean downsampling layer over NCHW
// tensors: each Kernel×Kernel window (stepped by Stride; 0 → Kernel,
// non-overlapping) is reduced to its arithmetic mean. The sibling of MaxPool2D
// and the layer wrapper over backend.OpAvgPool2D.
//
// Reach for AvgPool2D over MaxPool2D when you want every input in the window to
// influence the output — it keeps gradients flowing to all k² taps instead of
// routing them to a single argmax, which is why it is the standard head pooling
// in ResNet-style classifiers and the downsampler in most modern conv stacks.
// Prefer MaxPool2D when the signal is a sparse "was this feature present
// anywhere" detector, where averaging dilutes the response.
//
// NOT SUPPORTED, deliberately: padding and count_include_pad. The underlying
// backend op (backend.PoolAttrs) exposes only Kernel and Stride, so there is no
// option to set them here rather than an option that would be silently ignored.
// If you need padded pooling, pad the input tensor explicitly before Forward.
// Because there is no padding, windows that would run off the right/bottom edge
// are dropped (floor mode): out = (in − Kernel)/Stride + 1, and an input smaller
// than Kernel is an error, not a zero-padded window.
type AvgPool2D struct {
	Kernel int // pooling window size
	Stride int // window step; 0 → Kernel
}

// Forward pools x[N,C,H,W] through ctx.
func (l *AvgPool2D) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 4 {
		return nil, fmt.Errorf("nn: AvgPool2D expects x rank-4 [N,C,H,W], got %v", x.Shape())
	}
	if l.Kernel <= 0 {
		return nil, fmt.Errorf("nn: AvgPool2D kernel must be > 0, got %d", l.Kernel)
	}
	if l.Stride < 0 {
		return nil, fmt.Errorf("nn: AvgPool2D stride must be >= 0 (0 → kernel), got %d", l.Stride)
	}
	out, err := backend.Execute(ctx, backend.OpAvgPool2D, []*tensor.Tensor{x}, backend.PoolAttrs{Kernel: l.Kernel, Stride: l.Stride})
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Params returns nil — pooling has no trainable tensors.
func (l *AvgPool2D) Params() []*tensor.Tensor { return nil }
