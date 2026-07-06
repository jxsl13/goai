package backend

import (
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// Execute is the single dispatch choke-point (ADR-0003). It resolves the kernel
// for op at the inputs' dtype on ctx.Backend, falling back to the reference
// backend when the active one lacks it (§I4), runs the forward pass, and — if a
// Recorder is attached — hands the op to autograd for taping (§T13).
//
// All ops route through here, so it is the one place fallback, interception, and
// (later) profiling live. Inputs are assumed homogeneous in dtype.
func Execute(ctx *Context, op Op, inputs []*tensor.Tensor, attrs Attrs) ([]*tensor.Tensor, error) {
	if ctx == nil {
		ctx = NewContext()
	}
	dtype := tensor.Invalid
	if len(inputs) > 0 {
		dtype = inputs[0].Dtype()
	}

	k, ok := ctx.Backend.Kernel(op, dtype)
	if !ok {
		// Fallback to the reference backend (§I4). Run it in a context bound to
		// the reference so kernels allocate on the reference device.
		ref := Reference()
		if ref == nil {
			return nil, fmt.Errorf("backend %q: no kernel for %v/%v and no reference backend",
				ctx.Backend.Name(), op, dtype)
		}
		rk, rok := ref.Kernel(op, dtype)
		if !rok {
			return nil, fmt.Errorf("no kernel for %v/%v (active %q, reference %q)",
				op, dtype, ctx.Backend.Name(), ref.Name())
		}
		k = rk
		ctx = ctx.WithBackend(ref)
	}

	out, err := k(ctx, inputs, attrs)
	if err != nil {
		return nil, err
	}
	if ctx.Recorder != nil {
		ctx.Recorder.Record(op, inputs, out, attrs)
	}
	return out, nil
}
