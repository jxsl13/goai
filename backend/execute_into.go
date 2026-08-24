package backend

import (
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// ExecuteInto executes op into caller-owned outputs when the selected backend
// exposes a native IntoKernel. It preserves the ordinary Execute dispatch and
// fallback semantics; unsupported into forms run through Execute and copy the
// result into outputs. Outputs must be dense, offset-zero tensors with the same
// shape, dtype, and device kind as the produced values.
//
// A recorder sees the caller-owned outputs, so recorded graphs never retain the
// temporary tensors used by the compatibility fallback.
func ExecuteInto(ctx *Context, op Op, inputs, outputs []*tensor.Tensor, attrs Attrs) error {
	if ctx == nil {
		ctx = NewContext()
	}
	if len(outputs) == 0 {
		return fmt.Errorf("backend: ExecuteInto %v requires at least one output", op)
	}
	if attrs != nil {
		if err := checkAttrs(op, attrs); err != nil {
			return err
		}
	}

	dtype := tensor.Invalid
	if len(inputs) > 0 {
		dtype = inputs[0].Dtype()
	}
	// The registered, unrouted path can use the same cached resolution as
	// Execute. Custom backends and per-op routing take the compatibility path;
	// correctness stays identical while the common eager path remains direct.
	if ctx.opBackends == nil {
		table := contextDispatchTable(ctx)
		if table != uncachedDispatchTable {
			resolved, err := cachedDispatch(ctx, op, dtype, table)
			if err != nil {
				return err
			}
			if ib, ok := resolved.backend.(IntoBackend); ok {
				if k, ok := ib.IntoKernel(op, dtype); ok {
					if err := k(resolved.kernelCtx, inputs, outputs, attrs); err != nil {
						return err
					}
					if ctx.Recorder != nil {
						ctx.Recorder.Record(op, inputs, outputs, attrs)
					}
					return nil
				}
			}
		}
	}

	execCtx := ctx
	if ctx.Recorder != nil {
		execCtx = ctx.WithRecorder(nil)
	}
	produced, err := Execute(execCtx, op, inputs, attrs)
	if err != nil {
		return err
	}
	if err := copyIntoOutputs(outputs, produced); err != nil {
		return err
	}
	if ctx.Recorder != nil {
		ctx.Recorder.Record(op, inputs, outputs, attrs)
	}
	return nil
}

func copyIntoOutputs(dst, src []*tensor.Tensor) error {
	if len(dst) != len(src) {
		return fmt.Errorf("backend: ExecuteInto output count %d != produced %d", len(dst), len(src))
	}
	for i := range dst {
		d, s := dst[i], src[i]
		if d == nil || s == nil {
			return fmt.Errorf("backend: ExecuteInto output %d is nil", i)
		}
		if d.Dtype() != s.Dtype() || !d.Shape().Equal(s.Shape()) {
			return fmt.Errorf("backend: ExecuteInto output %d has %v/%v, want %v/%v",
				i, d.Shape(), d.Dtype(), s.Shape(), s.Dtype())
		}
		if !d.IsContiguous() || d.Offset() != 0 || d.Device().Kind() != s.Device().Kind() {
			return fmt.Errorf("backend: ExecuteInto output %d must be dense offset-zero on %v", i, s.Device().Kind())
		}
		s = s.Contiguous()
		switch d.Dtype() {
		case tensor.F32:
			copy(d.Storage().F32(), s.Storage().F32())
		case tensor.F64:
			copy(d.Storage().F64(), s.Storage().F64())
		case tensor.F16, tensor.BF16:
			copy(d.Storage().U16(), s.Storage().U16())
		default:
			return fmt.Errorf("backend: ExecuteInto output %d has unsupported dtype %v", i, d.Dtype())
		}
	}
	return nil
}
