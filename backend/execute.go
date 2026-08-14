package backend

import (
	"fmt"
	"os"
	"time"

	"github.com/jxsl13/goai/tensor"
)

// debugFallback (GOAI_LOG_FALLBACK=1) logs every op that falls back from the active backend to the
// reference (§I4) — the diagnostic that found the §T401 CrossEntropy + §T402 Embed silent CPU
// fallbacks. Off by default (a single bool test on the already-slow fallback path).
var debugFallback = os.Getenv("GOAI_LOG_FALLBACK") != ""

// debugTimeOps (GOAI_TIME_OPS=1) logs every op's wall time through the dispatch choke-point as
// "OPTIME <op> <backend> <shape> <ms>" — aggregate with sort|awk to profile a REAL workload's op
// mix (§V22/§T410: standalone op timing misleads; this measures what actually runs). Off by default.
var debugTimeOps = os.Getenv("GOAI_TIME_OPS") != ""

// prepareExecution applies the validation and backend selection shared by
// Execute and ExecuteInto. Keeping one resolver is important: an allocation-free
// path that silently bypassed per-op routing or the CPU-before-reference fallback
// would be fast but semantically different from ordinary eager execution.
func prepareExecution(ctx *Context, op Op, inputs []*tensor.Tensor, attrs Attrs) (*Context, Kernel, tensor.Dtype, error) {
	if ctx == nil {
		ctx = NewContext()
	}
	// Attrs type gate (§B): a MISMATCHED Attrs must not reach a kernel. Kernels read
	// their parameters with `pa, _ := attrs.(backend.FooAttrs)` and drop the ok — which
	// is right for nil (parameterless ops pass nil and the zero value feeds WithDefaults)
	// but turns a wrong type into the ZERO value, i.e. a wrong answer reported as
	// success. Execute is public, so this is reachable from outside the package.
	// Checked once here rather than in ~90 kernels, so it cannot be forgotten by the
	// next one. nil short-circuits before any work, keeping the common parameterless
	// call free.
	if attrs != nil {
		if err := checkAttrs(op, attrs); err != nil {
			return nil, nil, tensor.Invalid, err
		}
	}

	dtype := tensor.Invalid
	if len(inputs) > 0 {
		dtype = inputs[0].Dtype()
	}

	// Per-op backend routing (§C23/§T630): if an override names a backend for this
	// op, re-point ctx to it BEFORE kernel resolution, mirroring the fallback
	// path's ctx = ctx.WithBackend(fb) re-point below. OFF by default: opBackends
	// is nil, so this is a nil-map lookup that changes nothing and every op stays
	// on ctx.Backend (§V16-1, the zero-change-when-unused anchor). We only re-point
	// when the named backend is registered AND actually serves op at dtype;
	// otherwise we leave ctx.Backend untouched and fall through to the normal
	// resolution/fallback chain (§I4), so an override for an unsupported kernel
	// still produces a correct result and never crashes. The boundary transfer is
	// implicit — GPU backends upload/download host-resident inputs per Execute
	// (ADR-0021).
	if name, routed := ctx.opBackends[op]; routed {
		if rb, rok := Get(name); rok && rb != ctx.Backend {
			if _, has := rb.Kernel(op, dtype); has {
				ctx = ctx.WithBackend(rb)
			}
		}
	}

	k, ok := ctx.Backend.Kernel(op, dtype)
	if !ok {
		// Fallback chain (§I4/§T461): prefer the OPTIMIZED CPU backend — its kernels
		// are cross-validated against the reference (§V3) and typically orders of
		// magnitude faster (a GPU backend missing conv2d_backward would otherwise pay
		// the naive reference, §T459) — then the reference, which remains the
		// numerical truth and the guaranteed final fallback.
		var fb Backend
		if cpu, cok := Get(CPU); cok && cpu != ctx.Backend {
			if _, has := cpu.Kernel(op, dtype); has {
				fb = cpu
			}
		}
		if fb == nil {
			ref := Reference()
			if ref == nil {
				return nil, nil, dtype, fmt.Errorf("backend %q: no kernel for %v/%v and no reference backend",
					ctx.Backend.Name(), op, dtype)
			}
			if _, rok := ref.Kernel(op, dtype); !rok {
				return nil, nil, dtype, fmt.Errorf("no kernel for %v/%v (active %q, reference %q)",
					op, dtype, ctx.Backend.Name(), ref.Name())
			}
			fb = ref
		}
		if debugFallback {
			shp := ""
			if len(inputs) > 0 {
				shp = inputs[0].Shape().String()
			}
			fmt.Fprintf(os.Stderr, "FALLBACK[%s→%s] %v %v %s\n", ctx.Backend.Name(), fb.Name(), op, dtype, shp)
		}
		k, _ = fb.Kernel(op, dtype)
		ctx = ctx.WithBackend(fb)
	}
	return ctx, k, dtype, nil
}

// Execute is the single dispatch choke-point (ADR-0003). It resolves the kernel
// for op at the inputs' dtype on ctx.Backend, falling back to the reference
// backend when the active one lacks it (§I4), runs the forward pass, and — if a
// Recorder is attached — hands the op to autograd for taping (§T13).
//
// All ops route through here, so it is the one place fallback, interception, and
// (later) profiling live. Inputs are assumed homogeneous in dtype.
func Execute(ctx *Context, op Op, inputs []*tensor.Tensor, attrs Attrs) ([]*tensor.Tensor, error) {
	ctx, k, _, err := prepareExecution(ctx, op, inputs, attrs)
	if err != nil {
		return nil, err
	}

	var opStart time.Time
	if debugTimeOps {
		opStart = time.Now()
	}
	// Kernels never see the recorder (§V25): recording is Execute's job, done ONCE
	// below. A kernel that re-dispatches (in-kernel fallback or ADR-0008 routing)
	// would otherwise record the op twice and double its gradients — §B49 live,
	// §T537 found 46 latent sites. Enforced here by construction.
	kctx := ctx
	if ctx.Recorder != nil {
		kctx = ctx.WithRecorder(nil)
	}
	out, err := k(kctx, inputs, attrs)
	if debugTimeOps {
		shp := ""
		if len(inputs) > 0 {
			shp = inputs[0].Shape().String()
		}
		fmt.Fprintf(os.Stderr, "OPTIME %v %s %s %.3f\n", op, ctx.Backend.Name(), shp,
			float64(time.Since(opStart).Microseconds())/1000)
	}
	if err != nil {
		return nil, err
	}
	if ctx.Recorder != nil {
		ctx.Recorder.Record(op, inputs, out, attrs)
	}
	return out, nil
}

// validateOutputBase checks the ownership properties common to every
// caller-owned output before an IntoKernel can mutate it. Operation-specific
// dtype and shape checks are performed by ValidateOutput inside the selected
// IntoKernel, where the expected result contract is known.
func validateOutputBase(ctx *Context, out *tensor.Tensor, index int) error {
	if out == nil {
		if index >= 0 {
			return fmt.Errorf("backend: output %d is nil", index)
		}
		return fmt.Errorf("backend: output is nil")
	}
	if out.Storage() == nil {
		if index >= 0 {
			return fmt.Errorf("backend: output %d has no storage", index)
		}
		return fmt.Errorf("backend: output has no storage")
	}
	if out.Storage().IsReleased() {
		if index >= 0 {
			return fmt.Errorf("backend: output %d storage is released", index)
		}
		return fmt.Errorf("backend: output storage is released")
	}
	if out.Device() == nil {
		if index >= 0 {
			return fmt.Errorf("backend: output %d has no device", index)
		}
		return fmt.Errorf("backend: output has no device")
	}
	if out.Offset() != 0 || !out.IsContiguous() {
		if index >= 0 {
			return fmt.Errorf("backend: output %d must be a contiguous base tensor", index)
		}
		return fmt.Errorf("backend: output must be a contiguous base tensor")
	}
	if out.Storage().Len() != out.Numel() {
		if index >= 0 {
			return fmt.Errorf("backend: output %d storage length %d does not match tensor size %d (released or non-base storage)",
				index, out.Storage().Len(), out.Numel())
		}
		return fmt.Errorf("backend: output storage length %d does not match tensor size %d (released or non-base storage)",
			out.Storage().Len(), out.Numel())
	}
	if ctx == nil || ctx.Device() == nil {
		return fmt.Errorf("backend: selected execution context has no device")
	}
	if got, want := out.Device().Kind(), ctx.Device().Kind(); got != want {
		if index >= 0 {
			return fmt.Errorf("backend: output %d device %v does not match execution device %v", index, got, want)
		}
		return fmt.Errorf("backend: output device %v does not match execution device %v", got, want)
	}
	return nil
}

// ValidateOutput checks one operation-specific caller-owned output. IntoKernel
// implementations call it before taking a writable typed slice. ExecuteInto has
// already checked ownership and aliasing, but repeating the cheap base checks
// here keeps direct third-party IntoKernel implementations defensive.
func ValidateOutput(ctx *Context, out *tensor.Tensor, dtype tensor.Dtype, shape tensor.Shape) error {
	if err := validateOutputBase(ctx, out, -1); err != nil {
		return err
	}
	if got := out.Dtype(); got != dtype {
		return fmt.Errorf("backend: output dtype %v does not match result dtype %v", got, dtype)
	}
	if !out.Shape().Equal(shape) {
		return fmt.Errorf("backend: output shape %v does not match result shape %v", out.Shape(), shape)
	}
	return nil
}

// ExecuteInto runs op into caller-owned outputs without allocating result
// storage. Dispatch, routing, fallback, attrs validation, and diagnostics are
// identical to Execute; the backend selected by those rules must additionally
// implement IntoBackend for op/dtype or ErrIntoUnsupported is returned.
//
// ExecuteInto is inference-only. Recorder contexts are rejected because a tape
// may retain an output beyond the caller's next reuse, and input/output or
// output/output storage aliases are rejected until individual kernels prove
// their in-place semantics. Outputs must be live contiguous base tensors on the
// selected execution device. An IntoKernel performs the final count, dtype, and
// shape checks before it writes.
func ExecuteInto(ctx *Context, op Op, inputs, outputs []*tensor.Tensor, attrs Attrs) error {
	if ctx != nil && ctx.Recorder != nil {
		return fmt.Errorf("backend: ExecuteInto does not support recorder contexts")
	}
	ctx, _, dtype, err := prepareExecution(ctx, op, inputs, attrs)
	if err != nil {
		return err
	}
	if len(outputs) == 0 {
		return fmt.Errorf("backend: ExecuteInto requires at least one output")
	}
	for i, out := range outputs {
		if err := validateOutputBase(ctx, out, i); err != nil {
			return err
		}
		for j := 0; j < i; j++ {
			if outputs[j].Storage() == out.Storage() {
				return fmt.Errorf("backend: outputs %d and %d alias the same storage", j, i)
			}
		}
		for j, in := range inputs {
			if in != nil && in.Storage() != nil && in.Storage() == out.Storage() {
				return fmt.Errorf("backend: output %d aliases input %d storage", i, j)
			}
		}
	}

	ib, ok := ctx.Backend.(IntoBackend)
	if !ok {
		return fmt.Errorf("%w: backend %q has no into capability", ErrIntoUnsupported, ctx.Backend.Name())
	}
	k, ok := ib.KernelInto(op, dtype)
	if !ok {
		return fmt.Errorf("%w: backend %q has no into kernel for %v/%v",
			ErrIntoUnsupported, ctx.Backend.Name(), op, dtype)
	}

	var opStart time.Time
	if debugTimeOps {
		opStart = time.Now()
	}
	err = k(ctx, inputs, outputs, attrs)
	if debugTimeOps {
		shp := ""
		if len(inputs) > 0 {
			shp = inputs[0].Shape().String()
		}
		fmt.Fprintf(os.Stderr, "OPTIME %v %s %s %.3f\n", op, ctx.Backend.Name(), shp,
			float64(time.Since(opStart).Microseconds())/1000)
	}
	return err
}
