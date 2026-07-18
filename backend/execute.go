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
			return nil, err
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
				return nil, fmt.Errorf("backend %q: no kernel for %v/%v and no reference backend",
					ctx.Backend.Name(), op, dtype)
			}
			if _, rok := ref.Kernel(op, dtype); !rok {
				return nil, fmt.Errorf("no kernel for %v/%v (active %q, reference %q)",
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
