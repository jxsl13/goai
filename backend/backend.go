// Package backend is layer L1: the compute abstraction (ADR-0003).
//
// A Backend exposes kernels keyed by (Op, Dtype); all execution funnels through
// Execute, the single choke-point where fallback (§I4) and autograd interception
// (§T13) happen. The Pure-Go reference backend (subpackage ref) is the source of
// numeric truth (§V9).
package backend

import "github.com/jxsl13/goai/tensor"

// Kernel executes one primitive op: it reads inputs, allocates and writes its
// outputs (on ctx's device), and returns them. A Kernel is pure with respect to
// autograd — it records nothing; interception happens in Execute (ADR-0003).
type Kernel func(ctx *Context, inputs []*tensor.Tensor, attrs Attrs) ([]*tensor.Tensor, error)

// Backend is a named compute implementation for one device.
//
// Execution/sync model (§V14): Execute returns output tensors that are final
// once the backend has synchronized. Synchronous backends (the CPU reference,
// SIMD) complete work before Execute returns, so Synchronize is a no-op. An
// asynchronous backend (a future GPU) may enqueue work and return handle
// tensors whose data is valid only after Synchronize returns nil. Higher layers
// therefore call Synchronize before reading results across a backend boundary —
// a rule fixed here so adding a GPU backend never breaks the API (§V8, closes
// B7).
type Backend interface {
	Name() Name
	Device() tensor.Device
	// Kernel returns the kernel for op at dtype, or ok=false if unsupported.
	Kernel(op Op, dtype tensor.Dtype) (k Kernel, ok bool)
	// Synchronize blocks until all previously submitted work has completed.
	Synchronize() error
}

// Recorder is autograd's interception hook (§T13). When set on a Context, it is
// called by Execute after each successful forward op with the op, its inputs,
// its outputs and attrs — enough to append a tape node and a VJP rule. Nil in
// eager mode. This is the single seam that lets L2 record L1 ops without any L1
// changes (ADR-0003).
type Recorder interface {
	Record(op Op, inputs, outputs []*tensor.Tensor, attrs Attrs)
}

// Context threads the active backend and the optional autograd Recorder through
// a computation. Kernels allocate outputs on the context device.
type Context struct {
	Backend  Backend  // the backend that executes ops in this context
	Recorder Recorder // optional autograd tape recording ops for backprop; nil = no taping

	// opBackends is the optional per-op backend-routing override (§C23/§T630): a
	// map from Op to the Name of the backend that op must run on, overriding
	// Backend for exactly the listed ops. nil (the default) means no routing —
	// Execute reads it as a nil-map lookup that changes nothing, so a Context
	// with no override behaves BIT-IDENTICALLY to before the mechanism existed
	// (§V16-1). Populate it copy-on-write via WithOpBackend/WithLayerBackend;
	// never mutate a shared map in place. See Execute for the dispatch hook and
	// ADR-0021 for why a per-op split is OFF by default (transfer ≫ compute on a
	// fits-in-VRAM device — the mechanism exists for T631 low-VRAM layer offload
	// and heterogeneous spec-decoding, not as a default speedup).
	opBackends map[Op]Name

	// noRec is this context's recorder-free twin, built once at construction when a Recorder is
	// attached. Execute must hand kernels a context WITHOUT the recorder (§V25) — a kernel that
	// re-dispatches would otherwise record the op twice and double its gradients — and deriving
	// that twin per op allocated a Context on every dispatch of every training step. It is
	// immutable and shared, which is safe because a Context is read-only once built.
	noRec *Context
}

// NewContext returns an eager context bound to the default backend, with no
// recorder.
func NewContext() *Context { return &Context{Backend: Default()} }

// WithBackend returns a copy of ctx bound to b (recorder and per-op routing
// overrides preserved).
func (c *Context) WithBackend(b Backend) *Context {
	return newContext(b, c.Recorder, c.opBackends)
}

// newContext builds a Context and, when it records, its recorder-free twin. Every constructor goes
// through here so no path can produce a recording context whose twin is missing — Execute falls
// back to deriving one when noRec is nil, so a missed path would be correct but would quietly give
// back the allocation this exists to remove.
func newContext(b Backend, r Recorder, ob map[Op]Name) *Context {
	c := &Context{Backend: b, Recorder: r, opBackends: ob}
	if r != nil {
		c.noRec = &Context{Backend: b, opBackends: ob}
	}
	return c
}

// WithRecorder returns a copy of ctx with the given recorder (backend and per-op
// routing overrides preserved). Used by autograd to switch a computation into
// recording mode.
func (c *Context) WithRecorder(r Recorder) *Context {
	return newContext(c.Backend, r, c.opBackends)
}

// WithOpBackend returns a copy of ctx that routes op to the backend registered
// under name, leaving every other op on ctx.Backend (§C23/§T630). It is
// copy-on-write: the returned Context gets a fresh opBackends map, so derived
// contexts never interfere and a shared ctx's map is never mutated. Use the
// typed backend.Name constants (backend.CPU/Ref/Metal/CUDA/Vulkan), never a
// string literal (§C15). Routing is a no-op unless the named backend is
// registered and serves op at the input dtype; otherwise Execute falls through
// to the normal resolution/fallback chain (§I4), so an override never crashes.
//
// This is the whole-op routing primitive. The boundary transfer is IMPLICIT:
// GPU backends upload/download their host-resident inputs per Execute, so
// routing an op elsewhere just runs that backend's Execute path — no new
// transfer plumbing. OFF by default and a LOSS on a fits-in-VRAM device
// (ADR-0021); it exists for T631 low-VRAM offload and heterogeneous
// spec-decoding.
func (c *Context) WithOpBackend(op Op, name Name) *Context {
	m := make(map[Op]Name, len(c.opBackends)+1)
	for k, v := range c.opBackends {
		m[k] = v
	}
	m[op] = name
	return newContext(c.Backend, c.Recorder, m)
}

// WithLayerBackend is the per-layer convenience over WithOpBackend: it routes
// every op in ops to the backend named name in a single copy-on-write step —
// e.g. pinning a whole attention or MLP block to one device for T631 layer
// offload. Passing no ops returns an equivalent copy. See WithOpBackend for the
// routing semantics and the off-by-default rationale.
func (c *Context) WithLayerBackend(name Name, ops ...Op) *Context {
	m := make(map[Op]Name, len(c.opBackends)+len(ops))
	for k, v := range c.opBackends {
		m[k] = v
	}
	for _, op := range ops {
		m[op] = name
	}
	return newContext(c.Backend, c.Recorder, m)
}

// Device returns the device of the context's backend.
func (c *Context) Device() tensor.Device { return c.Backend.Device() }
