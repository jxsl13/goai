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
}

// NewContext returns an eager context bound to the default backend, with no
// recorder.
func NewContext() *Context { return &Context{Backend: Default()} }

// WithBackend returns a copy of ctx bound to b (recorder preserved).
func (c *Context) WithBackend(b Backend) *Context {
	return &Context{Backend: b, Recorder: c.Recorder}
}

// WithRecorder returns a copy of ctx with the given recorder (backend
// preserved). Used by autograd to switch a computation into recording mode.
func (c *Context) WithRecorder(r Recorder) *Context {
	return &Context{Backend: c.Backend, Recorder: r}
}

// Device returns the device of the context's backend.
func (c *Context) Device() tensor.Device { return c.Backend.Device() }
