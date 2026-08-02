package backend

import (
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// countingRecorder records how many times Execute taped an op, and what context the kernel saw.
type countingRecorder struct{ n int }

func (c *countingRecorder) Record(Op, []*tensor.Tensor, []*tensor.Tensor, Attrs) { c.n++ }

// redispatchBackend serves one op with a kernel that dispatches a SECOND op through Execute — the
// in-kernel fallback and ADR-0008 routing shape. It is the only way to observe whether the context
// a kernel receives still carries the recorder.
type redispatchBackend struct{ sawRecorder bool }

func (r *redispatchBackend) Name() Name            { return Name("redispatch-test") }
func (r *redispatchBackend) Device() tensor.Device { return tensor.CPU() }
func (r *redispatchBackend) Synchronize() error    { return nil }
func (r *redispatchBackend) Kernel(op Op, dt tensor.Dtype) (Kernel, bool) {
	if dt != tensor.F64 {
		return nil, false
	}
	switch op {
	case OpNeg: // the outer op: re-dispatches
		return func(ctx *Context, in []*tensor.Tensor, at Attrs) ([]*tensor.Tensor, error) {
			if ctx.Recorder != nil {
				r.sawRecorder = true
			}
			return Execute(ctx, OpAbs, in, at)
		}, true
	case OpAbs: // the inner op: plain identity
		return func(_ *Context, in []*tensor.Tensor, _ Attrs) ([]*tensor.Tensor, error) {
			return []*tensor.Tensor{in[0]}, nil
		}, true
	}
	return nil, false
}

// TestKernelsNeverSeeTheRecorder pins §V25 — the property Execute's recorder-free kernel context
// exists to guarantee, and the one the memoized twin must not break.
//
// A kernel that re-dispatches is the only observer of it. If the context handed to a kernel still
// carried the recorder, the inner Execute would tape a second node for one forward op and its
// gradient would be counted twice. Nothing else in the suite exercises that: a mutation making the
// twin carry the recorder left every autograd and nn test green, which is why this exists.
//
// Both halves are asserted: the kernel must see a nil Recorder, and exactly ONE node must be taped
// for the two dispatches.
func TestKernelsNeverSeeTheRecorder(t *testing.T) {
	be := &redispatchBackend{}
	rec := &countingRecorder{}
	ctx := NewContext().WithBackend(be).WithRecorder(rec)
	x := tensor.New(tensor.F64, tensor.Shape{2})
	if _, err := Execute(ctx, OpNeg, []*tensor.Tensor{x}, nil); err != nil {
		t.Fatal(err)
	}
	if be.sawRecorder {
		t.Fatal("the kernel was handed a context carrying the recorder — a re-dispatching kernel" +
			" will tape its op twice and double its gradient")
	}
	if rec.n != 1 {
		t.Fatalf("%d nodes taped for one forward op, want 1", rec.n)
	}
}

// TestKernelContextKeepsPerOpRouting pins the other half of the twin's contents: the recorder-free
// context must carry the per-op backend routing, or a kernel that re-dispatches would silently
// leave the backend its caller chose.
func TestKernelContextKeepsPerOpRouting(t *testing.T) {
	be := &redispatchBackend{}
	Register(be)
	ctx := NewContext().WithBackend(be).WithOpBackend(OpAbs, Name("redispatch-test")).
		WithRecorder(&countingRecorder{})
	if ctx.noRec == nil {
		t.Fatal("a recording context has no recorder-free twin")
	}
	if got := ctx.noRec.opBackends[OpAbs]; got != Name("redispatch-test") {
		t.Fatalf("the twin's per-op routing is %q, want the parent's", got)
	}
}
