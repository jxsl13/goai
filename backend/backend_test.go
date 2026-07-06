package backend

import (
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// idKernel is an identity kernel: it returns its inputs unchanged. Enough to
// exercise dispatch, fallback, and recording without real math (that is §T6+).
func idKernel(_ *Context, in []*tensor.Tensor, _ Attrs) ([]*tensor.Tensor, error) {
	return in, nil
}

type mockBackend struct {
	name  string
	table map[kernelKeyT]Kernel
}

type kernelKeyT struct {
	op    Op
	dtype tensor.Dtype
}

func (m *mockBackend) Name() string            { return m.name }
func (m *mockBackend) Device() tensor.Device   { return tensor.CPU() }
func (m *mockBackend) Synchronize() error      { return nil }
func (m *mockBackend) Kernel(op Op, dt tensor.Dtype) (Kernel, bool) {
	k, ok := m.table[kernelKeyT{op, dt}]
	return k, ok
}

var (
	refMock = &mockBackend{name: "refmock", table: map[kernelKeyT]Kernel{
		{OpNeg, tensor.F64}: idKernel, // only the reference serves OpNeg
	}}
	activeMock = &mockBackend{name: "active", table: map[kernelKeyT]Kernel{
		{OpAdd, tensor.F64}: idKernel, // active serves OpAdd
	}}
)

func init() {
	RegisterReference(refMock) // reference + fallback target
	Register(activeMock)
}

func TestRegistry(t *testing.T) {
	if _, ok := Get("active"); !ok {
		t.Error("active must be registered")
	}
	if Reference().Name() != "refmock" {
		t.Errorf("Reference = %q, want refmock", Reference().Name())
	}
	if Default().Name() != "refmock" {
		t.Errorf("Default = %q, want refmock", Default().Name())
	}
	names := Available()
	if len(names) < 2 || names[0] != "active" { // sorted
		t.Errorf("Available = %v, want sorted incl active,refmock", names)
	}
}

func TestExecuteHappyPath(t *testing.T) {
	ctx := NewContext().WithBackend(activeMock)
	x := tensor.New(tensor.F64, tensor.Shape{2})
	out, err := Execute(ctx, OpAdd, []*tensor.Tensor{x}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 1 || out[0] != x {
		t.Error("identity kernel must return its input")
	}
}

// §I4: active backend lacks OpNeg → Execute falls back to the reference.
func TestExecuteFallback(t *testing.T) {
	ctx := NewContext().WithBackend(activeMock)
	x := tensor.New(tensor.F64, tensor.Shape{2})
	if _, err := Execute(ctx, OpNeg, []*tensor.Tensor{x}, nil); err != nil {
		t.Fatalf("fallback to reference must succeed: %v", err)
	}
	// an op nobody serves → error, not panic
	if _, err := Execute(ctx, OpMatMul, []*tensor.Tensor{x}, nil); err == nil {
		t.Error("unserved op must error")
	}
}

type capRecorder struct {
	ops []Op
}

func (r *capRecorder) Record(op Op, _, _ []*tensor.Tensor, _ Attrs) {
	r.ops = append(r.ops, op)
}

// §T13 gate: a Recorder attached to the context sees every forward op — the seam
// autograd uses without any change to L1 ops.
func TestRecorderInterception(t *testing.T) {
	rec := &capRecorder{}
	ctx := NewContext().WithBackend(activeMock).WithRecorder(rec)
	x := tensor.New(tensor.F64, tensor.Shape{2})
	if _, err := Execute(ctx, OpAdd, []*tensor.Tensor{x}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := Execute(ctx, OpNeg, []*tensor.Tensor{x}, nil); err != nil { // via fallback
		t.Fatal(err)
	}
	if len(rec.ops) != 2 || rec.ops[0] != OpAdd || rec.ops[1] != OpNeg {
		t.Errorf("recorder saw %v, want [add neg]", rec.ops)
	}
}

func TestAttrs(t *testing.T) {
	a := Attrs{"axis": 1, "keepdims": true, "dims": []int{0, 2}}
	if a.Int("axis", -1) != 1 || a.Int("missing", 9) != 9 {
		t.Error("Int attr")
	}
	if !a.Bool("keepdims", false) || a.Bool("missing", true) != true {
		t.Error("Bool attr")
	}
	if got := a.Ints("dims"); len(got) != 2 || got[1] != 2 {
		t.Error("Ints attr")
	}
}
