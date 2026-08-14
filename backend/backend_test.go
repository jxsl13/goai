package backend

import (
	"fmt"
	"slices"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// idKernel is an identity kernel: it returns its inputs unchanged. Enough to
// exercise dispatch, fallback, and recording without real math (that is §T6+).
func idKernel(_ *Context, in []*tensor.Tensor, _ Attrs) ([]*tensor.Tensor, error) {
	return in, nil
}

func idIntoKernel(ctx *Context, in, out []*tensor.Tensor, _ Attrs) error {
	if len(in) != 1 || len(out) != 1 {
		return fmt.Errorf("identity into: want one input and output")
	}
	if err := ValidateOutput(ctx, out[0], in[0].Dtype(), in[0].Shape()); err != nil {
		return err
	}
	copy(out[0].Storage().F64(), in[0].Storage().F64())
	return nil
}

type mockBackend struct {
	name      Name
	table     map[kernelKeyT]Kernel
	intoTable map[kernelKeyT]IntoKernel
}

type kernelKeyT struct {
	op    Op
	dtype tensor.Dtype
}

func (m *mockBackend) Name() Name            { return m.name }
func (m *mockBackend) Device() tensor.Device { return tensor.CPU() }
func (m *mockBackend) Synchronize() error    { return nil }
func (m *mockBackend) Kernel(op Op, dt tensor.Dtype) (Kernel, bool) {
	k, ok := m.table[kernelKeyT{op, dt}]
	return k, ok
}
func (m *mockBackend) KernelInto(op Op, dt tensor.Dtype) (IntoKernel, bool) {
	k, ok := m.intoTable[kernelKeyT{op, dt}]
	return k, ok
}

var (
	refMock = &mockBackend{
		name: "refmock",
		table: map[kernelKeyT]Kernel{
			{OpNeg, tensor.F64}: idKernel, // only the reference serves OpNeg
			{OpQR, tensor.F64}:  idKernel, // reference-only fallback for ExecuteInto tests
		},
		intoTable: map[kernelKeyT]IntoKernel{
			{OpNeg, tensor.F64}: idIntoKernel,
			{OpQR, tensor.F64}:  idIntoKernel,
		},
	}
	activeMock = &mockBackend{
		name: "active",
		table: map[kernelKeyT]Kernel{
			{OpAdd, tensor.F64}: idKernel, // active serves OpAdd
		},
		intoTable: map[kernelKeyT]IntoKernel{
			{OpAdd, tensor.F64}: idIntoKernel,
		},
	}
)

func init() {
	RegisterReference(refMock) // reference + fallback target
	Register(activeMock)
}

func TestRegistry(t *testing.T) {
	if _, ok := Get("active"); !ok {
		t.Error("active must be registered")
	}
	if Reference() == nil {
		t.Error("a reference backend must be registered")
	}
	// With a preference matching nothing registered, Default() falls back to the
	// reference — the guaranteed last resort (§T46). (Robust to whichever accel/
	// cpu backends other test files in this binary happen to register.)
	saved := Preference()
	defer SetPreference(saved...)
	SetPreference("no-such-backend")
	if Default() != Reference() {
		t.Errorf("Default = %q, want reference fallback %q", Default().Name(), Reference().Name())
	}
	if names := Available(); !slices.Contains(names, "active") {
		t.Errorf("Available = %v, want to include active", names)
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

// TestAttrs exercises the sealed-interface op-params design (ADR-0014): each op has
// one concrete struct that satisfies Attrs, a kernel reads its params by comma-ok
// type-asserting to that struct (a nil Attrs asserts to the zero struct), and
// WithDefaults fills the documented non-zero defaults.
func TestAttrs(t *testing.T) {
	// concrete structs satisfy the sealed Attrs interface
	var _ Attrs = AttnAttrs{Heads: 2}
	var _ Attrs = NormAttrs{}

	// WithDefaults fills the documented non-zero defaults
	if d := (AttnAttrs{}).WithDefaults(); d.Heads != 1 || d.Scale != 1 {
		t.Errorf("AttnAttrs defaults: got heads=%d scale=%v, want 1 and 1", d.Heads, d.Scale)
	}
	if d := (NormAttrs{}).WithDefaults(); d.Eps != 1e-5 {
		t.Errorf("NormAttrs default Eps: got %v, want 1e-5", d.Eps)
	}
	// an explicitly set field survives WithDefaults
	if d := (AttnAttrs{Heads: 4}).WithDefaults(); d.Heads != 4 || d.KVHeads != 4 {
		t.Errorf("AttnAttrs{Heads:4}.WithDefaults(): got heads=%d kvheads=%d, want 4 and 4", d.Heads, d.KVHeads)
	}

	// the kernel-side read: comma-ok assert. A nil Attrs yields the zero struct.
	var nilAttrs Attrs
	if p, ok := nilAttrs.(AttnAttrs); ok || p != (AttnAttrs{}) {
		t.Errorf("nil Attrs asserted to AttnAttrs: got %+v ok=%v, want zero struct and ok=false", p, ok)
	}
	// a matching assert recovers the concrete value
	if p, ok := Attrs(AttnAttrs{Heads: 3, Causal: true}).(AttnAttrs); !ok || p.Heads != 3 || !p.Causal {
		t.Errorf("AttnAttrs assert: got %+v ok=%v, want {Heads:3 Causal:true} and ok=true", p, ok)
	}
}
