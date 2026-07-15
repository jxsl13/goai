package backend

import (
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// spyBackend records which ops actually ran on it, so a routing test can prove an
// override re-pointed dispatch here (rather than inferring it from results). It
// serves matmul/add at f64 with a real-enough kernel: element-wise identity is
// sufficient for routing-parity because the parity we assert is "the SAME kernel
// value flows regardless of which backend Execute selected".
type spyBackend struct {
	name Name
	ran  []Op
	kern Kernel
}

func (s *spyBackend) Name() Name            { return s.name }
func (s *spyBackend) Device() tensor.Device { return tensor.CPU() }
func (s *spyBackend) Synchronize() error    { return nil }
func (s *spyBackend) Kernel(op Op, dt tensor.Dtype) (Kernel, bool) {
	if s.kern == nil || dt != tensor.F64 {
		return nil, false
	}
	if op != OpMatMul && op != OpAdd {
		return nil, false
	}
	rec := func(ctx *Context, in []*tensor.Tensor, at Attrs) ([]*tensor.Tensor, error) {
		s.ran = append(s.ran, op)
		return s.kern(ctx, in, at)
	}
	return rec, true
}

// scaleKernel writes 2*x element-wise — a deterministic, backend-independent
// transform so routing-parity can compare bit values across backends.
func scaleKernel(_ *Context, in []*tensor.Tensor, _ Attrs) ([]*tensor.Tensor, error) {
	x := in[0]
	out := tensor.New(tensor.F64, x.Shape())
	xf := x.Storage().F64()
	of := out.Storage().F64()
	for i := range xf {
		of[i] = 2 * xf[i]
	}
	return []*tensor.Tensor{out}, nil
}

func newRoutingCtx(active Backend) *Context {
	return &Context{Backend: active}
}

// TestRoutingParity (§V16-2): with ctx.Backend = one backend, an override that
// routes OpMatMul to a DIFFERENT registered backend makes the op run there, and
// the RESULT is bit-identical to running it on either backend directly (same
// kernel value). The spy proves the override took effect.
func TestRoutingParity(t *testing.T) {
	home := &spyBackend{name: "route-home", kern: scaleKernel}
	target := &spyBackend{name: "route-target", kern: scaleKernel}
	Register(home)
	Register(target)

	x := tensor.FromFloat64(tensor.Shape{3}, []float64{1, 2, 3})

	// baseline: no override, runs on home.
	base := newRoutingCtx(home)
	bout, err := Execute(base, OpMatMul, []*tensor.Tensor{x}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(home.ran) != 1 || home.ran[0] != OpMatMul {
		t.Fatalf("baseline should run on home, home.ran=%v", home.ran)
	}
	if len(target.ran) != 0 {
		t.Fatalf("baseline must not touch target, target.ran=%v", target.ran)
	}

	// routed: same home backend, override sends OpMatMul to target.
	routed := newRoutingCtx(home).WithOpBackend(OpMatMul, target.Name())
	rout, err := Execute(routed, OpMatMul, []*tensor.Tensor{x}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(target.ran) != 1 || target.ran[0] != OpMatMul {
		t.Fatalf("override must run OpMatMul on target, target.ran=%v", target.ran)
	}
	// home must not have run the routed op (still just its one baseline run).
	if len(home.ran) != 1 {
		t.Fatalf("override must NOT run OpMatMul on home, home.ran=%v", home.ran)
	}

	// parity: bit-identical results (§V3/§V11 — exact for f64).
	bf, rf := bout[0].Storage().F64(), rout[0].Storage().F64()
	for i := range bf {
		if bf[i] != rf[i] {
			t.Errorf("routing parity mismatch at %d: home=%v target=%v", i, bf[i], rf[i])
		}
	}
}

// TestRoutingFunctionalFallback (§V16-3): routing an op to a backend that lacks
// that kernel must NOT crash and must still produce a correct result — the
// override is ignored and Execute falls through to the normal resolution/fallback
// chain (§I4).
func TestRoutingFunctionalFallback(t *testing.T) {
	home := &spyBackend{name: "fb-home", kern: scaleKernel}
	Register(home)

	x := tensor.FromFloat64(tensor.Shape{2}, []float64{4, 5})

	// route OpMatMul to the reference mock, which does NOT serve OpMatMul.
	ctx := newRoutingCtx(home).WithOpBackend(OpMatMul, refMock.Name())
	out, err := Execute(ctx, OpMatMul, []*tensor.Tensor{x}, nil)
	if err != nil {
		t.Fatalf("routing to a backend lacking the kernel must still succeed: %v", err)
	}
	// the override was ignored, so home ran it — correct result.
	if len(home.ran) != 1 || home.ran[0] != OpMatMul {
		t.Fatalf("unsupported override should fall through to home, home.ran=%v", home.ran)
	}
	of := out[0].Storage().F64()
	if of[0] != 8 || of[1] != 10 {
		t.Errorf("fallback result wrong: got %v, want [8 10]", of)
	}

	// route to a name that is not registered at all → also a no-op, no crash.
	ctx2 := newRoutingCtx(home).WithOpBackend(OpMatMul, "no-such-backend")
	if _, err := Execute(ctx2, OpMatMul, []*tensor.Tensor{x}, nil); err != nil {
		t.Fatalf("routing to an unregistered backend must fall through, got: %v", err)
	}
}

// TestRoutingCopyOnWrite (§V16-4): WithOpBackend/WithLayerBackend are
// copy-on-write — two contexts derived from the same base never interfere — and a
// routing override survives WithRecorder/WithBackend.
func TestRoutingCopyOnWrite(t *testing.T) {
	home := &spyBackend{name: "cow-home", kern: scaleKernel}
	target := &spyBackend{name: "cow-target", kern: scaleKernel}
	Register(home)
	Register(target)

	base := newRoutingCtx(home)
	a := base.WithOpBackend(OpMatMul, target.Name())
	b := base.WithOpBackend(OpAdd, target.Name())

	// base is untouched (no overrides leaked into it).
	if base.opBackends != nil {
		t.Errorf("base ctx map must stay nil, got %v", base.opBackends)
	}
	// a and b do not see each other's override.
	if _, ok := a.opBackends[OpAdd]; ok {
		t.Errorf("a must not see b's OpAdd override")
	}
	if _, ok := b.opBackends[OpMatMul]; ok {
		t.Errorf("b must not see a's OpMatMul override")
	}
	if n, ok := a.opBackends[OpMatMul]; !ok || n != target.Name() {
		t.Errorf("a must route OpMatMul to target, got %v/%v", n, ok)
	}

	// preservation: override survives WithRecorder and WithBackend copies.
	rec := &capRecorder{}
	derived := a.WithRecorder(rec).WithBackend(home)
	if n, ok := derived.opBackends[OpMatMul]; !ok || n != target.Name() {
		t.Errorf("override must survive WithRecorder/WithBackend, got %v/%v", n, ok)
	}
	// and it still routes: OpMatMul on derived runs on target.
	x := tensor.FromFloat64(tensor.Shape{1}, []float64{7})
	if _, err := Execute(derived, OpMatMul, []*tensor.Tensor{x}, nil); err != nil {
		t.Fatal(err)
	}
	if len(target.ran) == 0 || target.ran[len(target.ran)-1] != OpMatMul {
		t.Errorf("preserved override must still route to target, target.ran=%v", target.ran)
	}

	// WithLayerBackend routes a group in one step, also copy-on-write.
	layer := base.WithLayerBackend(target.Name(), OpMatMul, OpAdd)
	if layer.opBackends[OpMatMul] != target.Name() || layer.opBackends[OpAdd] != target.Name() {
		t.Errorf("WithLayerBackend must route every listed op, got %v", layer.opBackends)
	}
	if base.opBackends != nil {
		t.Errorf("WithLayerBackend must not mutate base, got %v", base.opBackends)
	}
}
