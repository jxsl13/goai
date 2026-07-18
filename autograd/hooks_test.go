package autograd_test

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/npz"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// Tests for Tape.RegisterGradHook (hooks.go): tier-1 torch parity against
// tensor.register_hook, the fan-out firing semantics (ONE call with the summed
// cotangent — the core of the finality guarantee), composition order, leaf
// firing, the upstream-only effect of a replacement, validation, and the
// anomaly-mode attribution of a hook that manufactures a non-finite value.

// scaleGrad returns a hook replacing the gradient with grad·s (a fresh tensor —
// hooks must not mutate in place).
func scaleGrad(t *testing.T, s float64) autograd.GradHook {
	t.Helper()
	return func(g *tensor.Tensor) *tensor.Tensor {
		out := tensor.New(g.Dtype(), g.Shape().Clone())
		src, dst := g.Storage().F64(), out.Storage().F64()
		for i := range src {
			dst[i] = src[i] * s
		}
		return out
	}
}

// clampGrad returns a hook clamping the gradient into [-lim, lim].
func clampGrad(t *testing.T, lim float64) autograd.GradHook {
	t.Helper()
	return func(g *tensor.Tensor) *tensor.Tensor {
		out := tensor.New(g.Dtype(), g.Shape().Clone())
		src, dst := g.Storage().F64(), out.Storage().F64()
		for i := range src {
			dst[i] = math.Max(-lim, math.Min(lim, src[i]))
		}
		return out
	}
}

// §V16 tier-1: the fixture graph P = A·B, y = tanh(P) + P (P fans out) with a
// mid-tensor hook scaling P's gradient by 0.5 and a leaf hook clamping A's to
// ±0.1 must reproduce torch's A.grad/B.grad from the same two register_hook
// calls, to ≤ 1e-12 (scripts/golden_gradhook.py).
func TestGradHookTorchParity(t *testing.T) {
	fix, err := npz.LoadFile("testdata/gradhook_torch.npz")
	if err != nil {
		t.Fatal(err)
	}
	for _, k := range []string{"a", "b", "ga", "gb"} {
		if fix[k] == nil {
			t.Fatalf("fixture missing %q", k)
		}
	}
	tape := autograd.NewTape()
	ctx := tape.Context()
	a, b := fix["a"], fix["b"]

	p, err := exec(ctx, backend.OpMatMul, nil, a, b)
	if err != nil {
		t.Fatal(err)
	}
	// Registered BEFORE the rest of the graph, exactly as torch does it.
	tape.RegisterGradHook(p, scaleGrad(t, 0.5))
	tape.RegisterGradHook(a, clampGrad(t, 0.1))
	th, err := exec(ctx, backend.OpTanh, nil, p)
	if err != nil {
		t.Fatal(err)
	}
	y, err := exec(ctx, backend.OpAdd, nil, th, p)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(y); err != nil { // loss = sum(y)
		t.Fatal(err)
	}
	const tol = 1e-12
	for _, tc := range []struct {
		name    string
		x, want *tensor.Tensor
	}{{"A.grad", a, fix["ga"]}, {"B.grad", b, fix["gb"]}} {
		g := tape.Grad(tc.x)
		if g == nil {
			t.Fatalf("%s: nil grad", tc.name)
		}
		got, want := g.Storage().F64(), tc.want.Storage().F64()
		for i := range got {
			if math.Abs(got[i]-want[i]) > tol {
				t.Errorf("%s elem %d: got %.17g, torch %.17g (|Δ| %.3g)",
					tc.name, i, got[i], want[i], math.Abs(got[i]-want[i]))
			}
		}
	}
}

// THE semantics test: a tensor consumed by two ops must have its hook called
// exactly ONCE, with the SUMMED (final) cotangent — never once per consumer.
// P feeds both tanh and the add, so its cotangent is dtanh + 1 elementwise.
func TestGradHookFanOutFiresOnceWithSummedGrad(t *testing.T) {
	tape := autograd.NewTape()
	ctx := tape.Context()
	a := tensor.New(tensor.F64, tensor.Shape{2, 3})
	for i, v := range []float64{0.1, -0.2, 0.3, 0.4, -0.5, 0.6} {
		a.Storage().F64()[i] = v
	}
	p, err := exec(ctx, backend.OpTanh, nil, a) // p is produced, then fans out
	if err != nil {
		t.Fatal(err)
	}
	calls := 0
	var seen []float64
	tape.RegisterGradHook(p, func(g *tensor.Tensor) *tensor.Tensor {
		calls++
		seen = append([]float64(nil), g.Storage().F64()...)
		return nil
	})
	e, err := exec(ctx, backend.OpExp, nil, p)
	if err != nil {
		t.Fatal(err)
	}
	y, err := exec(ctx, backend.OpAdd, nil, e, p) // second consumer of p
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(y); err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Fatalf("hook fired %d times, want exactly 1 (fan-out must be summed before firing)", calls)
	}
	// dL/dp = d(exp(p))/dp + 1 = exp(p) + 1 elementwise.
	pv := p.Storage().F64()
	for i := range seen {
		want := math.Exp(pv[i]) + 1
		if math.Abs(seen[i]-want) > 1e-12 {
			t.Errorf("summed cotangent elem %d: got %.17g, want %.17g", i, seen[i], want)
		}
	}
}

// An observe-only hook (nil return) must leave every gradient bit-identical to
// the no-hook run.
func TestGradHookObserveOnlyIsBitIdentical(t *testing.T) {
	run := func(hook bool) (ga, gb []float64) {
		tape := autograd.NewTape()
		ctx := tape.Context()
		a, b := seedTensor(t, tensor.Shape{3, 4}, 1), seedTensor(t, tensor.Shape{4, 2}, 2)
		p, err := exec(ctx, backend.OpMatMul, nil, a, b)
		if err != nil {
			t.Fatal(err)
		}
		if hook {
			tape.RegisterGradHook(p, func(*tensor.Tensor) *tensor.Tensor { return nil })
			tape.RegisterGradHook(a, func(*tensor.Tensor) *tensor.Tensor { return nil })
		}
		y, err := exec(ctx, backend.OpTanh, nil, p)
		if err != nil {
			t.Fatal(err)
		}
		if err := tape.Backward(y); err != nil {
			t.Fatal(err)
		}
		return append([]float64(nil), tape.Grad(a).Storage().F64()...),
			append([]float64(nil), tape.Grad(b).Storage().F64()...)
	}
	ga0, gb0 := run(false)
	ga1, gb1 := run(true)
	for i := range ga0 {
		if ga0[i] != ga1[i] {
			t.Fatalf("A.grad elem %d changed under an observe-only hook: %v != %v", i, ga1[i], ga0[i])
		}
	}
	for i := range gb0 {
		if gb0[i] != gb1[i] {
			t.Fatalf("B.grad elem %d changed under an observe-only hook: %v != %v", i, gb1[i], gb0[i])
		}
	}
}

// A replacement must propagate UPSTREAM (tensors before the hooked one in
// forward order) and leave already-computed downstream gradients untouched.
func TestGradHookReplacementIsUpstreamOnly(t *testing.T) {
	tape := autograd.NewTape()
	ctx := tape.Context()
	a, b := seedTensor(t, tensor.Shape{3, 4}, 3), seedTensor(t, tensor.Shape{4, 2}, 4)
	p, err := exec(ctx, backend.OpMatMul, nil, a, b) // upstream of the hook
	if err != nil {
		t.Fatal(err)
	}
	th, err := exec(ctx, backend.OpTanh, nil, p)
	if err != nil {
		t.Fatal(err)
	}
	y, err := exec(ctx, backend.OpExp, nil, th) // downstream of the hook
	if err != nil {
		t.Fatal(err)
	}
	// Baseline.
	if err := tape.Backward(y); err != nil {
		t.Fatal(err)
	}
	baseA := append([]float64(nil), tape.Grad(a).Storage().F64()...)
	baseTh := append([]float64(nil), tape.Grad(th).Storage().F64()...)

	tape.RegisterGradHook(th, scaleGrad(t, 2)) // hook on the middle tensor
	if err := tape.Backward(y); err != nil {
		t.Fatal(err)
	}
	// th's own stored grad is the POST-hook value (documented), and A (upstream)
	// doubles; the hook sits between them.
	gotA := tape.Grad(a).Storage().F64()
	for i := range baseA {
		if math.Abs(gotA[i]-2*baseA[i]) > 1e-12 {
			t.Fatalf("upstream A.grad elem %d: got %.17g, want 2× baseline %.17g", i, gotA[i], 2*baseA[i])
		}
	}
	gotTh := tape.Grad(th).Storage().F64()
	for i := range baseTh {
		if math.Abs(gotTh[i]-2*baseTh[i]) > 1e-12 {
			t.Fatalf("hooked tensor grad elem %d: got %.17g, want the post-hook 2× %.17g", i, gotTh[i], 2*baseTh[i])
		}
	}
}

// Multiple hooks on one tensor compose in registration order, each seeing the
// previous one's result (torch semantics): scale-by-2 then clamp-to-1 must
// clamp the DOUBLED gradient, not the original.
func TestGradHookCompositionOrder(t *testing.T) {
	tape := autograd.NewTape()
	ctx := tape.Context()
	a := seedTensor(t, tensor.Shape{2, 2}, 5)
	p, err := exec(ctx, backend.OpTanh, nil, a)
	if err != nil {
		t.Fatal(err)
	}
	var order []string
	tape.RegisterGradHook(p, func(g *tensor.Tensor) *tensor.Tensor {
		order = append(order, "first")
		return scaleGrad(t, 2)(g)
	})
	tape.RegisterGradHook(p, func(g *tensor.Tensor) *tensor.Tensor {
		order = append(order, "second")
		// Must see the doubled gradient from the first hook.
		for _, v := range g.Storage().F64() {
			if math.Abs(v-2.0) > 1e-12 { // seed cotangent is 1 → doubled = 2
				t.Errorf("second hook saw %v, want the first hook's doubled 2.0", v)
			}
		}
		return clampGrad(t, 1)(g)
	})
	if err := tape.Backward(p); err != nil {
		t.Fatal(err)
	}
	if len(order) != 2 || order[0] != "first" || order[1] != "second" {
		t.Fatalf("hook order %v, want [first second]", order)
	}
	for i, v := range tape.Grad(p).Storage().F64() {
		if math.Abs(v-1.0) > 1e-12 {
			t.Errorf("post-composition grad elem %d = %v, want the clamped 1.0", i, v)
		}
	}
}

// A leaf hook (on a tensor no taped op produced) fires at the end of the walk.
func TestGradHookLeafFires(t *testing.T) {
	tape := autograd.NewTape()
	ctx := tape.Context()
	a := seedTensor(t, tensor.Shape{2, 3}, 6)
	fired := false
	tape.RegisterGradHook(a, func(g *tensor.Tensor) *tensor.Tensor {
		fired = true
		return clampGrad(t, 0.25)(g)
	})
	y, err := exec(ctx, backend.OpTanh, nil, a)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(y); err != nil {
		t.Fatal(err)
	}
	if !fired {
		t.Fatal("leaf hook never fired")
	}
	for i, v := range tape.Grad(a).Storage().F64() {
		if v > 0.25+1e-12 || v < -0.25-1e-12 {
			t.Errorf("leaf grad elem %d = %v, want clamped into ±0.25", i, v)
		}
	}
}

// A replacement of the wrong shape or dtype fails the Backward call rather than
// silently corrupting the walk.
func TestGradHookReplacementValidation(t *testing.T) {
	for _, tc := range []struct {
		name string
		bad  func(*tensor.Tensor) *tensor.Tensor
		want string
	}{
		{"shape", func(*tensor.Tensor) *tensor.Tensor {
			return tensor.New(tensor.F64, tensor.Shape{9, 9})
		}, "shape"},
		{"dtype", func(g *tensor.Tensor) *tensor.Tensor {
			return tensor.New(tensor.F32, g.Shape().Clone())
		}, "dtype"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tape := autograd.NewTape()
			ctx := tape.Context()
			a := seedTensor(t, tensor.Shape{2, 2}, 7)
			y, err := exec(ctx, backend.OpTanh, nil, a)
			if err != nil {
				t.Fatal(err)
			}
			tape.RegisterGradHook(a, tc.bad)
			err = tape.Backward(y)
			if err == nil {
				t.Fatalf("a %s-mismatched replacement must fail Backward", tc.name)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error %q does not mention %q", err, tc.want)
			}
		})
	}
}

// Anomaly mode attributes a non-finite value MANUFACTURED by a hook to that
// hook, rather than blaming a neighboring op.
func TestGradHookAnomalyAttribution(t *testing.T) {
	tape := autograd.NewTape(autograd.WithAnomalyDetection())
	ctx := tape.Context()
	a := seedTensor(t, tensor.Shape{2, 2}, 8)
	y, err := exec(ctx, backend.OpTanh, nil, a)
	if err != nil {
		t.Fatal(err)
	}
	tape.RegisterGradHook(y, func(g *tensor.Tensor) *tensor.Tensor {
		out := tensor.New(g.Dtype(), g.Shape().Clone())
		for i := range out.Storage().F64() {
			out.Storage().F64()[i] = math.NaN()
		}
		return out
	})
	err = tape.Backward(y)
	if err == nil {
		t.Fatal("a NaN-manufacturing hook must be caught in anomaly mode")
	}
	msg := err.Error()
	for _, want := range []string{"hook", "CREATED"} {
		if !strings.Contains(msg, want) {
			t.Errorf("anomaly error %q missing %q (must attribute to the hook, not a neighbor op)", msg, want)
		}
	}
}

// seedTensor builds a small deterministic non-uniform f64 tensor.
func seedTensor(t *testing.T, shape tensor.Shape, seed int) *tensor.Tensor {
	t.Helper()
	x := tensor.New(tensor.F64, shape)
	d := x.Storage().F64()
	for i := range d {
		d[i] = math.Sin(float64(seed*17+i)) * 0.7
	}
	return x
}

// Hooks cost one nil-map test per backward node when none are registered.
func BenchmarkForwardBackwardHooksOff(b *testing.B) {
	x := tensor.New(tensor.F64, tensor.Shape{32, 32})
	for i := range x.Storage().F64() {
		x.Storage().F64()[i] = 0.01 * float64(i%7)
	}
	b.ResetTimer()
	for range b.N {
		tape := autograd.NewTape()
		ctx := tape.Context()
		p, err := backend.Execute(ctx, backend.OpTanh, []*tensor.Tensor{x}, nil)
		if err != nil {
			b.Fatal(err)
		}
		q, err := backend.Execute(ctx, backend.OpExp, p, nil)
		if err != nil {
			b.Fatal(err)
		}
		if err := tape.Backward(q[0]); err != nil {
			b.Fatal(err)
		}
	}
}

// Per-tensor gradient clipping during backward: the classic hook use — shrink
// one layer's gradient before it propagates into the layers below, without
// touching the rest of the graph.
func ExampleTape_RegisterGradHook() {
	tape := autograd.NewTape()
	ctx := tape.Context()

	w := tensor.New(tensor.F64, tensor.Shape{2, 2})
	for i, v := range []float64{0.5, -1.5, 2.5, -0.5} {
		w.Storage().F64()[i] = v
	}
	h, _ := backend.Execute(ctx, backend.OpExp, []*tensor.Tensor{w}, nil)

	// Clip this tensor's gradient into ±0.5 the moment it is final.
	tape.RegisterGradHook(h[0], func(g *tensor.Tensor) *tensor.Tensor {
		out := tensor.New(g.Dtype(), g.Shape().Clone())
		src, dst := g.Storage().F64(), out.Storage().F64()
		clipped := 0
		for i := range src {
			dst[i] = math.Max(-0.5, math.Min(0.5, src[i]))
			if dst[i] != src[i] {
				clipped++
			}
		}
		fmt.Printf("clipped %d of %d gradient elements\n", clipped, len(src))
		return out
	})

	y, _ := backend.Execute(ctx, backend.OpTanh, h, nil)
	if err := tape.Backward(y[0]); err != nil {
		panic(err)
	}
	fmt.Printf("w.grad finite: %v\n", allWithin(tape.Grad(w), 1e9))
	// Output:
	// clipped 2 of 4 gradient elements
	// w.grad finite: true
}

// allWithin reports whether every element of x lies in [-lim, lim].
func allWithin(x *tensor.Tensor, lim float64) bool {
	for _, v := range x.Storage().F64() {
		if v > lim || v < -lim {
			return false
		}
	}
	return true
}
