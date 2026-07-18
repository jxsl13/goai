package autograd_test

import (
	"math"
	"strings"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// (a) A forward op that overflows (exp(1000) = +Inf) is reported immediately:
// the error names the op, its tape index, and the first offending element, is
// visible via Err right after the forward, and is returned by Backward.
func TestAnomalyForwardInf(t *testing.T) {
	tape := autograd.NewTape(autograd.WithAnomalyDetection())
	ctx := tape.Context()
	x := tensor.FromFloat64(tensor.Shape{3}, []float64{0.5, 1000, 2})
	y, err := exec(ctx, backend.OpExp, nil, x)
	if err != nil {
		t.Fatal(err)
	}
	ferr := tape.Err()
	if ferr == nil {
		t.Fatal("anomaly mode must flag exp overflow right after the forward op")
	}
	for _, want := range []string{"exp", "tape index 0", "+Inf", "element 1"} {
		if !strings.Contains(ferr.Error(), want) {
			t.Errorf("forward anomaly error missing %q: %v", want, ferr)
		}
	}
	if berr := tape.Backward(y); berr == nil || berr.Error() != ferr.Error() {
		t.Errorf("Backward must surface the stored forward anomaly, got %v", berr)
	}
}

// (b) A VJP that creates a non-finite gradient from finite tensors — sqrt at 0,
// dx = g/(2·sqrt(0)) = +Inf — is blamed on the CREATING node (sqrt), not on the
// seed or the sum whose cotangent was perfectly finite.
func TestAnomalyBackwardCreated(t *testing.T) {
	tape := autograd.NewTape(autograd.WithAnomalyDetection())
	ctx := tape.Context()
	x := tensor.FromFloat64(tensor.Shape{3}, []float64{4, 0, 9})
	r, err := exec(ctx, backend.OpSqrt, nil, x)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := exec(ctx, backend.OpSum, nil, r)
	if err != nil {
		t.Fatal(err)
	}
	if ferr := tape.Err(); ferr != nil { // forward is finite: sqrt(0)=0
		t.Fatalf("no forward anomaly expected, got %v", ferr)
	}
	berr := tape.Backward(loss)
	if berr == nil {
		t.Fatal("anomaly mode must flag the Inf gradient sqrt's VJP produces at 0")
	}
	for _, want := range []string{"sqrt", "CREATED", "+Inf", "input 0", "element 1"} {
		if !strings.Contains(berr.Error(), want) {
			t.Errorf("backward anomaly error missing %q: %v", want, berr)
		}
	}
	if strings.Contains(berr.Error(), "PROPAGATED") {
		t.Errorf("a finite-cotangent hit must be labeled created, not propagated: %v", berr)
	}
}

// (c) Propagation: when the incoming cotangent is ALREADY non-finite (here an
// Inf Backward seed), the flagged node is labeled as propagating upstream
// damage — distinguishing it from a VJP that manufactured the value itself.
func TestAnomalyBackwardPropagated(t *testing.T) {
	tape := autograd.NewTape(autograd.WithAnomalyDetection())
	ctx := tape.Context()
	x := tensor.FromFloat64(tensor.Shape{2}, []float64{1, 2})
	w := tensor.FromFloat64(tensor.Shape{2}, []float64{3, 4})
	y, err := exec(ctx, backend.OpMul, nil, x, w)
	if err != nil {
		t.Fatal(err)
	}
	seed := tensor.FromFloat64(tensor.Shape{2}, []float64{math.Inf(1), 1})
	berr := tape.BackwardGrad(y, seed)
	if berr == nil {
		t.Fatal("anomaly mode must flag the non-finite cotangent flowing through mul")
	}
	for _, want := range []string{"mul", "PROPAGATED", "+Inf"} {
		if !strings.Contains(berr.Error(), want) {
			t.Errorf("propagation error missing %q: %v", want, berr)
		}
	}
	if strings.Contains(berr.Error(), "CREATED") {
		t.Errorf("a non-finite incoming cotangent must be labeled propagated: %v", berr)
	}
}

// (d) Off-mode: a plain tape and an anomaly tape produce bit-identical
// gradients on a finite graph, and the plain tape neither errors nor stores
// anything on a graph that DOES overflow — detection is strictly opt-in.
func TestAnomalyOffIdentical(t *testing.T) {
	run := func(tape *autograd.Tape) (*tensor.Tensor, *autograd.Tape, error) {
		ctx := tape.Context()
		x := tensor.FromFloat64(tensor.Shape{3}, []float64{0.3, -1.1, 2.4})
		w := tensor.FromFloat64(tensor.Shape{3}, []float64{1.2, 0.4, -0.9})
		y, err := exec(ctx, backend.OpMul, nil, x, w)
		if err != nil {
			return nil, nil, err
		}
		th, err := exec(ctx, backend.OpTanh, nil, y)
		if err != nil {
			return nil, nil, err
		}
		loss, err := exec(ctx, backend.OpSum, nil, th)
		if err != nil {
			return nil, nil, err
		}
		if err := tape.Backward(loss); err != nil {
			return nil, nil, err
		}
		return tape.Grad(x), tape, nil
	}
	gOff, _, err := run(autograd.NewTape())
	if err != nil {
		t.Fatal(err)
	}
	gOn, _, err := run(autograd.NewTape(autograd.WithAnomalyDetection()))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		if gOff.AtF64(i) != gOn.AtF64(i) {
			t.Errorf("elem %d: off %v != on %v", i, gOff.AtF64(i), gOn.AtF64(i))
		}
	}

	// overflow graph on a PLAIN tape: no error anywhere, NaN/Inf flow silently.
	tape := autograd.NewTape()
	ctx := tape.Context()
	x := tensor.FromFloat64(tensor.Shape{2}, []float64{1000, 1})
	e, err := exec(ctx, backend.OpExp, nil, x)
	if err != nil {
		t.Fatal(err)
	}
	if tape.Err() != nil {
		t.Errorf("off-mode tape must not store anomalies, got %v", tape.Err())
	}
	loss, err := exec(ctx, backend.OpSum, nil, e)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Errorf("off-mode Backward must not error on non-finite values, got %v", err)
	}
}

// benchStep runs one small forward+backward (mul → add → tanh → sum) on a fresh
// tape — the per-training-step idiom — and is shared by the anomaly on/off
// benchmarks so their numbers are directly comparable.
func benchStep(b *testing.B, tape *autograd.Tape, x, w, bias *tensor.Tensor) {
	ctx := tape.Context()
	xw, err := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{x, w}, nil)
	if err != nil {
		b.Fatal(err)
	}
	h, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{xw[0], bias}, nil)
	if err != nil {
		b.Fatal(err)
	}
	a, err := backend.Execute(ctx, backend.OpTanh, []*tensor.Tensor{h[0]}, nil)
	if err != nil {
		b.Fatal(err)
	}
	loss, err := backend.Execute(ctx, backend.OpSum, []*tensor.Tensor{a[0]}, nil)
	if err != nil {
		b.Fatal(err)
	}
	if err := tape.Backward(loss[0]); err != nil {
		b.Fatal(err)
	}
}

func benchInputs() (x, w, bias *tensor.Tensor) {
	n := 64
	xd := make([]float64, n)
	wd := make([]float64, n)
	bd := make([]float64, n)
	for i := range xd {
		xd[i] = 0.01 * float64(i%17)
		wd[i] = 0.02 * float64(i%13)
		bd[i] = 0.005 * float64(i%7)
	}
	s := tensor.Shape{n}
	return tensor.FromFloat64(s, xd), tensor.FromFloat64(s, wd), tensor.FromFloat64(s, bd)
}

// BenchmarkForwardBackwardAnomalyOff is the off-mode overhead gate: anomaly
// detection disabled (the default) must cost nothing beyond the pre-existing
// tape path — the guard is a single bool test per recorded op.
func BenchmarkForwardBackwardAnomalyOff(b *testing.B) {
	x, w, bias := benchInputs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStep(b, autograd.NewTape(), x, w, bias)
	}
}

// BenchmarkForwardBackwardAnomalyOn measures the scan cost when detection IS
// enabled — the price paid only while hunting a NaN.
func BenchmarkForwardBackwardAnomalyOn(b *testing.B) {
	x, w, bias := benchInputs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStep(b, autograd.NewTape(autograd.WithAnomalyDetection()), x, w, bias)
	}
}
