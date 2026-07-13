package autograd_test

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// segment y = relu(x·W1)·W2 — a small MLP block with intermediate activations (a, r) that
// gradient checkpointing must rematerialize.
func mlpSegment(ctx *backend.Context, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	x, w1, w2 := in[0], in[1], in[2]
	a, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{x, w1}, nil)
	if err != nil {
		return nil, err
	}
	r, err := backend.Execute(ctx, backend.OpReLU, []*tensor.Tensor{a[0]}, nil)
	if err != nil {
		return nil, err
	}
	return backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{r[0], w2}, nil)
}

func mlpInputs() (x, w1, w2 *tensor.Tensor) {
	x = tensor.FromFloat64(tensor.Shape{2, 3}, []float64{0.5, -1, 2, 0.3, 0.7, -0.4})
	w1 = tensor.FromFloat64(tensor.Shape{3, 4}, []float64{
		0.1, -0.2, 0.3, 0.4, 0.5, -0.6, 0.7, 0.8, -0.9, 1.0, -1.1, 1.2})
	w2 = tensor.FromFloat64(tensor.Shape{4, 2}, []float64{
		0.2, -0.3, 0.4, 0.5, -0.6, 0.7, 0.8, -0.9})
	return
}

// runMLP builds loss = Σ relu(x·W1)·W2 either directly (checkpoint=false) or with the segment
// wrapped in autograd.Checkpoint, and returns the gradients plus the recorded node count.
func runMLP(t *testing.T, x, w1, w2 *tensor.Tensor, checkpoint bool) (gx, gw1, gw2 *tensor.Tensor, nodes int) {
	t.Helper()
	tape := autograd.NewTape()
	ctx := tape.Context()
	var y []*tensor.Tensor
	var err error
	if checkpoint {
		y, err = autograd.Checkpoint(tape, mlpSegment, x, w1, w2)
	} else {
		y, err = mlpSegment(ctx, []*tensor.Tensor{x, w1, w2})
	}
	if err != nil {
		t.Fatal(err)
	}
	loss, err := backend.Execute(ctx, backend.OpSum, []*tensor.Tensor{y[0]}, nil)
	if err != nil {
		t.Fatal(err)
	}
	nodes = tape.Len()
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	return tape.Grad(x), tape.Grad(w1), tape.Grad(w2), nodes
}

// §V16 tier-1 (the correctness guarantee, Chen et al. 2016 §5.1 / §R109): gradient checkpointing
// computes MATHEMATICALLY IDENTICAL gradients to standard backprop — the segment forward is
// re-run during backward to rematerialize activations, and because the ops are deterministic the
// resulting gradients are bit-for-bit equal to the non-checkpointed ones. It also records far
// fewer tape nodes (the segment collapses to a single checkpoint node), proving the intermediate
// activations were not retained.
func TestCheckpointGradientsIdentical(t *testing.T) {
	x, w1, w2 := mlpInputs()
	gx0, gw10, gw20, n0 := runMLP(t, x, w1, w2, false)
	gx1, gw11, gw21, n1 := runMLP(t, x, w1, w2, true)

	for _, pair := range []struct {
		name      string
		ref, ckpt *tensor.Tensor
	}{{"dX", gx0, gx1}, {"dW1", gw10, gw11}, {"dW2", gw20, gw21}} {
		if pair.ref == nil || pair.ckpt == nil {
			t.Fatalf("%s: nil gradient (ref=%v ckpt=%v)", pair.name, pair.ref, pair.ckpt)
		}
		if !pair.ref.Shape().Equal(pair.ckpt.Shape()) {
			t.Fatalf("%s: shape %v != %v", pair.name, pair.ref.Shape(), pair.ckpt.Shape())
		}
		for i := range pair.ref.Numel() {
			idx := tensor.Unravel(i, pair.ref.Shape())
			if r, c := pair.ref.AtF64(idx...), pair.ckpt.AtF64(idx...); r != c {
				t.Errorf("%s[%v]: checkpoint %.17g != reference %.17g (must be bit-identical)", pair.name, idx, c, r)
			}
		}
	}
	// The checkpointed run records fewer nodes: the 3-op segment (matmul,relu,matmul) collapses to
	// one checkpoint node, so its intermediates were never taped.
	if n1 >= n0 {
		t.Errorf("checkpoint should record fewer nodes: ckpt %d vs plain %d", n1, n0)
	}
}

// §V16 tier-1: checkpointing composes — stacking several checkpointed segments (as one would wrap
// each transformer block) still yields gradients bit-identical to the fully-taped equivalent.
func TestCheckpointStacked(t *testing.T) {
	x, w1, w2 := mlpInputs()
	// two segments in series: h = seg(x,W1,W2) reshaped back to [2,3] is not conformable, so
	// instead stack seg twice with compatible shapes by reusing W-shaped blocks: block maps
	// [2,3]→[2,2]; feed through a second block [2,2]→[2,2].
	w2b := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{0.3, -0.4, 0.5, 0.6})
	build := func(ckpt bool) (*tensor.Tensor, int) {
		tape := autograd.NewTape()
		ctx := tape.Context()
		seg2 := func(c *backend.Context, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
			return backend.Execute(c, backend.OpMatMul, []*tensor.Tensor{in[0], in[1]}, nil)
		}
		var h []*tensor.Tensor
		var err error
		if ckpt {
			h, err = autograd.Checkpoint(tape, mlpSegment, x, w1, w2)
		} else {
			h, err = mlpSegment(ctx, []*tensor.Tensor{x, w1, w2})
		}
		if err != nil {
			t.Fatal(err)
		}
		var h2 []*tensor.Tensor
		if ckpt {
			h2, err = autograd.Checkpoint(tape, seg2, h[0], w2b)
		} else {
			h2, err = seg2(ctx, []*tensor.Tensor{h[0], w2b})
		}
		if err != nil {
			t.Fatal(err)
		}
		loss, err := backend.Execute(ctx, backend.OpSum, []*tensor.Tensor{h2[0]}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if err := tape.Backward(loss[0]); err != nil {
			t.Fatal(err)
		}
		return tape.Grad(w1), tape.Len()
	}
	gRef, _ := build(false)
	gCkpt, _ := build(true)
	for i := range gRef.Numel() {
		idx := tensor.Unravel(i, gRef.Shape())
		if r, c := gRef.AtF64(idx...), gCkpt.AtF64(idx...); r != c {
			t.Errorf("stacked dW1[%v]: %.17g != %.17g", idx, c, r)
		}
	}
}

func TestCheckpointErrors(t *testing.T) {
	tape := autograd.NewTape()
	empty := func(*backend.Context, []*tensor.Tensor) ([]*tensor.Tensor, error) {
		return nil, nil // no outputs
	}
	if _, err := autograd.Checkpoint(tape, empty); err == nil {
		t.Error("a segment producing no outputs must error")
	}
}

// Checkpoint runs a segment's forward without taping its intermediate activations, then re-runs it
// during Backward to recover them — yielding the same gradients for less memory. Here the segment
// is y = x², so d(Σy)/dx = 2x = [2,4,6].
func ExampleCheckpoint() {
	tape := autograd.NewTape()
	x := tensor.FromFloat64(tensor.Shape{3}, []float64{1, 2, 3})
	square := func(ctx *backend.Context, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
		return backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{in[0], in[0]}, nil)
	}
	y, _ := autograd.Checkpoint(tape, square, x)
	loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{y[0]}, nil)
	_ = tape.Backward(loss[0])
	g := tape.Grad(x)
	fmt.Printf("%.0f %.0f %.0f\n", g.AtF64(0), g.AtF64(1), g.AtF64(2))
	// Output: 2 4 6
}
