package nn_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §T489: EWC's defining claim in the classic two-task harness — an MLP learns task A, then task B;
// WITHOUT EWC the fine-tune on B catastrophically forgets A, WITH the Fisher-anchored penalty it
// retains A while still learning B. Tasks: XOR quadrants (A) and the unit circle (B) over 2-D
// points, with a TASK-INDICATOR input feature — without it the two labelings conflict pointwise
// on the same inputs and no weight setting can satisfy both (the literature's permuted/multi-head
// setups exist for exactly this reason; the first version of this test lacked the indicator and
// EWC could only retain 55%). Asserted: (1) baseline learns each
// task in isolation; (2) plain sequential training forgets A (accuracy drops far below its
// post-A level); (3) EWC training on B keeps A's accuracy SUBSTANTIALLY above the forgetting
// baseline while B still reaches usable accuracy.
func TestEWCReducesCatastrophicForgetting(t *testing.T) {
	if testing.Short() {
		t.Skip("trains several MLPs; skipped in -short")
	}
	rng := rand.New(rand.NewPCG(11, 0xe3c))
	sample := func(n int, task int) (*tensor.Tensor, []int) {
		x := tensor.New(tensor.F64, tensor.Shape{n, 3})
		labels := make([]int, n)
		for i := range n {
			a, b := rng.Float64()*4-2, rng.Float64()*4-2
			x.SetF64(a, i, 0)
			x.SetF64(b, i, 1)
			x.SetF64(float64(task), i, 2)
			switch task {
			case 0: // A: XOR quadrants
				if a*b > 0 {
					labels[i] = 1
				}
			case 1: // B: unit circle
				if a*a+b*b < 1 {
					labels[i] = 1
				}
			}
		}
		return x, labels
	}
	newNet := func() (*nn.Sequential, []*tensor.Tensor) {
		net := nn.NewSequential(
			nn.NewLinear(tensor.F64, 3, 32, 5), nn.ReLU(),
			nn.NewLinear(tensor.F64, 32, 32, 6), nn.ReLU(),
			nn.NewLinear(tensor.F64, 32, 2, 7),
		)
		return net, net.Params()
	}
	trainSteps := func(net *nn.Sequential, params []*tensor.Tensor, task, steps int,
		penalty func(ctx *backend.Context) (*tensor.Tensor, error)) {
		opt := nn.NewAdamW(params, 5e-3, 0)
		for range steps {
			x, labels := sample(64, task)
			targets := tensor.New(tensor.F64, tensor.Shape{64})
			for i, l := range labels {
				targets.SetF64(float64(l), i)
			}
			tape := autograd.NewTape()
			ctx := tape.Context()
			logits, err := net.Forward(ctx, x)
			if err != nil {
				t.Fatal(err)
			}
			loss, err := nn.CrossEntropy(ctx, logits, targets)
			if err != nil {
				t.Fatal(err)
			}
			if penalty != nil {
				pen, err := penalty(ctx)
				if err != nil {
					t.Fatal(err)
				}
				sum, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{loss, pen}, nil)
				if err != nil {
					t.Fatal(err)
				}
				loss = sum[0]
			}
			if err := tape.Backward(loss); err != nil {
				t.Fatal(err)
			}
			if err := opt.Step(tape.Grad); err != nil {
				t.Fatal(err)
			}
		}
	}
	accuracy := func(net *nn.Sequential, task int) float64 {
		x, labels := sample(600, task)
		logits, err := net.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		hit := 0
		for i, l := range labels {
			pred := 0
			if logits.AtF64(i, 1) > logits.AtF64(i, 0) {
				pred = 1
			}
			if pred == l {
				hit++
			}
		}
		return float64(hit) / 600
	}
	snapshot := func(params []*tensor.Tensor) []*tensor.Tensor {
		out := make([]*tensor.Tensor, len(params))
		for i, p := range params {
			c := tensor.New(p.Dtype(), p.Shape())
			for j := range p.Numel() {
				idx := tensor.Unravel(j, p.Shape())
				c.SetF64(p.AtF64(idx...), idx...)
			}
			out[i] = c
		}
		return out
	}
	// Fisher: mean squared gradient over task-A batches at the post-A optimum.
	fisherAt := func(net *nn.Sequential, params []*tensor.Tensor) []*tensor.Tensor {
		var samples [][]*tensor.Tensor
		for range 20 {
			x, labels := sample(64, 0)
			targets := tensor.New(tensor.F64, tensor.Shape{64})
			for i, l := range labels {
				targets.SetF64(float64(l), i)
			}
			tape := autograd.NewTape()
			ctx := tape.Context()
			logits, err := net.Forward(ctx, x)
			if err != nil {
				t.Fatal(err)
			}
			loss, err := nn.CrossEntropy(ctx, logits, targets)
			if err != nil {
				t.Fatal(err)
			}
			if err := tape.Backward(loss); err != nil {
				t.Fatal(err)
			}
			gs := make([]*tensor.Tensor, len(params))
			for i, p := range params {
				gs[i] = tape.Grad(p)
			}
			samples = append(samples, gs)
		}
		f, err := nn.EWCFisher(samples)
		if err != nil {
			t.Fatal(err)
		}
		return f
	}

	run := func(useEWC bool) (accA, accB float64) {
		net, params := newNet()
		trainSteps(net, params, 0, 400, nil)
		if a := accuracy(net, 0); a < 0.9 {
			t.Fatalf("task A not learned (%.0f%%)", 100*a)
		}
		var penalty func(ctx *backend.Context) (*tensor.Tensor, error)
		if useEWC {
			ref := snapshot(params)
			fisher := fisherAt(net, params)
			penalty = func(ctx *backend.Context) (*tensor.Tensor, error) {
				return nn.EWCPenalty(ctx, params, ref, fisher, 5000)
			}
		}
		trainSteps(net, params, 1, 400, penalty)
		return accuracy(net, 0), accuracy(net, 1)
	}

	plainA, plainB := run(false)
	ewcA, ewcB := run(true)
	t.Logf("after B: plain A=%.0f%% B=%.0f%% | EWC A=%.0f%% B=%.0f%%", 100*plainA, 100*plainB, 100*ewcA, 100*ewcB)
	if plainB < 0.85 {
		t.Fatalf("plain run failed to learn task B (%.0f%%)", 100*plainB)
	}
	if ewcB < 0.75 {
		t.Fatalf("EWC prevented learning task B (%.0f%%)", 100*ewcB)
	}
	if ewcA < plainA+0.1 {
		t.Fatalf("EWC did not retain task A meaningfully better (%.0f%% vs plain %.0f%%)", 100*ewcA, 100*plainA)
	}
	if math.Min(ewcA, ewcB) < 0.7 {
		t.Fatalf("EWC should keep BOTH tasks usable (A %.0f%%, B %.0f%%)", 100*ewcA, 100*ewcB)
	}
}
