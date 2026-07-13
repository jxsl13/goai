package vision_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
	"github.com/jxsl13/goai/vision"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// stripes builds a 3-class synthetic image set: horizontal bar, vertical bar, or both (cross) —
// classes only separable by SPATIAL pattern, the thing a conv layer must learn.
func stripes(n, hw int) (*tensor.Tensor, *tensor.Tensor, []int) {
	x := tensor.New(tensor.F64, tensor.Shape{n, 1, hw, hw})
	t := tensor.New(tensor.F64, tensor.Shape{n})
	labels := make([]int, n)
	for s := range n {
		cls := s % 3
		labels[s] = cls
		t.SetF64(float64(cls), s)
		row, col := (s*3)%(hw-1), (s*5)%(hw-1)
		for i := range hw {
			for j := range hw {
				v := 0.05 * math.Sin(float64(s*11+i*3+j*7))
				if (cls == 0 || cls == 2) && i == row {
					v += 1
				}
				if (cls == 1 || cls == 2) && j == col {
					v += 1
				}
				x.SetF64(v, s, 0, i, j)
			}
		}
	}
	return x, t, labels
}

// §T457: shape contract and error paths.
func TestCNNShapesAndErrors(t *testing.T) {
	m, err := vision.NewCNN(1, 3, 7, vision.WithDtype(tensor.F64))
	if err != nil {
		t.Fatal(err)
	}
	x := tensor.New(tensor.F64, tensor.Shape{2, 1, 8, 8})
	logits, err := m.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	if !logits.Shape().Equal(tensor.Shape{2, 3}) {
		t.Fatalf("logits %v, want [2 3]", logits.Shape())
	}
	if got := len(m.Params()); got != 6 { // 2 convs × (W,B) + head (W,B)
		t.Fatalf("params %d, want 6", got)
	}
	if _, err := vision.NewCNN(0, 3, 1); err == nil {
		t.Fatal("in=0 accepted")
	}
	if _, err := vision.NewCNN(1, 3, 1, vision.WithKernel(4)); err == nil {
		t.Fatal("even kernel accepted")
	}
	if _, err := m.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{2, 1, 6, 6})); err == nil {
		t.Fatal("6x6 through two pool stages accepted") // 6→3, 3 not divisible
	}
	if _, err := m.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{2, 3})); err == nil {
		t.Fatal("rank-2 accepted")
	}
}

// §T457: every parameter receives a gradient through the full stack.
func TestCNNGradFlow(t *testing.T) {
	m, err := vision.NewCNN(1, 3, 7, vision.WithDtype(tensor.F64))
	if err != nil {
		t.Fatal(err)
	}
	x, targets, _ := stripes(6, 8)
	tape := autograd.NewTape()
	ctx := tape.Context()
	logits, err := m.Forward(ctx, x)
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
	for i, p := range m.Params() {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("param %d: nil grad", i)
		}
		var norm float64
		for j := range g.Numel() {
			v := g.AtF64(tensor.Unravel(j, g.Shape())...)
			norm += v * v
		}
		if norm == 0 {
			t.Fatalf("param %d: zero grad — not connected", i)
		}
	}
}

// §T457 end-to-end: the CNN learns the spatial 3-class task to high train accuracy with the
// standard loop — the CV path (conv stack + fused backward + GAP head) works as a system.
func TestCNNLearnsSpatialClasses(t *testing.T) {
	m, err := vision.NewCNN(1, 3, 7, vision.WithDtype(tensor.F64), vision.WithChannels(8, 16))
	if err != nil {
		t.Fatal(err)
	}
	x, targets, labels := stripes(24, 8)
	opt := nn.NewAdamW(m.Params(), 0.01, 0)
	var first, last float64
	for step := range 150 {
		tape := autograd.NewTape()
		ctx := tape.Context()
		logits, err := m.Forward(ctx, x)
		if err != nil {
			t.Fatal(err)
		}
		loss, err := nn.CrossEntropy(ctx, logits, targets)
		if err != nil {
			t.Fatal(err)
		}
		lv := loss.AtF64()
		if step == 0 {
			first = lv
		}
		last = lv
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		clipped, _ := nn.ClipGradNorm(m.Params(), tape.Grad, 1.0)
		if err := opt.Step(clipped); err != nil {
			t.Fatal(err)
		}
	}
	logits, err := m.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	correct := 0
	for s, want := range labels {
		best, bestV := 0, math.Inf(-1)
		for c := range 3 {
			if v := logits.AtF64(s, c); v > bestV {
				best, bestV = c, v
			}
		}
		if best == want {
			correct++
		}
	}
	acc := float64(correct) / float64(len(labels))
	if acc < 0.9 {
		t.Fatalf("train accuracy %.0f%% < 90%% (CE %.3f → %.3f)", 100*acc, first, last)
	}
	t.Logf("CNN learned stripes: CE %.3f → %.3f, train accuracy %.0f%%", first, last, 100*acc)
}
