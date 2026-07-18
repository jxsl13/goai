package nn_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// equalTensors reports whether a and b hold bit-identical values (same shape).
func equalTensors(a, b *tensor.Tensor) bool {
	if !a.Shape().Equal(b.Shape()) {
		return false
	}
	for i := range a.Numel() {
		idx := tensor.Unravel(i, a.Shape())
		if a.AtF64(idx...) != b.AtF64(idx...) {
			return false
		}
	}
	return true
}

// A doubly-nested Dropout must be reached by Sequential.Eval — the footgun this
// mechanism exists to close (a nested dropout used to be unreachable generically).
func TestModePropagatesThroughNestedSequential(t *testing.T) {
	d := nn.NewDropout(0.9, 3)
	inner := nn.NewSequential(nn.ReLU(), d)
	mid := nn.NewSequential(inner, nn.Tanh())
	outer := nn.NewSequential(nn.ReLU(), mid)

	if !d.Training {
		t.Fatal("dropout must default to training mode")
	}
	outer.Eval()
	if d.Training {
		t.Error("Eval must reach a dropout nested two Sequentials deep")
	}
	outer.Train()
	if !d.Training {
		t.Error("Train must restore the nested dropout")
	}
	// SetTrain is the generic form and must behave identically.
	nn.SetTrain(outer, false)
	if d.Training {
		t.Error("SetTrain(false) must reach the nested dropout")
	}
}

// DropPath participates in the same walk, and layers that are NOT mode-dependent
// (ReLU, Linear) are skipped silently rather than erroring.
func TestModeSkipsNonModeLayers(t *testing.T) {
	dp := nn.NewDropPath(0.5, 5)
	lin := nn.NewLinear(tensor.F64, 3, 3, 1)
	m := nn.NewSequential(lin, nn.ReLU(), dp, nn.GELU())
	nn.SetTrain(m, false) // must not panic on the parameter-free / mode-free layers
	if dp.Training {
		t.Error("DropPath must follow the walk into eval mode")
	}
	nn.SetTrain(m, true)
	if !dp.Training {
		t.Error("DropPath must follow the walk back into training mode")
	}
	// A bare non-Mode layer as the root is a no-op, not an error.
	nn.SetTrain(nn.ReLU(), false)
	nn.SetTrain(nil, false)
}

// Eval makes Dropout the exact identity, and Train restores actual dropping.
func TestModeEvalIsBitExactIdentity(t *testing.T) {
	x := tensor.FromFloat64(tensor.Shape{2, 4}, []float64{1, -2, 3, 4, -5, 6, 7, -8})
	d := nn.NewDropout(0.5, 11)
	m := nn.NewSequential(d)

	m.Eval()
	out, err := m.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	if !equalTensors(out, x) {
		t.Error("eval-mode dropout inside Sequential must be bit-exactly the identity")
	}

	m.Train()
	dropped := false
	for range 20 {
		out, err := m.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		if !equalTensors(out, x) {
			dropped = true
			break
		}
	}
	if !dropped {
		t.Error("training-mode dropout must change the output at rate 0.5")
	}
}

// hiddenModel owns a Dropout but does NOT implement Composite: the documented
// limit of the generic walk.
type hiddenModel struct {
	drop *nn.Dropout
	lin  *nn.Linear
}

func (h *hiddenModel) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	y, err := h.lin.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	return h.drop.Forward(ctx, y)
}
func (h *hiddenModel) Params() []*tensor.Tensor { return h.lin.Params() }

// walkableModel is the same model with the four-line opt-in.
type walkableModel struct{ hiddenModel }

func (w *walkableModel) Sublayers() []nn.Layer { return []nn.Layer{w.lin, w.drop} }

// The documented limit: SetTrain cannot reach into a struct that hides its
// sublayers, and CAN reach into the same struct once it implements Composite.
// This test exists to keep the documentation honest.
func TestSetTrainDocumentedLimit(t *testing.T) {
	hidden := &hiddenModel{drop: nn.NewDropout(0.5, 1), lin: nn.NewLinear(tensor.F64, 2, 2, 1)}
	nn.SetTrain(hidden, false)
	if !hidden.drop.Training {
		t.Error("a struct that hides its sublayers is documented as UNREACHABLE; " +
			"if this now works, update the SetTrain godoc")
	}

	open := &walkableModel{hiddenModel{drop: nn.NewDropout(0.5, 1), lin: nn.NewLinear(tensor.F64, 2, 2, 1)}}
	nn.SetTrain(open, false)
	if open.drop.Training {
		t.Error("implementing Composite must make the nested dropout reachable")
	}
	nn.SetTrain(open, true)
	if !open.drop.Training {
		t.Error("SetTrain(true) must restore the dropout of a Composite model")
	}
}

// Sequential itself satisfies both optional interfaces, and a Sequential handed
// to SetTrain as a plain Layer still propagates.
func TestSequentialImplementsModeAndComposite(t *testing.T) {
	d := nn.NewDropout(0.3, 2)
	var l nn.Layer = nn.NewSequential(d)
	if _, ok := l.(nn.Mode); !ok {
		t.Fatal("Sequential must implement Mode")
	}
	c, ok := l.(nn.Composite)
	if !ok {
		t.Fatal("Sequential must implement Composite")
	}
	if len(c.Sublayers()) != 1 {
		t.Errorf("Sublayers() = %d layers, want 1", len(c.Sublayers()))
	}
	nn.SetTrain(l, false)
	if d.Training {
		t.Error("SetTrain through the Layer interface must propagate")
	}
}
