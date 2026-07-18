package nn_test

import (
	"fmt"
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

// Switching a whole model to inference. Dropout and DropPath DEFAULT TO TRAINING
// MODE and nothing switches them for you, so predicting without this call gives
// silently noisy outputs — no error, just worse numbers.
func ExampleSetTrain() {
	// A dropout nested one level down, to show that the walk recurses.
	model := nn.NewSequential(
		nn.NewLinear(tensor.F64, 4, 4, 1),
		nn.NewSequential(nn.NewDropout(0.5, 7)),
	)
	x := tensor.FromFloat64(tensor.Shape{1, 4}, []float64{1, 1, 1, 1})

	// Read the pre-dropout activations once, so we can say exactly whether the
	// model's output is the identity on them or a masked, rescaled version.
	clean, err := model.Layers[0].Forward(backend.NewContext(), x)
	if err != nil {
		panic(err)
	}
	describe := func() string {
		y, err := model.Forward(backend.NewContext(), x)
		if err != nil {
			panic(err)
		}
		zeros, untouched := 0, 0
		for i := range 4 {
			if y.AtF64(0, i) == 0 {
				zeros++
			}
			if y.AtF64(0, i) == clean.AtF64(0, i) {
				untouched++
			}
		}
		return fmt.Sprintf("%d/4 units zeroed, %d/4 passed through unchanged", zeros, untouched)
	}

	// A fresh model is in TRAINING mode: dropout is firing.
	fmt.Println("training:", describe())

	// nn.SetTrain(model, false) — equivalently model.Eval() — walks the whole
	// tree, including the nested Sequential, and reaches the dropout.
	nn.SetTrain(model, false)
	fmt.Println("eval:    ", describe())

	// And back, for the next epoch — a fresh mask, so a different subset drops.
	nn.SetTrain(model, true)
	fmt.Println("training:", describe())

	// Output:
	// training: 2/4 units zeroed, 0/4 passed through unchanged
	// eval:     0/4 units zeroed, 4/4 passed through unchanged
	// training: 1/4 units zeroed, 0/4 passed through unchanged
}

// tinyBlock is a hand-written model that owns layers in named fields. Four lines
// of Sublayers make it an nn.Composite, which is what lets a generic walk such
// as nn.SetTrain descend into it and reach the dropout inside.
type tinyBlock struct {
	Proj *nn.Linear
	Drop *nn.Dropout
}

func (b *tinyBlock) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := b.Proj.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	return b.Drop.Forward(ctx, h)
}

func (b *tinyBlock) Params() []*tensor.Tensor { return b.Proj.Params() }

// Sublayers is the opt-in: without it the walk sees an opaque leaf and CANNOT
// reach b.Drop, which then keeps firing at inference.
func (b *tinyBlock) Sublayers() []nn.Layer { return []nn.Layer{b.Proj, b.Drop} }

// A custom model type joins the mode-switching machinery by implementing
// nn.Composite. Go offers no sound way for a generic walk to discover the
// sublayers of an arbitrary struct, so the walk asks the type instead.
func ExampleComposite() {
	block := &tinyBlock{
		Proj: nn.NewLinear(tensor.F64, 4, 4, 1),
		Drop: nn.NewDropout(0.5, 7),
	}
	var _ nn.Composite = block // the opt-in the walk looks for

	// Probing for the optional interfaces is what SetTrain does internally; a
	// layer implementing neither is skipped, which is not a failure.
	if _, ok := nn.Layer(block).(nn.Mode); ok {
		fmt.Println("unexpected: tinyBlock itself is mode-dependent")
	}
	fmt.Println("block is a Composite, its Dropout is a Mode")
	fmt.Println("sublayers exposed to the walk:", len(block.Sublayers()))

	nn.SetTrain(block, false) // descends through Sublayers
	fmt.Println("nested dropout training after SetTrain(false):", block.Drop.Training)

	nn.SetTrain(block, true)
	fmt.Println("nested dropout training after SetTrain(true): ", block.Drop.Training)

	// Output:
	// block is a Composite, its Dropout is a Mode
	// sublayers exposed to the walk: 2
	// nested dropout training after SetTrain(false): false
	// nested dropout training after SetTrain(true):  true
}
