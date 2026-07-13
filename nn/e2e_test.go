package nn_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// §T18 / §G5: a small MLP trained end to end through the full stack
// (dispatch → tape → VJPs → optimizer) must converge on a deterministic
// synthetic 4-class problem: initial loss ≈ ln(4) (chance), final loss < 0.1,
// training accuracy ≥ 95%.

// blobs builds a deterministic 4-class Gaussian-blob dataset: centers at
// (±2, ±2), noise σ=0.5, n samples per class.
func blobs(nPerClass int, seed uint64) (x *tensor.Tensor, y *tensor.Tensor) {
	rng := rand.New(rand.NewPCG(seed, 0xda7a5e7))
	centers := [4][2]float64{{2, 2}, {-2, 2}, {2, -2}, {-2, -2}}
	n := 4 * nPerClass
	xd := make([]float64, n*2)
	yd := make([]float64, n)
	for c := range 4 {
		for s := range nPerClass {
			i := c*nPerClass + s
			xd[i*2] = centers[c][0] + rng.NormFloat64()*0.5
			xd[i*2+1] = centers[c][1] + rng.NormFloat64()*0.5
			yd[i] = float64(c)
		}
	}
	return tensor.FromFloat64(tensor.Shape{n, 2}, xd), tensor.FromFloat64(tensor.Shape{n}, yd)
}

func accuracy(t *testing.T, logits, labels *tensor.Tensor) float64 {
	t.Helper()
	pred, err := backend.Execute(backend.NewContext(), backend.OpArgMax,
		[]*tensor.Tensor{logits}, backend.ArgMaxAttrs{Axis: 1})
	if err != nil {
		t.Fatal(err)
	}
	n := labels.Numel()
	correct := 0
	for i := range n {
		if pred[0].AtF64(i) == labels.AtF64(i) {
			correct++
		}
	}
	return float64(correct) / float64(n)
}

func TestEndToEndClassification(t *testing.T) {
	x, y := blobs(40, 1) // 160 samples, 4 classes

	model := nn.NewSequential(
		nn.NewLinear(tensor.F64, 2, 16, 7),
		nn.ReLU(),
		nn.NewLinear(tensor.F64, 16, 4, 8),
	)
	opt := nn.NewAdam(model.Params(), 0.05)

	var first, last float64
	for step := range 300 {
		tape := autograd.NewTape()
		ctx := tape.Context()
		logits, err := model.Forward(ctx, x)
		if err != nil {
			t.Fatal(err)
		}
		loss, err := nn.CrossEntropy(ctx, logits, y)
		if err != nil {
			t.Fatal(err)
		}
		lv := loss.AtF64()
		if step == 0 {
			first = lv
			// untrained CE must sit near chance level ln(4)
			if math.Abs(first-math.Log(4)) > 0.5 {
				t.Fatalf("initial loss %.4f not near ln(4)=%.4f — CE or init broken", first, math.Log(4))
			}
		}
		last = lv
		if lv < 0.03 {
			break // converged early
		}
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		if err := opt.Step(tape.Grad); err != nil {
			t.Fatal(err)
		}
	}

	if last >= 0.1 {
		t.Fatalf("did not converge: first %.4f last %.4f (want < 0.1)", first, last)
	}
	if last >= first/5 {
		t.Fatalf("insufficient improvement: %.4f → %.4f", first, last)
	}

	logits, err := model.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	if acc := accuracy(t, logits, y); acc < 0.95 {
		t.Fatalf("training accuracy %.3f < 0.95", acc)
	} else {
		t.Logf("converged: loss %.4f → %.4f, accuracy %.3f", first, last, acc)
	}
}

// The same net in f32 must also train (accumulation/master-state guards §V10
// pay off here); slightly looser convergence bound.
func TestEndToEndClassificationF32(t *testing.T) {
	x64, y := blobs(40, 2)
	// narrow inputs to f32
	xf := make([]float32, x64.Numel())
	for i := range xf {
		xf[i] = float32(x64.AtF64(tensor.Unravel(i, x64.Shape())...))
	}
	x := tensor.FromFloat32(tensor.Shape(x64.Shape()), xf)

	model := nn.NewSequential(
		nn.NewLinear(tensor.F32, 2, 16, 7),
		nn.ReLU(),
		nn.NewLinear(tensor.F32, 16, 4, 8),
	)
	// targets must match logits dtype (homogeneous-dtype dispatch)
	yf := make([]float32, y.Numel())
	for i := range yf {
		yf[i] = float32(y.AtF64(i))
	}
	y32 := tensor.FromFloat32(tensor.Shape{y.Numel()}, yf)

	opt := nn.NewAdam(model.Params(), 0.05)
	var last float64
	for range 300 {
		tape := autograd.NewTape()
		ctx := tape.Context()
		logits, err := model.Forward(ctx, x)
		if err != nil {
			t.Fatal(err)
		}
		loss, err := nn.CrossEntropy(ctx, logits, y32)
		if err != nil {
			t.Fatal(err)
		}
		last = loss.AtF64()
		if last < 0.05 {
			break
		}
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		if err := opt.Step(tape.Grad); err != nil {
			t.Fatal(err)
		}
	}
	if last >= 0.15 {
		t.Fatalf("f32 net did not converge: final loss %.4f", last)
	}
}
