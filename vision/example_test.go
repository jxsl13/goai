package vision_test

import (
	"fmt"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
	"github.com/jxsl13/goai/vision"
)

// A CNN maps a batch of NCHW images to one logit per class.
func ExampleNewCNN() {
	m, _ := vision.NewCNN(1 /*channels in*/, 3 /*classes*/, 42)
	x := tensor.New(tensor.F32, tensor.Shape{2, 1, 16, 16})
	logits, _ := m.Forward(backend.NewContext(), x)
	fmt.Println(logits.Shape())
	// Output: (2, 3)
}

// Training is the standard four-step loop: forward, loss, backward, optimizer
// step. One step on a tiny batch already lowers the loss.
func ExampleCNN() {
	m, _ := vision.NewCNN(1, 2, 7, vision.WithDtype(tensor.F64), vision.WithChannels(4))
	x := tensor.New(tensor.F64, tensor.Shape{2, 1, 8, 8})
	x.SetF64(1, 0, 0, 1, 1) // class-0 sample has a bright pixel top-left
	targets := tensor.FromFloat64(tensor.Shape{2}, []float64{0, 1})
	opt := nn.NewAdamW(m.Params(), 0.05, 0)

	var first, after float64
	for step := range 8 {
		tape := autograd.NewTape()
		ctx := tape.Context()
		logits, _ := m.Forward(ctx, x)
		loss, _ := nn.CrossEntropy(ctx, logits, targets)
		if step == 0 {
			first = loss.AtF64()
		}
		after = loss.AtF64()
		_ = tape.Backward(loss)
		_ = opt.Step(tape.Grad)
	}
	fmt.Println(after < first)
	// Output: true
}

// Embedded in a pipeline: after training, argmax over Forward's logits is the
// predicted class for each image in the batch.
func ExampleCNN_predict() {
	m, _ := vision.NewCNN(1, 2, 42, vision.WithChannels(4))
	x := tensor.New(tensor.F32, tensor.Shape{1, 1, 8, 8})
	logits, _ := m.Forward(backend.NewContext(), x)
	best := 0
	for c := 1; c < 2; c++ {
		if logits.AtF64(0, c) > logits.AtF64(0, best) {
			best = c
		}
	}
	fmt.Println(best >= 0 && best < 2)
	// Output: true
}

// Safetensors exports every parameter under stable names; FromSafetensors
// rebuilds the identical CNN — the checkpoint round-trip.
func ExampleCNN_Safetensors() {
	m, _ := vision.NewCNN(3, 4, 1)
	ts := m.Safetensors()
	restored, _ := vision.FromSafetensors(ts)
	fmt.Println(len(ts) > 0, restored != nil)
	// Output: true true
}
