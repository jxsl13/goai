package vision_test

import (
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
	"github.com/jxsl13/goai/vision"
)

// §T590 end-to-end: the ViT learns the same spatial 3-class task the CNN e2e uses
// (apples to apples) to a large loss drop with the standard loop — patchify, class
// token, encoder stack and head train as a system through the tape.
func TestViTLearnsSpatialClasses(t *testing.T) {
	m, err := vision.NewViT(1, 8, 3, 7,
		vision.WithViTDtype(tensor.F64), vision.WithViTDim(16), vision.WithViTDepth(1), vision.WithViTHeads(2))
	if err != nil {
		t.Fatal(err)
	}
	x, targets, _ := stripes(12, 8)
	opt := nn.NewAdamW(m.Params(), 0.01, 0)
	var first, last float64
	for step := range 80 {
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
	if last > first*0.2 {
		t.Fatalf("ViT did not learn: loss %.4f → %.4f", first, last)
	}
}

// §T590: geometry validation — wrong shapes and non-dividing patch sizes error.
func TestViTValidation(t *testing.T) {
	if _, err := vision.NewViT(1, 8, 3, 7, vision.WithViTPatch(3)); err == nil {
		t.Error("patch must divide the image size")
	}
	if _, err := vision.NewViT(1, 8, 3, 7, vision.WithViTDim(30), vision.WithViTHeads(4)); err == nil {
		t.Error("dim must be divisible by heads")
	}
	m, err := vision.NewViT(1, 8, 3, 7)
	if err != nil {
		t.Fatal(err)
	}
	bad := tensor.New(tensor.F32, tensor.Shape{1, 6, 6})
	if _, err := m.Forward(nil, bad); err == nil {
		t.Error("wrong image geometry must error")
	}
}
