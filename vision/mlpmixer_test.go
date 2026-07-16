package vision_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
	"github.com/jxsl13/goai/vision"
)

// mixerGlobalPattern builds a 2-class task where the label depends on the GLOBAL arrangement of
// patches, not any single patch: class 0 = a bright block in the top-left, class 1 = a bright
// block in the bottom-right. Every image has one bright patch-sized block (matched intensity), so
// only a model that mixes across patches (token-mixing) can separate the two classes.
func mixerGlobalPattern(n, hw, patch int) (*tensor.Tensor, *tensor.Tensor, []int) {
	x := tensor.New(tensor.F64, tensor.Shape{n, 1, hw, hw})
	tg := tensor.New(tensor.F64, tensor.Shape{n})
	labels := make([]int, n)
	for s := range n {
		cls := s % 2
		labels[s] = cls
		for i := range hw {
			for j := range hw {
				x.SetF64(0.05*math.Sin(float64(s*17+i*3+j*7)), s, 0, i, j)
			}
		}
		oy, ox := 0, 0
		if cls == 1 {
			oy, ox = hw-patch, hw-patch
		}
		for i := range patch {
			for j := range patch {
				x.SetF64(1.0, s, 0, oy+i, ox+j)
			}
		}
		tg.SetF64(float64(cls), s)
	}
	return x, tg, labels
}

// §V16.5 THE VALUE: an MLP-Mixer trained a few hundred steps learns a task whose label depends on
// the GLOBAL arrangement of patches — loss drops sharply and train accuracy rises to 100%.
func TestMixerLearnsGlobalPattern(t *testing.T) {
	m, err := vision.NewMLPMixer(1, 8, 2, 3,
		vision.WithMixerDtype(tensor.F64), vision.WithMixerPatch(4), vision.WithMixerHidden(16),
		vision.WithMixerLayers(2), vision.WithMixerTokenDim(8), vision.WithMixerChannelDim(32))
	if err != nil {
		t.Fatal(err)
	}
	x, targets, labels := mixerGlobalPattern(16, 8, 4)
	opt := nn.NewAdamW(m.Params(), 0.01, 0)
	var first, last float64
	for step := range 250 {
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
		for c := range 2 {
			if v := logits.AtF64(s, c); v > bestV {
				best, bestV = c, v
			}
		}
		if best == want {
			correct++
		}
	}
	acc := float64(correct) / float64(len(labels))
	if last > first*0.3 {
		t.Fatalf("Mixer did not learn: loss %.4f → %.4f", first, last)
	}
	if acc < 0.9 {
		t.Fatalf("train accuracy %.0f%% < 90%%", 100*acc)
	}
	t.Logf("Mixer learned global pattern: CE %.3f → %.3f, train accuracy %.0f%%", first, last, 100*acc)
}

// §V16.6 SANITY: geometry/dimension validation, determinism and shape errors.
func TestMixerValidation(t *testing.T) {
	if _, err := vision.NewMLPMixer(1, 8, 3, 7, vision.WithMixerPatch(3)); err == nil {
		t.Error("patch must divide the image size")
	}
	if _, err := vision.NewMLPMixer(0, 8, 3, 7); err == nil {
		t.Error("channels must be > 0")
	}
	if _, err := vision.NewMLPMixer(1, 8, 3, 7, vision.WithMixerHidden(0)); err == nil {
		t.Error("hidden must be > 0")
	}
	if _, err := vision.NewMLPMixer(1, 8, 3, 7, vision.WithMixerLayers(0)); err == nil {
		t.Error("layers must be > 0")
	}
	if _, err := vision.NewMLPMixer(1, 8, 3, 7, vision.WithMixerTokenDim(-1)); err == nil {
		t.Error("token MLP dim must be > 0")
	}
	m, err := vision.NewMLPMixer(1, 8, 3, 7)
	if err != nil {
		t.Fatal(err)
	}
	bad := tensor.New(tensor.F32, tensor.Shape{1, 6, 6})
	if _, err := m.Forward(backend.NewContext(), bad); err == nil {
		t.Error("wrong image geometry must error")
	}
	if _, err := m.Forward(backend.NewContext(), tensor.New(tensor.F32, tensor.Shape{3})); err == nil {
		t.Error("rank-1 input must error")
	}
	// Determinism: same seed → identical logits.
	img := tensor.New(tensor.F64, tensor.Shape{1, 8, 8})
	for i := range 8 {
		for j := range 8 {
			img.SetF64(math.Sin(float64(i*3+j*5)), 0, i, j)
		}
	}
	a, _ := vision.NewMLPMixer(1, 8, 3, 7, vision.WithMixerDtype(tensor.F64))
	b, _ := vision.NewMLPMixer(1, 8, 3, 7, vision.WithMixerDtype(tensor.F64))
	la, _ := a.Forward(backend.NewContext(), img)
	lb, _ := b.Forward(backend.NewContext(), img)
	for c := range 3 {
		if la.AtF64(0, c) != lb.AtF64(0, c) {
			t.Fatalf("non-deterministic: %.15g vs %.15g", la.AtF64(0, c), lb.AtF64(0, c))
		}
	}
}

// A tiny MLP-Mixer classifies one 8×8 grayscale image with NO convolution and NO attention: the
// image becomes 4 patch rows, the Mixer layers alternate token-mixing (across patches) and
// channel-mixing (across features), and the head reads the pooled representation.
func ExampleMLPMixer() {
	m, err := vision.NewMLPMixer(1, 8, 3, 7,
		vision.WithMixerHidden(16), vision.WithMixerLayers(2))
	if err != nil {
		panic(err)
	}
	img := tensor.New(tensor.F32, tensor.Shape{1, 8, 8}) // zeros: just the shapes
	logits, err := m.Forward(backend.NewContext(), img)
	if err != nil {
		panic(err)
	}
	fmt.Println(logits.Shape(), len(m.Params()) > 0)
	// Output: (1, 3) true
}

// NewMLPMixer validates geometry: the patch size must divide the image size.
func ExampleNewMLPMixer() {
	_, err := vision.NewMLPMixer(1, 8, 3, 7, vision.WithMixerPatch(3))
	fmt.Println(err != nil)
	// Output: true
}
