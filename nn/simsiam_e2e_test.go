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

// §T492: self-supervised learning end-to-end — SimSiam pretrains a conv encoder on UNLABELED
// synthetic images (two augmented views per image: pixel jitter + circular shift), then a LINEAR
// PROBE on the frozen features must separate the three classes far above chance. Also asserted:
// the representation does not collapse (per-dimension std stays positive) — the failure mode
// SimSiam's stop-gradient exists to prevent.
func TestSimSiamPretrainsUsefulFeatures(t *testing.T) {
	if testing.Short() {
		t.Skip("SSL pretraining; skipped in -short")
	}
	const (
		hw     = 8
		feat   = 16
		proj   = 16
		batch  = 32
		steps  = 600
		probeN = 240
	)
	rng := rand.New(rand.NewPCG(5, 0x551))

	// base image of class c (0: horizontal bar, 1: vertical bar, 2: cross), bar at a random line.
	baseImg := func(c int) ([]float64, int, int) {
		img := make([]float64, hw*hw)
		row, col := 1+rng.IntN(hw-2), 1+rng.IntN(hw-2)
		for i := range hw {
			if c == 0 || c == 2 {
				img[row*hw+i] += 1
			}
			if c == 1 || c == 2 {
				img[i*hw+col] += 1
			}
		}
		return img, row, col
	}
	// augment: circular shift by ±1 in each axis + pixel noise.
	augment := func(img []float64) []float64 {
		dy, dx := rng.IntN(3)-1, rng.IntN(3)-1
		out := make([]float64, hw*hw)
		for y := range hw {
			for x := range hw {
				out[((y+dy+hw)%hw)*hw+(x+dx+hw)%hw] = img[y*hw+x] + rng.NormFloat64()*0.05
			}
		}
		return out
	}
	toTensor := func(imgs [][]float64) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape{len(imgs), 1, hw, hw})
		for i, img := range imgs {
			for j, v := range img {
				x.SetF64(v, i, 0, j/hw, j%hw)
			}
		}
		return x
	}

	// encoder: Conv(1→8) → ReLU → MaxPool2 → Conv(8→feat) → ReLU → GAP → [batch, feat].
	conv1, err := nn.NewConv2D(tensor.F64, 1, 8, 3, 1, 1, 3)
	if err != nil {
		t.Fatal(err)
	}
	conv2, err := nn.NewConv2D(tensor.F64, 8, feat, 3, 1, 1, 4)
	if err != nil {
		t.Fatal(err)
	}
	relu := nn.ReLU()
	pool := &nn.MaxPool2D{Kernel: 2}
	encode := func(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
		h, err := conv1.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		if h, err = relu.Forward(ctx, h); err != nil {
			return nil, err
		}
		if h, err = pool.Forward(ctx, h); err != nil {
			return nil, err
		}
		if h, err = conv2.Forward(ctx, h); err != nil {
			return nil, err
		}
		if h, err = relu.Forward(ctx, h); err != nil {
			return nil, err
		}
		g, err := backend.Execute(ctx, backend.OpAvgPool2D, []*tensor.Tensor{h}, backend.PoolAttrs{Kernel: hw / 2})
		if err != nil {
			return nil, err
		}
		out, err := backend.Execute(ctx, backend.OpReshape, []*tensor.Tensor{g[0]},
			backend.ReshapeAttrs{Shape: tensor.Shape{x.Shape()[0], feat}})
		if err != nil {
			return nil, err
		}
		return out[0], nil
	}
	projector := nn.NewLinear(tensor.F64, feat, proj, 5)
	predictor := nn.NewLinear(tensor.F64, proj, proj, 6)
	sslParams := append(append(append(conv1.Params(), conv2.Params()...), projector.Params()...), predictor.Params()...)

	opt := nn.NewAdamW(sslParams, 2e-3, 0)
	var first, last float64
	for step := range steps {
		imgs1 := make([][]float64, batch)
		imgs2 := make([][]float64, batch)
		for i := range batch {
			img, _, _ := baseImg(rng.IntN(3))
			imgs1[i] = augment(img)
			imgs2[i] = augment(img)
		}
		tape := autograd.NewTape()
		ctx := tape.Context()
		z1e, err := encode(ctx, toTensor(imgs1))
		if err != nil {
			t.Fatal(err)
		}
		z2e, err := encode(ctx, toTensor(imgs2))
		if err != nil {
			t.Fatal(err)
		}
		z1, err := projector.Forward(ctx, z1e)
		if err != nil {
			t.Fatal(err)
		}
		z2, err := projector.Forward(ctx, z2e)
		if err != nil {
			t.Fatal(err)
		}
		p1, err := predictor.Forward(ctx, z1)
		if err != nil {
			t.Fatal(err)
		}
		p2, err := predictor.Forward(ctx, z2)
		if err != nil {
			t.Fatal(err)
		}
		loss, err := nn.SimSiamLoss(ctx, p1, z1, p2, z2)
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
		if err := opt.Step(tape.Grad); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("SimSiam loss: %.3f → %.3f (−1 = perfectly aligned views)", first, last)
	if last > -0.5 {
		t.Fatalf("views did not align (loss %.3f)", last)
	}

	// anti-collapse: feature std across a fresh batch must be meaningfully positive.
	imgs := make([][]float64, 96)
	labels := make([]int, 96)
	for i := range imgs {
		c := i % 3
		img, _, _ := baseImg(c)
		imgs[i] = augment(img)
		labels[i] = c
	}
	featsT, err := encode(backend.NewContext(), toTensor(imgs))
	if err != nil {
		t.Fatal(err)
	}
	var meanStd float64
	for d := range feat {
		var m, s float64
		for i := range 96 {
			m += featsT.AtF64(i, d)
		}
		m /= 96
		for i := range 96 {
			v := featsT.AtF64(i, d) - m
			s += v * v
		}
		meanStd += math.Sqrt(s / 96)
	}
	meanStd /= feat
	if meanStd < 1e-3 {
		t.Fatalf("representation collapsed (mean feature std %.2g)", meanStd)
	}

	// linear probe on FROZEN features.
	probe := nn.NewLinear(tensor.F64, feat, 3, 7)
	pOpt := nn.NewAdamW(probe.Params(), 0.02, 0)
	targets := tensor.New(tensor.F64, tensor.Shape{96})
	for i, l := range labels {
		targets.SetF64(float64(l), i)
	}
	for range probeN {
		tape := autograd.NewTape()
		ctx := tape.Context()
		logits, err := probe.Forward(ctx, featsT) // frozen features: computed outside the tape
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
		if err := pOpt.Step(tape.Grad); err != nil {
			t.Fatal(err)
		}
	}
	// evaluate the probe on FRESH data.
	fresh := make([][]float64, 150)
	freshLabels := make([]int, 150)
	for i := range fresh {
		c := i % 3
		img, _, _ := baseImg(c)
		fresh[i] = augment(img)
		freshLabels[i] = c
	}
	ff, err := encode(backend.NewContext(), toTensor(fresh))
	if err != nil {
		t.Fatal(err)
	}
	logits, err := probe.Forward(backend.NewContext(), ff)
	if err != nil {
		t.Fatal(err)
	}
	hit := 0
	for i, l := range freshLabels {
		best := 0
		for c := 1; c < 3; c++ {
			if logits.AtF64(i, c) > logits.AtF64(i, best) {
				best = c
			}
		}
		if best == l {
			hit++
		}
	}
	acc := float64(hit) / 150
	t.Logf("linear probe on frozen SSL features: %.0f%% (chance 33%%), mean feature std %.3f", 100*acc, meanStd)
	if acc < 0.7 {
		t.Fatalf("probe accuracy %.0f%% — SSL features not linearly useful", 100*acc)
	}
}
