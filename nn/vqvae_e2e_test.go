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

// §T491: VQ-VAE end-to-end — encoder → VectorQuantizer (straight-through) → decoder trained
// jointly on an 8-Gaussian ring mixture. The discrete bottleneck must (1) reconstruct well (MSE
// far below the data variance), (2) actually USE the codebook (≥6 of 16 codes active — no
// codebook collapse), and (3) train the ENCODER through the straight-through estimator (its
// weights must move) — the part that silently breaks when the ST gradient path is miswired.
func TestVQVAETrainsEndToEnd(t *testing.T) {
	if testing.Short() {
		t.Skip("trains a VQ-VAE; skipped in -short")
	}
	const (
		dim   = 2
		hid   = 32
		lat   = 4
		codes = 16
		batch = 128
		steps = 3000
	)
	rng := rand.New(rand.NewPCG(3, 0x5a1))
	sample := func(n int) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape{n, dim})
		for i := range n {
			c := rng.IntN(8)
			th := 2 * math.Pi * float64(c) / 8
			x.SetF64(2*math.Cos(th)+rng.NormFloat64()*0.1, i, 0)
			x.SetF64(2*math.Sin(th)+rng.NormFloat64()*0.1, i, 1)
		}
		return x
	}

	enc1 := nn.NewLinear(tensor.F64, dim, hid, 3)
	enc2 := nn.NewLinear(tensor.F64, hid, lat, 4)
	dec1 := nn.NewLinear(tensor.F64, lat, hid, 5)
	dec2 := nn.NewLinear(tensor.F64, hid, dim, 6)
	relu := nn.ReLU()
	vq := nn.NewVectorQuantizer(tensor.F64, codes, lat, 0.25)
	params := append(append(append(append(enc1.Params(), enc2.Params()...), dec1.Params()...), dec2.Params()...), vq.Params()...)

	forward := func(ctx *backend.Context, x *tensor.Tensor) (recon, vqLoss *tensor.Tensor, idx []int, err error) {
		h, err := enc1.Forward(ctx, x)
		if err != nil {
			return nil, nil, nil, err
		}
		if h, err = relu.Forward(ctx, h); err != nil {
			return nil, nil, nil, err
		}
		ze, err := enc2.Forward(ctx, h)
		if err != nil {
			return nil, nil, nil, err
		}
		zq, vqLoss, idx, err := vq.Quantize(ctx, ze)
		if err != nil {
			return nil, nil, nil, err
		}
		if h, err = dec1.Forward(ctx, zq); err != nil {
			return nil, nil, nil, err
		}
		if h, err = relu.Forward(ctx, h); err != nil {
			return nil, nil, nil, err
		}
		recon, err = dec2.Forward(ctx, h)
		return recon, vqLoss, idx, err
	}

	encBefore := enc1.W.AtF64(0, 0)
	opt := nn.NewAdamW(params, 3e-3, 0)
	var first, last float64
	for step := range steps {
		x := sample(batch)
		tape := autograd.NewTape()
		ctx := tape.Context()
		recon, vqLoss, _, err := forward(ctx, x)
		if err != nil {
			t.Fatal(err)
		}
		mse, err := nn.MSE(ctx, recon, x)
		if err != nil {
			t.Fatal(err)
		}
		total, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{mse, vqLoss}, nil)
		if err != nil {
			t.Fatal(err)
		}
		lv := mse.AtF64()
		if step == 0 {
			first = lv
		}
		last = lv
		if err := tape.Backward(total[0]); err != nil {
			t.Fatal(err)
		}
		if err := opt.Step(tape.Grad); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("VQ-VAE reconstruction MSE: %.4f → %.4f (data variance ≈ 2)", first, last)
	if last > 0.15 {
		t.Fatalf("reconstruction MSE %.4f too high", last)
	}
	if last >= first*0.2 {
		t.Fatalf("did not train (%.4f → %.4f)", first, last)
	}
	if enc1.W.AtF64(0, 0) == encBefore {
		t.Fatal("encoder weight unchanged — straight-through gradient did not reach the encoder")
	}

	// codebook usage on fresh data.
	_, _, idx, err := forward(backend.NewContext(), sample(512))
	if err != nil {
		t.Fatal(err)
	}
	used := map[int]bool{}
	for _, k := range idx {
		used[k] = true
	}
	t.Logf("codebook usage: %d of %d codes active", len(used), codes)
	if len(used) < 6 {
		t.Fatalf("codebook collapse: only %d of %d codes used", len(used), codes)
	}
}
