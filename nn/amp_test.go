package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// §R42: dynamic loss scaling — clean steps grow the scale after the interval,
// an overflow halves it and signals "skip the step".
func TestDynamicScalerLogic(t *testing.T) {
	s := &nn.LossScaler{Scale: 1024, GrowthFactor: 2, Backoff: 0.5, Growth: 3}
	// 3 clean steps → grow ×2
	for i := range 3 {
		if skip := s.Update(false); skip {
			t.Fatalf("clean step %d must not skip", i)
		}
	}
	if s.Scale != 2048 {
		t.Errorf("after %d clean steps scale = %g, want 2048", 3, s.Scale)
	}
	// overflow → halve + skip
	if skip := s.Update(true); !skip {
		t.Fatal("overflow must signal skip")
	}
	if s.Scale != 1024 {
		t.Errorf("after overflow scale = %g, want 1024", s.Scale)
	}
	// default constructor matches torch/Apex defaults
	d := nn.NewDynamicScaler()
	if d.Scale != 65536 || d.GrowthFactor != 2 || d.Backoff != 0.5 || d.Growth != 2000 {
		t.Errorf("defaults = {%g,%g,%g,%d}, want {65536,2,0.5,2000}", d.Scale, d.GrowthFactor, d.Backoff, d.Growth)
	}
}

// Scale never underflows below 1 no matter how many overflows.
func TestScalerFloorAtOne(t *testing.T) {
	s := &nn.LossScaler{Scale: 2, Backoff: 0.5, GrowthFactor: 2, Growth: 100}
	for range 10 {
		s.Update(true)
	}
	if s.Scale < 1 {
		t.Errorf("scale underflowed to %g", s.Scale)
	}
}

// Micikevicius §3.1: the CORE reason for fp32 master weights. A weight update
// smaller than the fp16 ULP vanishes if applied directly to an fp16 weight, but
// accumulates in the fp32 master. Same tiny step, two accumulation strategies.
func TestMasterWeightsAccumulateTinyUpdates(t *testing.T) {
	// At w0=3.0 (interior of [2,4), symmetric fp16 spacing 2^-9 ≈ 1.95e-3) a step
	// below half the ULP (≈9.77e-4) rounds straight back — updates are lost.
	const w0 = 3.0
	const delta = 5e-4
	const steps = 100

	// (a) direct fp16 accumulation — each add rounds back to w0, weight frozen.
	directF16 := w0
	for range steps {
		h := tensor.FromFloat64(tensor.Shape{1}, []float64{directF16 - delta}).Cast(tensor.F16)
		directF16 = h.AtF64(0)
	}
	if directF16 != w0 {
		t.Fatalf("fp16-direct should stay frozen at %g (updates below half-ULP lost), got %g", w0, directF16)
	}

	// (b) fp32 master accumulation — updates accumulate, then round into fp16.
	master := w0
	for range steps {
		master -= delta
	}
	weightF16 := tensor.FromFloat64(tensor.Shape{1}, []float64{master}).Cast(tensor.F16).AtF64(0)
	if weightF16 == w0 {
		t.Fatal("fp32-master weight should have moved off w0")
	}
	if math.Abs(master-2.95) > 1e-9 {
		t.Errorf("master = %g, want 2.95", master)
	}
	t.Logf("100×%.0e: fp16-direct stayed %.5f, fp32-master → %.5f (weight %.5f)", delta, directF16, master, weightF16)
}

// Micikevicius §3.2: loss scaling rescues gradients that would underflow to zero
// in fp16. Unscaled, the tiny grad is lost; scaled up before fp16 storage then
// unscaled, it survives.
func TestLossScalingPreventsUnderflow(t *testing.T) {
	const g = 2e-8 // < half the smallest fp16 subnormal (2^-24 ≈ 5.96e-8)

	unscaled := tensor.FromFloat64(tensor.Shape{1}, []float64{g}).Cast(tensor.F16).AtF64(0)
	if unscaled != 0 {
		t.Fatalf("grad %.0e should underflow to 0 in fp16, got %g", g, unscaled)
	}

	const scale = 1024.0
	scaledStored := tensor.FromFloat64(tensor.Shape{1}, []float64{g * scale}).Cast(tensor.F16).AtF64(0)
	recovered := scaledStored / scale
	if recovered == 0 {
		t.Fatal("loss-scaled grad still underflowed — scaling failed")
	}
	if rel := math.Abs(recovered-g) / g; rel > 0.01 {
		t.Errorf("recovered grad %.3e vs %.3e (rel %.4f)", recovered, g, rel)
	}
	t.Logf("grad %.0e: unscaled→0, scaled×%.0f→%.4e recovered", g, scale, recovered)
}

// BackwardScaled seeds the loss cotangent with the scale, so every gradient is
// exactly `scale`× the unscaled gradient.
func TestBackwardScaledMultipliesGrads(t *testing.T) {
	x := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, 2, 3, 4})
	w := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{0.5, -1, 2, 0.3})

	gradAt := func(scale float64) *tensor.Tensor {
		tape := autograd.NewTape()
		ctx := tape.Context()
		y, _ := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{x, w}, nil)
		loss, err := nn.MSE(ctx, y[0], tensor.New(tensor.F64, y[0].Shape()))
		if err != nil {
			t.Fatal(err)
		}
		if err := tape.BackwardScaled(loss, scale); err != nil {
			t.Fatal(err)
		}
		return tape.Grad(w)
	}
	g1 := gradAt(1)
	g8 := gradAt(8)
	for i := range g1.Numel() {
		idx := tensor.Unravel(i, g1.Shape())
		if a, b := 8*g1.AtF64(idx...), g8.AtF64(idx...); math.Abs(a-b) > 1e-12*math.Max(1, math.Abs(a)) {
			t.Errorf("grad[%d]: 8×g1=%.12g, g8=%.12g", i, a, b)
		}
	}
}

// End-to-end: an fp16-compute linear model with fp32 masters + dynamic loss
// scaling + Adam over the masters trains MSE down toward zero (§V1 recipe works).
func TestMixedPrecisionTrainsRegression(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F16, tensor.BF16} {
		t.Run(dt.String(), func(t *testing.T) {
			W := tensor.FromFloat64(tensor.Shape{4, 3}, []float64{
				0.2, -0.5, 0.1, 0.7, 0.3, -0.2, -0.4, 0.6, 0.9, 0.1, -0.3, 0.5,
			})
			x := tensor.FromFloat64(tensor.Shape{3, 4}, []float64{
				1, -1, 0.5, 2, 0.3, 1, -2, 0.7, -0.5, 0.8, 1.2, -1,
			})
			target := tensor.FromFloat64(tensor.Shape{3, 3}, []float64{
				1, 0, -1, 0.5, 2, -0.5, -1, 1, 0.25,
			})

			mp := nn.NewMixedPrecision([]*tensor.Tensor{W}, dt)
			opt := nn.NewAdam(mp.Masters, 0.05)
			scaler := nn.NewDynamicScaler()
			scaler.Scale = 256 // modest — keeps the tiny problem well-conditioned

			var first, last float64
			for step := range 300 {
				mp.Sync() // fp16/bf16-round masters into the compute weight
				tape := autograd.NewTape()
				ctx := tape.Context()
				y, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{x, W}, nil)
				if err != nil {
					t.Fatal(err)
				}
				loss, err := nn.MSE(ctx, y[0], target)
				if err != nil {
					t.Fatal(err)
				}
				if step == 0 {
					first = loss.AtF64()
				}
				last = loss.AtF64()
				if err := tape.BackwardScaled(loss, scaler.Scale); err != nil {
					t.Fatal(err)
				}
				foundInf := false
				gfn := mp.UnscaleGrads(tape.Grad, scaler.Scale, &foundInf)
				if !foundInf {
					if err := opt.Step(gfn); err != nil {
						t.Fatal(err)
					}
				}
				scaler.Update(foundInf)
			}
			if last >= 0.15*first {
				t.Fatalf("%v mixed-precision did not converge: MSE %.4f → %.4f", dt, first, last)
			}
			t.Logf("%v AMP training: MSE %.4f → %.4f", dt, first, last)
		})
	}
}

func benchmarkAMP(b *testing.B, sync bool) {
	W := tensor.New(tensor.F32, tensor.Shape{512, 512})
	for i := range W.Storage().F32() {
		W.Storage().F32()[i] = float32(0.01 * float64(i%97-48))
	}
	mp := nn.NewMixedPrecision([]*tensor.Tensor{W}, tensor.F16)
	g := tensor.New(tensor.F32, tensor.Shape{512, 512})
	for i := range g.Storage().F32() {
		g.Storage().F32()[i] = float32(0.001 * float64(i%53-26))
	}
	gf := func(*tensor.Tensor) *tensor.Tensor { return g }
	var found bool
	b.ReportAllocs()
	for b.Loop() {
		if sync {
			mp.Sync()
		} else {
			mp.UnscaleGrads(gf, 65536, &found)(mp.Masters[0])
		}
	}
}
func BenchmarkAMPSync(b *testing.B)         { benchmarkAMP(b, true) }
func BenchmarkAMPUnscaleGrads(b *testing.B) { benchmarkAMP(b, false) }
