package nn_test

import (
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// benchTrainStep measures one full training step — tape-recorded forward,
// CrossEntropy loss, Backward, optimizer Step — through the pure-Go
// orchestration path (tape/dispatch/VJP/optimizer). Small MLP so Go-level
// overhead (allocations, per-element accessor dispatch, tape bookkeeping) is
// visible next to the kernels.
func benchTrainStep(b *testing.B, dt tensor.Dtype, newOpt func([]*tensor.Tensor) nn.Optimizer) {
	const (
		batch   = 64
		inDim   = 32
		hidden  = 64
		classes = 10
	)
	rng := rand.New(rand.NewPCG(42, 0xbe7c4))
	xd := make([]float64, batch*inDim)
	for i := range xd {
		xd[i] = rng.NormFloat64()
	}
	yd := make([]float64, batch)
	for i := range yd {
		yd[i] = float64(rng.IntN(classes))
	}
	x := tensor.FromFloat64(tensor.Shape{batch, inDim}, xd).Cast(dt)
	y := tensor.FromFloat64(tensor.Shape{batch}, yd).Cast(dt)

	model := nn.NewSequential(
		nn.NewLinear(dt, inDim, hidden, 7),
		nn.ReLU(),
		nn.NewLinear(dt, hidden, classes, 8),
	)
	opt := newOpt(model.Params())

	b.ReportAllocs()
	for b.Loop() {
		tape := autograd.NewTape()
		ctx := tape.Context()
		logits, err := model.Forward(ctx, x)
		if err != nil {
			b.Fatal(err)
		}
		loss, err := nn.CrossEntropy(ctx, logits, y)
		if err != nil {
			b.Fatal(err)
		}
		if err := tape.Backward(loss); err != nil {
			b.Fatal(err)
		}
		if err := opt.Step(tape.Grad); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTrainStepAdamF64(b *testing.B) {
	benchTrainStep(b, tensor.F64, func(p []*tensor.Tensor) nn.Optimizer { return nn.NewAdam(p, 1e-3) })
}

func BenchmarkTrainStepAdamF32(b *testing.B) {
	benchTrainStep(b, tensor.F32, func(p []*tensor.Tensor) nn.Optimizer { return nn.NewAdam(p, 1e-3) })
}

func BenchmarkTrainStepSGDF64(b *testing.B) {
	benchTrainStep(b, tensor.F64, func(p []*tensor.Tensor) nn.Optimizer { return nn.NewSGD(p, 1e-2, 0.9) })
}

// stepOnlyFixture builds transformer-ish parameters with fixed synthetic
// gradients so optimizer.Step can be timed in isolation (no tape/kernels).
func stepOnlyFixture(dt tensor.Dtype) ([]*tensor.Tensor, nn.GradFn) {
	shapes := []tensor.Shape{{256, 256}, {256}, {256, 512}, {512}}
	params := make([]*tensor.Tensor, len(shapes))
	grads := map[*tensor.Tensor]*tensor.Tensor{}
	for i, s := range shapes {
		params[i] = tensor.New(dt, s)
		g := tensor.New(tensor.F64, s)
		gd := g.Storage().F64()
		for j := range gd {
			gd[j] = float64(j%17) * 1e-3
		}
		grads[params[i]] = g.Cast(dt)
	}
	return params, func(p *tensor.Tensor) *tensor.Tensor { return grads[p] }
}

// benchStepOnly isolates optimizer.Step over ~264k parameters.
func benchStepOnly(b *testing.B, dt tensor.Dtype, newOpt func([]*tensor.Tensor) nn.Optimizer) {
	params, gf := stepOnlyFixture(dt)
	opt := newOpt(params)
	b.ReportAllocs()
	for b.Loop() {
		if err := opt.Step(gf); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkAdamStepOnly(b *testing.B) {
	benchStepOnly(b, tensor.F64, func(p []*tensor.Tensor) nn.Optimizer { return nn.NewAdam(p, 1e-3) })
}

func BenchmarkAdamStepOnlyF32(b *testing.B) {
	benchStepOnly(b, tensor.F32, func(p []*tensor.Tensor) nn.Optimizer { return nn.NewAdam(p, 1e-3) })
}

func BenchmarkSGDStepOnly(b *testing.B) {
	benchStepOnly(b, tensor.F64, func(p []*tensor.Tensor) nn.Optimizer { return nn.NewSGD(p, 1e-2, 0.9) })
}

func BenchmarkLionStepOnly(b *testing.B) {
	benchStepOnly(b, tensor.F64, func(p []*tensor.Tensor) nn.Optimizer { return nn.NewLion(p, 1e-4) })
}

func BenchmarkLAMBStepOnly(b *testing.B) {
	benchStepOnly(b, tensor.F64, func(p []*tensor.Tensor) nn.Optimizer { return nn.NewLAMB(p, 1e-3) })
}

func BenchmarkAdEMAMixStepOnly(b *testing.B) {
	benchStepOnly(b, tensor.F64, func(p []*tensor.Tensor) nn.Optimizer { return nn.NewAdEMAMix(p, 1e-3) })
}

func BenchmarkAdafactorStepOnly(b *testing.B) {
	benchStepOnly(b, tensor.F64, func(p []*tensor.Tensor) nn.Optimizer { return nn.NewAdafactor(p) })
}

// stepOnlyFixture2D is stepOnlyFixture restricted to matrix parameters (Muon
// accepts 2-D only).
func stepOnlyFixture2D(dt tensor.Dtype) ([]*tensor.Tensor, nn.GradFn) {
	shapes := []tensor.Shape{{256, 256}, {256, 512}}
	params := make([]*tensor.Tensor, len(shapes))
	grads := map[*tensor.Tensor]*tensor.Tensor{}
	for i, s := range shapes {
		params[i] = tensor.New(dt, s)
		g := tensor.New(tensor.F64, s)
		gd := g.Storage().F64()
		for j := range gd {
			gd[j] = float64(j%17) * 1e-3
		}
		grads[params[i]] = g.Cast(dt)
	}
	return params, func(p *tensor.Tensor) *tensor.Tensor { return grads[p] }
}

func BenchmarkMuonStepOnly(b *testing.B) {
	params, gf := stepOnlyFixture2D(tensor.F64)
	opt := nn.NewMuon(params, 0.02)
	b.ReportAllocs()
	for b.Loop() {
		if err := opt.Step(gf); err != nil {
			b.Fatal(err)
		}
	}
}

// stepOnlyFixtureSmall shrinks the matrices so the eigendecomposition-based
// optimizers (SOAP, Shampoo) stay benchmarkable; the element-wise loops are
// still a visible fraction of the step.
func stepOnlyFixtureSmall() ([]*tensor.Tensor, nn.GradFn) {
	shapes := []tensor.Shape{{64, 64}, {64}, {64, 128}, {128}}
	params := make([]*tensor.Tensor, len(shapes))
	grads := map[*tensor.Tensor]*tensor.Tensor{}
	for i, s := range shapes {
		params[i] = tensor.New(tensor.F64, s)
		g := tensor.New(tensor.F64, s)
		gd := g.Storage().F64()
		for j := range gd {
			gd[j] = float64(j%17) * 1e-3
		}
		grads[params[i]] = g
	}
	return params, func(p *tensor.Tensor) *tensor.Tensor { return grads[p] }
}

func BenchmarkSOAPStepOnly(b *testing.B) {
	params, gf := stepOnlyFixtureSmall()
	opt := nn.NewSOAP(params, 1e-3)
	b.ReportAllocs()
	for b.Loop() {
		if err := opt.Step(gf); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkShampooStepOnly(b *testing.B) {
	params, gf := stepOnlyFixtureSmall()
	opt := nn.NewShampoo(params, 1e-3, nn.WithShampooRootEvery(10))
	b.ReportAllocs()
	for b.Loop() {
		if err := opt.Step(gf); err != nil {
			b.Fatal(err)
		}
	}
}

// stepOnlyFixtureVec is stepOnlyFixture with vector (1-D) parameters only —
// for SOAP/Shampoo it isolates their element-wise diagonal path (plain Adam /
// AdaGrad), which is the entire Step for non-matrix parameters.
func stepOnlyFixtureVec() ([]*tensor.Tensor, nn.GradFn) {
	shapes := []tensor.Shape{{65536}, {256}, {131072}, {512}}
	params := make([]*tensor.Tensor, len(shapes))
	grads := map[*tensor.Tensor]*tensor.Tensor{}
	for i, s := range shapes {
		params[i] = tensor.New(tensor.F64, s)
		g := tensor.New(tensor.F64, s)
		gd := g.Storage().F64()
		for j := range gd {
			gd[j] = float64(j%17) * 1e-3
		}
		grads[params[i]] = g
	}
	return params, func(p *tensor.Tensor) *tensor.Tensor { return grads[p] }
}

func BenchmarkSOAPStepOnlyVec(b *testing.B) {
	params, gf := stepOnlyFixtureVec()
	opt := nn.NewSOAP(params, 1e-3)
	b.ReportAllocs()
	for b.Loop() {
		if err := opt.Step(gf); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkShampooStepOnlyVec(b *testing.B) {
	params, gf := stepOnlyFixtureVec()
	opt := nn.NewShampoo(params, 1e-3)
	b.ReportAllocs()
	for b.Loop() {
		if err := opt.Step(gf); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkClipGradNorm times the global-norm pass plus one clipped read of
// every gradient — the per-step cost of gradient clipping.
func BenchmarkClipGradNorm(b *testing.B) {
	params, gf := stepOnlyFixture(tensor.F64)
	b.ReportAllocs()
	for b.Loop() {
		clipped, _ := nn.ClipGradNorm(params, gf, 1e-9) // tiny max-norm → scaling path
		for _, p := range params {
			_ = clipped(p)
		}
	}
}

func BenchmarkSophiaStepOnly(b *testing.B) {
	benchStepOnly(b, tensor.F64, func(p []*tensor.Tensor) nn.Optimizer { return nn.NewSophia(p, 1e-3) })
}
func BenchmarkSophiaStepOnlyF32(b *testing.B) {
	benchStepOnly(b, tensor.F32, func(p []*tensor.Tensor) nn.Optimizer { return nn.NewSophia(p, 1e-3) })
}
func BenchmarkScheduleFreeStepOnly(b *testing.B) {
	benchStepOnly(b, tensor.F64, func(p []*tensor.Tensor) nn.Optimizer { return nn.NewScheduleFreeAdamW(p, 1e-3) })
}
func BenchmarkScheduleFreeStepOnlyF32(b *testing.B) {
	benchStepOnly(b, tensor.F32, func(p []*tensor.Tensor) nn.Optimizer { return nn.NewScheduleFreeAdamW(p, 1e-3) })
}

func BenchmarkCautiousStepOnly(b *testing.B) {
	benchStepOnly(b, tensor.F64, func(p []*tensor.Tensor) nn.Optimizer { return nn.NewCautiousAdamW(p, 1e-3) })
}
func BenchmarkCautiousStepOnlyF32(b *testing.B) {
	benchStepOnly(b, tensor.F32, func(p []*tensor.Tensor) nn.Optimizer { return nn.NewCautiousAdamW(p, 1e-3) })
}

func BenchmarkGrokfastStepOnly(b *testing.B) {
	benchStepOnly(b, tensor.F64, func(p []*tensor.Tensor) nn.Optimizer { return nn.NewGrokfast(nn.NewSGD(p, 1e-2, 0), p) })
}
func BenchmarkGrokfastMAStepOnly(b *testing.B) {
	benchStepOnly(b, tensor.F64, func(p []*tensor.Tensor) nn.Optimizer {
		return nn.NewGrokfastMA(nn.NewSGD(p, 1e-2, 0), p, nn.WithGrokfastMAWindow(8))
	})
}
func BenchmarkLookaheadStepOnly(b *testing.B) {
	benchStepOnly(b, tensor.F64, func(p []*tensor.Tensor) nn.Optimizer {
		return nn.NewLookahead(nn.NewSGD(p, 1e-2, 0), p, nn.WithLookaheadK(1))
	})
}
