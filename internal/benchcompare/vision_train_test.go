//go:build darwin && cgo && vulkan

// Vision end-to-end throughput benchmarks (§T884) — the computer-vision companion
// to the GPT training benchmarks above (§T883). A ViT and a CNN classifier are
// timed forward-only and for a full TRAINING STEP (forward + CrossEntropy +
// backward, NO optimizer update — the exact shape of BenchmarkGPTTrainingStep) on
// cpu / Metal / Vulkan, so GoAI's vision path can be compared head-to-head with
// torch-cpu and torch-mps (testdata/bench_vision_torch.py, BENCHMARKS.md).
//
// Geometry is PINNED to match the torch companion exactly (same layer dims/counts
// and param shapes — the fairness anchor). The reference backend is omitted (far
// too slow for a real model, as with the GPT benchmarks); cpu is the Pure-Go
// baseline. Throughput is img/s = batch·N / elapsed: one step processes the whole
// batch of `visBatch` images, so this is the images-classified-per-second a caller
// selecting that backend gets. Rows read Benchmark<Model><Phase>/<backend>.
//
// NOTE on ViT batching: vision.ViT.Forward loops over the batch INTERNALLY
// (slice → per-image encode → concat), i.e. it runs `visBatch` separate
// length-(N+1) sequences rather than one batched attention. The torch companion
// batches them ([B,N+1,D] in one pass). Both classify the same `visBatch` images
// per step — the img/s unit is identical — so the comparison is honest; torch's
// batched attention is simply the throughput a batched implementation reaches, and
// exposing that gap is the point. The CNN path is natively batched on both sides.
package benchcompare

import (
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
	"github.com/jxsl13/goai/vision"
)

// Pinned toy geometry, shared verbatim with testdata/bench_vision_torch.py (§T884).
// ViT: 807,306 params; CNN: 1,562 params (cross-check against Params() len·shapes).
const (
	visBatch   = 8  // images per step (the throughput unit)
	visChans   = 3  // input channels (RGB)
	visSize    = 32 // square image edge
	visClasses = 10 // classifier outputs

	vitPatch = 4   // 4×4 patches → (32/4)² = 64 patch tokens (+1 class token)
	vitDim   = 128 // encoder width D
	vitDepth = 4   // encoder blocks
	vitHeads = 4   // attention heads (D/heads = 32 per head)
)

// visionInputs builds the deterministic f32 image batch [batch,C,H,W] and the
// cycling class-index targets [batch] shared by every vision benchmark — the same
// deterministic-random pattern the GPT benchmarks use (bench.RandF32 + cycling
// targets), so runs are reproducible across backends and machines.
func visionInputs() (x, targets *tensor.Tensor) {
	x = bench.RandF32(tensor.Shape{visBatch, visChans, visSize, visSize}, 1)
	targets = tensor.New(tensor.F32, tensor.Shape{visBatch})
	for i := range visBatch {
		targets.SetF64(float64(i%visClasses), i)
	}
	return x, targets
}

// newViT builds the pinned ViT (fatal on error — geometry is a compile-time constant).
func newViT(b *testing.B) *vision.ViT {
	m, err := vision.NewViT(visChans, visSize, visClasses, 1,
		vision.WithViTPatch(vitPatch), vision.WithViTDim(vitDim),
		vision.WithViTDepth(vitDepth), vision.WithViTHeads(vitHeads))
	if err != nil {
		b.Fatal(err)
	}
	return m
}

// newCNN builds the pinned CNN: the default 2-stage [8,16] conv stack, kernel 3,
// pad 1, at 32×32×3 → 10 (two ×2 pools → 8×8 → global-avg-pool → head). Channels
// are pinned explicitly so the geometry is fixed independent of the library default.
func newCNN(b *testing.B) *vision.CNN {
	m, err := vision.NewCNN(visChans, visClasses, 1, vision.WithChannels(8, 16), vision.WithKernel(3))
	if err != nil {
		b.Fatal(err)
	}
	return m
}

// BenchmarkViTForward times a batched ViT inference (no tape) per backend.
func BenchmarkViTForward(b *testing.B) {
	m := newViT(b)
	x, _ := visionInputs()
	for _, name := range gptBackends {
		be, ok := backend.Get(name)
		if !ok {
			continue
		}
		ctx := backend.NewContext().WithBackend(be)
		if _, err := m.Forward(ctx, x); err != nil { // warm-up + shape check
			b.Fatal(err)
		}
		b.Run(string(name), func(b *testing.B) {
			for range b.N {
				if _, err := m.Forward(ctx, x); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(visBatch)*float64(b.N)/b.Elapsed().Seconds(), "img/s")
		})
	}
}

// BenchmarkViTTrainStep times a full ViT training step (forward + CrossEntropy +
// backward, no optimizer update) per backend — the exact structure of
// BenchmarkGPTTrainingStep, under autograd.NewTapeOn(be).
func BenchmarkViTTrainStep(b *testing.B) {
	m := newViT(b)
	x, targets := visionInputs()
	step := func(be backend.Backend) {
		tape := autograd.NewTapeOn(be)
		ctx := tape.Context()
		logits, err := m.Forward(ctx, x)
		if err != nil {
			b.Fatal(err)
		}
		loss, err := nn.CrossEntropy(ctx, logits, targets)
		if err != nil {
			b.Fatal(err)
		}
		if err := tape.Backward(loss); err != nil {
			b.Fatal(err)
		}
	}
	for _, name := range gptBackends {
		be, ok := backend.Get(name)
		if !ok {
			continue
		}
		step(be) // warm-up
		b.Run(string(name), func(b *testing.B) {
			for range b.N {
				step(be)
			}
			b.ReportMetric(float64(visBatch)*float64(b.N)/b.Elapsed().Seconds(), "img/s")
		})
	}
}

// BenchmarkCNNForward times a batched CNN inference (no tape) per backend.
func BenchmarkCNNForward(b *testing.B) {
	m := newCNN(b)
	x, _ := visionInputs()
	for _, name := range gptBackends {
		be, ok := backend.Get(name)
		if !ok {
			continue
		}
		ctx := backend.NewContext().WithBackend(be)
		if _, err := m.Forward(ctx, x); err != nil { // warm-up + shape check
			b.Fatal(err)
		}
		b.Run(string(name), func(b *testing.B) {
			for range b.N {
				if _, err := m.Forward(ctx, x); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(visBatch)*float64(b.N)/b.Elapsed().Seconds(), "img/s")
		})
	}
}

// BenchmarkCNNTrainStep times a full CNN training step (forward + CrossEntropy +
// backward, no optimizer update) per backend.
func BenchmarkCNNTrainStep(b *testing.B) {
	m := newCNN(b)
	x, targets := visionInputs()
	step := func(be backend.Backend) {
		tape := autograd.NewTapeOn(be)
		ctx := tape.Context()
		logits, err := m.Forward(ctx, x)
		if err != nil {
			b.Fatal(err)
		}
		loss, err := nn.CrossEntropy(ctx, logits, targets)
		if err != nil {
			b.Fatal(err)
		}
		if err := tape.Backward(loss); err != nil {
			b.Fatal(err)
		}
	}
	for _, name := range gptBackends {
		be, ok := backend.Get(name)
		if !ok {
			continue
		}
		step(be) // warm-up
		b.Run(string(name), func(b *testing.B) {
			for range b.N {
				step(be)
			}
			b.ReportMetric(float64(visBatch)*float64(b.N)/b.Elapsed().Seconds(), "img/s")
		})
	}
}
