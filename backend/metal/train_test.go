//go:build darwin && cgo

package metal_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// blobsF32 builds a deterministic 4-class Gaussian-blob dataset in f32.
func blobsF32(nPer int, seed uint64) (*tensor.Tensor, *tensor.Tensor) {
	rng := rand.New(rand.NewPCG(seed, 0xb10b5))
	centers := [4][2]float64{{2, 2}, {-2, 2}, {2, -2}, {-2, -2}}
	n := 4 * nPer
	xd := make([]float32, n*2)
	yd := make([]float32, n)
	for c := range 4 {
		for s := range nPer {
			i := c*nPer + s
			xd[i*2] = float32(centers[c][0] + rng.NormFloat64()*0.5)
			xd[i*2+1] = float32(centers[c][1] + rng.NormFloat64()*0.5)
			yd[i] = float32(c)
		}
	}
	return tensor.FromFloat32(tensor.Shape{n, 2}, xd), tensor.FromFloat32(tensor.Shape{n}, yd)
}

func trainMLP(t *testing.T, be backend.Backend, x, y *tensor.Tensor, steps int) float64 {
	t.Helper()
	model := nn.NewSequential(
		nn.NewLinear(tensor.F32, 2, 16, 7),
		nn.ReLU(),
		nn.NewLinear(tensor.F32, 16, 4, 8),
	)
	opt := nn.NewAdam(model.Params(), 0.05)
	var last float64
	for range steps {
		tape := autograd.NewTapeOn(be) // forward+backward GEMMs dispatch to `be`
		ctx := tape.Context()
		logits, err := model.Forward(ctx, x)
		if err != nil {
			t.Fatal(err)
		}
		loss, err := nn.CrossEntropy(ctx, logits, y)
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
	return last
}

// §T30: an MLP trained with a metal-backed tape — the Linear GEMMs (forward and
// both backward GEMMs of the matmul VJP) execute on the GPU, other ops fall back
// to cpu (§I4) — must converge, matching the cpu-trained reference. This is
// GPU-accelerated TRAINING validated against the CPU truth.
func TestMetalGPUTraining(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal: no MPS GPU — skipping (§V4)")
	}
	metalB, _ := backend.Get(backend.Metal)
	cpuB, _ := backend.Get(backend.CPU)

	x, y := blobsF32(40, 1)

	gpuLoss := trainMLP(t, metalB, x, y, 400)
	cpuLoss := trainMLP(t, cpuB, x, y, 400)

	if gpuLoss >= 0.15 {
		t.Fatalf("GPU-trained MLP did not converge: final loss %.4f", gpuLoss)
	}
	if math.Abs(gpuLoss-cpuLoss) > 0.1 {
		t.Errorf("GPU vs CPU training diverged: gpu %.4f vs cpu %.4f", gpuLoss, cpuLoss)
	}
	t.Logf("GPU training converged: gpu loss %.4f, cpu loss %.4f", gpuLoss, cpuLoss)
}

// Sanity: the metal backend really serves the matmul kernel used during training
// (so training GEMMs are on-GPU, not silently falling back to cpu).
func TestMetalServesMatmul(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal: no MPS GPU — skipping (§V4)")
	}
	mb, _ := backend.Get(backend.Metal)
	if _, ok := mb.Kernel(backend.OpMatMul, tensor.F32); !ok {
		t.Fatal("metal must serve f32 matmul for GPU training")
	}
}

// §B49: the ADR-0008 cpu-routing gate inside a GPU kernel must strip the tape
// recorder — with it attached the op records TWICE (inner + outer Execute) and
// every routed op's gradient doubles. y = x + x ⇒ dy/dx = 2 exactly; the buggy
// gate produced 4 (caught only by the full sweep's tight trained-model bars).
func TestBinaryRoutingRecordsOnce(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal: no gpu")
	}
	be, ok := backend.Get(backend.Metal)
	if !ok {
		t.Skip("metal not registered")
	}
	x := tensor.FromFloat32(tensor.Shape{4}, []float32{1, 2, 3, 4})
	tape := autograd.NewTapeOn(be)
	out, err := backend.Execute(tape.Context(), backend.OpAdd, []*tensor.Tensor{x, x}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(out[0]); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(x)
	if g == nil {
		t.Fatal("no gradient for x")
	}
	for i := range 4 {
		if got := g.AtF64(i); got != 2 {
			t.Fatalf("grad[%d] = %g, want exactly 2 (doubled recording gives 4)", i, got)
		}
	}
}

// §B49 class audit (§T537): EVERY in-kernel ref-fallback on a recorded forward had the
// same double-recording hazard. Exercise one representative fallback path under the tape —
// ALiBi attention (metal's mha kernel falls back to the reference) — and pin its gradients
// against a tape run directly on the reference backend: identical kernels, so any
// difference means the fallback recorded twice.
func TestFallbackUnderTapeRecordsOnce(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal: no gpu")
	}
	be, ok := backend.Get(backend.Metal)
	if !ok {
		t.Skip("metal not registered")
	}
	mk := func(seed int) *tensor.Tensor {
		x := tensor.New(tensor.F32, tensor.Shape{4, 8})
		for i := range x.Numel() {
			x.SetF64(math.Sin(float64(seed*100+i))*0.5, tensor.Unravel(i, x.Shape())...)
		}
		return x
	}
	attrs := backend.AttnAttrs{Heads: 2, Causal: true, ALiBi: true} // ALiBi → ref fallback
	grads := func(b backend.Backend) []*tensor.Tensor {
		q, k, v := mk(1), mk(2), mk(3)
		tape := autograd.NewTapeOn(b)
		out, err := backend.Execute(tape.Context(), backend.OpMHA, []*tensor.Tensor{q, k, v}, attrs)
		if err != nil {
			t.Fatal(err)
		}
		if err := tape.Backward(out[0]); err != nil {
			t.Fatal(err)
		}
		return []*tensor.Tensor{tape.Grad(q), tape.Grad(k), tape.Grad(v)}
	}
	gm := grads(be)
	gr := grads(backend.Reference())
	for n := range gm {
		if gm[n] == nil || gr[n] == nil {
			t.Fatalf("input %d: missing gradient", n)
		}
		for i := range gr[n].Numel() {
			idx := tensor.Unravel(i, gr[n].Shape())
			if gm[n].AtF64(idx...) != gr[n].AtF64(idx...) {
				t.Fatalf("input %d grad diverges at %v: fallback %g vs ref %g (2× = double-recorded)",
					n, idx, gm[n].AtF64(idx...), gr[n].AtF64(idx...))
			}
		}
	}
}
