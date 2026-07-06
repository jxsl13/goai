//go:build vulkan && cgo

package vulkan_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/vulkan"
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

// §T43: an MLP trained with a vulkan-backed tape — the Linear GEMMs (forward and
// both backward GEMMs of the matmul VJP) execute on the GPU via the compute
// shader, other ops fall back to cpu (§I4) — must converge, matching the
// cpu-trained reference. Portable GPU-accelerated TRAINING (user priority:
// accel for training AND inference) validated against the CPU truth.
func TestVulkanGPUTraining(t *testing.T) {
	if !vulkan.Available() {
		t.Skip("vulkan: no compute device — skipping (§V4)")
	}
	vulkanB, _ := backend.Get("vulkan")
	cpuB, _ := backend.Get("cpu")

	x, y := blobsF32(40, 1)

	gpuLoss := trainMLP(t, vulkanB, x, y, 400)
	cpuLoss := trainMLP(t, cpuB, x, y, 400)

	if gpuLoss >= 0.15 {
		t.Fatalf("GPU-trained MLP did not converge: final loss %.4f", gpuLoss)
	}
	if math.Abs(gpuLoss-cpuLoss) > 0.1 {
		t.Errorf("GPU vs CPU training diverged: gpu %.4f vs cpu %.4f", gpuLoss, cpuLoss)
	}
	t.Logf("GPU training converged: gpu loss %.4f, cpu loss %.4f", gpuLoss, cpuLoss)
}

// Sanity: the vulkan backend really serves the matmul kernel used during training
// (so training GEMMs are on-GPU, not silently falling back to cpu).
func TestVulkanServesMatmul(t *testing.T) {
	if !vulkan.Available() {
		t.Skip("vulkan: no compute device — skipping (§V4)")
	}
	vb, _ := backend.Get("vulkan")
	if _, ok := vb.Kernel(backend.OpMatMul, tensor.F32); !ok {
		t.Fatal("vulkan must serve f32 matmul for GPU training")
	}
}
