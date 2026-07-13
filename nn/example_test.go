package nn_test

import (
	"fmt"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// --- Level 1: trivial ---------------------------------------------------------

// A single linear layer: 3 inputs → 2 outputs. Forward runs the layer; the output
// has one row per input row and one column per output unit.
func ExampleLinear() {
	layer := nn.NewLinear(tensor.F64, 3, 2, 42) // in=3, out=2, seed=42
	x := tensor.FromFloat64(tensor.Shape{1, 3}, []float64{1, 2, 3})
	y, _ := layer.Forward(backend.NewContext(), x)
	fmt.Println(y.Shape())
	// Output: (1, 2)
}

// --- Level 2: realistic use-case ---------------------------------------------

// A small multi-layer perceptron (MLP): Linear → ReLU → Linear, composed with
// NewSequential. Two input rows in, two rows of 3-class scores out.
func ExampleSequential() {
	model := nn.NewSequential(
		nn.NewLinear(tensor.F64, 4, 8, 1),
		nn.ReLU(),
		nn.NewLinear(tensor.F64, 8, 3, 2),
	)
	x := tensor.FromFloat64(tensor.Shape{2, 4}, []float64{
		1, 0, -1, 2,
		0.5, 1, -2, 0.3,
	})
	y, _ := model.Forward(backend.NewContext(), x)
	fmt.Println(y.Shape())
	// Output: (2, 3)
}

// --- Level 3: embedded — a complete training loop ----------------------------

// The whole picture: fit a linear model to the rule y = 2·a + 3·b by training.
// Each step runs the four-part recipe — forward, loss, backward, optimizer step —
// on an autograd Tape. Because the target is linear, the model can fit it almost
// exactly, so the loss collapses toward zero. This same loop trains every model
// in GoAI; bigger networks just swap in more layers.
func Example_trainingLoop() {
	model := nn.NewLinear(tensor.F64, 2, 1, 7)
	opt := nn.NewAdam(model.Params(), 0.1)

	x := tensor.FromFloat64(tensor.Shape{4, 2}, []float64{1, 0, 0, 1, 1, 1, 2, 1})
	target := tensor.FromFloat64(tensor.Shape{4, 1}, []float64{2, 3, 5, 7}) // y = 2a + 3b

	var first, last float64
	for step := range 300 {
		tape := autograd.NewTape()
		ctx := tape.Context()
		pred, _ := model.Forward(ctx, x)
		loss, _ := nn.MSE(ctx, pred, target)
		if step == 0 {
			first = loss.AtF64()
		}
		last = loss.AtF64()
		_ = tape.Backward(loss)
		_ = opt.Step(tape.Grad)
	}
	fmt.Println("converged:", last < first*0.001)
	// Output: converged: true
}

// DPO (Direct Preference Optimization) aligns a model to preferences directly
// from (chosen, rejected) pairs — no reward model, no RL. Inputs are per-example
// sequence log-probabilities for the policy and the frozen reference on each
// response. Here the policy already prefers the chosen response over the rejected
// one by the same margin as the reference, so the loss is exactly -log σ(0) =
// log 2 ≈ 0.6931; training pushes it toward zero (see TestDPOTrainsPreference).
func ExampleDPO() {
	policyChosen := tensor.FromFloat64(tensor.Shape{2}, []float64{-2.0, -1.0})
	policyRejected := tensor.FromFloat64(tensor.Shape{2}, []float64{-3.0, -2.0})
	refChosen := tensor.FromFloat64(tensor.Shape{2}, []float64{-2.0, -1.0})
	refRejected := tensor.FromFloat64(tensor.Shape{2}, []float64{-3.0, -2.0})

	loss, _ := nn.DPO(backend.NewContext(), policyChosen, policyRejected, refChosen, refRejected, nn.Beta(0.1))
	fmt.Printf("%.4f\n", loss.AtF64())
	// Output: 0.6931
}

// PPOClipLoss is the PPO clipped surrogate objective (Schulman et al. 2017), the
// core policy loss of RLHF. With the policy unchanged from the rollout policy the
// ratio r = 1 (inside the clip range), so the loss is just −mean(advantage).
// Training minimizes it, raising the log-prob of high-advantage actions while the
// clip keeps each update inside a trust region.
func ExamplePPOClipLoss() {
	logpNew := tensor.FromFloat64(tensor.Shape{2}, []float64{-1.0, -2.0})
	logpOld := tensor.FromFloat64(tensor.Shape{2}, []float64{-1.0, -2.0}) // r = 1
	advantage := tensor.FromFloat64(tensor.Shape{2}, []float64{2.0, 1.0})

	loss, _ := nn.PPOClipLoss(backend.NewContext(), logpNew, logpOld, advantage, 0.2)
	fmt.Printf("%.4f\n", loss.AtF64())
	// Output: -1.5000
}

// DistillLoss (knowledge distillation, Hinton et al. 2015) trains a small student
// to mimic a large teacher's softened output. When the student already matches
// the teacher the soft loss is exactly zero (KL of a distribution with itself),
// for any temperature — the target training drives toward.
func ExampleDistillLoss() {
	logits := []float64{2.0, 0.5, -1.0, 0.0, 3.0, 1.0}
	student := tensor.FromFloat64(tensor.Shape{2, 3}, logits)
	teacher := tensor.FromFloat64(tensor.Shape{2, 3}, logits) // student == teacher

	loss, _ := nn.DistillLoss(backend.NewContext(), student, teacher, 4.0)
	fmt.Printf("%.4f\n", loss.AtF64())
	// Output: 0.0000
}

// CrossEntropySmooth adds label smoothing (Szegedy et al. 2016; used in the
// original Transformer with ε=0.1). On a confident-correct prediction it yields a
// HIGHER loss than plain cross-entropy — that penalty on over-confidence is the
// regularization that improves calibration and generalization.
func ExampleCrossEntropySmooth() {
	logits := tensor.FromFloat64(tensor.Shape{1, 3}, []float64{5, 0, 0}) // very confident on class 0
	targets := tensor.FromFloat64(tensor.Shape{1}, []float64{0})

	hard, _ := nn.CrossEntropy(backend.NewContext(), logits, targets)
	smooth, _ := nn.CrossEntropySmooth(backend.NewContext(), logits, targets, 0.1)
	fmt.Println("smoothing penalizes over-confidence:", smooth.AtF64() > hard.AtF64())
	// Output: smoothing penalizes over-confidence: true
}

// Dropout randomly zeroes activations during training (scaling the survivors so
// the expected value is unchanged) to reduce overfitting; at inference it is a
// no-op. Here in eval mode the activations pass through untouched.
func ExampleDropout() {
	x := tensor.FromFloat64(tensor.Shape{4}, []float64{1, 2, 3, 4})
	d := nn.NewDropout(0.5, 1)
	d.Eval() // inference: identity

	out, _ := d.Forward(backend.NewContext(), x)
	fmt.Println(out.AtF64(0), out.AtF64(1), out.AtF64(2), out.AtF64(3))
	// Output: 1 2 3 4
}

// GradAccumulator trains a large effective batch on limited memory: accumulate
// each microbatch's gradients, then take one optimizer step on their average.
// Here two microbatches contribute gradients [2,4] and [6,8]; the accumulator
// returns their average [4,6] — identical to one step on the combined batch.
func ExampleGradAccumulator() {
	w := tensor.FromFloat64(tensor.Shape{2}, []float64{0, 0})
	acc := nn.NewGradAccumulator([]*tensor.Tensor{w})

	acc.Add(func(*tensor.Tensor) *tensor.Tensor { return tensor.FromFloat64(tensor.Shape{2}, []float64{2, 4}) })
	acc.Add(func(*tensor.Tensor) *tensor.Tensor { return tensor.FromFloat64(tensor.Shape{2}, []float64{6, 8}) })

	g := acc.GradFn()(w)
	fmt.Println(g.AtF64(0), g.AtF64(1))
	// Output: 4 6
}

// IPO (Identity Preference Optimization, Azar et al. 2023) is a DPO variant with
// a squared loss toward a FINITE target margin 1/(2β), avoiding DPO's push toward
// an infinite gap. When the policy already sits at the target margin the loss is
// zero. Here β=0.5 → target 1, and the log-ratio difference is exactly 1.
func ExampleIPO() {
	pc := tensor.FromFloat64(tensor.Shape{1}, []float64{-1.0})
	pl := tensor.FromFloat64(tensor.Shape{1}, []float64{-2.0})
	rc := tensor.FromFloat64(tensor.Shape{1}, []float64{-1.0})
	rl := tensor.FromFloat64(tensor.Shape{1}, []float64{-1.0})

	loss, _ := nn.IPO(backend.NewContext(), pc, pl, rc, rl, nn.Beta(0.5))
	fmt.Printf("%.4f\n", loss.AtF64())
	// Output: 0.0000
}

// KTO (Kahneman-Tversky Optimization, Ethayarajh et al. 2024) aligns from UNPAIRED
// binary feedback — each example is just labeled desirable (1) or undesirable (0),
// no chosen/rejected pairs. With the policy equal to the reference (reward 0) and
// reference point 0, both examples sit at the σ(0)=0.5 decision point; training
// then raises desirable rewards and lowers undesirable ones.
func ExampleKTO() {
	pl := tensor.FromFloat64(tensor.Shape{2}, []float64{-1.0, -1.0})
	rl := tensor.FromFloat64(tensor.Shape{2}, []float64{-1.0, -1.0})
	labels := tensor.FromFloat64(tensor.Shape{2}, []float64{1, 0})

	loss, _ := nn.KTO(backend.NewContext(), pl, rl, labels, nn.Beta(0.1))
	fmt.Printf("%.4f\n", loss.AtF64())
	// Output: 0.5000
}

// TopKGating is the Mixture-of-Experts router: it sends a token to its top-k
// experts with softmax-over-the-selected gate weights. Here expert 1 (logit 3)
// and expert 3 (logit 2) win; their weights are softmax over just those two
// logits, so expert 1 gets exp(3)/(exp(3)+exp(2)) ≈ 0.73.
func ExampleTopKGating() {
	experts, weights := nn.TopKGating([]float64{1, 3, 0.5, 2}, 2)
	fmt.Printf("experts=%v weights=%.2f,%.2f\n", experts, weights[0], weights[1])
	// Output: experts=[1 3] weights=0.73,0.27
}

// Lion (EvoLved Sign Momentum, Chen et al. 2023) is a memory-efficient optimizer
// that keeps a SINGLE momentum buffer (half of Adam's state) and moves every
// parameter by the SAME step size lr, because the update direction is sign(c) ∈
// {-1,0,+1}. The magnitude of the gradient sets only the direction, never the
// step length — this is what makes Lion robust to gradient scale but sensitive to
// the learning rate (use ~3-10× smaller lr than Adam).
//
// For AI practitioners: with the momentum buffer starting at 0, the first update
// is c = (1-β1)·g, whose sign equals sign(g). So on step one every weight moves by
// exactly ±lr regardless of how large or small its gradient is — shown below.
func ExampleLion() {
	p := tensor.FromFloat64(tensor.Shape{2}, []float64{1.0, -2.0})
	// Wildly different gradient magnitudes: 0.01 and 500.
	grad := func(*tensor.Tensor) *tensor.Tensor {
		return tensor.FromFloat64(tensor.Shape{2}, []float64{0.01, 500})
	}
	opt := nn.NewLion([]*tensor.Tensor{p}, 0.1) // lr = 0.1, no weight decay
	_ = opt.Step(grad)

	// Both parameters move by exactly one lr step in the −sign(g) direction,
	// even though the gradients differ by five orders of magnitude.
	fmt.Printf("%.2f %.2f\n", p.AtF64(0), p.AtF64(1))
	// Output: 0.90 -2.10
}

// LayerNorm normalizes each row to zero mean and unit variance, then scales and
// shifts by learned γ,β. In plain terms it keeps a layer's outputs on a consistent
// scale so deep networks train stably; a [2,4] batch stays [2,4].
func ExampleLayerNorm() {
	ln := nn.NewLayerNorm(tensor.F64, 4)
	x := tensor.FromFloat64(tensor.Shape{2, 4}, []float64{1, 2, 3, 4, 4, 3, 2, 1})
	y, _ := ln.Forward(backend.NewContext(), x)
	fmt.Println(y.Shape())
	// Output: (2, 4)
}

// RMSNorm rescales each row by its root-mean-square (no mean subtraction) times a
// learned γ — the lighter normalization Llama uses. Like LayerNorm it keeps
// activations well-scaled; shape is preserved.
func ExampleRMSNorm() {
	rn := nn.NewRMSNorm(tensor.F64, 4)
	x := tensor.FromFloat64(tensor.Shape{2, 4}, []float64{1, 2, 3, 4, 4, 3, 2, 1})
	y, _ := rn.Forward(backend.NewContext(), x)
	fmt.Println(y.Shape())
	// Output: (2, 4)
}

// SwiGLU is the Llama feed-forward block: SiLU(x·Wgate) ⊙ (x·Wup), then ·Wdown. It
// is the "thinking" sublayer of a transformer block, expanding to a wider hidden
// size (here 8) and projecting back to the model width (4).
func ExampleSwiGLU() {
	ff := nn.NewSwiGLU(tensor.F64, 4, 8, 1)
	x := tensor.FromFloat64(tensor.Shape{2, 4}, []float64{1, 2, 3, 4, 4, 3, 2, 1})
	y, _ := ff.Forward(backend.NewContext(), x)
	fmt.Println(y.Shape())
	// Output: (2, 4)
}

// SGD is plain stochastic gradient descent: each Step nudges every parameter a
// little way down its gradient. One step on p=[1,2] with gradient [0.5,0.5] and
// learning rate 0.1 gives [0.95, 1.95].
func ExampleSGD() {
	p := tensor.FromFloat64(tensor.Shape{2}, []float64{1, 2})
	opt := nn.NewSGD([]*tensor.Tensor{p}, 0.1, 0)
	_ = opt.Step(func(*tensor.Tensor) *tensor.Tensor {
		return tensor.FromFloat64(tensor.Shape{2}, []float64{0.5, 0.5})
	})
	fmt.Println(p.AtF64(0), p.AtF64(1))
	// Output: 0.95 1.95
}

// LoRALinear fine-tunes a frozen weight with a small low-rank adapter (A·B). B
// starts at zero, so before any training the layer reproduces the base output x·W
// exactly — new behaviour is learned on top without disturbing the pretrained
// weights. Here x·W = [1·1−1·5, 1·2−1·6] = [−4, −4].
func ExampleLoRALinear() {
	w := tensor.FromFloat64(tensor.Shape{3, 2}, []float64{1, 2, 3, 4, 5, 6})
	lora, _ := nn.NewLoRA(w, 2, 4, 1)
	x := tensor.FromFloat64(tensor.Shape{1, 3}, []float64{1, 0, -1})
	y, _ := lora.Forward(backend.NewContext(), x)
	fmt.Printf("%.1f %.1f\n", y.AtF64(0, 0), y.AtF64(0, 1))
	// Output: -4.0 -4.0
}

// QLoRALinear stores the frozen base weight in 4-bit NF4 and trains only a small
// LoRA adapter, so a large layer can be fine-tuned in a fraction of the memory.
// The forward dequantizes the base on the fly and adds the adapter (a no-op at
// init); a [1,4] input maps to [1,2].
func ExampleQLoRALinear() {
	w := tensor.FromFloat64(tensor.Shape{4, 2}, []float64{0.5, -0.5, 1, -1, 0.25, -0.25, 0.75, -0.75})
	q, _ := nn.NewQLoRA(w, 8, 2, 4, 1)
	x := tensor.FromFloat64(tensor.Shape{1, 4}, []float64{1, 1, 1, 1})
	y, _ := q.Forward(backend.NewContext(), x)
	fmt.Println(y.Shape())
	// Output: (1, 2)
}

// Conv2D convolves NCHW images: one 3×3 kernel with padding 1 preserves the
// spatial size, so a [1,1,8,8] image maps to [1,4,8,8] with 4 output channels.
func ExampleConv2D() {
	conv, _ := nn.NewConv2D(tensor.F64, 1, 4, 3, 1, 1, 42)
	x := tensor.New(tensor.F64, tensor.Shape{1, 1, 8, 8})
	y, _ := conv.Forward(backend.NewContext(), x)
	fmt.Println(y.Shape())
	// Output: (1, 4, 8, 8)
}

// MaxPool2D downsamples each 2×2 window to its maximum — the standard CNN
// downsampler between convolution stages.
func ExampleMaxPool2D() {
	pool := &nn.MaxPool2D{Kernel: 2}
	x := tensor.FromFloat64(tensor.Shape{1, 1, 2, 2}, []float64{1, 5, 3, 2})
	y, _ := pool.Forward(backend.NewContext(), x)
	fmt.Println(y.Shape(), y.AtF64(0, 0, 0, 0))
	// Output: (1, 1, 1, 1) 5
}
