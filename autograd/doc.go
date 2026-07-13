// Package autograd is layer L2: reverse-mode automatic differentiation.
//
// # For the AI professional
//
// It provides a tape (Wengert list), Variable, and Backward, intercepting the op
// dispatch exposed by package backend so L1 kernels need no rewrite (§T13). Each
// differentiable op registers a vector-Jacobian-product (VJP (vector-Jacobian product — the per-operation backward rule)) rule, validated by
// central finite-difference gradient checking (§V2). A [Tape] records the forward
// ops executed through its Context; Backward walks them in reverse, seeding the
// output cotangent (1 by default, or an arbitrary scale via BackwardScaled — the
// loss-scaling hook for mixed-precision, §T41) and accumulating gradients at
// fan-out points. Gradients are plain tensors retrieved with Grad, ready for any
// optimizer in package nn.
//
// VJP coverage spans the full op surface: elementwise/broadcast/reduce/reshape,
// einsum, the fused transformer ops (attention, norms, RoPE, cross-entropy),
// linear-algebra decompositions (Cholesky, QR, SVD, eigh, logdet, SPD solves),
// SSM/retention/MoE/MLA blocks, and the preference-alignment losses (DPO, IPO,
// KTO, SimPO, CPO, PPO/GRPO, distillation). [Checkpoint] provides gradient
// checkpointing: a subgraph runs forward without recording and is recomputed
// during Backward, trading compute for activation memory.
//
// # For the newcomer
//
// Training a neural network means nudging its numbers (weights) in the direction
// that reduces its error. "Autograd" is what computes that direction — the
// gradient — automatically. You run your computation as usual through a [Tape];
// the tape quietly remembers every step. Then you call [Tape.Backward] on the
// final number (the loss), and it works backwards through those remembered steps
// to tell you how much each input contributed — that is the gradient. You then
// subtract a little of each gradient from each weight (a "gradient-descent step")
// and the network gets a bit better. You never write the calculus yourself; the
// tape does it. See the runnable examples below.
//
// Further reading: Baydin et al. 2018, "Automatic Differentiation in Machine Learning: a Survey" (JMLR), and Griewank & Walther, "Evaluating Derivatives" (SIAM 2008), the definitive treatments of reverse-mode AD (automatic differentiation) that this tape implements.
//
// In plain terms: training needs to know how much each parameter contributed to the error; this package records every operation on a tape while the model runs forward, then replays the tape backwards to compute exactly those contributions (gradients) automatically.
package autograd
