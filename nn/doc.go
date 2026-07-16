// Package nn is layer L3: neural-network building blocks.
//
// # For the AI professional
//
// Every layer implements Forward(ctx, x) and exposes its trainable tensors via
// Params(), so a model composes with NewSequential and trains against any
// optimizer. Built on autograd (L2) and backend (L1); it never imports backend
// internals (§I1). The canonical loop: build a Tape, Forward through it, compute
// a loss, Backward, then optimizer.Step(tape.Grad). The catalogue, grouped —
// every algorithm carries its paper citation (§R) in its own doc comment:
//
//   - Core layers & init: Linear, Conv2D/MaxPool2D (NCHW, fused conv backward),
//     Kolmogorov-Arnold (KAN) learnable-spline-edge layers,
//     activations, LayerNorm/RMSNorm/GroupNorm,
//     SwiGLU/GLU, dropout/droppath, spectral & weight norm, QK-norm, DyT,
//     DeepNorm, sinusoidal, T5 relative, and contextual (CoPE, content-gated
//     count) position encodings, Xavier/Kaiming
//     init, and µP (Maximal Update Parametrization) scaling rules.
//   - Losses & objectives: MSE, stable cross-entropy, focal, triplet, InfoNCE,
//     multi-token prediction, Matryoshka representations, ColBERT MaxSim,
//     Plackett-Luce/ListMLE ranking, and the R-Drop / SimCTG regularizers.
//   - Optimizers: SGD, Adam/AdamW, Adafactor, LAMB, Muon, Shampoo, SOAP, Sophia,
//     AdEMAMix, Schedule-Free — plus wrappers composing with any of them
//     (Lookahead, SAM sharpness-aware steps, cautious masking, Grokfast EMA (exponential moving average)
//     filtering, GaLore and APOLLO memory-efficient low-rank gradient projection), LR (learning rate) schedules (OneCycle),
//     gradient clipping and accumulation, and mixed-precision (AMP) helpers.
//   - Parameter-efficient fine-tuning: LoRA, DoRA, PiSSA, VeRA, (IA)³,
//     bottleneck adapters, prefix tuning, prompt tuning, and NEFTune noise.
//   - Quantization, pruning & discrete bottlenecks: AWQ, GPTQ, HQQ, NF4 (QLoRA),
//     LLM.int8, SmoothQuant, LSQ/FSQ, BitNet b1.58 ternary quantization-aware
//     training (BitLinear), SparseGPT and Wanda pruning, VQ-VAE and
//     Gumbel-Softmax.
//   - Architectures beyond the vanilla transformer: MoE routing with load
//     balancing (token-choice, expert-choice, soft MoE, loss-free balancing),
//     Mixture-of-Depths, Mamba selective SSM (+ Mamba-2's state-space duality,
//     SSDRecurrent/SSDQuadratic), RWKV (trainable WKV op, the full
//     RWKVBlock, and O(1)-state recurrent inference via RWKVBlock.Step),
//     RetNet retention, Kimi Delta Attention (per-channel gated delta rule),
//     DeepSeekMoE shared experts, and the sparse-attention trio
//     (MoBAAttention block routing, NSABranches compressed/selected/sliding,
//     DSAAttention lightning-indexer token selection),
//     gated/delta-rule linear attention, DeepSeek MLA, Differential Attention,
//     GatedAttention (sigmoid output gate on softmax attention), the normalized
//     Transformer (NGPTBlock, on-sphere weights), the Forgetting Transformer
//     (FoXBlock, log-cumulative forget bias), Hymba (parallel attention∥SSM
//     hybrid heads), and Mixture-of-Recursions (per-token adaptive recursion
//     depth over a weight-shared block); the softmax-alternative attentions
//     (SigmoidAttention, SelectiveAttention context masking, MultiTokenAttention
//     key-query convolution, StickBreakingAttention, and Aaren attention-as-RNN
//     with O(1)-state streaming), Tokenformer/Pattention (attention over
//     learnable, growable parameter tokens), PEER product-key retrieval over a
//     million single-neuron experts, the extended recurrent cells xLSTM
//     mLSTM (matrix memory), Griffin RG-LRU (real-gated linear recurrence), and
//     HGRN (hierarchically gated RNN, depth-increasing forget floor), and the
//     linear-complexity attention approximations Linformer (low-rank length
//     projection) and Nyströmformer (landmark Nyström approximation).
//   - Diffusion & generative: DDPM/DDIM schedules and samplers, EDM (with
//     preconditioning), and Flow Matching.
//   - Self-supervised learning: Barlow Twins, DINO, SimSiam, SwAV, VICReg.
//   - Alignment & distillation: Bradley-Terry reward models, GRPO, RLOO, RSO,
//     SLiC-HF, and generalized knowledge distillation (GKD) — fed from model
//     logits via TokenLogProbs / SequenceLogProbs, the differentiable bridge
//     every preference/RL loss consumes.
//   - Continual learning & model merging: EWC (elastic weight consolidation), MAS, SI penalties; TIES/DARE
//     merging, SLERP, model soups, and EMA/SWA weight averaging.
//   - Transport & assignment: Sinkhorn entropic optimal transport.
//
// # For the newcomer
//
// This is where you assemble and train a neural network. A "layer" transforms
// numbers; stacking layers (with a [NewSequential]) makes a network. [NewLinear]
// is the basic layer (a weighted sum plus a bias); an activation like [ReLU] adds
// the non-linearity that lets a network learn curves, not just straight lines. A
// "loss" (e.g. [MSE]) measures how wrong the network's output is, and an
// "optimizer" (e.g. [NewAdam]) adjusts the weights to make the loss smaller. The
// training recipe is always the same four steps — run the network forward,
// measure the loss, call Backward to get the gradients (see package autograd),
// and let the optimizer take one step — repeated until the loss is small. The
// runnable examples below show exactly that, from a single layer to a full
// training loop.
//
// Further reading: Goodfellow, Bengio & Courville, "Deep Learning" (MIT Press 2016) as the standard textbook; Ruder 2016, "An Overview of Gradient Descent Optimization Algorithms", for the optimizer family tree; Lialin et al. 2023, "Scaling Down to Scale Up", for a survey of the PEFT methods implemented here.
//
// In plain terms: this package is the training toolbox — the layers models are stacked from, the loss functions that say how wrong a prediction is, and the optimizers that turn gradients into better parameters.
package nn
