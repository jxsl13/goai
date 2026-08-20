package backend

// Op identifies a primitive operation. Kernels are registered per (Op, Dtype)
// and dispatched through Execute. New ops are appended (monotonic); kernels for
// them are added by the reference and accel backends without changing the
// Backend interface (ADR-0003).
type Op int

const (
	OpInvalid Op = iota // zero value: not a valid op

	// elementwise binary
	OpAdd // a + b
	OpSub // a − b
	OpMul // a ⊙ b (Hadamard)
	OpDiv // a / b

	// elementwise unary
	OpNeg      // −x
	OpExp      // eˣ
	OpLog      // natural log
	OpTanh     // hyperbolic tangent
	OpReLU     // max(0, x)
	OpGELU     // exact Gaussian Error Linear Unit
	OpSigmoid  // 1/(1+e⁻ˣ)
	OpSiLU     // x·sigmoid(x) (swish)
	OpSoftplus // log(1+eˣ), stable

	// reductions
	OpSum    // sum over axes
	OpMean   // mean over axes
	OpMax    // max over axes
	OpMin    // min over axes
	OpProd   // product over axes (numpy.prod); grad ∂/∂xᵢ = ∏_{j≠i} xⱼ
	OpArgMax // index of the max along an axis

	// blas-1
	OpDot  // vector dot product (f64 accum)
	OpAXPY // alpha·x + y
	OpNrm2 // Euclidean norm (overflow-safe)

	// blas-3
	OpMatMul // row-major matrix product (f64 accum)

	// nn
	OpAddBias      // x[m,n] + b[n] row-broadcast (general broadcasting: §B18)
	OpCrossEntropy // fused stable log-softmax + NLL, mean over batch (ADR-0007)
	OpSoftmax      // stable max-shift softmax over the LAST axis
	OpLayerNorm    // layer norm over the last axis w/ gamma/beta (torch semantics)

	// cv (NCHW)
	OpConv2D         // cross-correlation, zero-padding, stride; optional bias input
	OpConv2DBackward // conv2d backward: (X,W,dO)→(dX,dW,dBias); dispatched by conv2d's VJP so it runs on the active backend
	OpMaxPool2D      // window max, stride defaults to kernel
	OpAvgPool2D      // window mean, stride defaults to kernel

	// fused attention (heads split/concat/mask internal → one differentiable op)
	OpMHA         // multi-head scaled dot-product attention: (Q,K,V)[seq,dmodel]→[seq,dmodel]
	OpMHABackward // SDPA backward: (Q,K,V,dO)→(dQ,dK,dV); dispatched by mha's VJP so it runs on the active backend
	OpMHAMasked   // attention with a free additive mask input (Q,K,V,mask[sq,sk]); trainable — VJP via OpMHAMaskedBackward (§T507 — tree/grouped attention; §T730)
	OpMHASelect   // attention over two score sources with a per-pair selector (Q1,K1,Q2,K2,V,sel[sq,sk]: 0→source 1, 1→source 2, −Inf→excluded), ONE softmax; inference-only, no VJP (§T512 — Self-Extend merged neighbor/grouped scores)

	OpEmbed // row gather: (table[n,d], idx[m]) → out[m,d]; grad scatter-adds to table

	// llama-family
	OpRMSNorm // x/√(mean(x²)+eps)·γ over last axis (no mean-sub, no bias)
	OpRoPE    // rotary position embedding (rotate_half), position = row index

	// preference alignment / RLHF
	OpDPO     // fused Direct Preference Optimization loss (Rafailov et al. 2023)
	OpPPOClip // fused PPO clipped surrogate policy loss (Schulman et al. 2017)
	OpGRPO    // fused GRPO surrogate + k3 KL penalty (Shao et al. 2024, DeepSeekMath)
	OpGSPO    // fused GSPO sequence-level policy loss (Zheng et al. 2025, Qwen — §T549)

	// knowledge distillation
	OpDistill // fused soft-target distillation loss T²·KL(softmax(zt/T)‖softmax(zs/T))

	OpIPO // fused Identity Preference Optimization loss (Azar et al. 2023)
	OpKTO // fused Kahneman-Tversky Optimization loss (Ethayarajh et al. 2024)

	OpDoRAWeight        // fused DoRA weight decomposition W' = m·V/‖V‖_col (Liu et al. 2024)
	OpSimPO             // fused SimPO reference-free length-normalized preference loss (Meng 2024)
	OpORPO              // fused ORPO SFT + odds-ratio preference loss (Hong et al. 2024)
	OpFlashAttn         // fused FlashAttention-2 online-softmax tiled attention (Dao 2023)
	OpMLA               // fused Multi-head Latent Attention: content + decoupled-RoPE scores (DeepSeek-V2)
	OpSSM               // fused Mamba selective-scan (S6) SSM recurrence (Gu & Dao 2023)
	OpWKV               // RWKV-4 WKV time-mixing recurrence (k,v,w,u — Peng et al. 2023); trainable via the O(T²) reverse VJP (§T516)
	OpConv1D            // causal depthwise 1-D convolution (per-channel, left-padded)
	OpRetention         // fused RetNet retention: (QKᵀ⊙D)V with γ-decay mask (Sun et al. 2023)
	OpRetentionBackward // retention backward (Q,K,V,dO)→(dQ,dK,dV), runs on the active backend

	// mixture of experts
	OpMoEBalance // fused MoE load-balancing auxiliary loss (Fedus et al. 2021)
	OpMoECombine // fused MoE renormalize-and-combine over expert outputs (Mixtral)

	OpZLoss // standalone log-Z softmax regularizer coeff·mean(logsumexp²) (ST-MoE router, §R113)

	OpCPO // fused CPO: reference-free DPO preference + NLL/BC regularizer on chosen (Xu et al. 2024)

	OpIA3 // fused (IA)³ rescale: y[...,j] = x[...,j]·l[j], learned per-feature vector (Liu et al. 2022)

	OpConcat // join N tensors along an axis (numpy.concatenate); grad splits back along the axis

	OpSlice // extract a sub-range x[..,start:end,..] along an axis; grad scatters back (Split builds on it)

	OpReshape // change shape keeping row-major order + element count (numpy.reshape); grad reshapes back

	OpTranspose // rank-2 matrix transpose out[j,i]=in[i,j] (numpy.T, tape-recorded); grad = transpose(g)

	OpSqrt // elementwise √x (numpy.sqrt); grad g/(2√x)
	OpAbs  // elementwise |x| (numpy.abs); grad g·sign(x)
	OpClip // elementwise clamp to [Lo,Hi] (numpy.clip); grad passes through where unclipped

	OpStopGradient // identity forward, ZERO gradient backward (numpy/torch detach); severs the gradient path through x

	OpMaximum // elementwise max(a,b) (numpy.maximum); grad routes to the larger input
	OpMinimum // elementwise min(a,b) (numpy.minimum); grad routes to the smaller input
	OpWhere   // elementwise select cond?a:b (numpy.where); grad routes by the (non-diff) mask

	OpBroadcast // broadcast x to a larger shape (numpy.broadcast_to); grad sums over broadcast axes

	OpCumsum // cumulative sum along an axis (numpy.cumsum); grad is the reverse cumulative sum

	OpEinsum // Einstein-summation contraction (numpy.einsum); grad via the einsum-swap rule

	OpCholesky // Cholesky factor: SPD A → lower L with A=LLᵀ (numpy.linalg.cholesky); grad via Murray 2016

	OpLogDet // log-determinant of an SPD matrix: 2·Σ ln L_ii via Cholesky (numpy.linalg.slogdet); grad = A⁻¹ (Jacobi)

	OpSolveSPD // solve A·X=B for SPD A via Cholesky (numpy.linalg.solve); grads B̄=A⁻¹Ḡ, Ā=−½(B̄Xᵀ+XB̄ᵀ)

	OpQR // reduced Householder QR of m×n (m≥n): A→(Q[m,n],R[n,n]) (numpy.linalg.qr); grad via Seeger 2017 (multi-output)

	OpEigh // symmetric eigendecomposition: A→(w[n] ascending, V[n,n] cols) (numpy.linalg.eigh); grad via Ionescu 2015 (multi-output)

	OpSVD // reduced SVD of m×n (m≥n): A→(U[m,n],s[n] desc,V[n,n]) (numpy.linalg.svd); grad via Townsend 2016 (multi-output)

	OpSoftCap // logit soft-cap y=cap·tanh(x/cap), smoothly bounds to (−cap,cap) (Gemma-2); grad g·(1−(y/cap)²)

	OpRoPEBackward // RoPE backward: g→dq via the inverse (angle→−angle) rotation; dispatched by rope's VJP so it runs on the active backend

	OpRMSNormBackward // RMSNorm backward: (x,gamma,g)→(dx,dgamma); dispatched by rmsnorm's VJP so it runs on the active backend

	OpLayerNormBackward // LayerNorm backward: (x,gamma,g)→(dx,dgamma,dbeta); dispatched by layernorm's VJP so it runs on the active backend

	OpCrossEntropyBackward // cross-entropy backward: (z,targets,g)→dz = g·(softmax(z)−q')/b (+z-loss); dispatched by CE's VJP so it runs on the active backend

	OpEmbedBackward // embedding backward: (table,idx,g)→dtable, scatter-add dtable[idx[i]]+=g[i]; dispatched by embed's VJP so it runs on the active backend

	OpGELUBackward // GELU backward: (x,g)→dx = g·(Φ(x)+x·φ(x)); dispatched by gelu's VJP so it runs on the active backend (§T353)

	OpAddBiasBackward // bias-add backward: (g[rows,n])→dbias[n] = Σ_rows g; dispatched by addbias's VJP so the column-sum runs on the active backend (§T354)

	OpSiLUBackward // SiLU backward: (x,g)→dx = g·σ(x)·(1+x·(1−σ(x))); dispatched by silu's VJP so it runs on the active backend (§T362)

	OpMHAMaskedBackward // masked-attention backward: (Q,K,V,mask,dO)→(dQ,dK,dV,dmask); dispatched by OpMHAMasked's VJP (§T730 — makes T5's relative-position attention trainable)

	OpSoftplusBackward // softplus backward: (x,g)→dx = g·σ(x); dispatched by softplus's VJP so the vectorized exp runs on the active backend

	OpSmoothL1Core         // unscaled branch-free core d²−ReLU(|d|−1)²; grad via OpSmoothL1CoreBackward
	OpSmoothL1CoreBackward // fused core backward (pred,target,g)→(dpred,dtarget)

	numOps
)

var opName = [...]string{
	OpInvalid:              "invalid",
	OpAdd:                  "add",
	OpSub:                  "sub",
	OpMul:                  "mul",
	OpDiv:                  "div",
	OpNeg:                  "neg",
	OpExp:                  "exp",
	OpLog:                  "log",
	OpTanh:                 "tanh",
	OpReLU:                 "relu",
	OpGELU:                 "gelu",
	OpSigmoid:              "sigmoid",
	OpSiLU:                 "silu",
	OpSoftplus:             "softplus",
	OpSum:                  "sum",
	OpMean:                 "mean",
	OpMax:                  "max",
	OpMin:                  "min",
	OpProd:                 "prod",
	OpArgMax:               "argmax",
	OpDot:                  "dot",
	OpAXPY:                 "axpy",
	OpNrm2:                 "nrm2",
	OpMatMul:               "matmul",
	OpAddBias:              "addbias",
	OpCrossEntropy:         "crossentropy",
	OpSoftmax:              "softmax",
	OpLayerNorm:            "layernorm",
	OpConv2D:               "conv2d",
	OpConv2DBackward:       "conv2d_backward",
	OpMaxPool2D:            "maxpool2d",
	OpAvgPool2D:            "avgpool2d",
	OpMHA:                  "mha",
	OpMHABackward:          "mha_backward",
	OpMHAMasked:            "mha_masked",
	OpMHASelect:            "mha_select",
	OpEmbed:                "embed",
	OpRMSNorm:              "rmsnorm",
	OpRoPE:                 "rope",
	OpDPO:                  "dpo",
	OpPPOClip:              "ppoclip",
	OpGRPO:                 "grpo",
	OpGSPO:                 "gspo",
	OpDistill:              "distill",
	OpIPO:                  "ipo",
	OpKTO:                  "kto",
	OpDoRAWeight:           "doraweight",
	OpSimPO:                "simpo",
	OpORPO:                 "orpo",
	OpFlashAttn:            "flashattn",
	OpMLA:                  "mla",
	OpSSM:                  "ssm",
	OpWKV:                  "wkv",
	OpConv1D:               "conv1d",
	OpMoEBalance:           "moebalance",
	OpMoECombine:           "moecombine",
	OpZLoss:                "zloss",
	OpRetention:            "retention",
	OpRetentionBackward:    "retention_backward",
	OpCPO:                  "cpo",
	OpIA3:                  "ia3",
	OpConcat:               "concat",
	OpSlice:                "slice",
	OpReshape:              "reshape",
	OpTranspose:            "transpose",
	OpSqrt:                 "sqrt",
	OpAbs:                  "abs",
	OpSmoothL1Core:         "smoothl1_core",
	OpSmoothL1CoreBackward: "smoothl1_core_backward",
	OpClip:                 "clip",
	OpStopGradient:         "stopgradient",
	OpMaximum:              "maximum",
	OpMinimum:              "minimum",
	OpWhere:                "where",
	OpBroadcast:            "broadcast",
	OpCumsum:               "cumsum",
	OpEinsum:               "einsum",
	OpCholesky:             "cholesky",
	OpLogDet:               "logdet",
	OpSolveSPD:             "solvespd",
	OpQR:                   "qr",
	OpEigh:                 "eigh",
	OpSVD:                  "svd",
	OpSoftCap:              "softcap",
	OpRoPEBackward:         "rope_backward",
	OpRMSNormBackward:      "rmsnorm_backward",
	OpLayerNormBackward:    "layernorm_backward",
	OpCrossEntropyBackward: "crossentropy_backward",
	OpEmbedBackward:        "embed_backward",
	OpGELUBackward:         "gelu_backward",
	OpMHAMaskedBackward:    "mha_masked_backward",
	OpAddBiasBackward:      "addbias_backward",
	OpSiLUBackward:         "silu_backward",
	OpSoftplusBackward:     "softplus_backward",
}

// String implements fmt.Stringer.
func (o Op) String() string {
	if o < 0 || int(o) >= len(opName) {
		return "op(?)"
	}
	return opName[o]
}

// Attrs (the sealed op-parameter interface) and its per-op implementations live
// in attrs.go (ADR-0014).
