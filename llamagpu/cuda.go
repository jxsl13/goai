//go:build cuda && cgo && (linux || windows)

package llamagpu

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// cuda adapter: thin assertions from the backend-agnostic buffer/recorder interfaces to the
// concrete cuda.DeviceF32/cuda.Recorder, the same shape as the metal (§T408) and vulkan (§T409)
// adapters. The one difference is submit semantics: the CUDA recorder enqueues eagerly on the work
// stream (Commit is a no-op, Wait syncs), so asyncEncode stays false — a second recorder must not
// pre-encode onto the shared stream while one is in flight.

type cBuf struct{ *cuda.DeviceF32 }

type cRec struct{ r *cuda.Recorder }

func (c cRec) RMSNorm(x, g, o buffer, rows, dim int, eps float32) error {
	return c.r.RMSNorm(cb(x), cb(g), cb(o), rows, dim, eps)
}
func (c cRec) LayerNorm(x, g, b, o buffer, rows, dim int, eps float32) error {
	return c.r.LayerNorm(cb(x), cb(g), cb(b), cb(o), rows, dim, eps)
}
func (c cRec) AddBias(x, b, o buffer, rows, n int) error {
	return c.r.AddBias(cb(x), cb(b), cb(o), rows, n)
}
func (c cRec) MatMul(a, b, cc buffer, m, k, n int) error {
	return c.r.MatMul(cb(a), cb(b), cb(cc), m, k, n)
}
func (c cRec) MatMulAcc(a, b, cc buffer, m, k, n int) error {
	return c.r.MatMulAcc(cb(a), cb(b), cb(cc), m, k, n)
}
func (c cRec) RoPE(q, inv, o buffer, seq, width, heads, hd, half, pos int, posDiv float32) error {
	return c.r.RoPE(cb(q), cb(inv), cb(o), seq, width, heads, hd, half, pos, posDiv)
}
func (c cRec) RoPEAt(q, inv, o buffer, off, seq, width, heads, hd, half, pos int, posDiv float32) error {
	return c.r.RoPEAt(cb(q), cb(inv), cb(o), off, seq, width, heads, hd, half, pos, posDiv)
}
func (c cRec) RoPEPair(qkv, inv buffer, seq, stride, headsQ, offQ, headsK, offK, hd, half, pos int, posDiv float32) error {
	return c.r.RoPEPair(cb(qkv), cb(inv), seq, stride, headsQ, offQ, headsK, offK, hd, half, pos, posDiv)
}
func (c cRec) RoPEPartialPair(qkv, inv buffer, seq, stride, headsQ, offQ, headsK, offK, hd, rotaryDim, pos int, posDiv float32) error {
	return c.r.RoPEPartialPair(cb(qkv), cb(inv), seq, stride, headsQ, offQ, headsK, offK, hd, rotaryDim, pos, posDiv)
}
func (c cRec) Blit(src buffer, srcOff int, dst buffer, dstOff, n int) error {
	return c.r.Blit(cb(src), srcOff, cb(dst), dstOff, n)
}
func (c cRec) Copy2D(src buffer, srcOff, srcStride int, dst buffer, dstOff, dstStride, rows, rowFloats int) error {
	return c.r.Copy2D(cb(src), srcOff, srcStride, cb(dst), dstOff, dstStride, rows, rowFloats)
}
func (c cRec) MHA(q, k, v, o buffer, sq, sk, dm, heads, kvHeads, dk, causal, window int, scale float32) error {
	return c.r.MHA(cb(q), cb(k), cb(v), cb(o), sq, sk, dm, heads, kvHeads, dk, causal, window, scale)
}
func (c cRec) MHACap(q, k, v, o buffer, sq, sk, dm, heads, kvHeads, dk, causal, window int, scale, cap float32) error {
	return c.r.MHACap(cb(q), cb(k), cb(v), cb(o), sq, sk, dm, heads, kvHeads, dk, causal, window, scale, cap)
}
func (c cRec) MHAALiBi(q, k, v, o, slopes buffer, sq, sk, dm, heads, kvHeads, dk, causal, window int, scale float32) error {
	return c.r.MHAALiBi(cb(q), cb(k), cb(v), cb(o), cb(slopes), sq, sk, dm, heads, kvHeads, dk, causal, window, scale)
}
func (c cRec) MoEGate(logits, weights buffer, rows, e, k, raw int, scale float32) error {
	return c.r.MoEGate(cb(logits), cb(weights), rows, e, k, raw, scale)
}
func (c cRec) RowAxpy(dst, src, arow buffer, rows, cols int) error {
	return c.r.RowAxpy(cb(dst), cb(src), cb(arow), rows, cols)
}
func (c cRec) MHARect(q, k, v, o buffer, sq, sk, heads, kvHeads, dqk, dv, causal, window int, scale float32) error {
	return c.r.MHARect(cb(q), cb(k), cb(v), cb(o), sq, sk, heads, kvHeads, dqk, dv, causal, window, scale)
}
func (c cRec) SSMStep(u, delta, a, b, cc, dskip, h, y buffer, d, n int) error {
	return c.r.SSMStep(cb(u), cb(delta), cb(a), cb(b), cb(cc), cb(dskip), cb(h), cb(y), d, n)
}
func (c cRec) Conv1DStep(x, w, b, state, out buffer, d, k int) error {
	return c.r.Conv1DStep(cb(x), cb(w), cb(b), cb(state), cb(out), d, k)
}
func (c cRec) SSDStep(x, delta, a, b, cc, dskip, state, y buffer, heads, headDim, groups, n int) error {
	return c.r.SSDStep(cb(x), cb(delta), cb(a), cb(b), cb(cc), cb(dskip), cb(state), cb(y), heads, headDim, groups, n)
}
func (c cRec) Unary(x, o buffer, op int) error { return c.r.Unary(cb(x), cb(o), op) }
func (c cRec) Binary(a, b, o buffer, op int) error {
	return c.r.Binary(cb(a), cb(b), cb(o), op)
}
func (c cRec) QMatMulResident(x buffer, w qweight, o buffer, m int) error {
	return c.r.QMatMulResident(cb(x), w.(*cuda.ResidentBQ8), cb(o), m)
}
func (c cRec) Commit() error { return c.r.Commit() }
func (c cRec) Wait() error   { return c.r.Wait() }
func (c cRec) Finish() error { return c.r.Finish() }
func (c cRec) Free()         { c.r.Free() }

func cb(b buffer) *cuda.DeviceF32 { return b.(cBuf).DeviceF32 }

// NewCUDA uploads an nlp.Llama's weights into CUDA device buffers and prepares a KV cache up to
// m.Config.Ctx tokens — the NVIDIA variant of New/NewVulkan. It reuses the backend-agnostic batched
// Decoder core (§T409) via the cuda.Recorder, so Generate/Step/StepN and every sampler work
// unchanged. f32 path (quantized NewQuantCUDA is a follow-up: it needs cuda.Recorder.QMatMulResident).
func NewCUDA(m *nlp.Llama) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newDecoder(m, backendOps{
		name:        string(backend.CUDA),
		asyncEncode: false, // one shared work stream — pre-encoding a second recorder would interleave
		newBuffer: func(data []float32) (buffer, error) {
			b, err := cuda.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return cBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := cuda.NewRecorder()
			if err != nil {
				return nil, err
			}
			return cRec{r}, nil
		},
	})
}

// NewQwen2CUDA uploads a Qwen2 / Qwen2.5 model onto the batched Decoder core. Qwen2 shares
// nlp.Llama (SwiGLU MLP, RMSNorm, GQA, full rope) and departs only in carrying q/k/v projection
// biases (o_proj has none) — the newDecoder core adds them via qkvBias when b.Bq/Bk/Bv are set, so
// this is exactly NewCUDA with a Qwen2-typed entry point. Load with nlp.LlamaFromHF; the biases are
// picked up automatically from the checkpoint. (Qwen3's per-head QK-norm is a separate follow-up.)
func NewQwen2CUDA(m *nlp.Llama) (*Decoder, error) { return NewCUDA(m) }

// NewQwen3CUDA uploads a Qwen3 model onto the batched Decoder core. Qwen3 shares nlp.Llama and adds
// per-head RMSNorm on Q and K before RoPE (b.QNorm/b.KNorm), which newDecoder wires into recordQKNorm
// — it drops Qwen2's q/k/v projection bias, so this is again NewCUDA with a Qwen3-typed entry point.
// Load with nlp.LlamaFromHF; the QK-norm gains are picked up automatically from the checkpoint.
func NewQwen3CUDA(m *nlp.Llama) (*Decoder, error) { return NewCUDA(m) }

// NewPhi3CUDA uploads a Phi-3 model onto the batched Decoder core. Phi-3 is structurally a plain
// Llama (RMSNorm, SwiGLU, GQA, full rope, no biases) — nlp.Phi3FromHF just unpacks its row-packed
// qkv_proj / gate_up_proj into the standard projections and returns an *nlp.Llama — so this is
// NewCUDA with a Phi-3-typed entry point. Distinct from NewPhiCUDA (the older Phi-1/1.5/2 family,
// which is parallel-residual with LayerNorm and partial rotary).
func NewPhi3CUDA(m *nlp.Llama) (*Decoder, error) { return NewCUDA(m) }

// NewNemotronCUDA uploads an nlp.Nemotron onto the batched Decoder core: sequential two-norm
// residual with LayerNorm1P (mean-centered LayerNorm, γ=w+1/β folded at load), a squared-ReLU
// 2-layer MLP (down(relu²(up(x)))) and partial rotary. The relu² activation is a new cuda unary
// (cu_relu2_f32); everything else is reused plumbing. cuda-only.
func NewNemotronCUDA(m *nlp.Nemotron) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newNemotronDecoder(m, backendOps{
		name:        string(backend.CUDA),
		asyncEncode: false,
		newBuffer: func(data []float32) (buffer, error) {
			b, err := cuda.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return cBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := cuda.NewRecorder()
			if err != nil {
				return nil, err
			}
			return cRec{r}, nil
		},
	})
}

// NewFalconCUDA uploads an nlp.Falcon (falcon-7b class) onto the batched Decoder core: single-norm
// parallel residual, LayerNorm-with-bias, a 2-layer GELU MLP, multi-query attention (one KV head)
// and full rope — every departure a reused generalization. cuda-only.
func NewFalconCUDA(m *nlp.Falcon) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newFalconDecoder(m, backendOps{
		name:        string(backend.CUDA),
		asyncEncode: false,
		newBuffer: func(data []float32) (buffer, error) {
			b, err := cuda.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return cBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := cuda.NewRecorder()
			if err != nil {
				return nil, err
			}
			return cRec{r}, nil
		},
	})
}

// NewOLMo2CUDA uploads an nlp.OLMo2 (Allen AI) onto the batched Decoder core: post-norm blocks
// (each sublayer reads the raw residual, its output normed before the add), a full-width RMSNorm on
// the q/k projections before RoPE, SwiGLU, GQA, full rope and an untied lm_head. cuda-only.
func NewOLMo2CUDA(m *nlp.OLMo2) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newOLMo2Decoder(m, backendOps{
		name:        string(backend.CUDA),
		asyncEncode: false,
		newBuffer: func(data []float32) (buffer, error) {
			b, err := cuda.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return cBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := cuda.NewRecorder()
			if err != nil {
				return nil, err
			}
			return cRec{r}, nil
		},
	})
}

// NewGraniteMoECUDA uploads an nlp.GraniteMoE onto the batched Decoder core: the MoE sibling of
// dense Granite — a plain Llama attention core + sparse Mixture-of-Experts FFN, wrapped in the four
// Granite config scalars (embedding/attention/residual multipliers folded into the upload, logits
// scaling into the lm_head). cuda-only.
func NewGraniteMoECUDA(m *nlp.GraniteMoE) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newGraniteMoEDecoder(m, backendOps{
		name:        string(backend.CUDA),
		asyncEncode: false,
		newBuffer: func(data []float32) (buffer, error) {
			b, err := cuda.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return cBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := cuda.NewRecorder()
			if err != nil {
				return nil, err
			}
			return cRec{r}, nil
		},
	})
}

// NewQwen2MoECUDA uploads an nlp.Qwen2MoE onto the batched Decoder core: a routed sparse MoE PLUS a
// shared expert (a SwiGLU run on every token, sigmoid-gated), with Qwen2 q/k/v projection biases on
// the attention. cuda-only.
func NewQwen2MoECUDA(m *nlp.Qwen2MoE) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newQwen2MoEDecoder(m, backendOps{
		name:        string(backend.CUDA),
		asyncEncode: false,
		newBuffer: func(data []float32) (buffer, error) {
			b, err := cuda.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return cBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := cuda.NewRecorder()
			if err != nil {
				return nil, err
			}
			return cRec{r}, nil
		},
	})
}

// NewOLMoECUDA uploads an nlp.OLMoE (Allen AI sparse-MoE) onto the batched Decoder core: pre-norm
// Llama attention with FULL-WIDTH q/k RMSNorm and a sparse Mixture-of-Experts FFN, untied lm_head.
// cuda-only.
func NewOLMoECUDA(m *nlp.OLMoE) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newOLMoEDecoder(m, backendOps{
		name:        string(backend.CUDA),
		asyncEncode: false,
		newBuffer: func(data []float32) (buffer, error) {
			b, err := cuda.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return cBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := cuda.NewRecorder()
			if err != nil {
				return nil, err
			}
			return cRec{r}, nil
		},
	})
}

// NewDeepSeekV2CUDA uploads a DENSE (MLA-only) nlp.DeepSeekV2 onto the batched Decoder core:
// Multi-head Latent Attention (low-rank latent KV compression, decoupled RoPE, rectangular
// query/key vs value head dims) + SwiGLU FFN. The hardest attention in the catalogue, and the first
// non-fused-QKV decoder. cuda-only (the rectangular MHARect is cuda-only). The sparse-MoE variant is
// a follow-up (dense-only for now).
func NewDeepSeekV2CUDA(m *nlp.DeepSeekV2) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newDeepSeekV2Decoder(m, backendOps{
		name:        string(backend.CUDA),
		asyncEncode: false,
		newBuffer: func(data []float32) (buffer, error) {
			b, err := cuda.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return cBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := cuda.NewRecorder()
			if err != nil {
				return nil, err
			}
			return cRec{r}, nil
		},
	})
}

// NewMambaCUDA uploads an nlp.Mamba onto the batched Decoder core — the FIRST non-transformer GPU
// decoder. Each layer is a selective-state-space mixer (no attention, no KV cache): decode is a
// linear-time recurrence recorded from the per-timestep conv/SSM decode kernels (cu_conv1d_step /
// cu_ssm_step), carrying per-block conv/SSM state across Step calls. cuda-only.
func NewMambaCUDA(m *nlp.Mamba) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newMambaDecoder(m, backendOps{
		name:        string(backend.CUDA),
		asyncEncode: false,
		newBuffer: func(data []float32) (buffer, error) {
			b, err := cuda.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return cBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := cuda.NewRecorder()
			if err != nil {
				return nil, err
			}
			return cRec{r}, nil
		},
	})
}

// NewMamba2CUDA uploads an nlp.Mamba2 onto the batched Decoder core — the state-space-duality sibling
// of Mamba. Each layer is an SSD mixer (scalar per-head decay, B/C shared across a group, gated
// RMSNorm); decode is a linear-time recurrence recorded from cu_ssd_step + the shared conv1d/softplus
// primitives, carrying per-block conv/SSD state across Step calls. cuda-only.
func NewMamba2CUDA(m *nlp.Mamba2) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newMamba2Decoder(m, backendOps{
		name:        string(backend.CUDA),
		asyncEncode: false,
		newBuffer: func(data []float32) (buffer, error) {
			b, err := cuda.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return cBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := cuda.NewRecorder()
			if err != nil {
				return nil, err
			}
			return cRec{r}, nil
		},
	})
}

// NewMixtralCUDA uploads an nlp.Mixtral (sparse Mixture-of-Experts) onto the batched Decoder core.
// Attention is the plain Llama core (+ optional Qwen3-MoE QK-norm); the FFN is a sparse MoE routed
// per token — the routing weights are computed on-device (cu_moe_gate) so the whole step stays a
// pre-recorded batched command buffer. Dense expert eval (correct); sparse top-k gather is a
// follow-up. cuda-only.
func NewMixtralCUDA(m *nlp.Mixtral) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newMixtralDecoder(m, backendOps{
		name:        string(backend.CUDA),
		asyncEncode: false,
		newBuffer: func(data []float32) (buffer, error) {
			b, err := cuda.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return cBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := cuda.NewRecorder()
			if err != nil {
				return nil, err
			}
			return cRec{r}, nil
		},
	})
}

// NewQwen3MoECUDA uploads a Qwen3-MoE model onto the batched Decoder core. Qwen3-MoE loads as an
// nlp.Mixtral whose blocks additionally carry per-head q/k RMSNorm gains (the same optional QK-norm
// newMixtralDecoder already wires), so this is exactly NewMixtralCUDA with a Qwen3-MoE entry point.
// Load with nlp.Qwen3MoeFromHF.
func NewQwen3MoECUDA(m *nlp.Mixtral) (*Decoder, error) { return NewMixtralCUDA(m) }

// NewMPTCUDA uploads an nlp.MPT (MosaicML) onto the batched Decoder core: the first ALiBi decoder —
// position enters solely through a per-head linear bias on the attention scores (no RoPE), with
// weight-only LayerNorm, standard MHA, a bias-free 2-layer GELU MLP and a tied lm_head. cuda-only
// (the ALiBi attention kernel is cuda-only).
func NewMPTCUDA(m *nlp.MPT) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newMPTDecoder(m, backendOps{
		name:        string(backend.CUDA),
		asyncEncode: false,
		newBuffer: func(data []float32) (buffer, error) {
			b, err := cuda.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return cBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := cuda.NewRecorder()
			if err != nil {
				return nil, err
			}
			return cRec{r}, nil
		},
	})
}

// NewGemma2CUDA uploads an nlp.Gemma2 onto the batched Decoder core: Gemma's √dim embed / (1+w)
// RMSNorm / GeGLU / decoupled head_dim / tied lm_head, plus sandwich norms (pre + post per sublayer),
// an attention-logit soft-cap (via MHACap) and query_pre_attn_scalar. The final-logit soft-cap is
// monotonic (greedy-invariant) and omitted. cuda-only (the soft-cap kernel is cuda-only).
func NewGemma2CUDA(m *nlp.Gemma2) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newGemma2Decoder(m, backendOps{
		name:        string(backend.CUDA),
		asyncEncode: false,
		newBuffer: func(data []float32) (buffer, error) {
			b, err := cuda.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return cBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := cuda.NewRecorder()
			if err != nil {
				return nil, err
			}
			return cRec{r}, nil
		},
	})
}

// NewGemmaCUDA uploads an nlp.Gemma (Gemma v1) onto the batched Decoder core: RMSNorm (with the
// (1+w) gain folded at load), RoPE, GQA, a √dim embedding normalizer (d.embMult), a GeGLU FFN and
// a tied lm_head — every departure a reused generalization or a load-time fold. cuda-only.
func NewGemmaCUDA(m *nlp.Gemma) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newGemmaDecoder(m, backendOps{
		name:        string(backend.CUDA),
		asyncEncode: false,
		newBuffer: func(data []float32) (buffer, error) {
			b, err := cuda.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return cBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := cuda.NewRecorder()
			if err != nil {
				return nil, err
			}
			return cRec{r}, nil
		},
	})
}

// NewCohereCUDA uploads an nlp.Cohere (Command-R) onto the batched Decoder core: one-norm parallel
// residual, weight-only mean-centered LayerNorm, SwiGLU, GQA, full rope (its interleaved rotary is
// pre-permuted into the q/k weights by CohereFromHF) and a logit_scale-folded tied lm_head — every
// piece a reused generalization. cuda-only (parallel residual + LayerNorm are the Phi plumbing).
func NewCohereCUDA(m *nlp.Cohere) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newCohereDecoder(m, backendOps{
		name:        string(backend.CUDA),
		asyncEncode: false,
		newBuffer: func(data []float32) (buffer, error) {
			b, err := cuda.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return cBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := cuda.NewRecorder()
			if err != nil {
				return nil, err
			}
			return cRec{r}, nil
		},
	})
}

// NewGraniteCUDA uploads an IBM Granite model onto the batched Decoder core. Granite is a plain
// Llama plus four learned-at-config scalars — embedding_multiplier, attention_multiplier,
// residual_multiplier and logits_scaling — which newDecoder folds into the upload (embedding gather
// scale, softmax scale override, Wo/Wdown pre-scale, and 1/logits_scaling into the lm_head), so this
// is NewCUDA with a Granite-typed entry point. Load with nlp.GraniteFromHF.
func NewGraniteCUDA(m *nlp.Llama) (*Decoder, error) { return NewCUDA(m) }

// NewStableLMCUDA uploads an nlp.StableLM into CUDA device buffers and runs it through the same
// batched Decoder core as NewCUDA — but with LayerNorm-with-bias norms and PARTIAL rotary (the
// StableLM/Phi/StarCoder2-class departures from Llama). The first of the new-architecture GPU
// graph decoders enabled by the cu_rope_partial* kernels. cuda-only for now.
func NewStableLMCUDA(m *nlp.StableLM) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newStableLMDecoder(m, backendOps{
		name:        string(backend.CUDA),
		asyncEncode: false,
		newBuffer: func(data []float32) (buffer, error) {
			b, err := cuda.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return cBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := cuda.NewRecorder()
			if err != nil {
				return nil, err
			}
			return cRec{r}, nil
		},
	})
}

// NewStarCoder2CUDA uploads an nlp.StarCoder2 onto the batched Decoder core: LayerNorm-with-bias,
// biased q/k/v/o projections, a biased 2-layer GELU MLP, full rope and GQA. The second new-arch GPU
// graph decoder, exercising every Decoder-core generalization except partial rotary. cuda-only.
func NewStarCoder2CUDA(m *nlp.StarCoder2) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newStarCoder2Decoder(m, backendOps{
		name:        string(backend.CUDA),
		asyncEncode: false,
		newBuffer: func(data []float32) (buffer, error) {
			b, err := cuda.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return cBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := cuda.NewRecorder()
			if err != nil {
				return nil, err
			}
			return cRec{r}, nil
		},
	})
}

// NewPhiCUDA uploads an nlp.Phi onto the batched Decoder core: one-norm parallel residual, biased
// q/k/v/dense projections, a biased GELU MLP, partial rotary, a biased untied lm_head and a final
// LayerNorm — every Decoder-core generalization at once. The third new-arch GPU graph decoder.
func NewPhiCUDA(m *nlp.Phi) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newPhiDecoder(m, backendOps{
		name:        string(backend.CUDA),
		asyncEncode: false,
		newBuffer: func(data []float32) (buffer, error) {
			b, err := cuda.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return cBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := cuda.NewRecorder()
			if err != nil {
				return nil, err
			}
			return cRec{r}, nil
		},
	})
}

// NewGPTNeoXCUDA uploads an nlp.GPTNeoX onto the batched Decoder core: TWO-norm parallel residual
// (input_layernorm → attn and post_attention_layernorm → mlp, both over the raw residual),
// LayerNorm-with-bias, biased q/k/v/dense projections, a biased GELU MLP, (partial or full) rotary
// and an untied unbiased lm_head. The fourth new-arch GPU graph decoder — the two-norm sibling of
// Phi's one-norm parallel residual. cuda-only.
func NewGPTNeoXCUDA(m *nlp.GPTNeoX) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newGPTNeoXDecoder(m, backendOps{
		name:        string(backend.CUDA),
		asyncEncode: false,
		newBuffer: func(data []float32) (buffer, error) {
			b, err := cuda.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return cBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := cuda.NewRecorder()
			if err != nil {
				return nil, err
			}
			return cRec{r}, nil
		},
	})
}

// NewGPTCUDA uploads an nlp.GPT (GPT-2-style: learned positional embeddings, LayerNorm, GELU MLP)
// into CUDA device buffers for batched decoding — the NVIDIA variant of NewGPT/NewGPTVulkan (§T422).
// Closes the last backend gap for the GPT-2 decoder, which previously ran only on metal and vulkan.
func NewGPTCUDA(m *nlp.GPT) (*GPTDecoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newGPTDecoder(m, backendOps{
		name:        string(backend.CUDA),
		asyncEncode: false,
		newBuffer: func(data []float32) (buffer, error) {
			b, err := cuda.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return cBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := cuda.NewRecorder()
			if err != nil {
				return nil, err
			}
			return cRec{r}, nil
		},
	})
}

// cudaUploadQWeight makes a ggml quantized [Out,In] weight resident on the GPU as Q8.
// The CUDA backend has ONE quant kernel (Q8 GEMV), so — unlike metal/vulkan, which keep
// each native ggml type and dequantize in-kernel — any source type (Q4_K, Q6_K, …) is
// dequantized to f32 and re-quantized to a uniform resident Q8_0. That's a faithful Q8
// inference path (Q8→f32→Q8 is near-lossless; lower-bit sources lose only what they
// already lost). NewResidentBQ8 wants [In,Out], so the dequantized [Out,In] is transposed.
func cudaUploadQWeight(weight []byte, qt uint32, n, k int) (qweight, error) {
	f32, err := gguf.QuantTensor{Data: weight, GGType: qt, Shape: tensor.Shape{n, k}}.Dequantize()
	if err != nil {
		return nil, fmt.Errorf("llamagpu: CUDA dequant qt=%d [%d,%d]: %w", qt, n, k, err)
	}
	tin, err := f32.Transpose(0, 1) // [Out,In] → [In,Out]
	if err != nil {
		return nil, fmt.Errorf("llamagpu: CUDA weight transpose: %w", err)
	}
	rw, err := cuda.NewResidentBQ8(tin)
	if err != nil {
		return nil, fmt.Errorf("llamagpu: CUDA Q8 upload: %w", err)
	}
	return rw, nil
}

// NewQuantCUDA uploads a quantized Llama with every projection as a resident Q8 weight
// (source types re-quantized to Q8) — the NVIDIA variant of NewQuant/NewQuantVulkan.
// Same batched Decoder core, so Generate/Step and every sampler work unchanged.
func NewQuantCUDA(m *nlp.QuantLlama) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newQuantDecoder(m, backendOps{
		name:        string(backend.CUDA),
		asyncEncode: false,
		newBuffer: func(data []float32) (buffer, error) {
			b, err := cuda.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return cBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := cuda.NewRecorder()
			if err != nil {
				return nil, err
			}
			return cRec{r}, nil
		},
		uploadQWeight: cudaUploadQWeight,
	})
}
