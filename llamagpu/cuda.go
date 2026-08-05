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
func (c cRec) MHABias(q, k, v, o, bias buffer, sq, sk, dm, heads, kvHeads, dk, causal, window int, scale float32) error {
	return c.r.MHABias(cb(q), cb(k), cb(v), cb(o), cb(bias), sq, sk, dm, heads, kvHeads, dk, causal, window, scale)
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
func (c cRec) WKVStep(k, v, w, u, aa, bb, pp, out buffer, d int) error {
	return c.r.WKVStep(cb(k), cb(v), cb(w), cb(u), cb(aa), cb(bb), cb(pp), cb(out), d)
}
func (c cRec) Unary(x, o buffer, op int) error { return c.r.Unary(cb(x), cb(o), op) }
func (c cRec) Binary(a, b, o buffer, op int) error {
	return c.r.Binary(cb(a), cb(b), cb(o), op)
}

// SwiGLUHalves fuses SwiGLU over the two halves of a column-fused gate|up buffer (§gate-up fusion):
// out[r,i] = silu(gu[r,i])·gu[r,hidden+i]. Satisfies the decoder's optional swiGLUHalvesRecorder so
// recordFFN can fuse ffn_gate|ffn_up into one GEMV + this op instead of two GEMVs + a Binary SwiGLU.
func (c cRec) SwiGLUHalves(gu, out buffer, rows, hidden int) error {
	return c.r.SwiGLUHalves(cb(gu), cb(out), rows, hidden)
}

// GeGLUHalves fuses GeGLU over the halves of a column-fused gate|up buffer (Gemma): out[r,i] =
// GELU(gu[r,i])·gu[r,hidden+i]. Satisfies the decoder's optional geGLUHalvesRecorder.
func (c cRec) GeGLUHalves(gu, out buffer, rows, hidden int) error {
	return c.r.GeGLUHalves(cb(gu), cb(out), rows, hidden)
}

// QMatMulResidentMoE routes a Q8 expert GEMV through the moeW-gated kernel (skips non-selected tokens'
// weight-stream, bit-identical to dense+RowAxpy-w0). Non-Q8 quant → dense QMatMulResident (correct).
func (c cRec) QMatMulResidentMoE(x buffer, w qweight, o buffer, m int, moeGate buffer, gateStride, ex int) error {
	switch rw := w.(type) {
	case *cuda.ResidentBQ8:
		return rw.QMatMulMoeInto(cb(x), cb(o), cb(moeGate), m, gateStride, ex)
	case *cuda.ResidentBQ4K:
		return rw.QMatMulMoeInto(cb(x), cb(o), cb(moeGate), m, gateStride, ex)
	}
	return c.QMatMulResident(x, w, o, m)
}

func (c cRec) QMatMulResident(x buffer, w qweight, o buffer, m int) error {
	switch rw := w.(type) {
	case *cuda.ResidentBQ8:
		return c.r.QMatMulResident(cb(x), rw, cb(o), m) // Q8: GEMV decode + int8-tensor-core MMQ prefill
	case *cuda.ResidentBQ4K:
		return c.r.QMatMulResidentQ4K(cb(x), rw, cb(o), m) // Q4_K: aggressive-quant decode GEMV + tensor-core prefill
	case *cuda.ResidentBQ6K:
		return c.r.QMatMulResidentQ6K(cb(x), rw, cb(o), m) // Q6_K: 6-bit k-quant decode GEMV + tensor-core prefill
	case *cuda.ResidentBQ5K:
		return c.r.QMatMulResidentQ5K(cb(x), rw, cb(o), m) // Q5_K: 5-bit k-quant decode GEMV + tensor-core prefill
	case *cuda.ResidentBQ2K:
		return c.r.QMatMulResidentQ2K(cb(x), rw, cb(o), m) // Q2_K: 2-bit k-quant (0.25 B/w) decode GEMV + tensor-core prefill
	case *cuda.ResidentBQ3K:
		return c.r.QMatMulResidentQ3K(cb(x), rw, cb(o), m) // Q3_K: 3-bit k-quant decode GEMV + tensor-core prefill
	default:
		return fmt.Errorf("llamagpu: cuda QMatMulResident: unsupported resident weight %T", w)
	}
}

// QMatMulResidentAcc fuses the residual add into the resident quant GEMV (dst += x·dequant(w), beta=1)
// for the decode epilogue. All K-quants + Q8 are wired (Q8/Q4_K/Q6_K/Q5_K/Q2_K/Q3_K — the mixes
// spread projections across these); other formats return errQuantAccUnsupported → GEMV+Binary fallback.
func (c cRec) QMatMulResidentAcc(x buffer, w qweight, dst buffer, m int) error {
	switch rw := w.(type) {
	case *cuda.ResidentBQ8:
		return c.r.QMatMulResidentAcc(cb(x), rw, cb(dst), m)
	case *cuda.ResidentBQ4K:
		return c.r.QMatMulResidentAccQ4K(cb(x), rw, cb(dst), m)
	case *cuda.ResidentBQ6K:
		return c.r.QMatMulResidentAccQ6K(cb(x), rw, cb(dst), m)
	case *cuda.ResidentBQ5K:
		return c.r.QMatMulResidentAccQ5K(cb(x), rw, cb(dst), m)
	case *cuda.ResidentBQ2K:
		return c.r.QMatMulResidentAccQ2K(cb(x), rw, cb(dst), m)
	case *cuda.ResidentBQ3K:
		return c.r.QMatMulResidentAccQ3K(cb(x), rw, cb(dst), m)
	}
	return errQuantAccUnsupported
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

// NewLlamaQ8CUDA is NewCUDA with every projection (fused QKV, o_proj, gate/up/down, lm_head) quantized
// to resident Q8_0 — a direct f32→Q8 decode path that needs NO ggml QuantLlama conversion. Transformer
// decode is weight-bandwidth-bound, so streaming ~4× smaller Q8 projections speeds it up substantially.
// Unlike nlp.QuantLlama (Llama-only), this reaches the Qwen2 (qkv-bias) and Qwen3 (QK-norm) variants —
// biases and QK-norm gains stay f32 and apply after the Q8 matmul. Q8 is not bit-exact (cosine-validated).
func NewLlamaQ8CUDA(m *nlp.Llama) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newDecoder(m, backendOps{
		name:        string(backend.CUDA),
		asyncEncode: false,
		fusedGateUp: true, // cRec implements SwiGLUHalves; newDecoder column-fuses ffn_gate|ffn_up
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
		quantizeF32: func(w *tensor.Tensor) (qweight, error) {
			return cuda.NewResidentBQ8(w)
		},
	})
}

// cudaQ4KResident quantizes a host f32 weight [K,N] to a resident Q4_K (ggml 4-bit k-quant, 256-element
// affine super-blocks) device weight — the aggressive-quant option. It transposes [K,N]→[N,K] (Q4_K blocks
// are per output row), quantizes with the tested gguf.Quantize(Q4_K), and uploads via NewResidentBQ4KFromBlocks.
// Q4_K is 2× smaller than Q8 (a 13B model fits the 12GB 3060 at Q4_K but not Q8) and matches ggml Q4_K_M
// accuracy — but decode-only (no int-tensor-core prefill), so prefill is slower than Q8. K must be a mult of 256.
func cudaQ4KResident(w *tensor.Tensor) (qweight, error) {
	kk, nn := w.Shape()[0], w.Shape()[1]
	if kk%256 != 0 {
		return nil, fmt.Errorf("llamagpu: Q4_K needs K%%256==0, got K=%d (weight %dx%d)", kk, kk, nn)
	}
	bf := w.Cast(tensor.F32).Contiguous().Storage().F32() // [K,N] row-major: bf[r*nn+col]
	wt := make([]float32, nn*kk)                          // [N,K]
	for col := 0; col < nn; col++ {
		base := col * kk
		//perfscan:ignore PS4004 one-time Q4_K weight repack/quantize at model load
		for r := 0; r < kk; r++ {
			//perfscan:ignore PS6011 load-time resident-block construction; cold
			wt[base+r] = bf[r*nn+col]
		}
	}
	t := tensor.New(tensor.F32, tensor.Shape{nn, kk})
	copy(t.Storage().F32(), wt)
	blocks, err := gguf.Quantize(t, gguf.Q4_K)
	if err != nil {
		return nil, err
	}
	return cuda.NewResidentBQ4KFromBlocks(blocks, kk, nn)
}

// NewLlamaQ4KCUDA is NewCUDA with every projection quantized to resident Q4_K (4-bit k-quant) — 2× smaller
// than Q8, matching ggml Q4_K_M accuracy, with a direct f32 path (no ggml file needed). It lets the 12GB
// 3060 run models that don't fit at Q8. Decode is a Q4_K GEMV (weight-bandwidth win over Q8); prefill has no
// int-tensor-core GEMM so is slower than Q8's MMQ — prefer Q8 when prefill throughput dominates. Reaches the
// Qwen2/Qwen3 variants nlp.QuantLlama can't. K%256==0 required (typical transformer dims satisfy it). cuda-only.
func NewLlamaQ4KCUDA(m *nlp.Llama) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newDecoder(m, backendOps{
		name:        string(backend.CUDA),
		asyncEncode: false,
		fusedGateUp: true, // cRec implements SwiGLUHalves; newDecoder column-fuses ffn_gate|ffn_up
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
		quantizeF32: cudaQ4KResident,
	})
}

// NewQwen2Q4KCUDA / NewQwen3Q4KCUDA / NewPhi3Q4KCUDA are NewLlamaQ4KCUDA typed entry points for the popular
// Llama variants (Qwen2/2.5 qkv-bias, Qwen3 QK-norm, Phi-3) — their 4-bit path so a bigger model fits the
// 12GB 3060. Load with nlp.LlamaFromHF / Phi3FromHF; biases/QK-norm stay f32.
func NewQwen2Q4KCUDA(m *nlp.Llama) (*Decoder, error) { return NewLlamaQ4KCUDA(m) }

// NewQwen3Q4KCUDA is NewQwen2Q4KCUDA's Qwen3 sibling (see above): NewLlamaQ4KCUDA with a Qwen3-typed
// signature, so it picks up Qwen3's per-head QK-norm automatically from the checkpoint.
func NewQwen3Q4KCUDA(m *nlp.Llama) (*Decoder, error) { return NewLlamaQ4KCUDA(m) }

// NewPhi3Q4KCUDA is NewQwen2Q4KCUDA's Phi-3 sibling (see above): NewLlamaQ4KCUDA with a Phi-3-typed
// signature. Load with nlp.Phi3FromHF.
func NewPhi3Q4KCUDA(m *nlp.Llama) (*Decoder, error) { return NewLlamaQ4KCUDA(m) }

// NewQwen2Q8CUDA / NewQwen3Q8CUDA / NewPhi3Q8CUDA are NewLlamaQ8CUDA typed entry points for the Llama
// variants that nlp.QuantLlama cannot represent (Qwen2 qkv-bias, Qwen3 QK-norm) or that only had an f32
// GPU path (Phi-3). They give those models their first quantized decode. Load with nlp.LlamaFromHF /
// Phi3FromHF; biases and QK-norm gains are picked up and kept f32.
func NewQwen2Q8CUDA(m *nlp.Llama) (*Decoder, error) { return NewLlamaQ8CUDA(m) }

// NewQwen3Q8CUDA is NewQwen2Q8CUDA's Qwen3 sibling (see above): NewLlamaQ8CUDA with a Qwen3-typed
// signature, so it picks up Qwen3's per-head QK-norm automatically from the checkpoint.
func NewQwen3Q8CUDA(m *nlp.Llama) (*Decoder, error) { return NewLlamaQ8CUDA(m) }

// NewPhi3Q8CUDA is NewQwen2Q8CUDA's Phi-3 sibling (see above): NewLlamaQ8CUDA with a Phi-3-typed
// signature — Phi-3's first quantized GPU decode path. Load with nlp.Phi3FromHF.
func NewPhi3Q8CUDA(m *nlp.Llama) (*Decoder, error) { return NewLlamaQ8CUDA(m) }

// cudaQ8Ops is the cuda backendOps with resident-Q8_0 quantization enabled (ops.quantizeF32 set) —
// shared by the dense-transformer NewXQ8CUDA entry points below. Each routes its arch's f32 checkpoint
// through the same weight-bandwidth lever (~2-3× faster decode): mkLin/mkFused quantize every projection
// and fused QKV; biases, QK-norm, ALiBi and RoPE stay f32. Q8 is not bit-exact (cosine-validated). These
// reach arches with NO prior quant path (nlp.QuantLlama is Llama-only).
func cudaQ8Ops() backendOps {
	return backendOps{
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
		quantizeF32: func(w *tensor.Tensor) (qweight, error) { return cuda.NewResidentBQ8(w) },
	}
}

// NewGemma2Q8CUDA is NewGemma2CUDA with every projection (fused QKV, o_proj, gate/up/down) and the
// tied lm_head quantized to resident Q8_0 — Gemma2's vocab is huge (256k), so newGemma2Decoder routes
// the tied head through the same quantize-aware builder rather than leaving it f32. The attention-logit
// soft-cap (MHACap), √dim embed scaling and (1+w) RMSNorm gains stay f32. cuda-only.
func NewGemma2Q8CUDA(m *nlp.Gemma2) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newGemma2Decoder(m, cudaQ8Ops())
}

// NewGemmaQ8CUDA is NewGemmaCUDA (Gemma v1) with all projections + the tied lm_head quantized to resident
// Q8_0. Gemma's √dim embed scaling and GeGLU are unchanged; the lm_head goes Q8 too (huge 256k vocab).
func NewGemmaQ8CUDA(m *nlp.Gemma) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	o := cudaQ8Ops()
	o.fusedGateUp = true // cRec implements GeGLUHalves; newGemmaDecoder column-fuses ffn_gate|ffn_up
	return newGemmaDecoder(m, o)
}

// NewGraniteQ8CUDA is the Q8 entry point for IBM Granite (dense) — Granite maps to nlp.Llama (Llama core +
// embMult/attn-scale/residual-mult/logit-scale config scalars, all folded into the upload), so this is
// NewLlamaQ8CUDA with a Granite-typed signature. The scaled Wo/Wdown are quantized post-scale.
func NewGraniteQ8CUDA(m *nlp.Llama) (*Decoder, error) { return NewLlamaQ8CUDA(m) }

// NewCohereQ8CUDA is NewCohereCUDA with every projection (fused QKV, o_proj, gate/up/down) quantized to
// resident Q8_0 — Command-R's weight-bandwidth win. The logit_scale-folded tied lm_head is quantized
// too (via mkLinS); the parallel-residual weight-only mean-centered LayerNorm gains stay f32. cuda-only.
func NewCohereQ8CUDA(m *nlp.Cohere) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newCohereDecoder(m, cudaQ8Ops())
}

// NewNemotronQ8CUDA is NewNemotronCUDA with every projection (fused QKV, o_proj, the 2-layer relu²
// MLP's up/down) and the lm_head quantized to resident Q8_0. The LayerNorm1P gain/bias (γ=w+1, β
// folded in by NemotronFromHF) and the squared-ReLU activation stay f32. cuda-only.
func NewNemotronQ8CUDA(m *nlp.Nemotron) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newNemotronDecoder(m, cudaQ8Ops())
}

// NewOLMo2Q8CUDA is NewOLMo2CUDA with every projection (fused QKV, o_proj, gate/up/down) and the
// untied lm_head quantized to resident Q8_0. The post-norm placement and the full-width q/k RMSNorm
// gains (applied before RoPE) stay f32. cuda-only.
func NewOLMo2Q8CUDA(m *nlp.OLMo2) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newOLMo2Decoder(m, cudaQ8Ops())
}

// NewFalconQ8CUDA is NewFalconCUDA with every projection (fused QKV, dense, the 2-layer GELU MLP's
// up/down) and the lm_head quantized to resident Q8_0. The parallel-residual LayerNorm-with-bias gains
// stay f32. cuda-only.
func NewFalconQ8CUDA(m *nlp.Falcon) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newFalconDecoder(m, cudaQ8Ops())
}

// NewStableLMQ8CUDA is NewStableLMCUDA with every projection (fused QKV, o_proj, gate/up/down) and the
// lm_head quantized to resident Q8_0. The LayerNorm-with-bias gains and the partial-rotary split stay
// f32. cuda-only.
func NewStableLMQ8CUDA(m *nlp.StableLM) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newStableLMDecoder(m, cudaQ8Ops())
}

// NewStarCoder2Q8CUDA is NewStarCoder2CUDA with every projection (fused QKV, o_proj, the 2-layer GELU
// MLP's fc/proj) and the untied lm_head quantized to resident Q8_0. Every projection bias and the
// LayerNorm-with-bias gains stay f32. cuda-only.
func NewStarCoder2Q8CUDA(m *nlp.StarCoder2) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newStarCoder2Decoder(m, cudaQ8Ops())
}

// NewMPTQ8CUDA is NewMPTCUDA with every projection and the tied lm_head quantized to resident Q8_0.
// The per-head ALiBi slopes and the weight-only LayerNorm gains stay f32. cuda-only.
func NewMPTQ8CUDA(m *nlp.MPT) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newMPTDecoder(m, cudaQ8Ops())
}

// NewGPTNeoXQ8CUDA is NewGPTNeoXCUDA with every projection (fused QKV, o_proj, the 2-layer GELU MLP's
// up/down) and the untied, unbiased lm_head quantized to resident Q8_0. Every projection bias, the
// two-norm LayerNorm gains and the (partial or full) rotary split stay f32. cuda-only.
func NewGPTNeoXQ8CUDA(m *nlp.GPTNeoX) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newGPTNeoXDecoder(m, cudaQ8Ops())
}

// NewPhiQ8CUDA is NewPhiCUDA with every projection (fused QKV, dense, the 2-layer GELU MLP's fc1/fc2)
// and the untied lm_head quantized to resident Q8_0. Every projection bias (including the lm_head's),
// the final LayerNorm and the partial-rotary split stay f32. cuda-only.
func NewPhiQ8CUDA(m *nlp.Phi) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newPhiDecoder(m, cudaQ8Ops())
}

// The MoE Q8 entry points quantize the (bandwidth-dominant) expert gate/up/down matrices and attention
// projections to resident Q8_0 while the top-k ROUTER stays f32 (d.f32Lin) — Q8 rounding in the router
// logits could flip expert selection. Sparse-MoE decode streams many expert matrices per token, so this
// is the largest Q8 decode win. Qwen3-MoE loads as an nlp.Mixtral, so NewQwen3MoEQ8CUDA == NewMixtralQ8CUDA.
// cudaQ4KOps is cudaQ8Ops with resident-Q4_K quantization (0.5625 B/weight vs Q8 1.0625 — ~2x less
// VRAM). Every projection with K%256==0 becomes a resident Q4_K weight; MoE experts (ResidentBQ4K) then
// get the gated sparse decode (cu_qmatmul_q4k_moe) automatically via cRec.QMatMulResidentMoE.
func cudaQ4KOps() backendOps {
	o := cudaQ8Ops()
	o.quantizeF32 = cudaQ4KResident
	return o
}

// NewMixtralQ4KCUDA uploads an nlp.Mixtral with Q4_K-quantized weights (experts included) — the Q4_K_M
// MoE path: ~2x less VRAM than NewMixtralQ8CUDA plus the ~E/K gated expert decode. Requires every
// projection K (=Dim; per-expert Dim/Hidden) %256==0. Qwen3-MoE loads as an nlp.Mixtral, so it applies too.
func NewMixtralQ4KCUDA(m *nlp.Mixtral) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newMixtralDecoder(m, cudaQ4KOps())
}

func NewMixtralQ8CUDA(m *nlp.Mixtral) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newMixtralDecoder(m, cudaQ8Ops())
}

// NewQwen3MoEQ8CUDA is NewMixtralQ8CUDA's Qwen3-MoE entry point (see the MoE Q8 note above) —
// Qwen3-MoE loads as an nlp.Mixtral, so this is exactly NewMixtralQ8CUDA. Load with nlp.Qwen3MoeFromHF.
func NewQwen3MoEQ8CUDA(m *nlp.Mixtral) (*Decoder, error) { return NewMixtralQ8CUDA(m) }

// NewQwen2MoEQ8CUDA is the Q8 entry point for nlp.Qwen2MoE (see the MoE Q8 note above): the routed and
// shared-expert matrices plus the Qwen2 attention (with its q/k/v projection biases) go resident Q8_0;
// the shared-expert sigmoid gate is quantized too. The top-k router stays f32. cuda-only.
func NewQwen2MoEQ8CUDA(m *nlp.Qwen2MoE) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newQwen2MoEDecoder(m, cudaQ8Ops())
}

// NewGraniteMoEQ8CUDA is the Q8 entry point for nlp.GraniteMoE (see the MoE Q8 note above): the
// attention projections and the routed expert gate/up/down matrices go resident Q8_0, with
// ResidualMult/LogitsScale folded pre-quantization into Wo/expert-Wdown/lm_head (as NewGraniteQ8CUDA
// does for dense Granite). The top-k router stays f32 (routing is selection-sensitive). cuda-only.
func NewGraniteMoEQ8CUDA(m *nlp.GraniteMoE) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newGraniteMoEDecoder(m, cudaQ8Ops())
}

// NewOLMoEQ8CUDA is the Q8 entry point for nlp.OLMoE (see the MoE Q8 note above): the attention
// projections and the routed expert matrices go resident Q8_0; the full-width q/k RMSNorm gains and
// the top-k router stay f32. cuda-only.
func NewOLMoEQ8CUDA(m *nlp.OLMoE) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newOLMoEDecoder(m, cudaQ8Ops())
}

// NewDeepSeekV2Q8CUDA quantizes the flagship MLA + DeepSeekMoE decoder — the last Q8 gap. The low-rank
// MLA projections (q_a/q_b/kv_a/kv_b, o_proj), the routed + shared expert matrices all go resident Q8_0;
// the latent RMSNorms, decoupled RoPE and the top-k router stay f32 (the router carved out, as for the
// other MoE arches). Completes Q8 decode across the entire f32 transformer catalogue.
func NewDeepSeekV2Q8CUDA(m *nlp.DeepSeekV2) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newDeepSeekV2Decoder(m, cudaQ8Ops())
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

// NewMambaQ8CUDA is NewMambaCUDA with the projection weights quantized to resident Q8_0. Mamba decode
// is weight-bandwidth-bound (~230 MB of f32 weights streamed per token), so the Q8_0 GEMVs cut the
// dominant cost ~4× at the price of Q8 rounding (not bit-exact — validated to a tolerance, not parity).
// The tiny per-channel state (A, dt_bias, conv, dskip, norms) stays f32. cuda-only.
func NewMambaQ8CUDA(m *nlp.Mamba) (*Decoder, error) {
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
		quantizeF32: func(w *tensor.Tensor) (qweight, error) {
			return cuda.NewResidentBQ8(w)
		},
	})
}

// NewJambaCUDA uploads an nlp.Jamba (hybrid Mamba + NoPE-attention + MoE/dense) onto the batched
// Decoder core. Each layer is a Mamba selective-scan mixer OR grouped-query causal attention (no rotary
// — the Mamba layers carry position), followed by a sparse-MoE or dense SwiGLU FFN. Attention layers
// keep a growing KV cache; Mamba layers carry conv/SSM state across Step, so the whole model decodes
// sequentially (rows==1 Step; StepN loops Step). cuda-only.
func NewJambaCUDA(m *nlp.Jamba) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newJambaDecoder(m, backendOps{
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

// NewJambaQ8CUDA is NewJambaCUDA with the projection weights quantized to resident Q8_0 — the hybrid's
// Mamba mixers, NoPE-attention q/k/v/o, MoE expert gate/up/down, dense SwiGLU and lm_head all go Q8 via
// the shared mkLin, while the MoE top-k router stays f32 (d.f32Lin) so routing is exact. Jamba decodes
// rows==1 (Step), so this is a decode-bandwidth win (the GEMV path); StepN loops Step. cuda-only.
func NewJambaQ8CUDA(m *nlp.Jamba) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newJambaDecoder(m, cudaQ8Ops())
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

// NewMamba2Q8CUDA is NewMamba2CUDA with the in/out projections and LM head quantized to resident Q8_0 —
// the same weight-bandwidth lever as NewMambaQ8CUDA. Mamba-2's InProj/OutProj are stored [out,in] (torch
// layout), so newMamba2Decoder transposes each to [in,out] before quantizing (NewResidentBQ8 materializes
// the transposed view). The SSD recurrence (per-head decay, conv, gated RMSNorm) stays f32. cuda-only.
func NewMamba2Q8CUDA(m *nlp.Mamba2) (*Decoder, error) {
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
		quantizeF32: func(w *tensor.Tensor) (qweight, error) {
			return cuda.NewResidentBQ8(w)
		},
	})
}

// NewRWKVCUDA uploads an nlp.RWKV onto the batched Decoder core — GoAI's third recurrent family. Each
// layer is a WKV time-mix (recorded from cu_wkv_step) + a gated squared-ReLU channel-mix; decode is
// O(1) recurrent with no attention and no KV cache, carrying per-block token-shift + WKV state across
// Step calls. cuda-only.
func NewRWKVCUDA(m *nlp.RWKV) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newRWKVDecoder(m, backendOps{
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

// NewRWKVQ8CUDA is NewRWKVCUDA with the time-mix (Wr/Wk/Wv/Wo) and channel-mix (CWr/CWk/CWv) GEMVs plus
// the LM head quantized to resident Q8_0 — the same weight-bandwidth lever as NewMambaQ8CUDA. RWKV decode
// streams these projection weights per token; Q8_0 cuts that ~4× at the price of Q8 rounding (validated to
// a cosine tolerance, not parity). The tiny per-channel mix params (μ/decay/bonus) and LayerNorms stay f32.
func NewRWKVQ8CUDA(m *nlp.RWKV) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newRWKVDecoder(m, backendOps{
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
		quantizeF32: func(w *tensor.Tensor) (qweight, error) {
			return cuda.NewResidentBQ8(w)
		},
	})
}

// NewBertCUDA uploads an nlp.Bert bidirectional encoder onto the GPU — the first non-decoder GPU
// model in llamagpu. It runs one bidirectional forward (post-LN, learned absolute positions, no
// causal mask, no KV cache) returning the [seq, dim] hidden states, reusing the existing recorder
// ops (MHA with causal=0, LayerNorm, GELU MLP) with no new kernel. cuda-only.
func NewBertCUDA(m *nlp.Bert) (*GPUBert, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newBertEncoder(m, backendOps{
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

// NewBertQ8CUDA is NewBertCUDA with the attention (q/k/v/o) and FFN (w1/w2) projections quantized to
// resident Q8_0. BERT encodes the whole sequence at once (M=seq>1), so each Q8 projection runs on the
// int8 tensor-core MMQ GEMM (QMatMulResident's m>1 path) — faster than the f32 cuBLAS GEMM and 4× less
// weight memory. Biases and LayerNorm stay f32. Works for RoBERTa/DistilBERT too (all *nlp.Bert). cuda-only.
func NewBertQ8CUDA(m *nlp.Bert) (*GPUBert, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newBertEncoder(m, cudaQ8Ops())
}

// NewT5CUDA uploads an nlp.T5 bidirectional encoder onto the GPU. T5 is the second non-decoder GPU
// model: PRE-LN residuals, RMSNorm (T5LayerNorm), a learned per-head RELATIVE-position bias added to
// the scores (via cu_attn_softmax_bias / MHABias), NO 1/√d scaling, NO absolute/rotary position, a
// gated-GELU (v1.1) or ReLU (v1.0) FFN and no projection biases. Returns [seq, dim] hidden states.
// cuda-only (MHABias is cuda-only).
func NewT5CUDA(m *nlp.T5) (*GPUT5, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newT5Encoder(m, backendOps{
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

// NewT5Q8CUDA is NewT5CUDA with the attention (q/k/v/o) and FFN (wi0/wi1/wOut) projections quantized to
// resident Q8_0. T5 encodes the whole sequence at once (M=seq>1), so each Q8 projection runs on the int8
// tensor-core MMQ GEMM (QMatMulResident's m>1 path) — faster than the f32 cuBLAS GEMM and 4× less weight
// memory. RMSNorm and the relative-position bias stay f32. Covers T5 v1.0 (ReLU) and v1.1 (GeGLU). cuda-only.
func NewT5Q8CUDA(m *nlp.T5) (*GPUT5, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newT5Encoder(m, cudaQ8Ops())
}

// NewT5DecoderCUDA uploads an nlp.T5Decoder onto the GPU — the seq2seq (encoder-decoder) decoder, the
// first encoder-decoder GPU model. Each block runs causal self-attention with the T5 relpos bias
// (MHABias over a growing KV cache), cross-attention over the encoder output (plain MHA), and a
// gated-GELU/ReLU FFN, all PRE-LN/RMSNorm/unscaled. Pair with NewT5CUDA (the encoder). cuda-only.
func NewT5DecoderCUDA(m *nlp.T5Decoder) (*GPUT5Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newT5Decoder(m, backendOps{
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

// NewT5DecoderQ8CUDA is NewT5DecoderCUDA with the self/cross-attention (q/k/v/o), FFN (wi0/wi1/wOut) and
// lm_head projections quantized to resident Q8_0 — the seq2seq generation decode-bandwidth win. Autoregressive
// decode is M=1 (Q8 GEMV); the cross K/V projections run once over the encoder output (M=eseq>1 → int8
// tensor-core MMQ). RMSNorm and the relpos bias stay f32. Pair with NewT5Q8CUDA (the Q8 encoder). cuda-only.
func NewT5DecoderQ8CUDA(m *nlp.T5Decoder) (*GPUT5Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newT5Decoder(m, cudaQ8Ops())
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

// cudaUploadQWeight makes a ggml quantized [Out,In] weight resident on the GPU.
//
// Q4_K sources stay NATIVE: the raw GGUF Q4_K blocks (row-major [Out][In/256] super-blocks — exactly
// what NewResidentBQ4KFromBlocks wants) upload as-is, keeping the format's 0.5625 B/weight (vs 1.0625
// upcast to Q8) — ~2× less VRAM and less decode bandwidth, at IDENTICAL accuracy (Q4_K→f32→Q8 adds
// nothing a Q4_K weight didn't already lose). Q4_K's resident dispatch (QMatMulResidentQ4K) already
// serves the GEMV decode AND the tensor-core WMMA prefill, and the GGUF-block path is exactly the one
// TestCUDAQ4KMatMulParity validates against the exact-dequant reference. Any error falls through to Q8.
//
// Other source types (Q5_K, Q6_K, …) have no native CUDA resident dispatch yet, so they are
// dequantized to f32 and re-quantized to a uniform resident Q8_0 (a faithful, near-lossless path).
// NewResidentBQ8 wants [In,Out], so the dequantized [Out,In] is transposed.
func cudaUploadQWeight(weight []byte, qt uint32, n, k int) (qweight, error) {
	if qt == 12 /* tQ4_K */ && k%256 == 0 {
		if rw, err := cuda.NewResidentBQ4KFromBlocks(weight, k, n); err == nil {
			return rw, nil
		}
		// fall through to the always-correct Q8 path on any error
	}
	if qt == 14 /* tQ6_K */ && k%256 == 0 {
		// Q6_K (0.8203 B/w vs Q8 1.0625): keep native — its resident dispatch (QMatMulResidentQ6K)
		// serves the GEMV decode AND the tensor-core WMMA prefill (#874/#880). Raw GGUF Q6_K blocks
		// are row-major [Out][In/256] 210-byte super-blocks = NewResidentBQ6KFromBlocks's input.
		if rw, err := cuda.NewResidentBQ6KFromBlocks(weight, k, n); err == nil {
			return rw, nil
		}
		// fall through to Q8 on any error
	}
	if qt == 13 /* tQ5_K */ && k%256 == 0 {
		// Q5_K (0.6875 B/w vs Q8 1.0625, ~1.5× less VRAM): native via QMatMulResidentQ5K (GEMV decode
		// + tensor-core WMMA prefill). Raw GGUF Q5_K blocks are row-major [Out][In/256] 176-byte
		// super-blocks = NewResidentBQ5KFromBlocks's input.
		if rw, err := cuda.NewResidentBQ5KFromBlocks(weight, k, n); err == nil {
			return rw, nil
		}
		// fall through to Q8 on any error
	}
	if qt == 11 /* tQ3_K */ && k%256 == 0 {
		// Q3_K (0.43 B/w vs Q8 1.0625, ~2.5× less VRAM): native via QMatMulResidentQ3K (GEMV decode +
		// tensor-core WMMA prefill). NewResidentBQ3KFromBlocks repacks the raw GGUF Q3_K 110-byte blocks
		// (row-major [Out][In/256]) into the aligned meta/qs/hmask planes the kernels want.
		if rw, err := cuda.NewResidentBQ3KFromBlocks(weight, k, n); err == nil {
			return rw, nil
		}
		// fall through to Q8 on any error
	}
	if qt == 10 /* tQ2_K */ && k%256 == 0 {
		// Q2_K (0.25 B/w vs Q8 1.0625, ~4× less VRAM): the aggressive-quant option — a Q2_K GGUF upcast
		// to Q8 may not even fit the 12GB 3060, so keeping it native is enabling, not just faster. Native
		// via QMatMulResidentQ2K (GEMV decode + tensor-core WMMA prefill). Raw GGUF Q2_K blocks are
		// row-major [Out][In/256] 84-byte super-blocks = NewResidentBQ2KFromBlocks's input.
		if rw, err := cuda.NewResidentBQ2KFromBlocks(weight, k, n); err == nil {
			return rw, nil
		}
		// fall through to Q8 on any error
	}
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
