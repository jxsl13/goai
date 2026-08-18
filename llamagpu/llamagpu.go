//go:build darwin && cgo

// Further reading: Pope et al. 2022, "Efficiently Scaling Transformer Inference", for the batched-decode systems view; Dao et al. 2022, "FlashAttention", for the memory-aware attention lineage behind the cooperative kernels.
//
// In plain terms: the fast lane for text generation — whole decode steps run as single pre-recorded GPU programs over weights that stay resident on the device, instead of shuttling tensors back and forth per operation.
package llamagpu

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/nlp"
)

// metal adapter (§T409): thin assertions from the backend-agnostic buffer/recorder interfaces to
// the concrete metal.DeviceBuffer/metal.Recorder (whose API vulkan mirrors method-for-method).

type mBuf struct{ *metal.DeviceBuffer }

type mRec struct{ r *metal.Recorder }

// mProfileRec is used only by ProfileMetalStep. It captures the completed recorder before the
// normal Step lifecycle frees it; embedding mRec keeps the production adapter and operation set
// identical.
type mProfileRec struct {
	mRec
	profile *metal.RecorderProfile
}

func (m mRec) RMSNorm(x, g, o buffer, rows, dim int, eps float32) error {
	return m.r.RMSNorm(mb(x), mb(g), mb(o), rows, dim, eps)
}
func (m mRec) LayerNorm(x, g, bb, o buffer, rows, dim int, eps float32) error {
	return m.r.LayerNorm(mb(x), mb(g), mb(bb), mb(o), rows, dim, eps)
}
func (m mRec) AddBias(x, bb, o buffer, rows, n int) error {
	return m.r.AddBias(mb(x), mb(bb), mb(o), rows, n)
}
func (m mRec) MatMul(a, b, c buffer, mm, k, n int) error {
	return m.r.MatMul(mb(a), mb(b), mb(c), mm, k, n)
}
func (m mRec) MatMulAcc(a, b, c buffer, mm, k, n int) error {
	return m.r.MatMulAcc(mb(a), mb(b), mb(c), mm, k, n)
}
func (m mRec) RoPE(q, inv, o buffer, seq, width, heads, hd, half, pos int, posDiv float32) error {
	return m.r.RoPE(mb(q), mb(inv), mb(o), seq, width, heads, hd, half, pos, posDiv)
}
func (m mRec) RoPEAt(q, inv, o buffer, off, seq, width, heads, hd, half, pos int, posDiv float32) error {
	return m.r.RoPEAt(mb(q), mb(inv), mb(o), off, seq, width, heads, hd, half, pos, posDiv)
}
func (m mRec) RoPEPair(qkv, inv buffer, seq, stride, headsQ, offQ, headsK, offK, hd, half, pos int, posDiv float32) error {
	return m.r.RoPEPair(mb(qkv), mb(inv), seq, stride, headsQ, offQ, headsK, offK, hd, half, pos, posDiv)
}
func (m mRec) RoPEPairSplit(qkv, inv, q, k, v buffer,
	seq, stride, headsQ, offQ, headsK, offK, hd, half, vOff, vDim, pos int, posDiv float32,
) error {
	return m.r.RoPEPairSplit(mb(qkv), mb(inv), mb(q), mb(k), mb(v),
		seq, stride, headsQ, offQ, headsK, offK, hd, half, vOff, vDim, pos, posDiv)
}
func (mRec) RoPEPartialPair(qkv, inv buffer, seq, stride, headsQ, offQ, headsK, offK, hd, rotaryDim, pos int, posDiv float32) error {
	return fmt.Errorf("llamagpu(metal): partial-rotary RoPE not implemented (partial-rotary decoders are cuda-only for now)")
}
func (m mRec) Blit(src buffer, srcOff int, dst buffer, dstOff, n int) error {
	return m.r.Blit(mb(src), srcOff, mb(dst), dstOff, n)
}
func (m mRec) Copy2D(src buffer, srcOff, srcStride int, dst buffer, dstOff, dstStride, rows, rowFloats int) error {
	return m.r.Copy2D(mb(src), srcOff, srcStride, mb(dst), dstOff, dstStride, rows, rowFloats)
}
func (m mRec) Copy2DF32ToF16Pair(
	kSrc buffer, kSrcOff, kSrcStride int,
	vSrc buffer, vSrcOff, vSrcStride int,
	kDst buffer, kDstOff, kDstStride int,
	vDst buffer, vDstOff, vDstStride int,
	rows, rowFloats int,
) error {
	return m.r.Copy2DF32ToF16Pair(
		mb(kSrc), kSrcOff, kSrcStride,
		mb(vSrc), vSrcOff, vSrcStride,
		mb(kDst), kDstOff, kDstStride,
		mb(vDst), vDstOff, vDstStride,
		rows, rowFloats,
	)
}
func (m mRec) MHA(q, k, v, o buffer, sq, sk, dm, heads, kvHeads, dk, causal, window int, scale float32) error {
	return m.r.MHA(mb(q), mb(k), mb(v), mb(o), sq, sk, dm, heads, kvHeads, dk, causal, window, scale)
}
func (m mRec) MHAF16KV(q, k, v, o buffer, sq, sk, dm, heads, kvHeads, dk, causal, window int, scale float32) error {
	return m.r.MHAF16KV(mb(q), mb(k), mb(v), mb(o), sq, sk, dm, heads, kvHeads, dk, causal, window, scale)
}
func (mRec) MHACap(q, k, v, o buffer, sq, sk, dm, heads, kvHeads, dk, causal, window int, scale, cap float32) error {
	return fmt.Errorf("llamagpu(metal): attention-logit soft-cap not implemented (Gemma-2-class softcap decoders are cuda-only for now)")
}
func (mRec) MHAALiBi(q, k, v, o, slopes buffer, sq, sk, dm, heads, kvHeads, dk, causal, window int, scale float32) error {
	return fmt.Errorf("llamagpu(metal): ALiBi attention not implemented (ALiBi decoders are cuda-only for now)")
}
func (mRec) MHABias(q, k, v, o, bias buffer, sq, sk, dm, heads, kvHeads, dk, causal, window int, scale float32) error {
	return fmt.Errorf("llamagpu(metal): per-head bias attention not implemented (T5 encoder is cuda-only for now)")
}
func (mRec) MoEGate(logits, weights buffer, rows, e, k, raw int, scale float32) error {
	return fmt.Errorf("llamagpu(metal): MoE routing not implemented (sparse-MoE decoders are cuda-only for now)")
}
func (mRec) RowAxpy(dst, src, arow buffer, rows, cols int) error {
	return fmt.Errorf("llamagpu(metal): MoE combine not implemented (sparse-MoE decoders are cuda-only for now)")
}
func (mRec) MHARect(q, k, v, o buffer, sq, sk, heads, kvHeads, dqk, dv, causal, window int, scale float32) error {
	return fmt.Errorf("llamagpu(metal): rectangular MHA not implemented (MLA decoders are cuda-only for now)")
}
func (mRec) SSMStep(u, delta, a, b, c, dskip, h, y buffer, d, n int) error {
	return fmt.Errorf("llamagpu(metal): SSM step not implemented (Mamba decoders are cuda-only for now)")
}
func (mRec) Conv1DStep(x, w, b, state, out buffer, d, k int) error {
	return fmt.Errorf("llamagpu(metal): conv1d step not implemented (Mamba decoders are cuda-only for now)")
}
func (mRec) SSDStep(x, delta, a, b, c, dskip, state, y buffer, heads, headDim, groups, n int) error {
	return fmt.Errorf("llamagpu(metal): SSD step not implemented (Mamba-2 decoders are cuda-only for now)")
}
func (mRec) WKVStep(k, v, w, u, aa, bb, pp, out buffer, d int) error {
	return fmt.Errorf("llamagpu(metal): WKV step not implemented (RWKV decoders are cuda-only for now)")
}
func (m mRec) Unary(x, o buffer, op int) error { return m.r.Unary(mb(x), mb(o), op) }
func (m mRec) Binary(a, b, o buffer, op int) error {
	return m.r.Binary(mb(a), mb(b), mb(o), op)
}
func (m mRec) BinaryN(a, b, o buffer, op, n int) error {
	return m.r.BinaryN(mb(a), mb(b), mb(o), op, n)
}
func (m mRec) QMatMulResident(x buffer, w qweight, o buffer, mm int) error {
	switch w := w.(type) {
	case *metal.ResidentQWeight:
		return m.r.QMatMulResident(mb(x), w, mb(o), mm)
	case *metal.ResidentQGroup:
		return m.r.QMatMulResidentGroup(mb(x), w, mb(o), mm)
	default:
		return fmt.Errorf("llamagpu(metal): unsupported resident quant weight %T", w)
	}
}
func (m mRec) Commit() error { return m.r.Commit() }
func (m mRec) Wait() error   { return m.r.Wait() }
func (m mRec) Finish() error { return m.r.Finish() }
func (m mRec) Free()         { m.r.Free() }

func (m mProfileRec) Finish() error {
	if err := m.r.Finish(); err != nil {
		return err
	}
	p, err := m.r.Profile()
	if err == nil {
		*m.profile = p
	}
	return err
}

func (m mProfileRec) Wait() error {
	if err := m.r.Wait(); err != nil {
		return err
	}
	p, err := m.r.Profile()
	if err == nil {
		*m.profile = p
	}
	return err
}

func mb(b buffer) *metal.DeviceBuffer { return b.(mBuf).DeviceBuffer }

// New uploads m's weights into metal device buffers and prepares a KV (key/value attention cache) cache up to m.Config.Ctx
// tokens (the 24× batched decode, §T404).
func New(m *nlp.Llama) (*Decoder, error) {
	if !metal.Available() {
		return nil, fmt.Errorf("llamagpu: no metal GPU")
	}
	return newDecoder(m, backendOps{
		name:        string(backend.Metal),
		asyncEncode: true, // metal command buffers are independent objects (§T614)
		newBuffer: func(data []float32) (buffer, error) {
			b, err := metal.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return mBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := metal.NewRecorder()
			if err != nil {
				return nil, err
			}
			return mRec{r}, nil
		},
		uploadQWeight: metalUploadQWeight,
	})
}

// NewGPT uploads an nlp.GPT's weights into metal device buffers for batched decoding — the
// GPT-2-style sibling of New (§T422).
func NewGPT(m *nlp.GPT) (*GPTDecoder, error) {
	if !metal.Available() {
		return nil, fmt.Errorf("llamagpu: no metal GPU")
	}
	return newGPTDecoder(m, backendOps{
		name: string(backend.Metal),
		newBuffer: func(data []float32) (buffer, error) {
			b, err := metal.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return mBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := metal.NewRecorder()
			if err != nil {
				return nil, err
			}
			return mRec{r}, nil
		},
	})
}

func metalUploadQWeight(weight []byte, qt uint32, n, k int) (qweight, error) {
	rw, err := metal.Backend{}.UploadQuant(weight, qt, n, k)
	if err != nil {
		return nil, err
	}
	return rw.(*metal.ResidentQWeight), nil
}

func metalGroupQWeights(weights []qweight) (qweight, error) {
	if len(weights) != 3 {
		return nil, fmt.Errorf("llamagpu(metal): mixed QKV group needs 3 weights, got %d", len(weights))
	}
	concrete := make([]*metal.ResidentQWeight, 3)
	for i, w := range weights {
		var ok bool
		concrete[i], ok = w.(*metal.ResidentQWeight)
		if !ok {
			return nil, fmt.Errorf("llamagpu(metal): mixed QKV weight %d has type %T", i, w)
		}
	}
	return metal.NewResidentQGroup(concrete...)
}

// NewQuant uploads a quantized Llama (nlp.QuantizeLlama / nlp.QuantLlamaFromGGUF) with every
// projection as a device-resident quantized weight — the 4-8× smaller weights of quantization
// combined with the batched-decode speedup (§T415). Metal.
func NewQuant(m *nlp.QuantLlama) (*Decoder, error) {
	return newQuantMetal(m, false)
}

// NewQuantF16KV is NewQuant with an opt-in IEEE-binary16 K/V cache. It halves retained cache
// storage and uses half-reading Metal attention while Q/O and accumulation remain f32. The initial
// prompt still uses the established f32 FlashMM path as its K/V rows are converted into the cache.
// NewQuant and every non-Metal backend remain unchanged. The current M2-specialized path requires
// head dimension 64 and returns an explicit error for other geometries.
func NewQuantF16KV(m *nlp.QuantLlama) (*Decoder, error) {
	return newQuantMetal(m, true)
}

func newQuantMetal(m *nlp.QuantLlama, f16KV bool) (*Decoder, error) {
	return newQuantMetalWithMixedQKV(m, f16KV, true)
}

// newQuantMetalWithMixedQKV is the production constructor plus a narrow control seam for the
// trained-model promotion campaign. Public constructors always enable the grouped path; tests can
// build the otherwise-identical established separate-projection arm without environment switches.
func newQuantMetalWithMixedQKV(m *nlp.QuantLlama, f16KV, mixedQKV bool) (*Decoder, error) {
	if !metal.Available() {
		return nil, fmt.Errorf("llamagpu: no metal GPU")
	}
	ops := backendOps{
		name:        string(backend.Metal),
		asyncEncode: true, // metal command buffers are independent objects (§T614)
		newBuffer: func(data []float32) (buffer, error) {
			b, err := metal.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return mBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := metal.NewRecorder()
			if err != nil {
				return nil, err
			}
			return mRec{r}, nil
		},
		uploadQWeight: metalUploadQWeight,
	}
	if mixedQKV {
		ops.groupQWeights = metalGroupQWeights
	}
	if f16KV {
		ops.newF16KVBuffer = func(n int) (buffer, error) {
			b, err := metal.NewDeviceBufferF16Zeros(n)
			if err != nil {
				return nil, err
			}
			return mBuf{b}, nil
		}
	}
	return newQuantDecoder(m, ops)
}

// ProfileMetalStep runs one ordinary Decoder.Step through an opt-in Metal timestamp recorder and
// returns both its logits and per-encoder GPU durations. It disables next-step pre-encoding for
// this diagnostic call so exactly one command buffer is attributed, then restores the decoder's
// production recorder factory. Any pending pre-encoded step is discarded; Decoder is already
// documented as not safe for concurrent use.
func (d *Decoder) ProfileMetalStep(token, pos, maxEvents int) ([]float32, metal.RecorderProfile, error) {
	if d.ops.name != string(backend.Metal) {
		return nil, metal.RecorderProfile{}, fmt.Errorf("llamagpu: ProfileMetalStep requires the Metal backend, got %s", d.ops.name)
	}
	if pos < 0 || pos >= d.maxLen {
		return nil, metal.RecorderProfile{}, fmt.Errorf("llamagpu(%s): pos %d out of [0,%d)", d.ops.name, pos, d.maxLen)
	}
	if token < 0 || token >= d.v {
		return nil, metal.RecorderProfile{}, fmt.Errorf("llamagpu(%s): token %d out of vocab %d", d.ops.name, token, d.v)
	}
	if maxEvents < 1 || maxEvents > 2048 {
		return nil, metal.RecorderProfile{}, fmt.Errorf("llamagpu: ProfileMetalStep maxEvents=%d outside [1,2048]", maxEvents)
	}
	d.dropPending()
	oldNewRecorder, oldAsyncEncode := d.ops.newRecorder, d.ops.asyncEncode
	defer func() {
		d.ops.newRecorder = oldNewRecorder
		d.ops.asyncEncode = oldAsyncEncode
	}()
	var profile metal.RecorderProfile
	d.ops.newRecorder = func() (recorder, error) {
		r, err := metal.NewProfilingRecorder(maxEvents)
		if err != nil {
			return nil, err
		}
		return mProfileRec{mRec: mRec{r: r}, profile: &profile}, nil
	}
	d.ops.asyncEncode = false
	logits, err := d.Step(token, pos)
	if err != nil {
		return nil, metal.RecorderProfile{}, err
	}
	return logits, profile, nil
}
