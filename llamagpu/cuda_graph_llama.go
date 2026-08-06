//go:build cuda && cgo && (linux || windows)

package llamagpu

import (
	"fmt"
	"math"
	"runtime"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// GraphLlamaDecoder is a CUDA-graph-captured decode path for a Llama-family model with native Q4_K
// projections — the production form of the q4kGraphDecoder proof-of-concept (backend/cuda,
// TestCUDAQ4KGraphDecodeSpeed = 253.5 tok/s, beating llama.cpp Vulkan Q8's 244). The per-token decode
// op chain (device-pos RoPE / KV-append / flash attention, so pos lives in a device buffer) is
// captured into ONE CUDA graph and replayed per token, updating only the pos buffer and the token
// embedding — eliminating the per-token kernel-launch overhead that bounds the eager NewQuantCUDA path
// on small models.
//
// This is an ADDITIVE, opt-in path (Llama + Q4_K only); it does not touch the generic backend-agnostic
// Decoder. Prefill is eager per-token (fine for short prompts / long generations); decode is the graph.
// Mamba/RWKV-style stateful models are out of scope (their per-step state mutation is not graph-safe).
type GraphLlamaDecoder struct {
	emb    *cuda.ResidentB
	norm   *cuda.ResidentVec
	out    *cuda.ResidentBQ4K
	layers []*graphLlamaLayer
	pos    *cuda.DevicePos

	dx, dh, dh2             *cuda.DeviceF32
	dqkv, dq, dk, dv, da    *cuda.DeviceF32 // dq/dk/dv are Views into the fused q|k|v buffer dqkv
	dgu, dgate, dup, logits *cuda.DeviceF32 // dgate/dup are Views into the fused gate|up buffer dgu
	inv                     *cuda.DeviceF32
	graph                   *cuda.CapturedGraph

	rec   *cuda.Recorder
	scale float32

	heads, kv, hd, dim, hidden, vocab, maxLen int
	eps                                       float64
	useF16KV                                  bool // opt-in: store the decode KV cache in f16 (llama.cpp default) — half the KV bytes
}

type graphLlamaLayer struct {
	gAttn, gFFN *cuda.ResidentVec
	wqkv, wo    *cuda.ResidentBQ4K // wqkv = col-fused attn_q|attn_k|attn_v (one N=(heads+2·kv)·hd GEMV → RoPE/append over slices)
	wgu, wd     *cuda.ResidentBQ4K // wgu = col-fused ffn_gate|ffn_up (one N=2·hidden GEMV → SwiGLU over halves)
	cache       *cuda.KVCache
	cacheF16    *cuda.KVCacheF16 // set instead of cache when useF16KV (f16 storage, flash-only attend)
}

// NewLlamaQ4KGraphCUDA builds a graph-captured Q4_K decode path for a Llama model. Every projection
// K must be a multiple of 256 (Q4_K super-block); typical transformer dims satisfy it. maxLen caps
// prompt+generation. GQA (KVHeads<Heads) is supported.
func NewLlamaQ4KGraphCUDA(m *nlp.Llama, maxLen int) (*GraphLlamaDecoder, error) {
	return newLlamaQ4KGraph(m, maxLen, false)
}

// NewLlamaQ4KGraphCUDAF16 is NewLlamaQ4KGraphCUDA with the decode KV cache stored in f16 (llama.cpp's
// default) — half the K/V VRAM (⇒ ~2× the max context/batch that fits) and the flash decode kernel
// reads half the bytes (bandwidth-bound at long context). Compute stays f32 (tiles h2f in shared);
// only KV STORAGE rounds to f16 (~2^-11 rel), the incumbent tolerance. Best for long-context / large
// models; short-context decode is weight-bandwidth-bound so it's ~neutral there.
func NewLlamaQ4KGraphCUDAF16(m *nlp.Llama, maxLen int) (*GraphLlamaDecoder, error) {
	return newLlamaQ4KGraph(m, maxLen, true)
}

func newLlamaQ4KGraph(m *nlp.Llama, maxLen int, useF16KV bool) (*GraphLlamaDecoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	cfg := m.Config
	if maxLen <= 0 || maxLen > cfg.Ctx {
		maxLen = cfg.Ctx
	}
	kv := cfg.KVHeads
	if kv == 0 {
		kv = cfg.Heads
	}
	hd := cfg.Dim / cfg.Heads
	wqW, wkv := cfg.Heads*hd, kv*hd
	cast := func(t *tensor.Tensor) *tensor.Tensor { return t.Cast(tensor.F32) }
	q := func(w *tensor.Tensor) (*cuda.ResidentBQ4K, error) {
		qi, err := cudaQ4KResident(cast(w))
		if err != nil {
			return nil, err
		}
		r, ok := qi.(*cuda.ResidentBQ4K)
		if !ok {
			return nil, fmt.Errorf("llamagpu: cudaQ4KResident returned %T, want *cuda.ResidentBQ4K", qi)
		}
		return r, nil
	}
	gd := &GraphLlamaDecoder{
		heads: cfg.Heads, kv: kv, hd: hd, dim: cfg.Dim, hidden: cfg.Hidden,
		vocab: cfg.Vocab, maxLen: maxLen, eps: cfg.Eps,
		scale:    float32(1.0 / math.Sqrt(float64(hd))),
		useF16KV: useF16KV,
	}
	// A failure part-way must not leak the device buffers already allocated.
	ok := false
	defer func() {
		if !ok {
			gd.Release()
		}
	}()
	var err error
	if gd.rec, err = cuda.NewRecorder(); err != nil {
		return nil, err
	}
	if gd.emb, err = cuda.NewResidentB(cast(m.TokEmb)); err != nil {
		return nil, err
	}
	if gd.norm, err = cuda.NewResidentVec(cast(m.Norm.Gamma)); err != nil {
		return nil, err
	}
	if gd.out, err = q(m.Out); err != nil {
		return nil, err
	}
	if gd.pos, err = cuda.NewDevicePos(); err != nil {
		return nil, err
	}
	buf := func(c int) *cuda.DeviceF32 {
		if err != nil {
			return nil
		}
		var d *cuda.DeviceF32
		d, err = cuda.NewDeviceF32(1, c)
		return d
	}
	gd.dx, gd.dh, gd.dh2 = buf(cfg.Dim), buf(cfg.Dim), buf(cfg.Dim)
	gd.da = buf(wqW)
	gd.dqkv = buf(wqW + 2*wkv)   // fused q|k|v output; dq/dk/dv View its three slices (q=heads·hd, k/v=kv·hd)
	gd.dgu = buf(2 * cfg.Hidden) // fused gate|up output; dgate/dup View its two halves
	gd.logits = buf(cfg.Vocab)
	if err != nil {
		return nil, err
	}
	if gd.dq, err = gd.dqkv.View(0, 1, wqW); err != nil {
		return nil, err
	}
	if gd.dk, err = gd.dqkv.View(wqW, 1, wkv); err != nil {
		return nil, err
	}
	if gd.dv, err = gd.dqkv.View(wqW+wkv, 1, wkv); err != nil {
		return nil, err
	}
	if gd.dgate, err = gd.dgu.View(0, 1, cfg.Hidden); err != nil {
		return nil, err
	}
	if gd.dup, err = gd.dgu.View(cfg.Hidden, 1, cfg.Hidden); err != nil {
		return nil, err
	}
	if gd.inv, err = cuda.BuildRoPEInv(hd, cfg.RopeBase); err != nil {
		return nil, err
	}
	gd.layers = make([]*graphLlamaLayer, len(m.Blocks))
	for i, blk := range m.Blocks {
		l := &graphLlamaLayer{}
		if l.gAttn, err = cuda.NewResidentVec(cast(blk.AttnNorm.Gamma)); err != nil {
			return nil, err
		}
		if l.gFFN, err = cuda.NewResidentVec(cast(blk.FFNNorm.Gamma)); err != nil {
			return nil, err
		}
		if useF16KV {
			if l.cacheF16, err = cuda.NewKVCacheF16(maxLen, wkv); err != nil {
				return nil, err
			}
			if err = l.cacheF16.ZeroCache(); err != nil { // ZeroCache (unlike int8's omission) — clean pool state
				return nil, err
			}
		} else {
			if l.cache, err = cuda.NewKVCache(maxLen, wkv); err != nil {
				return nil, err
			}
			if err = l.cache.ZeroCache(); err != nil {
				return nil, err
			}
			l.cache.SetLen(maxLen)
		}
		for _, wp := range []struct {
			dst **cuda.ResidentBQ4K
			w   *tensor.Tensor
		}{
			{&l.wo, blk.Wo}, {&l.wd, blk.FFN.Wdown},
		} {
			if *wp.dst, err = q(wp.w); err != nil {
				return nil, err
			}
		}
		// Column-fuse attn_q|attn_k|attn_v (and ffn_gate|ffn_up) into ONE Q4_K projection each. Q4_K
		// quantizes each output row's K independently, so concatenating along the output dimension leaves
		// every row's bytes identical to encoding the projections separately — the fused GEMV is bit-exact,
		// only the launch (and weight-stream setup) is merged. Decode does one N=(heads+2·kv)·hd GEMV then
		// RoPE/KV-append over the q/k/v slices, and one N=2·hidden GEMV then SwiGLU over the gate/up halves.
		if l.wqkv, err = q(concatCols1F32(cast(blk.Wq), cast(blk.Wk), cast(blk.Wv))); err != nil {
			return nil, err
		}
		if l.wgu, err = q(concatCols1F32(cast(blk.FFN.Wgate), cast(blk.FFN.Wup))); err != nil {
			return nil, err
		}
		gd.layers[i] = l
	}
	// Warm lazy device allocations (notably the flash-decode split-K scratch, allocated on first use)
	// with one throwaway eager forwardBody, so a later CUDA-graph capture never cudaMallocs DURING
	// capture — which fails. Inputs are uninitialized (the logits are discarded); the KV caches touched
	// at pos 0 are re-zeroed. Without this, Generate/GenerateGreedy fail when they are the first flash
	// user (masked in the test suite only because an earlier eager test warms the scratch first).
	if err = gd.pos.Set(0); err != nil {
		return nil, err
	}
	if err = gd.forwardBody(); err != nil {
		return nil, err
	}
	for _, l := range gd.layers { // the warmup wrote garbage into the cache — re-zero before real use
		var ze error
		if gd.useF16KV {
			ze = l.cacheF16.ZeroCache()
		} else {
			ze = l.cache.ZeroCache()
		}
		if ze != nil {
			return nil, ze
		}
	}
	ok = true
	return gd, nil
}

// forwardBody records/executes one decode step over the current dx (embedding) and pos buffer.
func (gd *GraphLlamaDecoder) forwardBody() error {
	for _, l := range gd.layers {
		if err := gd.dx.RMSNormInto(l.gAttn, float32(gd.eps), gd.dh); err != nil {
			return err
		}
		if err := l.wqkv.QMatMulInto(gd.dh, gd.dqkv); err != nil { // one N=(heads+2·kv)·hd GEMV → [q|k|v]
			return err
		}
		if err := gd.dq.RoPEDposInv(gd.heads, gd.inv, gd.pos, 0); err != nil {
			return err
		}
		if err := gd.dk.RoPEDposInv(gd.kv, gd.inv, gd.pos, 0); err != nil {
			return err
		}
		if gd.useF16KV {
			if err := l.cacheF16.AppendDpos(gd.dk, gd.dv, gd.pos); err != nil {
				return err
			}
			if err := cuda.GroupedQueryAttentionKVF16DposFlashInto(gd.dq, l.cacheF16, gd.heads, gd.kv, gd.pos, gd.da); err != nil {
				return err
			}
		} else {
			if err := l.cache.AppendDpos(gd.dk, gd.dv, gd.pos); err != nil {
				return err
			}
			kF, vF := l.cache.FullView()
			if err := cuda.GroupedQueryAttentionKVDposFlashInto(gd.dq, kF, vF, gd.heads, gd.kv, gd.pos, gd.da); err != nil {
				return err
			}
		}
		if err := l.wo.QMatMulAccInto(gd.da, gd.dx); err != nil {
			return err
		}
		if err := gd.dx.RMSNormInto(l.gFFN, float32(gd.eps), gd.dh2); err != nil {
			return err
		}
		if err := l.wgu.QMatMulInto(gd.dh2, gd.dgu); err != nil { // one N=2·hidden GEMV → [gate|up]
			return err
		}
		if err := gd.dgate.SwiGLU(gd.dup); err != nil { // SwiGLU over the two halves of dgu, in place into dgate
			return err
		}
		if err := l.wd.QMatMulAccInto(gd.dgate, gd.dx); err != nil {
			return err
		}
	}
	if err := gd.dx.RMSNormInto(gd.norm, float32(gd.eps), gd.dh); err != nil {
		return err
	}
	return gd.out.QMatMulInto(gd.dh, gd.logits)
}

// prefillForward processes ALL prompt tokens (M = len(tokens)) in ONE batched pass — weight-read-once
// WMMA matmuls (QMatMulInto routes M>=48 to the tensor-core prefill GEMM) + a single causal GQA
// attention per layer, instead of M eager decode steps that re-read every weight M times. It writes
// the prompt's RoPE'd K and V into each layer's KV cache (rows 0..M-1) so the subsequent graph decode
// attends to them, and returns the LAST token's logits (host) — the logits for the first generated
// token. M-sized scratch is allocated here and freed on return.
func (gd *GraphLlamaDecoder) prefillForward(tokens []int) ([]float32, error) {
	m := len(tokens)
	ids := make([]int32, m)
	for i, t := range tokens {
		ids[i] = int32(t)
	}
	var err error
	nb := func(cols int) *cuda.DeviceF32 {
		if err != nil {
			return nil
		}
		var d *cuda.DeviceF32
		d, err = cuda.NewDeviceF32(m, cols)
		return d
	}
	dxM, dhM, dh2M := nb(gd.dim), nb(gd.dim), nb(gd.dim)
	dqM, daM := nb(gd.heads*gd.hd), nb(gd.heads*gd.hd)
	wkv := gd.kv * gd.hd
	dkM, dvM := nb(wkv), nb(wkv)
	dgM, duM := nb(gd.hidden), nb(gd.hidden)
	dguM := nb(2 * gd.hidden)           // fused gate|up prefill output [m,2·hidden]; split per-row into dgM/duM
	dqkvM := nb(gd.heads*gd.hd + 2*wkv) // fused q|k|v prefill output [m,(heads+2·kv)·hd]; split into dqM/dkM/dvM
	dlM := nb(gd.vocab)
	bufs := []*cuda.DeviceF32{dxM, dhM, dh2M, dqM, daM, dkM, dvM, dgM, duM, dguM, dqkvM, dlM}
	defer func() {
		for _, b := range bufs {
			if b != nil {
				b.Free()
			}
		}
	}()
	if err != nil {
		return nil, err
	}
	if err = gd.emb.EmbedInto(ids, dxM); err != nil {
		return nil, err
	}
	for _, l := range gd.layers {
		if err = dxM.RMSNormInto(l.gAttn, float32(gd.eps), dhM); err != nil {
			return nil, err
		}
		// WMMA tensor-core prefill (#877) via the recorder — ~2.4x the scalar MT that QMatMulInto routes
		// M>1 to. f16-accum; the eager reference (generateEager in the test) shares this same
		// prefillForward, so the graph-vs-eager oracle stays exact regardless.
		// One fused q|k|v GEMM → [m,(heads+2·kv)·hd], then split each row's slices (columns
		// [0,q)=q, [q,q+kv)=k, [q+kv,q+2kv)=v) into the contiguous [m,·] q/k/v buffers the RoPE/MHA
		// path expects. Prefill is weight-read-once, so the extra strided copies are cheap.
		qN, totN := gd.heads*gd.hd, gd.heads*gd.hd+2*wkv
		if err = gd.rec.QMatMulResidentQ4K(dhM, l.wqkv, dqkvM, m); err != nil {
			return nil, err
		}
		if err = gd.rec.Copy2D(dqkvM, 0, totN, dqM, 0, qN, m, qN); err != nil {
			return nil, err
		}
		if err = gd.rec.Copy2D(dqkvM, qN, totN, dkM, 0, wkv, m, wkv); err != nil {
			return nil, err
		}
		if err = gd.rec.Copy2D(dqkvM, qN+wkv, totN, dvM, 0, wkv, m, wkv); err != nil {
			return nil, err
		}
		// RoPE the M query/key rows at positions 0..M-1 (posDiv=1, the RoPEDposInv default the decode
		// path uses — so the prompt K cached here rotates identically to decode-appended K).
		if err = dqM.RoPEAtBand(gd.inv, 0, gd.heads, gd.hd, 0, 1, gd.heads*gd.hd); err != nil {
			return nil, err
		}
		if err = dkM.RoPEAtBand(gd.inv, 0, gd.kv, gd.hd, 0, 1, wkv); err != nil {
			return nil, err
		}
		// causal GQA attention over the prompt's own K/V (no cache needed for prefill attention).
		if err = gd.rec.MHA(dqM, dkM, dvM, daM, m, m, gd.heads*gd.hd, gd.heads, gd.kv, gd.hd, 1, 0, gd.scale); err != nil {
			return nil, err
		}
		// PREFILL: route the residual-add o/down projections to the tensor-core WMMA acc (dequant→cuBLAS
		// f16 GEMM, beta=1) — ~3× the scalar-MT that QMatMulAccInto uses (MT has no WMMA path, so wo/wd
		// were silently stuck on it while wqkv/wgu already used WMMA, making graph prefill 3× slower than
		// the eager Decoder). Matches what the eager Decoder does; f16-accum incumbent tolerance. m<48
		// (tiny prompts) keeps the bit-exact MT path.
		if m >= 48 {
			err = l.wo.QMatMulWMMAAccInto(daM, dxM, m)
		} else {
			err = l.wo.QMatMulAccInto(daM, dxM)
		}
		if err != nil {
			return nil, err
		}
		// populate the decode KV cache with the prompt's RoPE'd K and un-rotated V (rows 0..M-1).
		if gd.useF16KV {
			if err = l.cacheF16.PrefillKV(dkM, dvM, m); err != nil { // convert the m prompt K/V rows to f16
				return nil, err
			}
		} else {
			if err = gd.rec.Blit(dkM, 0, l.cache.K(), 0, m*wkv); err != nil {
				return nil, err
			}
			if err = gd.rec.Blit(dvM, 0, l.cache.V(), 0, m*wkv); err != nil {
				return nil, err
			}
		}
		if err = dxM.RMSNormInto(l.gFFN, float32(gd.eps), dh2M); err != nil {
			return nil, err
		}
		// One fused gate|up GEMM → [m,2·hidden], then split each row's halves (columns [0,h)=gate,
		// [h,2h)=up) into the contiguous [m,h] SwiGLU inputs. Prefill is batched (weight-read-once),
		// so the extra strided copies are cheap; the win is fewer weight-stream setups + one GEMM.
		if err = gd.rec.QMatMulResidentQ4K(dh2M, l.wgu, dguM, m); err != nil {
			return nil, err
		}
		if err = gd.rec.Copy2D(dguM, 0, 2*gd.hidden, dgM, 0, gd.hidden, m, gd.hidden); err != nil {
			return nil, err
		}
		if err = gd.rec.Copy2D(dguM, gd.hidden, 2*gd.hidden, duM, 0, gd.hidden, m, gd.hidden); err != nil {
			return nil, err
		}
		if err = dgM.SwiGLU(duM); err != nil {
			return nil, err
		}
		if m >= 48 { // PREFILL: tensor-core WMMA acc for the down projection too (see wo above)
			err = l.wd.QMatMulWMMAAccInto(dgM, dxM, m)
		} else {
			err = l.wd.QMatMulAccInto(dgM, dxM)
		}
		if err != nil {
			return nil, err
		}
	}
	if err = dxM.RMSNormInto(gd.norm, float32(gd.eps), dhM); err != nil {
		return nil, err
	}
	if err = gd.rec.QMatMulResidentQ4K(dhM, gd.out, dlM, m); err != nil {
		return nil, err
	}
	if err = gd.rec.Wait(); err != nil {
		return nil, err
	}
	full, err := dlM.ToHost()
	if err != nil {
		return nil, err
	}
	fs := full.Storage().F32()
	return fs[(m-1)*gd.vocab:], nil // last row = logits for the first generated token
}

// stepEager runs one full decode step (embedding token at pos), executing eagerly, and returns the
// host logits — used for prefill and as the fallback before the graph is captured.
func (gd *GraphLlamaDecoder) stepEager(token, pos int) ([]float32, error) {
	if err := gd.pos.Set(pos); err != nil {
		return nil, err
	}
	if err := gd.emb.EmbedInto([]int32{int32(token)}, gd.dx); err != nil {
		return nil, err
	}
	if err := gd.forwardBody(); err != nil {
		return nil, err
	}
	l, err := gd.logits.ToHost()
	if err != nil {
		return nil, err
	}
	return l.Storage().F32(), nil
}

// captureGraph records the decode op chain into a replayable CUDA graph (records only — no execution;
// pos/embedding are read from device buffers at Launch time). Idempotent.
func (gd *GraphLlamaDecoder) captureGraph() error {
	if gd.graph != nil {
		return nil
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := cuda.CaptureBegin(); err != nil {
		return err
	}
	if err := gd.forwardBody(); err != nil {
		return err
	}
	g, err := cuda.CaptureEnd()
	if err != nil {
		return err
	}
	gd.graph = g
	return nil
}

// stepGraphLaunch replays the captured decode graph for one token at pos, leaving the result in the
// device logits buffer (gd.logits) — no host copy. The greedy/top-k sampling paths consume gd.logits
// on device (Argmax / TopK); only the flexible CPU-sampler fallback copies the whole vector to host.
func (gd *GraphLlamaDecoder) stepGraphLaunch(token, pos int) error {
	if err := gd.pos.Set(pos); err != nil {
		return err
	}
	if err := gd.emb.EmbedInto([]int32{int32(token)}, gd.dx); err != nil {
		return err
	}
	if err := gd.graph.Launch(); err != nil {
		return err
	}
	return cuda.GraphSync()
}

// stepGraph replays the captured decode graph for one token at pos, returning host logits.
func (gd *GraphLlamaDecoder) stepGraph(token, pos int) ([]float32, error) {
	if err := gd.stepGraphLaunch(token, pos); err != nil {
		return nil, err
	}
	l, err := gd.logits.ToHost()
	if err != nil {
		return nil, err
	}
	return l.Storage().F32(), nil
}

// Generate decodes maxNew tokens after the prompt with the given sampler. Prompt tokens are prefilled
// eagerly (per token); generated tokens are decoded by replaying the captured graph.
func (gd *GraphLlamaDecoder) Generate(prompt []int, maxNew int, s nlp.TokenSampler) ([]int, error) {
	if len(prompt) == 0 {
		return nil, fmt.Errorf("llamagpu: GraphLlamaDecoder.Generate needs a non-empty prompt")
	}
	if len(prompt) >= gd.maxLen {
		return nil, fmt.Errorf("llamagpu: prompt (%d) >= maxLen (%d)", len(prompt), gd.maxLen)
	}
	out := append([]int(nil), prompt...)
	logits, err := gd.prefillForward(prompt) // host logits for the first token
	if err != nil {
		return nil, err
	}
	if err := gd.captureGraph(); err != nil {
		return nil, err
	}
	// On-device top-k sampling fast-path (§topk): for a penalty-free TopK sampler the whole-vocab D2H +
	// CPU sort per token collapses to a GPU TopK(K) (K·8 bytes out) + a CPU draw over K candidates —
	// bit-identical to the full-vocab sampler (K⊇top-k; candidates fed in vocab-index order so the draw
	// order and rng match). sp==nil ⟹ the flexible CPU-sampler fallback (full ToHost).
	sp, fastK := fastTopKSampler(s, gd.vocab)
	spP, fastC := fastTopPSampler(s, gd.vocab) // pure-top-p device path (mutually exclusive with sp: it needs top-k off)
	fast := sp != nil || spP != nil            // either device fast-path avoids the whole-vocab D2H + host sort per token
	pos := len(prompt)
	buf := make([]float64, gd.vocab)
	for range maxNew {
		if pos >= gd.maxLen {
			break
		}
		var next int
		if fast && logits == nil { // decode step, logits device-resident — sample on-device
			if sp != nil { // top-k (± post-shrink top-p/min-p): draw over the K-candidate superset
				gi, gv, e := gd.logits.TopK(fastK)
				if e != nil {
					return nil, e
				}
				next = sampleTopKCandidates(sp, gi, gv)
			} else { // pure top-p: resolve the nucleus over the top-C candidates using full-vocab stats
				gi, gv, e := gd.logits.TopK(fastC)
				if e != nil {
					return nil, e
				}
				maxL, zexp, e2 := gd.logits.SoftmaxStatsN(gd.vocab, spP.Temperature)
				if e2 != nil {
					return nil, e2
				}
				cl := make([]float64, len(gv))
				for i, v := range gv {
					cl[i] = float64(v)
				}
				if tok, ok := spP.SampleTopPFromCandidates(cl, gi, maxL, zexp); ok {
					next = tok
				} else { // nucleus exceeded the C candidates → fall back to the full-vocab host sample
					l, e3 := gd.logits.ToHost()
					if e3 != nil {
						return nil, e3
					}
					for i, x := range l.Storage().F32() {
						buf[i] = float64(x)
					}
					next = s.SampleWithHistory(buf, out)
				}
			}
		} else { // first token (prefill host logits) or CPU-sampler fallback
			for i, x := range logits {
				buf[i] = float64(x)
			}
			next = s.SampleWithHistory(buf, out)
		}
		out = append(out, next)
		if fast {
			if e := gd.stepGraphLaunch(next, pos); e != nil { // no host copy — next iter reads gd.logits
				return nil, e
			}
			logits = nil
		} else {
			l, e := gd.stepGraph(next, pos)
			if e != nil {
				return nil, e
			}
			logits = l
		}
		pos++
	}
	return out, nil
}

// GenerateGreedy decodes maxNew tokens by argmax, doing the argmax ON THE GPU (cu_argmax_f32) so each
// decode step transfers ONE int back to the host instead of the whole vocab-wide logit row — removing
// the per-token device→host copy + conversion that dominates the sampler path at large vocab. This is
// the throughput path (matches llama-bench's greedy token-generation); use Generate for arbitrary
// samplers. Prefill stays eager; decode replays the captured graph.
func (gd *GraphLlamaDecoder) GenerateGreedy(prompt []int, maxNew int) ([]int, error) {
	if len(prompt) == 0 {
		return nil, fmt.Errorf("llamagpu: GraphLlamaDecoder.GenerateGreedy needs a non-empty prompt")
	}
	if len(prompt) >= gd.maxLen {
		return nil, fmt.Errorf("llamagpu: prompt (%d) >= maxLen (%d)", len(prompt), gd.maxLen)
	}
	out := append([]int(nil), prompt...)
	logits, err := gd.prefillForward(prompt) // batched prefill → host logits for the first token
	if err != nil {
		return nil, err
	}
	if err := gd.captureGraph(); err != nil {
		return nil, err
	}
	next, best := 0, float32(-1e38)
	for i, x := range logits {
		if x > best {
			best, next = x, i
		}
	}
	pos := len(prompt)
	for range maxNew {
		out = append(out, next)
		if pos+1 >= gd.maxLen {
			break
		}
		if err := gd.pos.Set(pos); err != nil { // process `next` at pos → gd.logits for the following token
			return nil, err
		}
		if err := gd.emb.EmbedInto([]int32{int32(next)}, gd.dx); err != nil {
			return nil, err
		}
		if err := gd.graph.Launch(); err != nil {
			return nil, err
		}
		if err := cuda.GraphSync(); err != nil {
			return nil, err
		}
		next = gd.logits.Argmax() // GPU argmax, 1-int transfer
		pos++
	}
	return out, nil
}

// Release frees all device resources.
func (gd *GraphLlamaDecoder) Release() {
	if gd == nil {
		return
	}
	free := func(d *cuda.DeviceF32) {
		if d != nil {
			d.Free()
		}
	}
	if gd.emb != nil {
		gd.emb.Free()
	}
	if gd.norm != nil {
		gd.norm.Free()
	}
	if gd.out != nil {
		gd.out.Free()
	}
	if gd.pos != nil {
		gd.pos.Free()
	}
	if gd.rec != nil {
		gd.rec.Free()
	}
	// dgate/dup are non-owning Views into dgu (their Free is a no-op); dgu owns the buffer.
	for _, d := range []*cuda.DeviceF32{gd.dx, gd.dh, gd.dh2, gd.dqkv, gd.dq, gd.dk, gd.dv, gd.da, gd.dgu, gd.dgate, gd.dup, gd.logits, gd.inv} {
		free(d)
	}
	if gd.graph != nil {
		gd.graph.Free()
	}
	for _, l := range gd.layers {
		if l == nil {
			continue
		}
		if l.gAttn != nil {
			l.gAttn.Free()
		}
		if l.gFFN != nil {
			l.gFFN.Free()
		}
		if l.cache != nil {
			l.cache.Free()
		}
		if l.cacheF16 != nil {
			l.cacheF16.Free()
		}
		for _, w := range []*cuda.ResidentBQ4K{l.wqkv, l.wo, l.wgu, l.wd} {
			if w != nil {
				w.Free()
			}
		}
	}
	gd.layers = nil
}

// concatCols1F32 concatenates [K,Ni] f32 weight tensors (the [in,out] layout cudaQ4KResident wants,
// K=contracted dim on axis 0, output width on axis 1) along axis 1 into one [K,ΣNi] tensor — the
// output-fusion that merges ffn_gate|ffn_up (and could merge q|k|v) into a single Q4_K projection.
// cudaQ4KResident transposes to [N,K] and quantizes each output row's K independently, so the fused
// weight's output row n<N0 is input 0's output n (its same K values) — bit-exact vs the separate
// projections; only the GEMV launch and weight-stream setup are merged.
func concatCols1F32(ts ...*tensor.Tensor) *tensor.Tensor {
	rows := ts[0].Shape()[0]
	widths := make([]int, len(ts))
	srcs := make([][]float32, len(ts))
	ntot := 0
	for i, t := range ts {
		widths[i] = t.Shape()[1]
		ntot += widths[i]
		srcs[i] = t.Cast(tensor.F32).Contiguous().Storage().F32()
	}
	out := tensor.New(tensor.F32, tensor.Shape{rows, ntot})
	of := out.Storage().F32()
	for r := 0; r < rows; r++ {
		off := r * ntot
		for i := range ts {
			w := widths[i]
			copy(of[off:off+w], srcs[i][r*w:r*w+w])
			off += w
		}
	}
	return out
}
