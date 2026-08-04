//go:build cuda && cgo && (linux || windows)

package llamagpu

import (
	"fmt"
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

	dx, dh, dh2        *cuda.DeviceF32
	dq, dk, dv, da     *cuda.DeviceF32
	dgate, dup, logits *cuda.DeviceF32
	inv                *cuda.DeviceF32
	graph              *cuda.CapturedGraph

	heads, kv, vocab, maxLen int
	eps                      float64
}

type graphLlamaLayer struct {
	gAttn, gFFN    *cuda.ResidentVec
	wq, wk, wv, wo *cuda.ResidentBQ4K
	wg, wu, wd     *cuda.ResidentBQ4K
	cache          *cuda.KVCache
}

// NewLlamaQ4KGraphCUDA builds a graph-captured Q4_K decode path for a Llama model. Every projection
// K must be a multiple of 256 (Q4_K super-block); typical transformer dims satisfy it. maxLen caps
// prompt+generation. GQA (KVHeads<Heads) is supported.
func NewLlamaQ4KGraphCUDA(m *nlp.Llama, maxLen int) (*GraphLlamaDecoder, error) {
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
	gd := &GraphLlamaDecoder{heads: cfg.Heads, kv: kv, vocab: cfg.Vocab, maxLen: maxLen, eps: cfg.Eps}
	// A failure part-way must not leak the device buffers already allocated.
	ok := false
	defer func() {
		if !ok {
			gd.Release()
		}
	}()
	var err error
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
	gd.dq, gd.da = buf(wqW), buf(wqW)
	gd.dk, gd.dv = buf(wkv), buf(wkv)
	gd.dgate, gd.dup = buf(cfg.Hidden), buf(cfg.Hidden)
	gd.logits = buf(cfg.Vocab)
	if err != nil {
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
		if l.cache, err = cuda.NewKVCache(maxLen, wkv); err != nil {
			return nil, err
		}
		if err = l.cache.ZeroCache(); err != nil {
			return nil, err
		}
		l.cache.SetLen(maxLen)
		for _, wp := range []struct {
			dst **cuda.ResidentBQ4K
			w   *tensor.Tensor
		}{
			{&l.wq, blk.Wq}, {&l.wk, blk.Wk}, {&l.wv, blk.Wv}, {&l.wo, blk.Wo},
			{&l.wg, blk.FFN.Wgate}, {&l.wu, blk.FFN.Wup}, {&l.wd, blk.FFN.Wdown},
		} {
			if *wp.dst, err = q(wp.w); err != nil {
				return nil, err
			}
		}
		gd.layers[i] = l
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
		if err := l.wq.QMatMulInto(gd.dh, gd.dq); err != nil {
			return err
		}
		if err := l.wk.QMatMulInto(gd.dh, gd.dk); err != nil {
			return err
		}
		if err := l.wv.QMatMulInto(gd.dh, gd.dv); err != nil {
			return err
		}
		if err := gd.dq.RoPEDposInv(gd.heads, gd.inv, gd.pos, 0); err != nil {
			return err
		}
		if err := gd.dk.RoPEDposInv(gd.kv, gd.inv, gd.pos, 0); err != nil {
			return err
		}
		if err := l.cache.AppendDpos(gd.dk, gd.dv, gd.pos); err != nil {
			return err
		}
		kF, vF := l.cache.FullView()
		if err := cuda.GroupedQueryAttentionKVDposFlashInto(gd.dq, kF, vF, gd.heads, gd.kv, gd.pos, gd.da); err != nil {
			return err
		}
		if err := l.wo.QMatMulAccInto(gd.da, gd.dx); err != nil {
			return err
		}
		if err := gd.dx.RMSNormInto(l.gFFN, float32(gd.eps), gd.dh2); err != nil {
			return err
		}
		if err := l.wg.QMatMulInto(gd.dh2, gd.dgate); err != nil {
			return err
		}
		if err := l.wu.QMatMulInto(gd.dh2, gd.dup); err != nil {
			return err
		}
		if err := gd.dgate.SwiGLU(gd.dup); err != nil {
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

// stepGraph replays the captured decode graph for one token at pos, returning host logits.
func (gd *GraphLlamaDecoder) stepGraph(token, pos int) ([]float32, error) {
	if err := gd.pos.Set(pos); err != nil {
		return nil, err
	}
	if err := gd.emb.EmbedInto([]int32{int32(token)}, gd.dx); err != nil {
		return nil, err
	}
	if err := gd.graph.Launch(); err != nil {
		return nil, err
	}
	if err := cuda.GraphSync(); err != nil {
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
	var logits []float32
	for i, tok := range prompt {
		l, err := gd.stepEager(tok, i)
		if err != nil {
			return nil, err
		}
		logits = l
	}
	if err := gd.captureGraph(); err != nil {
		return nil, err
	}
	pos := len(prompt)
	buf := make([]float64, gd.vocab)
	for range maxNew {
		if pos >= gd.maxLen {
			break
		}
		for i, x := range logits {
			buf[i] = float64(x)
		}
		next := s.SampleWithHistory(buf, out)
		out = append(out, next)
		l, err := gd.stepGraph(next, pos)
		if err != nil {
			return nil, err
		}
		logits = l
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
	for i, tok := range prompt {
		if err := gd.pos.Set(i); err != nil {
			return nil, err
		}
		if err := gd.emb.EmbedInto([]int32{int32(tok)}, gd.dx); err != nil {
			return nil, err
		}
		if err := gd.forwardBody(); err != nil {
			return nil, err
		}
	}
	if err := gd.captureGraph(); err != nil {
		return nil, err
	}
	pos := len(prompt)
	for range maxNew {
		if pos >= gd.maxLen {
			break
		}
		next := gd.logits.Argmax() // GPU argmax over the current logits (device-resident)
		out = append(out, next)
		if pos+1 >= gd.maxLen {
			break
		}
		if err := gd.pos.Set(pos); err != nil {
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
	for _, d := range []*cuda.DeviceF32{gd.dx, gd.dh, gd.dh2, gd.dq, gd.dk, gd.dv, gd.da, gd.dgate, gd.dup, gd.logits, gd.inv} {
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
		for _, w := range []*cuda.ResidentBQ4K{l.wq, l.wk, l.wv, l.wo, l.wg, l.wu, l.wd} {
			if w != nil {
				w.Free()
			}
		}
	}
	gd.layers = nil
}
