//go:build cuda && cgo && (linux || windows)

package llamagpu

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// BatchedDecoderF16 is a continuous-batching f16 decoder. Every Decode call advances the WHOLE active
// batch by one token in a single forward, with each concurrent sequence at its OWN position (ragged) —
// the regime that turns bandwidth-bound single-stream decode into high aggregate serving throughput
// (measured ~30× at batch 32 vs batch 1 on GA106; BenchmarkBatchedDecodeThroughput). Sequences may be
// admitted or dropped between steps, so the batch composition can change every call. Per-sequence
// positions ride DeviceF16.RoPERagged; the ragged attention reads each sequence's own paged KV; greedy
// sampling is the on-device BatchArgmax (only B ints cross back to the host).
//
// This is the f16-resident B4 forward. Correctness is covered by TestBatchedDecodeRaggedPositions
// (ragged, mid-stream admit) and TestA1RealGenerateBatched (synchronous, seq-0 == batch-1 reference).
type BatchedDecoderF16 struct {
	dim, heads, kv, hd, vocab int
	eps                       float32
	ropeBase                  float64
	emb                       *cuda.ResidentB
	fnorm                     *cuda.ResidentVec
	outW                      *cuda.ResidentBF16
	layers                    []*b4Layer
	pools                     []*cuda.PagedKVPool
	invDev                    *cuda.DeviceF32
}

type b4Layer struct {
	gA, gF                     *cuda.ResidentVec
	wq, wk, wv, wo, wg, wu, wd *cuda.ResidentBF16
}

// BatchedSeq is one in-flight sequence: a paged-KV handle per layer. Its position/length is its KV
// length; RoPE and attention derive the position from it, so sequences at different lengths are handled
// in the same batch without any per-sequence padding.
type BatchedSeq struct {
	kv []*cuda.SeqKV
}

// Len returns the sequence's current token count (== its next decode position).
func (s *BatchedSeq) Len() int { return s.kv[0].Len() }

// Release returns the sequence's KV blocks to the pools.
func (s *BatchedSeq) Release() {
	for _, k := range s.kv {
		k.Release()
	}
}

// NewBatchedDecoderF16 uploads m's weights as f16-resident and allocates one paged-KV pool per layer
// sized for up to maxSeqs concurrent sequences of up to maxLen tokens. Requires headDim==64.
func NewBatchedDecoderF16(m *nlp.Llama, maxSeqs, maxLen int) (*BatchedDecoderF16, error) {
	cfg := m.Config
	dim, heads := cfg.Dim, cfg.Heads
	kv := cfg.KVHeads
	if kv == 0 {
		kv = heads
	}
	hd := dim / heads
	if hd != 64 {
		return nil, fmt.Errorf("llamagpu: BatchedDecoderF16 needs headDim==64, got %d", hd)
	}
	cast := func(tt *tensor.Tensor) *tensor.Tensor { return tt.Cast(tensor.F32) }
	d := &BatchedDecoderF16{
		dim: dim, heads: heads, kv: kv, hd: hd, vocab: cfg.Vocab,
		eps: float32(cfg.Eps), ropeBase: cfg.RopeBase,
	}
	var err error
	fail := func(e error) (*BatchedDecoderF16, error) { d.Close(); return nil, e }
	if d.emb, err = cuda.NewResidentB(cast(m.TokEmb)); err != nil {
		return fail(fmt.Errorf("llamagpu: emb: %w", err))
	}
	if d.fnorm, err = cuda.NewResidentVec(cast(m.Norm.Gamma)); err != nil {
		return fail(fmt.Errorf("llamagpu: fnorm: %w", err))
	}
	if d.outW, err = cuda.NewResidentBF16(cast(m.Out)); err != nil {
		return fail(fmt.Errorf("llamagpu: out: %w", err))
	}
	mkW := func(tt *tensor.Tensor) *cuda.ResidentBF16 {
		if err != nil {
			return nil
		}
		var r *cuda.ResidentBF16
		r, err = cuda.NewResidentBF16(cast(tt))
		return r
	}
	mkV := func(tt *tensor.Tensor) *cuda.ResidentVec {
		if err != nil {
			return nil
		}
		var r *cuda.ResidentVec
		r, err = cuda.NewResidentVec(cast(tt))
		return r
	}
	kvW := kv * hd
	blocksPerPool := maxSeqs * (maxLen/16 + 2)
	d.layers = make([]*b4Layer, len(m.Blocks))
	d.pools = make([]*cuda.PagedKVPool, len(m.Blocks))
	for i, b := range m.Blocks {
		d.layers[i] = &b4Layer{
			gA: mkV(b.AttnNorm.Gamma), gF: mkV(b.FFNNorm.Gamma),
			wq: mkW(b.Wq), wk: mkW(b.Wk), wv: mkW(b.Wv), wo: mkW(b.Wo),
			wg: mkW(b.FFN.Wgate), wu: mkW(b.FFN.Wup), wd: mkW(b.FFN.Wdown),
		}
		if err != nil {
			return fail(fmt.Errorf("llamagpu: layer %d weights: %w", i, err))
		}
		if d.pools[i], err = cuda.NewPagedKVPool(blocksPerPool, 16, kvW); err != nil {
			return fail(fmt.Errorf("llamagpu: layer %d pool: %w", i, err))
		}
	}
	invD, _ := backend.RoPEFreqs(hd, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: heads})
	inv32 := make([]float32, len(invD))
	for i := range invD {
		inv32[i] = float32(invD[i])
	}
	if d.invDev, err = cuda.NewDeviceF32(1, len(inv32)); err != nil {
		return fail(fmt.Errorf("llamagpu: inv: %w", err))
	}
	if err = d.invDev.UploadF32(inv32); err != nil {
		return fail(fmt.Errorf("llamagpu: inv upload: %w", err))
	}
	return d, nil
}

// NewSequence allocates a fresh in-flight sequence (one paged-KV handle per layer).
func (d *BatchedDecoderF16) NewSequence() *BatchedSeq {
	s := &BatchedSeq{kv: make([]*cuda.SeqKV, len(d.layers))}
	for li := range d.layers {
		s.kv[li] = d.pools[li].NewSeqKV()
	}
	return s
}

// Close frees all resident weights and KV pools.
func (d *BatchedDecoderF16) Close() {
	if d.emb != nil {
		d.emb.Free()
	}
	if d.fnorm != nil {
		d.fnorm.Free()
	}
	if d.outW != nil {
		d.outW.Free()
	}
	if d.invDev != nil {
		d.invDev.Free()
	}
	for _, l := range d.layers {
		if l == nil {
			continue
		}
		for _, v := range []*cuda.ResidentVec{l.gA, l.gF} {
			if v != nil {
				v.Free()
			}
		}
		for _, w := range []*cuda.ResidentBF16{l.wq, l.wk, l.wv, l.wo, l.wg, l.wu, l.wd} {
			if w != nil {
				w.Free()
			}
		}
	}
	for _, p := range d.pools {
		if p != nil {
			p.Free()
		}
	}
}

// Logits runs one ragged batched decode step and returns the on-device f32 logits [B, vocab]. active[b]
// is the sequence for row b (at position active[b].Len()); toks[b] is its new token. Each sequence's K/V
// for this token is appended to its paged cache. The caller owns the returned buffer (Free it). Use
// Decode for greedy tokens directly.
func (d *BatchedDecoderF16) Logits(active []*BatchedSeq, toks []int32) (*cuda.DeviceF32, error) {
	B := len(active)
	if B == 0 || len(toks) != B {
		return nil, fmt.Errorf("llamagpu: Logits: %d seqs, %d toks", B, len(toks))
	}
	positions := make([]int32, B)
	for b := range active {
		positions[b] = int32(active[b].Len())
	}
	xf, err := d.emb.Embed(toks)
	if err != nil {
		return nil, fmt.Errorf("llamagpu: embed: %w", err)
	}
	x, err := cuda.F16FromF32(xf)
	xf.Free()
	if err != nil {
		return nil, err
	}
	defer x.Free()
	rqa := backend.RoPEAttrs{Base: d.ropeBase, Heads: d.heads}
	rka := backend.RoPEAttrs{Base: d.ropeBase, Heads: d.kv}
	layerSeqs := make([]*cuda.SeqKV, B)
	for li, l := range d.layers {
		for b := range active {
			layerSeqs[b] = active[b].kv[li]
		}
		dh, err := cuda.NewDeviceF16(B, d.dim)
		if err != nil {
			return nil, err
		}
		if err := x.RMSNormInto(l.gA, d.eps, dh); err != nil {
			dh.Free()
			return nil, err
		}
		dq, err := l.wq.MatMulF16(dh)
		if err != nil {
			dh.Free()
			return nil, err
		}
		dk, err := l.wk.MatMulF16(dh)
		if err != nil {
			dh.Free()
			dq.Free()
			return nil, err
		}
		dv, err := l.wv.MatMulF16(dh)
		dh.Free()
		if err != nil {
			dq.Free()
			dk.Free()
			return nil, err
		}
		if err := dq.RoPERagged(d.invDev, positions, rqa); err != nil {
			dq.Free()
			dk.Free()
			dv.Free()
			return nil, err
		}
		if err := dk.RoPERagged(d.invDev, positions, rka); err != nil {
			dq.Free()
			dk.Free()
			dv.Free()
			return nil, err
		}
		dkf, _ := dk.ToF32()
		dvf, _ := dv.ToF32()
		dk.Free()
		dv.Free()
		for b := range layerSeqs {
			if err := layerSeqs[b].Reserve1(); err != nil {
				dq.Free()
				dkf.Free()
				dvf.Free()
				return nil, err
			}
		}
		viewPre, err := d.pools[li].UploadBatchView(layerSeqs)
		if err != nil {
			dq.Free()
			dkf.Free()
			dvf.Free()
			return nil, err
		}
		err = d.pools[li].AppendBatched(layerSeqs, dkf, dvf, viewPre)
		viewPre.Free()
		dkf.Free()
		dvf.Free()
		if err != nil {
			dq.Free()
			return nil, err
		}
		viewPost, err := d.pools[li].UploadBatchView(layerSeqs)
		if err != nil {
			dq.Free()
			return nil, err
		}
		dqf, _ := dq.ToF32()
		dq.Free()
		da, err := d.pools[li].BatchedDecodeAttnViewGQA(dqf, viewPost, d.heads, d.kv)
		dqf.Free()
		viewPost.Free()
		if err != nil {
			return nil, err
		}
		da16, err := cuda.F16FromF32(da)
		da.Free()
		if err != nil {
			return nil, err
		}
		tmp, err := l.wo.MatMulF16(da16)
		da16.Free()
		if err != nil {
			return nil, err
		}
		err = x.Add(tmp)
		tmp.Free()
		if err != nil {
			return nil, err
		}
		dh2, err := cuda.NewDeviceF16(B, d.dim)
		if err != nil {
			return nil, err
		}
		if err := x.RMSNormInto(l.gF, d.eps, dh2); err != nil {
			dh2.Free()
			return nil, err
		}
		dg, err := l.wg.MatMulF16(dh2)
		if err != nil {
			dh2.Free()
			return nil, err
		}
		du, err := l.wu.MatMulF16(dh2)
		dh2.Free()
		if err != nil {
			dg.Free()
			return nil, err
		}
		err = dg.SwiGLU(du)
		du.Free()
		if err != nil {
			dg.Free()
			return nil, err
		}
		tmp2, err := l.wd.MatMulF16(dg)
		dg.Free()
		if err != nil {
			return nil, err
		}
		err = x.Add(tmp2)
		tmp2.Free()
		if err != nil {
			return nil, err
		}
	}
	x32, err := x.ToF32()
	if err != nil {
		return nil, err
	}
	nh, err := x32.RMSNormTo(d.fnorm, d.eps)
	x32.Free()
	if err != nil {
		return nil, err
	}
	lg, err := d.outW.MatMulDevice(nh)
	nh.Free()
	if err != nil {
		return nil, err
	}
	return lg, nil
}

// Decode runs one ragged batched decode step and returns the greedy (argmax) next token for each
// sequence, computed on-device (only B ints cross to the host). Appends each sequence's K/V.
func (d *BatchedDecoderF16) Decode(active []*BatchedSeq, toks []int32) ([]int32, error) {
	lg, err := d.Logits(active, toks)
	if err != nil {
		return nil, err
	}
	defer lg.Free()
	lg16, err := cuda.F16FromF32(lg)
	if err != nil {
		return nil, err
	}
	defer lg16.Free()
	return lg16.BatchArgmax()
}
