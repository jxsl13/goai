//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// TestA1RealGenerate: the productization capstone — real TinyLlama weights driven through the actual
// A1 SERVING DECODE path (per-layer paged KV + device-resident growing KV + paged GQA attention +
// f16 activations), processing a real prompt token-by-token then autoregressively generating. Unlike
// TestA1RealPromptTokenMatch (full-attention PREFILL only), this exercises the paged single-query
// decode kernel over growing KV with real weights, and validates its first prediction against the
// trusted f32 full-prefill reference — proving the assembled serving loop produces correct tokens.
func TestA1RealGenerate(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	f, err := gguf.ReadFile(tinyLlamaPath)
	if err != nil {
		t.Skipf("gguf: %v", err)
	}
	m, err := nlp.LlamaFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	cfg := m.Config
	dim, heads := cfg.Dim, cfg.Heads
	kv := cfg.KVHeads
	if kv == 0 {
		kv = heads
	}
	hd := dim / heads
	if hd != 64 {
		t.Skipf("needs hd==64, got %d", hd)
	}
	kvW := kv * hd
	eps := float32(cfg.Eps)
	cast := func(tt *tensor.Tensor) *tensor.Tensor { return tt.Cast(tensor.F32) }
	mkW := func(tt *tensor.Tensor) *cuda.ResidentBF16 {
		r, e := cuda.NewResidentBF16(cast(tt))
		mustTB(t, e)
		return r
	}
	mkV := func(tt *tensor.Tensor) *cuda.ResidentVec { r, _ := cuda.NewResidentVec(cast(tt)); return r }
	emb, _ := cuda.NewResidentB(cast(m.TokEmb))
	defer emb.Free()
	fnorm := mkV(m.Norm.Gamma)
	defer fnorm.Free()
	outW := mkW(m.Out)
	defer outW.Free()
	type L struct {
		gA, gF                     *cuda.ResidentVec
		wq, wk, wv, wo, wg, wu, wd *cuda.ResidentBF16
	}
	ls := make([]*L, len(m.Blocks))
	for i, b := range m.Blocks {
		ls[i] = &L{gA: mkV(b.AttnNorm.Gamma), gF: mkV(b.FFNNorm.Gamma),
			wq: mkW(b.Wq), wk: mkW(b.Wk), wv: mkW(b.Wv), wo: mkW(b.Wo),
			wg: mkW(b.FFN.Wgate), wu: mkW(b.FFN.Wup), wd: mkW(b.FFN.Wdown)}
	}
	defer func() {
		for _, l := range ls {
			l.gA.Free()
			l.gF.Free()
			for _, w := range []*cuda.ResidentBF16{l.wq, l.wk, l.wv, l.wo, l.wg, l.wu, l.wd} {
				w.Free()
			}
		}
	}()
	rq := backend.RoPEAttrs{Base: cfg.RopeBase, Heads: heads}
	rk := backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv}
	invD, _ := backend.RoPEFreqs(hd, rq)
	inv32 := make([]float32, len(invD))
	for i := range invD {
		inv32[i] = float32(invD[i])
	}
	invDev, _ := cuda.NewDeviceF32(1, len(inv32))
	invDev.UploadF32(inv32)
	defer invDev.Free()

	const promptLen = 16 // multiple of 16 for the WMMA f32 reference
	ids := make([]int32, promptLen)
	for i := range ids {
		ids[i] = int32((i*2129 + 41) % cfg.Vocab)
	}
	argmaxRow := func(host []float32, vocab int) int {
		best, bi := host[0], 0
		for c := 1; c < vocab; c++ {
			if host[c] > best {
				best, bi = host[c], c
			}
		}
		return bi
	}

	// f32 full-prefill reference: next-token after the prompt (argmax of last position).
	refNext := func() int {
		x, _ := emb.Embed(ids)
		for _, l := range ls {
			dh, _ := x.RMSNormTo(l.gA, eps)
			dq, _ := l.wq.MatMulDevice(dh)
			dk, _ := l.wk.MatMulDevice(dh)
			dv, _ := l.wv.MatMulDevice(dh)
			dh.Free()
			dq.RoPE(rq)
			dk.RoPE(rk)
			da, _ := cuda.GroupedQueryAttentionWMMA(dq, dk, dv, heads, kv)
			dq.Free()
			dk.Free()
			dv.Free()
			l.wo.MatMulAccInto(da, x)
			da.Free()
			dh2, _ := x.RMSNormTo(l.gF, eps)
			dg, _ := l.wg.MatMulDevice(dh2)
			du, _ := l.wu.MatMulDevice(dh2)
			dh2.Free()
			dg.SwiGLU(du)
			du.Free()
			l.wd.MatMulAccInto(dg, x)
			dg.Free()
		}
		nh, _ := x.RMSNormTo(fnorm, eps)
		x.Free()
		lg, _ := outW.MatMulDevice(nh)
		nh.Free()
		vocab := cfg.Vocab
		host := make([]float32, promptLen*vocab)
		lg.DownloadF32(host)
		lg.Free()
		return argmaxRow(host[(promptLen-1)*vocab:], vocab) // last position
	}()

	// A1 paged serving decode: per-layer pools, process prompt + generate.
	pools := make([]*cuda.PagedKVPool, len(ls))
	seqs := make([]*cuda.SeqKV, len(ls))
	maxLen := promptLen + 8
	for i := range ls {
		pools[i], _ = cuda.NewPagedKVPool((maxLen/16 + 2), 16, kvW)
		seqs[i] = pools[i].NewSeqKV()
	}
	defer func() {
		for i := range pools {
			seqs[i].Release()
			pools[i].Free()
		}
	}()

	a1step := func(tok int32, pos int) int {
		xf, _ := emb.Embed([]int32{tok})
		x, _ := cuda.F16FromF32(xf)
		xf.Free()
		rqp := backend.RoPEAttrs{Base: cfg.RopeBase, Heads: heads, PosOffset: pos}
		rkp := backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv, PosOffset: pos}
		for li, l := range ls {
			dh, _ := cuda.NewDeviceF16(1, dim)
			x.RMSNormInto(l.gA, eps, dh)
			dq, _ := l.wq.MatMulF16(dh)
			dk, _ := l.wk.MatMulF16(dh)
			dv, _ := l.wv.MatMulF16(dh)
			dh.Free()
			dq.RoPE(invDev, rqp)
			dk.RoPE(invDev, rkp)
			dkf, _ := dk.ToF32()
			dvf, _ := dv.ToF32()
			dk.Free()
			dv.Free()
			seqs[li].Append(dkf, dvf) // grow this layer's KV
			dkf.Free()
			dvf.Free()
			view, _ := pools[li].UploadBatchView([]*cuda.SeqKV{seqs[li]})
			dqf, _ := dq.ToF32()
			dq.Free()
			da, _ := pools[li].BatchedDecodeAttnViewGQA(dqf, view, heads, kv)
			dqf.Free()
			view.Free()
			da16, _ := cuda.F16FromF32(da)
			da.Free()
			tmp, _ := l.wo.MatMulF16(da16)
			da16.Free()
			x.Add(tmp)
			tmp.Free()
			dh2, _ := cuda.NewDeviceF16(1, dim)
			x.RMSNormInto(l.gF, eps, dh2)
			dg, _ := l.wg.MatMulF16(dh2)
			du, _ := l.wu.MatMulF16(dh2)
			dh2.Free()
			dg.SwiGLU(du)
			du.Free()
			tmp2, _ := l.wd.MatMulF16(dg)
			dg.Free()
			x.Add(tmp2)
			tmp2.Free()
		}
		x32, _ := x.ToF32()
		x.Free()
		nh, _ := x32.RMSNormTo(fnorm, eps)
		x32.Free()
		lg, _ := outW.MatMulDevice(nh)
		nh.Free()
		vocab := cfg.Vocab
		host := make([]float32, vocab)
		lg.DownloadF32(host)
		lg.Free()
		return argmaxRow(host, vocab)
	}

	const maxGen = 8
	gen := make([]int32, 0, maxGen)
	a1FirstNext := -1
	for pos := 0; pos < promptLen+maxGen; pos++ {
		var tok int32
		if pos < promptLen {
			tok = ids[pos]
		} else {
			tok = gen[len(gen)-1]
		}
		next := a1step(tok, pos)
		if pos == promptLen-1 {
			a1FirstNext = next
		}
		if pos >= promptLen-1 && len(gen) < maxGen {
			gen = append(gen, int32(next))
		}
	}
	t.Logf("A1 paged-decode generation (real TinyLlama 22L): first-next A1=%d vs f32-prefill ref=%d", a1FirstNext, refNext)
	t.Logf("  generated %d tokens: %v", len(gen), gen)
	// non-degeneracy: not all identical
	allSame := true
	for _, g := range gen {
		if g != gen[0] {
			allSame = false
			break
		}
	}
	if allSame {
		t.Fatalf("A1 generation degenerate (all %d)", gen[0])
	}
	if a1FirstNext != refNext {
		t.Fatalf("A1 paged-decode first prediction %d != f32 prefill reference %d — serving decode path incorrect", a1FirstNext, refNext)
	}
}
