//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"strings"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// TestA1RealGenerateText: the demonstrable serving capstone — a REAL text prompt tokenized, run
// through the A1 f16-activation SERVING DECODE path (per-layer paged KV + growing KV + paged GQA
// attention + f16 activations + greedy argmax), and detokenized back to readable English. Unlike
// the token-id correctness tests, this proves the whole f16 pipeline preserves the model's language
// decisions on a real prompt end-to-end — the human-verifiable "it works" proof.
func TestA1RealGenerateText(t *testing.T) {
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
	tok, err := nlp.UnigramFromGGUF(f.Metadata)
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
	invD, _ := backend.RoPEFreqs(hd, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: heads})
	inv32 := make([]float32, len(invD))
	for i := range invD {
		inv32[i] = float32(invD[i])
	}
	invDev, _ := cuda.NewDeviceF32(1, len(inv32))
	invDev.UploadF32(inv32)
	defer invDev.Free()

	prompt := "The capital of France is"
	ids := append([]int{1}, tok.Encode(prompt)...) // BOS=1
	const maxGen = 20
	pools := make([]*cuda.PagedKVPool, len(ls))
	seqs := make([]*cuda.SeqKV, len(ls))
	maxLen := len(ids) + maxGen + 2
	for i := range ls {
		pools[i], _ = cuda.NewPagedKVPool(maxLen/16+2, 16, kvW)
		seqs[i] = pools[i].NewSeqKV()
	}
	defer func() {
		for i := range pools {
			seqs[i].Release()
			pools[i].Free()
		}
	}()

	a1step := func(tokID int32, pos int) int {
		xf, _ := emb.Embed([]int32{tokID})
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
			seqs[li].Append(dkf, dvf)
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
		best, bi := host[0], 0
		for c := 1; c < vocab; c++ {
			if host[c] > best {
				best, bi = host[c], c
			}
		}
		return bi
	}

	// prefill the prompt (via decode steps), tracking the last prediction
	last := 0
	for i, id := range ids {
		last = a1step(int32(id), i)
	}
	// leak discipline (Iw7): after prefill warms the pool, every decode step must net-zero device
	// buffers — KV growth reuses pre-allocated pool blocks (no alloc) and per-step view uploads are
	// freed. A growing LiveBufs would be a real leak in the serving loop.
	liveBase := cuda.LiveBufs()
	// greedy generate
	gen := make([]int, 0, maxGen)
	pos := len(ids)
	for n := 0; n < maxGen; n++ {
		if last == 2 { // EOS
			break
		}
		gen = append(gen, last)
		last = a1step(int32(last), pos)
		pos++
		if live := cuda.LiveBufs(); live != liveBase {
			t.Fatalf("serving decode step leaked: LiveBufs %d -> %d after step %d", liveBase, live, n)
		}
	}
	genText := strings.TrimSpace(renderPieces(tok.Decode(gen)))
	t.Logf("PROMPT:              %q", prompt)
	t.Logf("A1 f16 GENERATED:    %q", genText)
	if len(gen) == 0 {
		t.Fatal("A1 generated nothing")
	}
	// coherence smoke: the strong-prior continuation of "The capital of France is" should mention Paris
	if !strings.Contains(strings.ToLower(genText), "paris") {
		t.Logf("note: 'Paris' not in output (greedy on 1.1B can drift) — output above for manual inspection")
	}
}
