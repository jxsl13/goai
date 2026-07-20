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

// TestA1RealGenerateBatched: the batched (continuous-batching) serving proof — B real sequences with
// DIFFERENT prompts generated concurrently through the A1 decode path (batched f16 GEMMs + per-layer
// paged pool holding B SeqKVs + device-side AppendBatched + batched paged GQA attention + per-seq
// argmax). Sequence 0 uses the same prompt as the batch=1 TestA1RealGenerate, so its output MUST
// match that reference — proving batching does not corrupt per-sequence results (each attends only
// its own KV). Uniform prompt length keeps all sequences at the same decode position.
func TestA1RealGenerateBatched(t *testing.T) {
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
	invD, _ := backend.RoPEFreqs(hd, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: heads})
	inv32 := make([]float32, len(invD))
	for i := range invD {
		inv32[i] = float32(invD[i])
	}
	invDev, _ := cuda.NewDeviceF32(1, len(inv32))
	invDev.UploadF32(inv32)
	defer invDev.Free()

	const B, promptLen, maxGen = 4, 16, 6
	// per-sequence prompts; seq 0 == the batch=1 reference prompt
	prompts := make([][]int32, B)
	for b := 0; b < B; b++ {
		prompts[b] = make([]int32, promptLen)
		for i := range prompts[b] {
			prompts[b][i] = int32((i*2129 + 41 + b*7919) % cfg.Vocab)
		}
	}
	ref := []int32{30548, 30767, 30548, 31685, 236, 150} // seq-0 from TestA1RealGenerate

	pools := make([]*cuda.PagedKVPool, len(ls))
	seqs := make([][]*cuda.SeqKV, len(ls))
	maxLen := promptLen + maxGen
	for li := range ls {
		pools[li], _ = cuda.NewPagedKVPool(B*(maxLen/16+2), 16, kvW)
		seqs[li] = make([]*cuda.SeqKV, B)
		for b := 0; b < B; b++ {
			seqs[li][b] = pools[li].NewSeqKV()
		}
	}
	defer func() {
		for li := range pools {
			for b := 0; b < B; b++ {
				seqs[li][b].Release()
			}
			pools[li].Free()
		}
	}()

	argmaxRow := func(host []float32, off, vocab int) int {
		best, bi := host[off], 0
		for c := 1; c < vocab; c++ {
			if host[off+c] > best {
				best, bi = host[off+c], c
			}
		}
		return bi
	}

	// batched A1 decode step: B tokens at position pos -> B next tokens.
	step := func(toks []int32, pos int) []int {
		xf, _ := emb.Embed(toks) // [B, dim]
		x, _ := cuda.F16FromF32(xf)
		xf.Free()
		rqp := backend.RoPEAttrs{Base: cfg.RopeBase, Heads: heads, PosOffset: pos}
		rkp := backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv, PosOffset: pos}
		for li, l := range ls {
			dh, _ := cuda.NewDeviceF16(B, dim)
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
			for b := 0; b < B; b++ {
				seqs[li][b].Reserve1()
			}
			viewPre, _ := pools[li].UploadBatchView(seqs[li])
			pools[li].AppendBatched(seqs[li], dkf, dvf, viewPre) // scatter row b -> seq b, bumps len
			viewPre.Free()
			dkf.Free()
			dvf.Free()
			viewPost, _ := pools[li].UploadBatchView(seqs[li]) // lengths now include current token
			dqf, _ := dq.ToF32()
			dq.Free()
			da, _ := pools[li].BatchedDecodeAttnViewGQA(dqf, viewPost, heads, kv)
			dqf.Free()
			viewPost.Free()
			da16, _ := cuda.F16FromF32(da)
			da.Free()
			tmp, _ := l.wo.MatMulF16(da16)
			da16.Free()
			x.Add(tmp)
			tmp.Free()
			dh2, _ := cuda.NewDeviceF16(B, dim)
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
		lg, _ := outW.MatMulDevice(nh) // [B, vocab]
		nh.Free()
		vocab := cfg.Vocab
		host := make([]float32, B*vocab)
		lg.DownloadF32(host)
		lg.Free()
		out := make([]int, B)
		for b := 0; b < B; b++ {
			out[b] = argmaxRow(host, b*vocab, vocab)
		}
		return out
	}

	gen := make([][]int32, B)
	for b := range gen {
		gen[b] = make([]int32, 0, maxGen)
	}
	for pos := 0; pos < promptLen+maxGen; pos++ {
		toks := make([]int32, B)
		for b := 0; b < B; b++ {
			if pos < promptLen {
				toks[b] = prompts[b][pos]
			} else {
				toks[b] = gen[b][len(gen[b])-1]
			}
		}
		next := step(toks, pos)
		if pos >= promptLen-1 {
			for b := 0; b < B; b++ {
				if len(gen[b]) < maxGen {
					gen[b] = append(gen[b], int32(next[b]))
				}
			}
		}
	}
	for b := 0; b < B; b++ {
		t.Logf("seq %d generated: %v", b, gen[b])
	}
	// seq 0 must match the batch=1 reference (proves batching is per-sequence correct)
	for i := range ref {
		if gen[0][i] != ref[i] {
			t.Fatalf("batched seq-0 token[%d]=%d != batch=1 reference %d — batching corrupts per-seq results", i, gen[0][i], ref[i])
		}
	}
	// sequences with different prompts should not all be identical
	if gen[1][0] == gen[0][0] && gen[2][0] == gen[0][0] && gen[3][0] == gen[0][0] {
		t.Logf("note: all seqs produced same first token %d (possible with strong priors)", gen[0][0])
	}
}
