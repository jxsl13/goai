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

// BenchmarkBatchedDecodeThroughput measures the AGGREGATE decode throughput (tokens/sec across the
// whole batch) of the batched model forward (the B4 path prototyped in TestA1RealGenerateBatched) at
// several batch sizes B. Decode is memory-bandwidth-bound at B=1; batching amortizes the weight reads
// across B tokens, so if the win is real, aggregate tok/s should rise super-linearly-to-linearly with B
// while per-step latency grows sub-linearly. This is the MEASURE-FIRST gate for promoting B4 to a
// production continuous-batching API: it quantifies whether batched serving actually beats batch-1 on
// GA106 (28 SMs) before investing in the API/scheduler wiring.
//
// The forward here uses f16-resident weights (like the batched test) and keeps logits on-device (no
// per-step D2H), syncing once per benchmark iteration so we measure the forward, not the host round-trip.
func BenchmarkBatchedDecodeThroughput(b *testing.B) {
	for _, B := range []int{1, 4, 8, 16, 32} {
		b.Run(benchB4Name(B), func(b *testing.B) { benchB4Throughput(b, B) })
	}
}

func benchB4Name(B int) string {
	switch B {
	case 1:
		return "B1"
	case 4:
		return "B4"
	case 8:
		return "B8"
	case 16:
		return "B16"
	default:
		return "B32"
	}
}

func benchB4Throughput(b *testing.B, B int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	f, err := gguf.ReadFile(tinyLlamaPath)
	if err != nil {
		b.Skipf("gguf: %v", err)
	}
	m, err := nlp.LlamaFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		b.Fatal(err)
	}
	cfg := m.Config
	dim, heads := cfg.Dim, cfg.Heads
	kv := cfg.KVHeads
	if kv == 0 {
		kv = heads
	}
	hd := dim / heads
	if hd != 64 {
		b.Skipf("needs hd==64, got %d", hd)
	}
	kvW := kv * hd
	eps := float32(cfg.Eps)
	cast := func(tt *tensor.Tensor) *tensor.Tensor { return tt.Cast(tensor.F32) }
	mkW := func(tt *tensor.Tensor) *cuda.ResidentBF16 {
		r, e := cuda.NewResidentBF16(cast(tt))
		mustTB(b, e)
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
	for i, blk := range m.Blocks {
		ls[i] = &L{gA: mkV(blk.AttnNorm.Gamma), gF: mkV(blk.FFNNorm.Gamma),
			wq: mkW(blk.Wq), wk: mkW(blk.Wk), wv: mkW(blk.Wv), wo: mkW(blk.Wo),
			wg: mkW(blk.FFN.Wgate), wu: mkW(blk.FFN.Wup), wd: mkW(blk.FFN.Wdown)}
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

	const seed = 64 // steady-state KV length before timing (realistic attention work)
	maxLen := seed + b.N + 4
	pools := make([]*cuda.PagedKVPool, len(ls))
	seqs := make([][]*cuda.SeqKV, len(ls))
	for li := range ls {
		pools[li], _ = cuda.NewPagedKVPool(B*(maxLen/16+2), 16, kvW)
		seqs[li] = make([]*cuda.SeqKV, B)
		for bi := 0; bi < B; bi++ {
			seqs[li][bi] = pools[li].NewSeqKV()
		}
	}
	defer func() {
		for li := range pools {
			for bi := 0; bi < B; bi++ {
				seqs[li][bi].Release()
			}
			pools[li].Free()
		}
	}()

	// fwd runs one batched decode step (B tokens at position pos) WITHOUT the host logits download —
	// returns the device logits so the caller frees them; only sync at the end of the loop.
	fwd := func(toks []int32, pos int) *cuda.DeviceF32 {
		xf, _ := emb.Embed(toks)
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
			for bi := 0; bi < B; bi++ {
				seqs[li][bi].Reserve1()
			}
			viewPre, _ := pools[li].UploadBatchView(seqs[li])
			pools[li].AppendBatched(seqs[li], dkf, dvf, viewPre)
			viewPre.Free()
			dkf.Free()
			dvf.Free()
			viewPost, _ := pools[li].UploadBatchView(seqs[li])
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
		return lg
	}

	// Seed each sequence's KV to `seed` tokens so the timed steps do realistic attention reads.
	toks := make([]int32, B)
	for pos := 0; pos < seed; pos++ {
		for bi := 0; bi < B; bi++ {
			toks[bi] = int32((pos*131 + bi*911 + 7) % cfg.Vocab)
		}
		fwd(toks, pos).Free()
	}
	cuda.GraphSync()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for bi := 0; bi < B; bi++ {
			toks[bi] = int32((i*577 + bi*911 + 3) % cfg.Vocab)
		}
		fwd(toks, seed+i).Free()
	}
	cuda.GraphSync()
	b.StopTimer()

	// Aggregate throughput: B tokens per step, b.N steps.
	toksPerSec := float64(B) * float64(b.N) / b.Elapsed().Seconds()
	b.ReportMetric(toksPerSec, "tok/s-agg")
	b.ReportMetric(b.Elapsed().Seconds()*1e3/float64(b.N), "ms/step")
}
