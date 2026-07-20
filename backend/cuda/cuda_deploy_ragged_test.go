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

// TestA1DeployDecoderRagged: the continuous-batching decode core — B sequences at DIFFERENT positions
// (different prompt lengths) decoded together in ONE joint step using PER-SEQUENCE positions
// (RoPEF16DposArrRaw) + per-seq paged KV lengths (AppendBatchedDev writes slot=dsl[b], attention reads
// dsl[b]). Each seq is prefilled separately (ragged), then joint-decoded. Verifies each seq's tokens
// match its own eager (batch=1) reference. Kept within one 16-block (len<=10, gen<=5). This proves
// ragged joint decode works — the technical heart of continuous batching.
func TestA1DeployDecoderRagged(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	f, err := gguf.ReadFile(tinyLlamaPath)
	if err != nil {
		t.Skipf("gguf: %v", err)
	}
	m, _ := nlp.LlamaFromGGUF(f.Metadata, f.Tensors)
	tok, _ := nlp.UnigramFromGGUF(f.Metadata)
	cfg := m.Config
	dim, heads := cfg.Dim, cfg.Heads
	kv := cfg.KVHeads
	if kv == 0 {
		kv = heads
	}
	hd := dim / heads
	if hd != 64 {
		t.Skipf("needs hd==64")
	}
	qW, kvW := heads*hd, kv*hd
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
	rqAttr := backend.RoPEAttrs{Base: cfg.RopeBase, Heads: heads}
	rkAttr := backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv}
	invD, _ := backend.RoPEFreqs(hd, rqAttr)
	inv32 := make([]float32, len(invD))
	for i := range invD {
		inv32[i] = float32(invD[i])
	}
	invDev, _ := cuda.NewDeviceF32(1, len(inv32))
	invDev.UploadF32(inv32)
	defer invDev.Free()

	// two sequences with DIFFERENT prompt lengths
	enc := func(s string) []int32 {
		ids := append([]int{1}, tok.Encode(s)...)
		o := make([]int32, len(ids))
		for i, v := range ids {
			o[i] = int32(v)
		}
		return o
	}
	prompts := [][]int32{
		enc("The capital"),              // short
		enc("The capital of France is"), // long
		enc("Hello there"),              // different content/length
		enc("Once upon a time"),         // 4 concurrent seqs at different positions
	}
	B := len(prompts)
	const maxGen = 5
	argmaxRow := func(host []float32, off int) int {
		best, bi := host[off], 0
		for c := 1; c < cfg.Vocab; c++ {
			if host[off+c] > best {
				best, bi = host[off+c], c
			}
		}
		return bi
	}
	// eager batch=1 reference for one sequence.
	eagerOne := func(prompt []int32) []int {
		pools := make([]*cuda.PagedKVPool, len(ls))
		seqs := make([]*cuda.SeqKV, len(ls))
		for i := range ls {
			pools[i], _ = cuda.NewPagedKVPool(4, 16, kvW)
			seqs[i] = pools[i].NewSeqKV()
		}
		defer func() {
			for i := range pools {
				seqs[i].Release()
				pools[i].Free()
			}
		}()
		step := func(tokID int32, pos int) int {
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
				view, _ := pools[li].UploadBatchView([]*cuda.SeqKV{seqs[li]})
				dqf, _ := dq.ToF32()
				dq.Free()
				da, _ := pools[li].BatchedDecodeAttnViewGQA(dqf, view, heads, kv)
				dqf.Free()
				view.Free()
				dkf.Free()
				dvf.Free()
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
			host := make([]float32, cfg.Vocab)
			lg.DownloadF32(host)
			lg.Free()
			return argmaxRow(host, 0)
		}
		out := []int{}
		last := 0
		for i, id := range prompt {
			last = step(id, i)
		}
		for n := 0; n < maxGen; n++ {
			out = append(out, last)
			last = step(int32(last), len(prompt)+n)
		}
		return out
	}
	refs := make([][]int, B)
	for b := 0; b < B; b++ {
		refs[b] = eagerOne(prompts[b])
	}

	// ---- ragged joint decode ----
	pools := make([]*cuda.PagedKVPool, len(ls))
	seqs := make([][]*cuda.SeqKV, len(ls))
	for i := range ls {
		pools[i], _ = cuda.NewPagedKVPool(B*2, 16, kvW)
		seqs[i] = make([]*cuda.SeqKV, B)
		for b := 0; b < B; b++ {
			seqs[i][b] = pools[i].NewSeqKV()
		}
	}
	defer func() {
		for i := range pools {
			for b := 0; b < B; b++ {
				seqs[i][b].Release()
			}
			pools[i].Free()
		}
	}()
	// per-seq eager prefill into the shared pool (batch=1 forward, SeqKV.Append), gives first tokens.
	prefillSeq := func(b int, prompt []int32) int {
		last := 0
		for pos, id := range prompt {
			xf, _ := emb.Embed([]int32{id})
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
				seqs[li][b].Append(dkf, dvf)
				view, _ := pools[li].UploadBatchView([]*cuda.SeqKV{seqs[li][b]})
				dqf, _ := dq.ToF32()
				dq.Free()
				da, _ := pools[li].BatchedDecodeAttnViewGQA(dqf, view, heads, kv)
				dqf.Free()
				view.Free()
				dkf.Free()
				dvf.Free()
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
			host := make([]float32, cfg.Vocab)
			lg.DownloadF32(host)
			lg.Free()
			last = argmaxRow(host, 0)
		}
		return last
	}
	last := make([]int, B)
	for b := 0; b < B; b++ {
		last[b] = prefillSeq(b, prompts[b])
	}
	// joint view over all B seqs (per-seq dsl = each seq's prompt length)
	views := make([]*cuda.PagedBatchView, len(ls))
	attnOut := make([]*cuda.DeviceF32, len(ls))
	for i := range ls {
		views[i], _ = pools[i].UploadBatchView(seqs[i]) // dsl[b] = len(prompt_b)
		attnOut[i], _ = cuda.NewDeviceF32(B, qW)
	}
	defer func() {
		for i := range ls {
			views[i].Free()
			attnOut[i].Free()
		}
	}()
	// joint decode step with PER-SEQ positions.
	jointStep := func(toks []int32, positions []int32) []int {
		xf, _ := emb.Embed(toks) // [B, dim]
		x := cuda.AllocU16(B * dim)
		cuda.CvtF32ToF16(x, xf.DevPtr(), B*dim)
		xf.Free()
		dpos := cuda.UploadI32(positions) // per-seq positions
		for li, l := range ls {
			dh := cuda.AllocU16(B * dim)
			cuda.RMSNormF16(x, dh, l.gA.VecPtr(), B, dim, eps)
			dq16, dk16, dv16 := cuda.AllocU16(B*qW), cuda.AllocU16(B*kvW), cuda.AllocU16(B*kvW)
			cuda.GemmF16Pure(dh, l.wq.WPtr(), dq16, B, dim, qW)
			cuda.GemmF16Pure(dh, l.wk.WPtr(), dk16, B, dim, kvW)
			cuda.GemmF16Pure(dh, l.wv.WPtr(), dv16, B, dim, kvW)
			cuda.FreeDev(dh)
			cuda.RoPEF16DposArrRaw(dq16, invDev.DevPtr(), dpos, B, heads, hd, rqAttr)
			cuda.RoPEF16DposArrRaw(dk16, invDev.DevPtr(), dpos, B, kv, hd, rkAttr)
			dkf, _ := cuda.NewDeviceF32(B, kvW)
			dvf, _ := cuda.NewDeviceF32(B, kvW)
			cuda.CvtF16ToF32(dkf.DevPtr(), dk16, B*kvW)
			cuda.CvtF16ToF32(dvf.DevPtr(), dv16, B*kvW)
			cuda.FreeDev(dk16)
			cuda.FreeDev(dv16)
			pools[li].AppendBatchedDev(dkf, dvf, views[li])
			views[li].BumpLens(1)
			dkf.Free()
			dvf.Free()
			dqf, _ := cuda.NewDeviceF32(B, qW)
			cuda.CvtF16ToF32(dqf.DevPtr(), dq16, B*qW)
			cuda.FreeDev(dq16)
			pools[li].BatchedDecodeAttnViewInto(dqf, views[li], heads, kv, attnOut[li])
			dqf.Free()
			da := cuda.AllocU16(B * qW)
			cuda.CvtF32ToF16(da, attnOut[li].DevPtr(), B*qW)
			tmp := cuda.AllocU16(B * dim)
			cuda.GemmF16Pure(da, l.wo.WPtr(), tmp, B, qW, dim)
			cuda.FreeDev(da)
			cuda.AddF16(x, tmp, B*dim)
			cuda.FreeDev(tmp)
			dh2 := cuda.AllocU16(B * dim)
			cuda.RMSNormF16(x, dh2, l.gF.VecPtr(), B, dim, eps)
			hidden := cfg.Hidden
			dg, du := cuda.AllocU16(B*hidden), cuda.AllocU16(B*hidden)
			cuda.GemmF16Pure(dh2, l.wg.WPtr(), dg, B, dim, hidden)
			cuda.GemmF16Pure(dh2, l.wu.WPtr(), du, B, dim, hidden)
			cuda.FreeDev(dh2)
			cuda.SwiGLUF16(dg, du, B*hidden)
			cuda.FreeDev(du)
			tmp2 := cuda.AllocU16(B * dim)
			cuda.GemmF16Pure(dg, l.wd.WPtr(), tmp2, B, hidden, dim)
			cuda.FreeDev(dg)
			cuda.AddF16(x, tmp2, B*dim)
			cuda.FreeDev(tmp2)
		}
		xn := cuda.AllocU16(B * dim)
		cuda.RMSNormF16(x, xn, fnorm.VecPtr(), B, dim, eps)
		cuda.FreeDev(x)
		x32, _ := cuda.NewDeviceF32(B, dim)
		cuda.CvtF16ToF32(x32.DevPtr(), xn, B*dim)
		cuda.FreeDev(xn)
		cuda.FreeDev(dpos)
		lg, _ := outW.MatMulDevice(x32)
		x32.Free()
		host := make([]float32, B*cfg.Vocab)
		lg.DownloadF32(host)
		lg.Free()
		out := make([]int, B)
		for b := 0; b < B; b++ {
			out[b] = argmaxRow(host, b*cfg.Vocab)
		}
		return out
	}
	got := make([][]int, B)
	for b := 0; b < B; b++ {
		got[b] = []int{last[b]}
	}
	for n := 0; n < maxGen-1; n++ {
		toks := make([]int32, B)
		positions := make([]int32, B)
		for b := 0; b < B; b++ {
			toks[b] = int32(got[b][len(got[b])-1])
			positions[b] = int32(len(prompts[b]) + n) // per-seq absolute position
		}
		nx := jointStep(toks, positions)
		for b := 0; b < B; b++ {
			got[b] = append(got[b], nx[b])
		}
	}
	for b := 0; b < B; b++ {
		t.Logf("seq %d (len %d) eager: %v", b, len(prompts[b]), refs[b])
		t.Logf("seq %d (len %d) ragged:%v", b, len(prompts[b]), got[b])
		for i := range got[b] {
			if refs[b][i] != got[b][i] {
				t.Fatalf("ragged seq %d diverged at %d: eager %d vs ragged %d", b, i, refs[b][i], got[b][i])
			}
		}
	}
	t.Logf("=> RAGGED joint decode (per-seq positions + per-seq KV lengths) == eager for all %d seqs: continuous-batch decode core WORKS", B)
}
