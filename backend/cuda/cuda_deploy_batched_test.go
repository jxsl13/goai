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

// TestA1DeployDecoderBatched: the serving-scale deployable decoder — B concurrent sequences generated
// together through the GRAPH-CAPTURED batched A1 decode step (batched f16 GEMMs M=B, per-layer paged
// pool with B SeqKVs, in-graph AppendBatchedDev + BumpLens + batched attention, device-pos RoPE),
// replayed with per-sequence argmax sampling between replays. Uniform prompt length keeps all seqs at
// the same decode position (shared BumpLens). Each sequence's graph tokens must match its own eager
// reference — proving the full batched serving decode path (batched + graph + correct + real) works.
func TestA1DeployDecoderBatched(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	f, err := gguf.ReadFile(tinyLlamaPath)
	if err != nil {
		t.Skipf("gguf: %v", err)
	}
	m, _ := nlp.LlamaFromGGUF(f.Metadata, f.Tensors)
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

	const B, promptLen, maxGen = 3, 6, 16
	total := promptLen + maxGen
	prompts := make([][]int32, B)
	for b := 0; b < B; b++ {
		prompts[b] = make([]int32, promptLen)
		prompts[b][0] = 1 // BOS
		for i := 1; i < promptLen; i++ {
			prompts[b][i] = int32((i*1013 + 71 + b*4099) % cfg.Vocab)
		}
	}
	argmaxRow := func(host []float32, off int) int {
		best, bi := host[off], 0
		for c := 1; c < cfg.Vocab; c++ {
			if host[off+c] > best {
				best, bi = host[off+c], c
			}
		}
		return bi
	}

	// per-sequence eager references (batch=1 each, correct ordering).
	eagerOne := func(prompt []int32) []int {
		pools := make([]*cuda.PagedKVPool, len(ls))
		seqs := make([]*cuda.SeqKV, len(ls))
		for i := range ls {
			pools[i], _ = cuda.NewPagedKVPool(total/16+2, 16, kvW)
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

	// batched graph decoder
	pools := make([]*cuda.PagedKVPool, len(ls))
	seqs := make([][]*cuda.SeqKV, len(ls))
	views := make([]*cuda.PagedBatchView, len(ls))
	attnOut := make([]*cuda.DeviceF32, len(ls))
	zeroKV := make([]float32, total*kvW)
	for i := range ls {
		pools[i], _ = cuda.NewPagedKVPool(B*(total/16+2), 16, kvW)
		seqs[i] = make([]*cuda.SeqKV, B)
		for b := 0; b < B; b++ {
			seqs[i][b] = pools[i].NewSeqKV()
			d, _ := cuda.NewDeviceF32(total, kvW)
			d.UploadF32(zeroKV)
			seqs[i][b].Append(d, d) // pre-grow: allocate all blocks
			d.Free()
		}
		views[i], _ = pools[i].UploadBatchView(seqs[i]) // dbt=all, dsl=total (per seq)
		views[i].BumpLens(-total)                       // dsl=0
		attnOut[i], _ = cuda.NewDeviceF32(B, qW)
	}
	defer func() {
		for i := range pools {
			views[i].Free()
			attnOut[i].Free()
			for b := 0; b < B; b++ {
				seqs[i][b].Release()
			}
			pools[i].Free()
		}
	}()
	x0, _ := cuda.NewDeviceF32(B, dim)
	defer x0.Free()
	hidOut, _ := cuda.NewDeviceF32(B, dim)
	defer hidOut.Free()
	pos, _ := cuda.NewDevicePos()
	defer pos.Free()

	step := func() {
		x := cuda.AllocU16(B * dim)
		cuda.CvtF32ToF16(x, x0.DevPtr(), B*dim)
		for li, l := range ls {
			dh := cuda.AllocU16(B * dim)
			cuda.RMSNormF16(x, dh, l.gA.VecPtr(), B, dim, eps)
			dq16, dk16, dv16 := cuda.AllocU16(B*qW), cuda.AllocU16(B*kvW), cuda.AllocU16(B*kvW)
			cuda.GemmF16Pure(dh, l.wq.WPtr(), dq16, B, dim, qW)
			cuda.GemmF16Pure(dh, l.wk.WPtr(), dk16, B, dim, kvW)
			cuda.GemmF16Pure(dh, l.wv.WPtr(), dv16, B, dim, kvW)
			cuda.FreeDev(dh)
			cuda.RoPEF16DposRaw(dq16, invDev.DevPtr(), B, heads, hd, pos, rqAttr)
			cuda.RoPEF16DposRaw(dk16, invDev.DevPtr(), B, kv, hd, pos, rkAttr)
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
		cuda.CvtF16ToF32(hidOut.DevPtr(), xn, B*dim)
		cuda.FreeDev(xn)
	}
	sample := func() []int {
		lg, _ := outW.MatMulDevice(hidOut) // [B, vocab]
		host := make([]float32, B*cfg.Vocab)
		lg.DownloadF32(host)
		lg.Free()
		out := make([]int, B)
		for b := 0; b < B; b++ {
			out[b] = argmaxRow(host, b*cfg.Vocab)
		}
		return out
	}

	// prefill (eager on the batched buffers)
	var last []int
	for i := 0; i < promptLen; i++ {
		toks := make([]int32, B)
		for b := 0; b < B; b++ {
			toks[b] = prompts[b][i]
		}
		emb.EmbedInto(toks, x0)
		pos.Set(i)
		step()
		cuda.GraphSync()
		last = sample()
	}
	// capture + replay
	toks := make([]int32, B)
	copy(toks, int32s(last))
	emb.EmbedInto(toks, x0)
	pos.Set(promptLen)
	cuda.GraphSync()
	if err := cuda.CaptureBegin(); err != nil {
		t.Skipf("capture: %v", err)
	}
	step()
	g, err := cuda.CaptureEnd()
	if err != nil {
		t.Skipf("capture end: %v", err)
	}
	defer g.Free()
	got := make([][]int, B)
	for b := 0; b < B; b++ {
		got[b] = []int{last[b]}
	}
	for n := 0; n < maxGen-1; n++ {
		for b := 0; b < B; b++ {
			toks[b] = int32(got[b][len(got[b])-1])
		}
		emb.EmbedInto(toks, x0)
		pos.Set(promptLen + n)
		g.Launch()
		cuda.GraphSync()
		nx := sample()
		for b := 0; b < B; b++ {
			got[b] = append(got[b], nx[b])
		}
	}
	for b := 0; b < B; b++ {
		t.Logf("seq %d eager: %v", b, refs[b])
		t.Logf("seq %d graph: %v", b, got[b])
		for i := range refs[b] {
			if refs[b][i] != got[b][i] {
				t.Fatalf("seq %d diverged at %d: eager %d vs graph %d", b, i, refs[b][i], got[b][i])
			}
		}
	}
	t.Logf("=> batched graph-captured deployable decoder == eager for all %d seqs: serving-scale decode WORKS", B)
}

func int32s(a []int) []int32 {
	o := make([]int32, len(a))
	for i, v := range a {
		o[i] = int32(v)
	}
	return o
}
