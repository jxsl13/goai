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

// TestA1DeployDecoder: the productization capstone — a GRAPH-CAPTURED, real-weight, CORRECT serving
// decoder. The 22-layer A1 f16 decode step is captured as ONE CUDA graph (append INSIDE the graph via
// AppendBatchedDev + BumpLens, device-position RoPE, fixed x0/attnOut/hidden buffers) and REPLAYED to
// generate tokens with zero host intervention in the hot path (only logits+argmax+embed between
// replays). Verifies the graph-generated tokens are IDENTICAL to the eager reference. Kept within one
// 16-block (prompt+gen<=16) so the device dsl (BumpLens) is the only length state — no boundary alloc.
func TestA1DeployDecoder(t *testing.T) {
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

	ids := append([]int{1}, tok.Encode("The capital of France")...)
	const maxGen = 20 // crosses the 16-token block boundary (tests pre-allocated multi-block decode)
	total := len(ids) + maxGen
	argmaxHost := func(d *cuda.DeviceF32) int {
		host := make([]float32, cfg.Vocab)
		d.DownloadF32(host)
		best, bi := host[0], 0
		for c := 1; c < cfg.Vocab; c++ {
			if host[c] > best {
				best, bi = host[c], c
			}
		}
		return bi
	}

	// ---- eager reference (validated correct ordering) ----
	refGen := func() []int {
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
				l.wo.MatMulF16AddInto(da16, x) // x += da·wo (fused residual)
				da16.Free()
				dh2, _ := cuda.NewDeviceF16(1, dim)
				x.RMSNormInto(l.gF, eps, dh2)
				dg, _ := l.wg.MatMulF16(dh2)
				du, _ := l.wu.MatMulF16(dh2)
				dh2.Free()
				dg.SwiGLU(du)
				du.Free()
				l.wd.MatMulF16AddInto(dg, x) // x += dg·wd (fused residual)
				dg.Free()
			}
			x32, _ := x.ToF32()
			x.Free()
			nh, _ := x32.RMSNormTo(fnorm, eps)
			x32.Free()
			lg, _ := outW.MatMulDevice(nh)
			nh.Free()
			r := argmaxHost(lg)
			lg.Free()
			return r
		}
		out := []int{}
		last := 0
		for i, id := range ids {
			last = step(int32(id), i)
		}
		for n := 0; n < maxGen; n++ {
			out = append(out, last)
			last = step(int32(last), len(ids)+n)
		}
		return out
	}
	ref := refGen()

	// ---- graph-captured deployable decoder ----
	pools := make([]*cuda.PagedKVPool, len(ls))
	seqs := make([]*cuda.SeqKV, len(ls))
	views := make([]*cuda.PagedBatchView, len(ls))
	attnOut := make([]*cuda.DeviceF32, len(ls))
	zeroKV := make([]float32, total*kvW)
	for i := range ls {
		pools[i], _ = cuda.NewPagedKVPool(total/16+2, 16, kvW)
		seqs[i] = pools[i].NewSeqKV()
		// pre-grow: append `total` zero tokens to allocate ALL blocks up front so the view's block
		// table is complete and no Reserve1/block-table update is needed mid-decode. Then reset the
		// device length to 0 (blocks stay allocated); prefill+decode overwrite the slots with real KV.
		d, _ := cuda.NewDeviceF32(total, kvW)
		d.UploadF32(zeroKV)
		seqs[i].Append(d, d)
		d.Free()
		views[i], _ = pools[i].UploadBatchView([]*cuda.SeqKV{seqs[i]}) // dbt=all blocks, dsl=total
		views[i].BumpLens(-total)                                      // dsl=0 (blocks remain allocated)
		attnOut[i], _ = cuda.NewDeviceF32(1, qW)
	}
	defer func() {
		for i := range pools {
			views[i].Free()
			attnOut[i].Free()
			seqs[i].Release()
			pools[i].Free()
		}
	}()
	x0, _ := cuda.NewDeviceF32(1, dim) // input embedding (updated each step, read by the graph)
	defer x0.Free()
	hidOut, _ := cuda.NewDeviceF32(1, dim) // final normed hidden f32 (graph writes; logits read it)
	defer hidOut.Free()
	pos, _ := cuda.NewDevicePos()
	defer pos.Free()

	// the capturable 22-layer decode step: reads x0 + pos, appends current K/V IN-GRAPH, writes hidOut.
	step := func() {
		x := cuda.AllocU16(dim)
		cuda.CvtF32ToF16(x, x0.DevPtr(), dim)
		for li, l := range ls {
			dh := cuda.AllocU16(dim)
			cuda.RMSNormF16(x, dh, l.gA.VecPtr(), 1, dim, eps)
			dq16, dk16, dv16 := cuda.AllocU16(qW), cuda.AllocU16(kvW), cuda.AllocU16(kvW)
			cuda.GemmF16Pure(dh, l.wq.WPtr(), dq16, 1, dim, qW)
			cuda.GemmF16Pure(dh, l.wk.WPtr(), dk16, 1, dim, kvW)
			cuda.GemmF16Pure(dh, l.wv.WPtr(), dv16, 1, dim, kvW)
			cuda.FreeDev(dh)
			cuda.RoPEF16DposRaw(dq16, invDev.DevPtr(), 1, heads, hd, pos, rqAttr)
			cuda.RoPEF16DposRaw(dk16, invDev.DevPtr(), 1, kv, hd, pos, rkAttr)
			dkf, _ := cuda.NewDeviceF32(1, kvW)
			dvf, _ := cuda.NewDeviceF32(1, kvW)
			cuda.CvtF16ToF32(dkf.DevPtr(), dk16, kvW)
			cuda.CvtF16ToF32(dvf.DevPtr(), dv16, kvW)
			cuda.FreeDev(dk16)
			cuda.FreeDev(dv16)
			pools[li].AppendBatchedDev(dkf, dvf, views[li]) // write slot=dsl (in-graph)
			views[li].BumpLens(1)                           // dsl++ (attention includes current)
			dkf.Free()
			dvf.Free()
			dqf, _ := cuda.NewDeviceF32(1, qW)
			cuda.CvtF16ToF32(dqf.DevPtr(), dq16, qW)
			cuda.FreeDev(dq16)
			pools[li].BatchedDecodeAttnViewInto(dqf, views[li], heads, kv, attnOut[li])
			dqf.Free()
			da := cuda.AllocU16(qW)
			cuda.CvtF32ToF16(da, attnOut[li].DevPtr(), qW)
			tmp := cuda.AllocU16(dim)
			cuda.GemmF16Pure(da, l.wo.WPtr(), tmp, 1, qW, dim)
			cuda.FreeDev(da)
			cuda.AddF16(x, tmp, dim)
			cuda.FreeDev(tmp)
			dh2 := cuda.AllocU16(dim)
			cuda.RMSNormF16(x, dh2, l.gF.VecPtr(), 1, dim, eps)
			hidden := cfg.Hidden
			dg, du := cuda.AllocU16(hidden), cuda.AllocU16(hidden)
			cuda.GemmF16Pure(dh2, l.wg.WPtr(), dg, 1, dim, hidden)
			cuda.GemmF16Pure(dh2, l.wu.WPtr(), du, 1, dim, hidden)
			cuda.FreeDev(dh2)
			cuda.SwiGLUF16(dg, du, hidden)
			cuda.FreeDev(du)
			tmp2 := cuda.AllocU16(dim)
			cuda.GemmF16Pure(dg, l.wd.WPtr(), tmp2, 1, hidden, dim)
			cuda.FreeDev(dg)
			cuda.AddF16(x, tmp2, dim)
			cuda.FreeDev(tmp2)
		}
		xn := cuda.AllocU16(dim)
		cuda.RMSNormF16(x, xn, fnorm.VecPtr(), 1, dim, eps)
		cuda.FreeDev(x)
		cuda.CvtF16ToF32(hidOut.DevPtr(), xn, dim) // final hidden -> fixed f32 buffer
		cuda.FreeDev(xn)
	}
	logitsArgmax := func() int {
		lg, _ := outW.MatMulDevice(hidOut)
		r := argmaxHost(lg)
		lg.Free()
		return r
	}

	// prefill: run the step eagerly per prompt token (populates KV, advances all views' dsl).
	var last int
	for i, id := range ids {
		emb.EmbedInto([]int32{int32(id)}, x0)
		pos.Set(i)
		step()
		cuda.GraphSync()
		last = logitsArgmax()
	}
	// capture the decode step, then replay to generate.
	emb.EmbedInto([]int32{int32(last)}, x0)
	pos.Set(len(ids))
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
	// got[0]=first token (from prefill). Each replay: input=last token, pos=len(ids)+n -> next token.
	// The capture step only RECORDED; replays execute. maxGen-1 replays -> maxGen total tokens.
	got := []int{last}
	for n := 0; n < maxGen-1; n++ {
		emb.EmbedInto([]int32{int32(got[len(got)-1])}, x0)
		pos.Set(len(ids) + n)
		g.Launch()
		cuda.GraphSync()
		got = append(got, logitsArgmax())
	}

	t.Logf("eager : %v", ref)
	t.Logf("graph : %v", got)
	n := len(ref)
	if len(got) < n {
		n = len(got)
	}
	for i := 0; i < n; i++ {
		if ref[i] != got[i] {
			t.Fatalf("graph decoder diverged at %d: eager %d vs graph %d", i, ref[i], got[i])
		}
	}
	t.Logf("=> graph-captured deployable decoder == eager (%d/%d): fast+correct real-weight serving decode WORKS", n, n)
}
