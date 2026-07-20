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

// TestA1PrimitiveOrdering validates the deployable decoder's core: the CAPTURABLE primitive path
// (Reserve1 -> UploadBatchView(dsl=pos) -> AppendBatched(writes slot=pos) -> BumpLens(1)(dsl=pos+1)
// -> attention(reads pos+1)) must produce the SAME correct tokens as the eager SeqKV.Append path.
// All device ops (AppendBatched, BumpLens, attention) are capturable, so if this matches eager, the
// correct-ordering decode step is graph-capturable => deployable fast decoder is unblocked.
func TestA1PrimitiveOrdering(t *testing.T) {
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
	ids := append([]int{1}, tok.Encode("The capital of France is")...)
	const maxGen = 16

	gen := func(prim bool) []int {
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
				var da *cuda.DeviceF32
				if prim {
					// CAPTURABLE primitive path: append via AppendBatched(slot=dsl) + BumpLens
					seqs[li].Reserve1()
					view, _ := pools[li].UploadBatchView([]*cuda.SeqKV{seqs[li]}) // dsl=pos
					pools[li].AppendBatched([]*cuda.SeqKV{seqs[li]}, dkf, dvf, view)
					view.BumpLens(1) // dsl=pos+1 (attention includes the just-appended token)
					dqf, _ := dq.ToF32()
					dq.Free()
					da, _ = pools[li].BatchedDecodeAttnViewGQA(dqf, view, heads, kv)
					dqf.Free()
					view.Free()
				} else {
					// eager reference: host-side SeqKV.Append then attend
					seqs[li].Append(dkf, dvf)
					view, _ := pools[li].UploadBatchView([]*cuda.SeqKV{seqs[li]})
					dqf, _ := dq.ToF32()
					dq.Free()
					da, _ = pools[li].BatchedDecodeAttnViewGQA(dqf, view, heads, kv)
					dqf.Free()
					view.Free()
				}
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
		out := []int{}
		last := 0
		for i, id := range ids {
			last = step(int32(id), i)
		}
		for n := 0; n < maxGen; n++ {
			if last == 2 {
				break
			}
			out = append(out, last)
			last = step(int32(last), len(ids)+n)
		}
		return out
	}
	ref := gen(false)
	prim := gen(true)
	t.Logf("eager     : %v", ref)
	t.Logf("primitives: %v", prim)
	n := len(ref)
	if len(prim) < n {
		n = len(prim)
	}
	for i := 0; i < n; i++ {
		if ref[i] != prim[i] {
			t.Fatalf("primitive path diverged at %d: eager %d vs prim %d — AppendBatched+BumpLens ordering wrong", i, ref[i], prim[i])
		}
	}
	t.Logf("=> AppendBatched+BumpLens == eager (%d/%d): capturable correct-ordering VALIDATED for graph decode", n, n)
}
