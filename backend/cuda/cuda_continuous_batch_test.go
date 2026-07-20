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

// TestA1ContinuousBatchAdmit: continuous batching with ADMIT mid-stream. Seq A decodes solo for a few
// steps, THEN seq B is admitted (prefilled) and both decode together — a DYNAMIC active set. Because
// the active set changes, the paged view is re-uploaded each step from SeqKV.n; the eager decode step
// is: Reserve1 -> view(dsl=Len) -> forward -> AppendBatchedDev(slot=Len) -> Advance(1) -> view(dsl=Len+1)
// -> attention, at per-seq positions (RoPEF16DposArrRaw). Each seq's full stream matches its own eager
// (batch=1) reference. This is the continuous-batching serving loop (eager path).
func TestA1ContinuousBatchAdmit(t *testing.T) {
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
	argmaxRow := func(host []float32, off int) int {
		best, bi := host[off], 0
		for c := 1; c < cfg.Vocab; c++ {
			if host[off+c] > best {
				best, bi = host[off+c], c
			}
		}
		return bi
	}
	enc := func(s string) []int32 {
		ids := append([]int{1}, tok.Encode(s)...)
		o := make([]int32, len(ids))
		for i, v := range ids {
			o[i] = int32(v)
		}
		return o
	}

	// per-layer shared pools with 2 seq slots (0=A, 1=B).
	pools := make([]*cuda.PagedKVPool, len(ls))
	seqs := make([][]*cuda.SeqKV, len(ls))
	for i := range ls {
		pools[i], _ = cuda.NewPagedKVPool(2*2, 16, kvW)
		seqs[i] = []*cuda.SeqKV{pools[i].NewSeqKV(), pools[i].NewSeqKV()}
	}
	defer func() {
		for i := range pools {
			for _, s := range seqs[i] {
				s.Release()
			}
			pools[i].Free()
		}
	}()

	// forward one decode step over `active` slots (input token + absolute position per active seq).
	// Uses the eager dynamic-set pattern: Reserve1 -> view(dsl=Len) -> AppendBatchedDev -> Advance ->
	// view(dsl=Len+1) -> attention. Returns each active seq's argmax next token.
	forward := func(active []int, toks []int32, positions []int32) []int {
		B := len(active)
		xf, _ := emb.Embed(toks)
		x := cuda.AllocU16(B * dim)
		cuda.CvtF32ToF16(x, xf.DevPtr(), B*dim)
		xf.Free()
		dpos := cuda.UploadI32(positions)
		for li, l := range ls {
			as := make([]*cuda.SeqKV, B)
			for k, sl := range active {
				as[k] = seqs[li][sl]
				as[k].Reserve1()
			}
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
			v1, _ := pools[li].UploadBatchView(as) // dsl = Len (pre-append)
			pools[li].AppendBatchedDev(dkf, dvf, v1)
			v1.Free()
			for _, s := range as {
				s.Advance(1) // host length now includes the appended token
			}
			dkf.Free()
			dvf.Free()
			v2, _ := pools[li].UploadBatchView(as) // dsl = Len+1 -> attention includes current token
			dqf, _ := cuda.NewDeviceF32(B, qW)
			cuda.CvtF16ToF32(dqf.DevPtr(), dq16, B*qW)
			cuda.FreeDev(dq16)
			attnOut, _ := cuda.NewDeviceF32(B, qW)
			pools[li].BatchedDecodeAttnViewInto(dqf, v2, heads, kv, attnOut)
			dqf.Free()
			v2.Free()
			da := cuda.AllocU16(B * qW)
			cuda.CvtF32ToF16(da, attnOut.DevPtr(), B*qW)
			attnOut.Free()
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
		res := make([]int, B)
		for b := 0; b < B; b++ {
			res[b] = argmaxRow(host, b*cfg.Vocab)
		}
		return res
	}

	// eager reference (batch=1) for one sequence, reusing `forward` at B=1 on a private slot pair.
	promptA := enc("The capital of France")
	promptB := enc("Hello there")
	const soloA, together = 3, 4

	// drive: A prefill + soloA decode; then admit B (prefill) + `together` joint decode.
	// track per-slot generated tokens and current length (Len()).
	gen := map[int][]int{}
	lenOf := func(slot int) int { return seqs[0][slot].Len() }
	// prefill+decode helpers operate on the shared pools via `forward`.
	feed := func(active []int, toks []int32) []int {
		positions := make([]int32, len(active))
		for i, sl := range active {
			positions[i] = int32(lenOf(sl))
		}
		return forward(active, toks, positions)
	}
	// prefill A
	var lastA int
	for _, id := range promptA {
		lastA = feed([]int{0}, []int32{id})[0]
	}
	gen[0] = []int{lastA}
	// A solo decode (soloA-1 more so A has soloA generated tokens total incl first)
	for n := 0; n < soloA-1; n++ {
		lastA = feed([]int{0}, []int32{int32(gen[0][len(gen[0])-1])})[0]
		gen[0] = append(gen[0], lastA)
	}
	// ADMIT B: prefill B on slot 1
	var lastB int
	for _, id := range promptB {
		lastB = feed([]int{1}, []int32{id})[0]
	}
	gen[1] = []int{lastB}
	// joint decode A+B for `together` steps (A already has soloA; give it `together` more here for B's count)
	for n := 0; n < together-1; n++ {
		nx := feed([]int{0, 1}, []int32{int32(gen[0][len(gen[0])-1]), int32(gen[1][len(gen[1])-1])})
		gen[0] = append(gen[0], nx[0])
		gen[1] = append(gen[1], nx[1])
	}
	// EVICT A (as if finished): release its blocks; B continues SOLO — the active set shrinks 2->1 and
	// B's KV/position are unaffected. This is the evict half of continuous batching.
	for li := range ls {
		seqs[li][0].Release()
	}
	for n := 0; n < 2; n++ {
		lastB = feed([]int{1}, []int32{int32(gen[1][len(gen[1])-1])})[0]
		gen[1] = append(gen[1], lastB)
	}

	// eager references
	eagerOne := func(prompt []int32, maxGen int) []int {
		p := make([]*cuda.PagedKVPool, len(ls))
		s := make([]*cuda.SeqKV, len(ls))
		for i := range ls {
			p[i], _ = cuda.NewPagedKVPool(4, 16, kvW)
			s[i] = p[i].NewSeqKV()
		}
		defer func() {
			for i := range p {
				s[i].Release()
				p[i].Free()
			}
		}()
		step := func(tokID int32, pos int) int {
			xf, _ := emb.Embed([]int32{tokID})
			xx, _ := cuda.F16FromF32(xf)
			xf.Free()
			rqp := backend.RoPEAttrs{Base: cfg.RopeBase, Heads: heads, PosOffset: pos}
			rkp := backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv, PosOffset: pos}
			for li, l := range ls {
				dh, _ := cuda.NewDeviceF16(1, dim)
				xx.RMSNormInto(l.gA, eps, dh)
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
				s[li].Append(dkf, dvf)
				view, _ := p[li].UploadBatchView([]*cuda.SeqKV{s[li]})
				dqf, _ := dq.ToF32()
				dq.Free()
				da, _ := p[li].BatchedDecodeAttnViewGQA(dqf, view, heads, kv)
				dqf.Free()
				view.Free()
				dkf.Free()
				dvf.Free()
				da16, _ := cuda.F16FromF32(da)
				da.Free()
				l.wo.MatMulF16AddInto(da16, xx) // xx += da·wo (fused residual)
				da16.Free()
				dh2, _ := cuda.NewDeviceF16(1, dim)
				xx.RMSNormInto(l.gF, eps, dh2)
				dg, _ := l.wg.MatMulF16(dh2)
				du, _ := l.wu.MatMulF16(dh2)
				dh2.Free()
				dg.SwiGLU(du)
				du.Free()
				l.wd.MatMulF16AddInto(dg, xx) // xx += dg·wd (fused residual)
				dg.Free()
			}
			x32, _ := xx.ToF32()
			xx.Free()
			nh, _ := x32.RMSNormTo(fnorm, eps)
			x32.Free()
			lg, _ := outW.MatMulDevice(nh)
			nh.Free()
			hh := make([]float32, cfg.Vocab)
			lg.DownloadF32(hh)
			lg.Free()
			return argmaxRow(hh, 0)
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
	refA := eagerOne(promptA, len(gen[0]))
	refB := eagerOne(promptB, len(gen[1]))
	t.Logf("A eager: %v", refA)
	t.Logf("A cbatch:%v", gen[0])
	t.Logf("B eager: %v", refB)
	t.Logf("B cbatch:%v", gen[1])
	for i := range refA {
		if refA[i] != gen[0][i] {
			t.Fatalf("A diverged at %d: eager %d vs cbatch %d", i, refA[i], gen[0][i])
		}
	}
	for i := range refB {
		if refB[i] != gen[1][i] {
			t.Fatalf("B diverged at %d: eager %d vs cbatch %d", i, refB[i], gen[1][i])
		}
	}
	t.Logf("=> CONTINUOUS BATCHING (ADMIT + EVICT) WORKS: A(solo->joint->evicted) + B(admitted->joint->solo) both == eager")
}
