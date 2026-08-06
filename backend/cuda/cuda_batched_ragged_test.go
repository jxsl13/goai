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

// raggedModel holds the f16-resident TinyLlama weights + a per-layer paged KV pool, enough to run the
// batched decode forward with RAGGED per-sequence positions (the B4 continuous-batching path).
type raggedModel struct {
	dim, heads, kv, hd, vocab int
	eps                       float32
	ropeBase                  float64
	emb                       *cuda.ResidentB
	fnorm                     *cuda.ResidentVec
	outW                      *cuda.ResidentBF16
	layers                    []*raggedLayer
	pools                     []*cuda.PagedKVPool
	invDev                    *cuda.DeviceF32
}

type raggedLayer struct {
	gA, gF                     *cuda.ResidentVec
	wq, wk, wv, wo, wg, wu, wd *cuda.ResidentBF16
}

func loadRaggedTinyLlama(tb testing.TB, maxBlocks int) *raggedModel {
	f, err := gguf.ReadFile(tinyLlamaPath)
	if err != nil {
		tb.Skipf("gguf: %v", err)
	}
	m, err := nlp.LlamaFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		tb.Fatal(err)
	}
	cfg := m.Config
	dim, heads := cfg.Dim, cfg.Heads
	kv := cfg.KVHeads
	if kv == 0 {
		kv = heads
	}
	hd := dim / heads
	if hd != 64 {
		tb.Skipf("needs hd==64, got %d", hd)
	}
	cast := func(tt *tensor.Tensor) *tensor.Tensor { return tt.Cast(tensor.F32) }
	mkW := func(tt *tensor.Tensor) *cuda.ResidentBF16 {
		r, e := cuda.NewResidentBF16(cast(tt))
		mustTB(tb, e)
		return r
	}
	mkV := func(tt *tensor.Tensor) *cuda.ResidentVec { r, _ := cuda.NewResidentVec(cast(tt)); return r }
	rm := &raggedModel{
		dim: dim, heads: heads, kv: kv, hd: hd, vocab: cfg.Vocab,
		eps: float32(cfg.Eps), ropeBase: cfg.RopeBase,
	}
	rm.emb, _ = cuda.NewResidentB(cast(m.TokEmb))
	rm.fnorm = mkV(m.Norm.Gamma)
	rm.outW = mkW(m.Out)
	rm.layers = make([]*raggedLayer, len(m.Blocks))
	rm.pools = make([]*cuda.PagedKVPool, len(m.Blocks))
	kvW := kv * hd
	for i, b := range m.Blocks {
		rm.layers[i] = &raggedLayer{gA: mkV(b.AttnNorm.Gamma), gF: mkV(b.FFNNorm.Gamma),
			wq: mkW(b.Wq), wk: mkW(b.Wk), wv: mkW(b.Wv), wo: mkW(b.Wo),
			wg: mkW(b.FFN.Wgate), wu: mkW(b.FFN.Wup), wd: mkW(b.FFN.Wdown)}
		rm.pools[i], _ = cuda.NewPagedKVPool(maxBlocks, 16, kvW)
	}
	invD, _ := backend.RoPEFreqs(hd, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: heads})
	inv32 := make([]float32, len(invD))
	for i := range invD {
		inv32[i] = float32(invD[i])
	}
	rm.invDev, _ = cuda.NewDeviceF32(1, len(inv32))
	rm.invDev.UploadF32(inv32)
	return rm
}

func (rm *raggedModel) free() {
	rm.emb.Free()
	rm.fnorm.Free()
	rm.outW.Free()
	rm.invDev.Free()
	for _, l := range rm.layers {
		l.gA.Free()
		l.gF.Free()
		for _, w := range []*cuda.ResidentBF16{l.wq, l.wk, l.wv, l.wo, l.wg, l.wu, l.wd} {
			w.Free()
		}
	}
	for _, p := range rm.pools {
		p.Free()
	}
}

// newSeq allocates one SeqKV per layer for a fresh sequence.
func (rm *raggedModel) newSeq() []*cuda.SeqKV {
	s := make([]*cuda.SeqKV, len(rm.layers))
	for li := range rm.layers {
		s[li] = rm.pools[li].NewSeqKV()
	}
	return s
}

func releaseSeq(s []*cuda.SeqKV) {
	for _, k := range s {
		k.Release()
	}
}

// step runs ONE ragged batched decode iteration: active[b] is a per-layer SeqKV slice, toks[b] its new
// token. Each sequence's RoPE position = its current length (they may all differ — the ragged case).
// Appends each sequence's K/V, returns the argmax token per row.
func (rm *raggedModel) step(active [][]*cuda.SeqKV, toks []int32) []int {
	B := len(active)
	positions := make([]int32, B)
	for b := range active {
		positions[b] = int32(active[b][0].Len())
	}
	xf, _ := rm.emb.Embed(toks)
	x, _ := cuda.F16FromF32(xf)
	xf.Free()
	rqa := backend.RoPEAttrs{Base: rm.ropeBase, Heads: rm.heads}
	rka := backend.RoPEAttrs{Base: rm.ropeBase, Heads: rm.kv}
	for li, l := range rm.layers {
		layerSeqs := make([]*cuda.SeqKV, B)
		for b := range active {
			layerSeqs[b] = active[b][li]
		}
		dh, _ := cuda.NewDeviceF16(B, rm.dim)
		x.RMSNormInto(l.gA, rm.eps, dh)
		dq, _ := l.wq.MatMulF16(dh)
		dk, _ := l.wk.MatMulF16(dh)
		dv, _ := l.wv.MatMulF16(dh)
		dh.Free()
		dq.RoPERagged(rm.invDev, positions, rqa)
		dk.RoPERagged(rm.invDev, positions, rka)
		dkf, _ := dk.ToF32()
		dvf, _ := dv.ToF32()
		dk.Free()
		dv.Free()
		for b := range layerSeqs {
			layerSeqs[b].Reserve1()
		}
		viewPre, _ := rm.pools[li].UploadBatchView(layerSeqs)
		rm.pools[li].AppendBatched(layerSeqs, dkf, dvf, viewPre)
		viewPre.Free()
		dkf.Free()
		dvf.Free()
		viewPost, _ := rm.pools[li].UploadBatchView(layerSeqs)
		dqf, _ := dq.ToF32()
		dq.Free()
		da, _ := rm.pools[li].BatchedDecodeAttnViewGQA(dqf, viewPost, rm.heads, rm.kv)
		dqf.Free()
		viewPost.Free()
		da16, _ := cuda.F16FromF32(da)
		da.Free()
		tmp, _ := l.wo.MatMulF16(da16)
		da16.Free()
		x.Add(tmp)
		tmp.Free()
		dh2, _ := cuda.NewDeviceF16(B, rm.dim)
		x.RMSNormInto(l.gF, rm.eps, dh2)
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
	nh, _ := x32.RMSNormTo(rm.fnorm, rm.eps)
	x32.Free()
	lg, _ := rm.outW.MatMulDevice(nh)
	nh.Free()
	host := make([]float32, B*rm.vocab)
	lg.DownloadF32(host)
	lg.Free()
	out := make([]int, B)
	for b := 0; b < B; b++ {
		best, bi := host[b*rm.vocab], 0
		for c := 1; c < rm.vocab; c++ {
			if host[b*rm.vocab+c] > best {
				best, bi = host[b*rm.vocab+c], c
			}
		}
		out[b] = bi
	}
	return out
}

// TestBatchedDecodeRaggedPositions proves continuous batching is per-sequence correct at RAGGED
// positions: sequence A decoded SOLO must produce the exact same token stream as sequence A decoded in
// a BATCH where a second sequence B is admitted mid-stream (so A and B sit at DIFFERENT positions). If
// the per-row RoPE position or the ragged paged attention were wrong, A's tokens would diverge.
func TestBatchedDecodeRaggedPositions(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const promptLen, gen, admitDelay = 8, 8, 3
	rm := loadRaggedTinyLlama(t, 4*(64/16+4)) // room for a few sequences
	defer rm.free()

	promptA := make([]int32, promptLen)
	promptB := make([]int32, promptLen)
	for i := range promptA {
		promptA[i] = int32((i*2129 + 41) % rm.vocab)
		promptB[i] = int32((i*1327 + 613) % rm.vocab)
	}

	// SOLO: decode A alone for `gen` tokens.
	solo := func(prompt []int32) []int {
		seq := rm.newSeq()
		defer releaseSeq(seq)
		var toks []int32
		out := make([]int, 0, gen)
		last := 0
		for pos := 0; pos < promptLen+gen; pos++ {
			var tok int32
			if pos < promptLen {
				tok = prompt[pos]
			} else {
				tok = int32(last)
			}
			toks = []int32{tok}
			nx := rm.step([][]*cuda.SeqKV{seq}, toks)
			last = nx[0]
			if pos >= promptLen-1 && len(out) < gen {
				out = append(out, last)
			}
		}
		return out
	}
	soloA := solo(promptA)

	// BATCHED: decode A from step 0; admit B at step `admitDelay`. A and B are then at different
	// positions every step. A's stream must equal soloA.
	seqA := rm.newSeq()
	seqB := rm.newSeq()
	defer releaseSeq(seqA)
	defer releaseSeq(seqB)
	batchedA := make([]int, 0, gen)
	lastA, lastB := 0, 0
	bStarted := false
	bStep := 0
	for pos := 0; pos < promptLen+gen; pos++ {
		var tokA int32
		if pos < promptLen {
			tokA = promptA[pos]
		} else {
			tokA = int32(lastA)
		}
		if pos >= admitDelay && !bStarted {
			bStarted = true
		}
		if !bStarted {
			nx := rm.step([][]*cuda.SeqKV{seqA}, []int32{tokA})
			lastA = nx[0]
		} else {
			var tokB int32
			if bStep < promptLen {
				tokB = promptB[bStep]
			} else {
				tokB = int32(lastB)
			}
			nx := rm.step([][]*cuda.SeqKV{seqA, seqB}, []int32{tokA, tokB})
			lastA, lastB = nx[0], nx[1]
			bStep++
		}
		if pos >= promptLen-1 && len(batchedA) < gen {
			batchedA = append(batchedA, lastA)
		}
	}

	if len(soloA) != len(batchedA) {
		t.Fatalf("length mismatch solo %d batched %d", len(soloA), len(batchedA))
	}
	for i := range soloA {
		if soloA[i] != batchedA[i] {
			t.Fatalf("ragged batching corrupts seq A at token %d: solo=%d batched=%d (full solo=%v batched=%v)",
				i, soloA[i], batchedA[i], soloA, batchedA)
		}
	}
	t.Logf("ragged parity OK — seq A identical solo vs batched-with-mid-stream-admit: %v", soloA)
}
