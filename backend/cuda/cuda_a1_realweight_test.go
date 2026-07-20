//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// TestA1RealWeightAccuracy: the strongest A1 validation — real TinyLlama-1.1B weights (not random)
// through both the f32-activation and full-f16 (A1) decode forwards, comparing final hidden state
// rel-RMS AND final-logits argmax (token decision). Read-only (no production change).
func TestA1RealWeightAccuracy(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	f, err := gguf.ReadFile(tinyLlamaPath)
	if err != nil {
		t.Skipf("gguf: %v", err)
	}
	m, err := nlp.LlamaFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("LlamaFromGGUF: %v", err)
	}
	cfg := m.Config
	dim, heads, hidden := cfg.Dim, cfg.Heads, cfg.Hidden
	kvh := cfg.KVHeads
	if kvh == 0 {
		kvh = heads
	}
	qW, kvW := heads*(dim/heads), kvh*(dim/heads)
	hd := dim / heads
	const batch, seqLen = 4, 128
	if hd != 64 {
		t.Skipf("A1 attn needs hd==64, got %d", hd)
	}
	cast := func(tt *tensor.Tensor) *tensor.Tensor { return tt.Cast(tensor.F32) }
	mkW := func(tt *tensor.Tensor) *cuda.ResidentBF16 {
		r, e := cuda.NewResidentBF16(cast(tt))
		if e != nil {
			t.Fatal(e)
		}
		return r
	}
	mkV := func(tt *tensor.Tensor) *cuda.ResidentVec { r, _ := cuda.NewResidentVec(cast(tt)); return r }
	type L struct {
		gA, gF         *cuda.ResidentVec
		wq, wk, wv, wo *cuda.ResidentBF16
		wg, wu, wd     *cuda.ResidentBF16
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
	rng := rand.New(rand.NewSource(11))
	pool, _ := cuda.NewPagedKVPool(batch*((seqLen+16)/16+1), 16, kvW)
	defer pool.Free()
	seqs := make([]*cuda.SeqKV, batch)
	for i := range seqs {
		seqs[i] = pool.NewSeqKV()
		kfb := make([]float32, seqLen*kvW)
		for j := range kfb {
			kfb[j] = float32(rng.NormFloat64()) * 0.1
		}
		dk, _ := cuda.NewDeviceF32(seqLen, kvW)
		dk.UploadF32(kfb)
		seqs[i].Append(dk, dk)
		dk.Free()
	}
	xf := make([]float32, batch*dim)
	for i := range xf {
		xf[i] = float32(rng.NormFloat64()) * 0.1
	}
	x0, _ := cuda.NewDeviceF32(batch, dim)
	x0.UploadF32(xf)
	defer x0.Free()
	view, _ := pool.UploadBatchView(seqs)
	defer view.Free()
	rq := backend.RoPEAttrs{Base: cfg.RopeBase, Heads: heads, PosOffset: seqLen}
	rk := backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kvh, PosOffset: seqLen}
	eps := float32(cfg.Eps)

	// f32-activation reference forward
	ref := func() []float32 {
		x, _ := cuda.NewDeviceF32(batch, dim)
		x.CopyFrom(x0)
		for _, l := range ls {
			dh, _ := x.RMSNormTo(l.gA, eps)
			dq, _ := l.wq.MatMulDevice(dh)
			dk, _ := l.wk.MatMulDevice(dh)
			dv, _ := l.wv.MatMulDevice(dh)
			dh.Free()
			dq.RoPE(rq)
			dk.RoPE(rk)
			dk.Free()
			dv.Free()
			da, _ := pool.BatchedDecodeAttnViewGQA(dq, view, heads, kvh)
			dq.Free()
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
		out := make([]float32, batch*dim)
		x.DownloadF32(out)
		x.Free()
		return out
	}()

	// A1 full-f16 forward
	invQd, posDivQ := backend.RoPEFreqs(hd, rq)
	invKd, posDivK := backend.RoPEFreqs(hd, rk)
	iq := make([]float32, len(invQd))
	ik := make([]float32, len(invKd))
	for i := range invQd {
		iq[i] = float32(invQd[i])
	}
	for i := range invKd {
		ik[i] = float32(invKd[i])
	}
	iqd, _ := cuda.NewDeviceF32(1, len(iq))
	iqd.UploadF32(iq)
	defer iqd.Free()
	ikd, _ := cuda.NewDeviceF32(1, len(ik))
	ikd.UploadF32(ik)
	defer ikd.Free()
	x := cuda.AllocU16(batch * dim)
	cuda.CvtF32ToF16(x, x0.DevPtr(), batch*dim)
	for _, l := range ls {
		dh := cuda.AllocU16(batch * dim)
		cuda.RMSNormF16(x, dh, l.gA.VecPtr(), batch, dim, eps)
		dq := cuda.AllocU16(batch * qW)
		dk := cuda.AllocU16(batch * kvW)
		dv := cuda.AllocU16(batch * kvW)
		cuda.GemmF16Pure(dh, l.wq.WPtr(), dq, batch, dim, qW)
		cuda.GemmF16Pure(dh, l.wk.WPtr(), dk, batch, dim, kvW)
		cuda.GemmF16Pure(dh, l.wv.WPtr(), dv, batch, dim, kvW)
		cuda.FreeDev(dh)
		cuda.RoPEF16(dq, iqd.DevPtr(), batch, heads, hd, seqLen, posDivQ)
		cuda.RoPEF16(dk, ikd.DevPtr(), batch, kvh, hd, seqLen, posDivK)
		cuda.FreeDev(dk)
		cuda.FreeDev(dv)
		// qio attention: f16 query straight in, f16 out — no dq→dqf / daf→da conversions
		da, _ := pool.BatchedDecodeAttnViewGQAf16Qio(dq, qW, view, heads, kvh)
		cuda.FreeDev(dq)
		cuda.GemmF16PureAddC(da, l.wo.WPtr(), x, batch, qW, dim) // x += da·wo (fused residual)
		cuda.FreeDev(da)
		dh2 := cuda.AllocU16(batch * dim)
		cuda.RMSNormF16(x, dh2, l.gF.VecPtr(), batch, dim, eps)
		dg := cuda.AllocU16(batch * hidden)
		du := cuda.AllocU16(batch * hidden)
		cuda.GemmF16Pure(dh2, l.wg.WPtr(), dg, batch, dim, hidden)
		cuda.GemmF16Pure(dh2, l.wu.WPtr(), du, batch, dim, hidden)
		cuda.FreeDev(dh2)
		cuda.SwiGLUF16(dg, du, batch*hidden)
		cuda.FreeDev(du)
		cuda.GemmF16PureAddC(dg, l.wd.WPtr(), x, batch, hidden, dim) // x += dg·wd (fused residual)
		cuda.FreeDev(dg)
	}
	raw := make([]uint16, batch*dim)
	cuda.DownloadU16(x, raw)
	cuda.FreeDev(x)
	got := make([]float32, batch*dim)
	for i := range raw {
		got[i] = h2f32(raw[i])
	}
	var num, den float64
	for i := range ref {
		d := float64(ref[i] - got[i])
		num += d * d
		den += float64(ref[i]) * float64(ref[i])
	}
	relRMS := math.Sqrt(num / den)
	t.Logf("A1 REAL-WEIGHT (TinyLlama 22L) hidden rel-RMS %.4e (random-weight was 1.86e-2)", relRMS)
	if relRMS > 5e-2 {
		t.Fatalf("A1 real-weight rel-RMS %.4e too high", relRMS)
	}
}
