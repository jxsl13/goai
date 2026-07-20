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

// TestA1RealPromptTokenMatch: the capstone A1 validation — real TinyLlama weights + a real token
// sequence through both the f32-activation and A1 (DeviceF16) PREFILL forwards, comparing the
// argmax next-token prediction at every position. If A1's f16 activations preserved the model's
// decisions, the predicted tokens match. WMMA causal attention (f32) is shared; only the GEMMs +
// elementwise ops differ (f16 vs f32). Read-only test.
func TestA1RealPromptTokenMatch(t *testing.T) {
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
		t.Skipf("WMMA attn needs hd==64, got %d", hd)
	}
	const seq = 16 // WMMA needs seq%16==0
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
	ids := make([]int32, seq)
	for i := range ids {
		ids[i] = int32((i*2129 + 41) % cfg.Vocab) // deterministic real token ids
	}
	rq := backend.RoPEAttrs{Base: cfg.RopeBase, Heads: heads}
	rk := backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv}
	eps := float32(cfg.Eps)
	invD, _ := backend.RoPEFreqs(hd, rq)
	inv32 := make([]float32, len(invD))
	for i := range invD {
		inv32[i] = float32(invD[i])
	}
	invDev, _ := cuda.NewDeviceF32(1, len(inv32))
	invDev.UploadF32(inv32)
	defer invDev.Free()

	logitsToTokens := func(hidden32 *cuda.DeviceF32) []int {
		nh, _ := hidden32.RMSNormTo(fnorm, eps)
		lg, _ := outW.MatMulDevice(nh)
		nh.Free()
		vocab := cfg.Vocab
		host := make([]float32, seq*vocab)
		lg.DownloadF32(host)
		lg.Free()
		toks := make([]int, seq)
		for r := 0; r < seq; r++ {
			best, bi := host[r*vocab], 0
			for c := 1; c < vocab; c++ {
				if host[r*vocab+c] > best {
					best, bi = host[r*vocab+c], c
				}
			}
			toks[r] = bi
		}
		return toks
	}

	// f32-activation prefill
	tokF32 := func() []int {
		x, _ := emb.Embed(ids)
		for _, l := range ls {
			dh, _ := x.RMSNormTo(l.gA, eps)
			dq, _ := l.wq.MatMulDevice(dh)
			dk, _ := l.wk.MatMulDevice(dh)
			dv, _ := l.wv.MatMulDevice(dh)
			dh.Free()
			dq.RoPE(rq)
			dk.RoPE(rk)
			da, _ := cuda.GroupedQueryAttentionWMMA(dq, dk, dv, heads, kv)
			dq.Free()
			dk.Free()
			dv.Free()
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
		toks := logitsToTokens(x)
		x.Free()
		return toks
	}()

	// A1 (DeviceF16) prefill
	tokA1 := func() []int {
		xf, _ := emb.Embed(ids)
		x, _ := cuda.F16FromF32(xf)
		xf.Free()
		for _, l := range ls {
			dh, _ := cuda.NewDeviceF16(seq, dim)
			x.RMSNormInto(l.gA, eps, dh)
			dq, _ := l.wq.MatMulF16(dh)
			dk, _ := l.wk.MatMulF16(dh)
			dv, _ := l.wv.MatMulF16(dh)
			dh.Free()
			dq.RoPE(invDev, rq)
			dk.RoPE(invDev, rk)
			dqf, _ := dq.ToF32()
			dkf, _ := dk.ToF32()
			dvf, _ := dv.ToF32()
			dq.Free()
			dk.Free()
			dv.Free()
			da, _ := cuda.GroupedQueryAttentionWMMA(dqf, dkf, dvf, heads, kv)
			dqf.Free()
			dkf.Free()
			dvf.Free()
			da16, _ := cuda.F16FromF32(da)
			da.Free()
			tmp, _ := l.wo.MatMulF16(da16)
			da16.Free()
			x.Add(tmp)
			tmp.Free()
			dh2, _ := cuda.NewDeviceF16(seq, dim)
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
		toks := logitsToTokens(x32)
		x32.Free()
		return toks
	}()

	match := 0
	for i := range tokF32 {
		if tokF32[i] == tokA1[i] {
			match++
		}
	}
	t.Logf("A1 real-prompt token match: %d/%d positions agree (f32 vs A1 prefill, TinyLlama 22L)", match, seq)
	t.Logf("  f32 next-token(last): %d   A1 next-token(last): %d", tokF32[seq-1], tokA1[seq-1])
	if match < seq*3/4 {
		t.Fatalf("A1 token match too low: %d/%d", match, seq)
	}
}
