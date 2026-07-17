//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

func mustTB(tb testing.TB, err error) {
	tb.Helper()
	if err != nil {
		tb.Fatal(err)
	}
}

// resLayerQ8 is resLayer with the seven projection weights Q8_0-quantized (norms
// and attention stay f32) — the decode-bandwidth variant.
type resLayerQ8 struct {
	gAttn, gFFN            *cuda.ResidentVec
	wq, wk, wv, wo         *cuda.ResidentBQ8
	wg, wu, wd             *cuda.ResidentBQ8
	heads, kv, dim, hidden int
	ropeBase, eps          float64
}

func (l *resLayerQ8) free() {
	for _, f := range []interface{ Free() }{l.gAttn, l.gFFN, l.wq, l.wk, l.wv, l.wo, l.wg, l.wu, l.wd} {
		f.Free()
	}
}

func (l *resLayerQ8) forwardKV(dx *cuda.DeviceF32, kc *cuda.KVCache) (*cuda.DeviceF32, error) {
	pos := kc.Len()
	rq := backend.RoPEAttrs{Base: l.ropeBase, Heads: l.heads, PosOffset: pos}
	rk := backend.RoPEAttrs{Base: l.ropeBase, Heads: l.kv, PosOffset: pos}
	dh, err := dx.RMSNormTo(l.gAttn, float32(l.eps))
	if err != nil {
		return nil, err
	}
	dq, err := l.wq.QMatMulDevice(dh)
	if err != nil {
		return nil, err
	}
	dk, _ := l.wk.QMatMulDevice(dh)
	dv, _ := l.wv.QMatMulDevice(dh)
	dh.Free()
	dq.RoPE(rq)
	dk.RoPE(rk)
	if err := kc.Append(dk, dv); err != nil {
		return nil, err
	}
	da, err := cuda.GroupedQueryAttentionKV(dq, kc.K(), kc.V(), l.heads, l.kv)
	if err != nil {
		return nil, err
	}
	dq.Free()
	dk.Free()
	dv.Free()
	if err := l.wo.QMatMulAccInto(da, dx); err != nil {
		return nil, err
	}
	da.Free()
	dh2, _ := dx.RMSNormTo(l.gFFN, float32(l.eps))
	dgate, _ := l.wg.QMatMulDevice(dh2)
	dup, _ := l.wu.QMatMulDevice(dh2)
	dh2.Free()
	dgate.SwiGLU(dup)
	dup.Free()
	if err := l.wd.QMatMulAccInto(dgate, dx); err != nil {
		return nil, err
	}
	dgate.Free()
	return dx, nil
}

type residentLlamaQ8 struct {
	emb        *cuda.ResidentB // embedding gather stays f32
	norm       *cuda.ResidentVec
	out        *cuda.ResidentBQ8
	layers     []*resLayerQ8
	dim, vocab int
	eps        float64
}

func buildResidentLlamaQ8(tb testing.TB, m *nlp.Llama) *residentLlamaQ8 {
	cfg := m.Config
	kv := cfg.KVHeads
	if kv == 0 {
		kv = cfg.Heads
	}
	cast := func(t *tensor.Tensor) *tensor.Tensor { return t.Cast(tensor.F32) }
	q := func(w *tensor.Tensor) *cuda.ResidentBQ8 {
		r, e := cuda.NewResidentBQ8(cast(w))
		mustTB(tb, e)
		return r
	}
	rl := &residentLlamaQ8{dim: cfg.Dim, vocab: cfg.Vocab, eps: cfg.Eps}
	rl.emb, _ = cuda.NewResidentB(cast(m.TokEmb))
	rl.norm, _ = cuda.NewResidentVec(cast(m.Norm.Gamma))
	rl.out = q(m.Out)
	rl.layers = make([]*resLayerQ8, len(m.Blocks))
	for i, blk := range m.Blocks {
		ga, _ := cuda.NewResidentVec(cast(blk.AttnNorm.Gamma))
		gf, _ := cuda.NewResidentVec(cast(blk.FFNNorm.Gamma))
		rl.layers[i] = &resLayerQ8{
			gAttn: ga, gFFN: gf,
			wq: q(blk.Wq), wk: q(blk.Wk), wv: q(blk.Wv), wo: q(blk.Wo),
			wg: q(blk.FFN.Wgate), wu: q(blk.FFN.Wup), wd: q(blk.FFN.Wdown),
			heads: cfg.Heads, kv: kv, dim: cfg.Dim, hidden: cfg.Hidden,
			ropeBase: cfg.RopeBase, eps: cfg.Eps,
		}
	}
	return rl
}

func (rl *residentLlamaQ8) free() {
	rl.emb.Free()
	rl.norm.Free()
	rl.out.Free()
	for _, l := range rl.layers {
		l.free()
	}
}

func (rl *residentLlamaQ8) step(tb testing.TB, ids []int32, caches []*cuda.KVCache) *tensor.Tensor {
	dx, err := rl.emb.Embed(ids)
	mustTB(tb, err)
	for i, l := range rl.layers {
		if dx, err = l.forwardKV(dx, caches[i]); err != nil {
			tb.Fatal(err)
		}
	}
	mustTB(tb, dx.RMSNorm(rl.norm, float32(rl.eps)))
	dl, err := rl.out.QMatMulDevice(dx)
	mustTB(tb, err)
	dx.Free()
	out, err := dl.ToHost()
	mustTB(tb, err)
	dl.Free()
	return out
}

func (rl *residentLlamaQ8) newCaches(maxSeq int) []*cuda.KVCache {
	wkv := rl.layers[0].kv * (rl.layers[0].dim / rl.layers[0].heads)
	c := make([]*cuda.KVCache, len(rl.layers))
	for i := range c {
		c[i], _ = cuda.NewKVCache(maxSeq, wkv)
	}
	return c
}

// Q8 full-model decode vs the f32 model: greedy tokens should match (Q8 is lossy
// ~0.4%/matmul, but the top logit is usually well separated); the per-position
// logit L1 error quantifies the accuracy cost of quantization end-to-end.
func TestCUDATinyLlamaQ8DecodeVsF32(t *testing.T) {
	if testing.Short() {
		t.Skip("model-loading integration test; -short = kernel-level gate")
	}
	m := loadTinyLlama(t)
	q8 := buildResidentLlamaQ8(t, m)
	defer q8.free()
	f32 := buildResidentLlama(t, m)
	defer f32.free()

	prompt := []int32{1, 15043, 29892, 590, 1024, 338}
	const steps = 5
	qc := q8.newCaches(len(prompt) + steps + 1)
	fc, _ := f32.newCaches(len(prompt) + steps + 1)
	defer func() {
		for _, c := range qc {
			c.Free()
		}
		for _, c := range fc {
			c.Free()
		}
	}()

	qlog := q8.step(t, prompt, qc)
	flog := f32.step(t, prompt, fc)
	match := 0
	for s := 0; s < steps; s++ {
		qt, ft := argmaxRow(qlog, qlog.Shape()[0]-1), argmaxRow(flog, flog.Shape()[0]-1)
		var l1, ref float64
		for c := 0; c < qlog.Shape()[1]; c++ {
			a, b := qlog.AtF64(qlog.Shape()[0]-1, c), flog.AtF64(flog.Shape()[0]-1, c)
			l1 += math.Abs(a - b)
			ref += math.Abs(b)
		}
		if qt == ft {
			match++
		}
		t.Logf("step %d: Q8 token %d, f32 token %d, %s (logit L1 rel %.3f)", s, qt, ft,
			map[bool]string{true: "MATCH", false: "differ"}[qt == ft], l1/(ref+1e-9))
		qlog = q8.step(t, []int32{int32(qt)}, qc)
		flog = f32.step(t, []int32{int32(ft)}, fc)
	}
	t.Logf("Q8 vs f32 greedy: %d/%d tokens match", match, steps)
	if match < steps-1 { // allow at most one late divergence
		t.Fatalf("Q8 decode diverged too much: only %d/%d tokens match f32", match, steps)
	}
}

// End-to-end decode throughput: Q8 (int8 weights, 4× less bandwidth) vs f32.
func benchDecodeQ8(b *testing.B, prefill, decode int) {
	m := loadTinyLlama(b)
	rl := buildResidentLlamaQ8(b, m)
	defer rl.free()
	prompt := make([]int32, prefill)
	for i := range prompt {
		prompt[i] = int32((i*131 + 1) % m.Config.Vocab)
	}
	for range b.N {
		b.StopTimer()
		caches := rl.newCaches(prefill + decode + 1)
		kv := rl.step(b, prompt, caches)
		tok := int32(argmaxRow(kv, kv.Shape()[0]-1))
		b.StartTimer()
		for d := 0; d < decode; d++ {
			kv = rl.step(b, []int32{tok}, caches)
			tok = int32(argmaxRow(kv, 0))
		}
		b.StopTimer()
		for _, c := range caches {
			c.Free()
		}
		b.StartTimer()
	}
	b.ReportMetric(float64(decode)*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
}

func BenchmarkTinyLlamaDecodeGPUQ8_p32(b *testing.B) { benchDecodeQ8(b, 32, 64) }
