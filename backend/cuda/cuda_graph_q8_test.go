//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"runtime"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// q8GraphDecoder is graphDecoder with Q8_0-quantized projection weights — the
// fixed-buffer/graph decode path where Q8's 4× weight-bandwidth advantage is
// unmasked (decode is now GPU-memory-bound, not alloc/launch-bound).
type q8gdLayer struct {
	gAttn, gFFN    *cuda.ResidentVec
	wq, wk, wv, wo *cuda.ResidentBQ8
	wg, wu, wd     *cuda.ResidentBQ8
	cache          *cuda.KVCache
}

type q8GraphDecoder struct {
	emb                    *cuda.ResidentB
	norm                   *cuda.ResidentVec
	out                    *cuda.ResidentBQ8
	layers                 []*q8gdLayer
	pos                    *cuda.DevicePos
	dx, dh, dh2            *cuda.DeviceF32
	dq, dk, dv, da         *cuda.DeviceF32
	dgate, dup, scores     *cuda.DeviceF32
	logits, inv            *cuda.DeviceF32
	graph                  *cuda.CapturedGraph
	heads, kv, dim, hidden int
	ropeBase, eps          float64
}

func buildQ8GraphDecoder(tb testing.TB, m *nlp.Llama, maxSeq int) *q8GraphDecoder {
	cfg := m.Config
	kv := cfg.KVHeads
	if kv == 0 {
		kv = cfg.Heads
	}
	hd := cfg.Dim / cfg.Heads
	wq, wkv := cfg.Heads*hd, kv*hd
	cast := func(t *tensor.Tensor) *tensor.Tensor { return t.Cast(tensor.F32) }
	q := func(w *tensor.Tensor) *cuda.ResidentBQ8 {
		r, e := cuda.NewResidentBQ8(cast(w))
		mustTB(tb, e)
		return r
	}
	buf := func(r, c int) *cuda.DeviceF32 { d, e := cuda.NewDeviceF32(r, c); mustTB(tb, e); return d }

	gd := &q8GraphDecoder{heads: cfg.Heads, kv: kv, dim: cfg.Dim, hidden: cfg.Hidden, ropeBase: cfg.RopeBase, eps: cfg.Eps}
	gd.emb, _ = cuda.NewResidentB(cast(m.TokEmb))
	gd.norm, _ = cuda.NewResidentVec(cast(m.Norm.Gamma))
	gd.out = q(m.Out)
	gd.pos, _ = cuda.NewDevicePos()
	gd.dx, gd.dh, gd.dh2 = buf(1, cfg.Dim), buf(1, cfg.Dim), buf(1, cfg.Dim)
	gd.dq, gd.da = buf(1, wq), buf(1, wq)
	gd.dk, gd.dv = buf(1, wkv), buf(1, wkv)
	gd.dgate, gd.dup = buf(1, cfg.Hidden), buf(1, cfg.Hidden)
	gd.scores = buf(1, cfg.Heads*maxSeq)
	gd.logits = buf(1, cfg.Vocab)
	gd.inv, _ = cuda.BuildRoPEInv(hd, cfg.RopeBase)
	gd.layers = make([]*q8gdLayer, len(m.Blocks))
	for i, blk := range m.Blocks {
		ga, _ := cuda.NewResidentVec(cast(blk.AttnNorm.Gamma))
		gf, _ := cuda.NewResidentVec(cast(blk.FFNNorm.Gamma))
		cache, _ := cuda.NewKVCache(maxSeq, wkv)
		mustTB(tb, cache.ZeroCache())
		cache.SetLen(maxSeq)
		gd.layers[i] = &q8gdLayer{
			gAttn: ga, gFFN: gf,
			wq: q(blk.Wq), wk: q(blk.Wk), wv: q(blk.Wv), wo: q(blk.Wo),
			wg: q(blk.FFN.Wgate), wu: q(blk.FFN.Wup), wd: q(blk.FFN.Wdown),
			cache: cache,
		}
	}
	return gd
}

func (gd *q8GraphDecoder) free() {
	gd.emb.Free()
	gd.norm.Free()
	gd.out.Free()
	gd.pos.Free()
	for _, d := range []*cuda.DeviceF32{gd.dx, gd.dh, gd.dh2, gd.dq, gd.dk, gd.dv, gd.da, gd.dgate, gd.dup, gd.scores, gd.logits, gd.inv} {
		d.Free()
	}
	if gd.graph != nil {
		gd.graph.Free()
	}
	for _, l := range gd.layers {
		l.gAttn.Free()
		l.gFFN.Free()
		l.cache.Free()
		for _, w := range []*cuda.ResidentBQ8{l.wq, l.wk, l.wv, l.wo, l.wg, l.wu, l.wd} {
			w.Free()
		}
	}
}

func (gd *q8GraphDecoder) forwardBody(tb testing.TB) {
	for _, l := range gd.layers {
		mustTB(tb, gd.dx.RMSNormInto(l.gAttn, float32(gd.eps), gd.dh))
		mustTB(tb, l.wq.QMatMulInto(gd.dh, gd.dq))
		mustTB(tb, l.wk.QMatMulInto(gd.dh, gd.dk))
		mustTB(tb, l.wv.QMatMulInto(gd.dh, gd.dv))
		mustTB(tb, gd.dq.RoPEDposInv(gd.heads, gd.inv, gd.pos, 0))
		mustTB(tb, gd.dk.RoPEDposInv(gd.kv, gd.inv, gd.pos, 0))
		mustTB(tb, l.cache.AppendDpos(gd.dk, gd.dv, gd.pos))
		kF, vF := l.cache.FullView()
		mustTB(tb, cuda.GroupedQueryAttentionKVDposFlashInto(gd.dq, kF, vF, gd.heads, gd.kv, gd.pos, gd.da))
		mustTB(tb, l.wo.QMatMulAccInto(gd.da, gd.dx))
		mustTB(tb, gd.dx.RMSNormInto(l.gFFN, float32(gd.eps), gd.dh2))
		mustTB(tb, l.wg.QMatMulInto(gd.dh2, gd.dgate))
		mustTB(tb, l.wu.QMatMulInto(gd.dh2, gd.dup))
		mustTB(tb, gd.dgate.SwiGLU(gd.dup))
		mustTB(tb, l.wd.QMatMulAccInto(gd.dgate, gd.dx))
	}
	mustTB(tb, gd.dx.RMSNormInto(gd.norm, float32(gd.eps), gd.dh))
	mustTB(tb, gd.out.QMatMulInto(gd.dh, gd.logits))
}

func (gd *q8GraphDecoder) step(tb testing.TB, tok int32, pos int) *tensor.Tensor {
	mustTB(tb, gd.pos.Set(pos))
	mustTB(tb, gd.emb.EmbedInto([]int32{tok}, gd.dx))
	gd.forwardBody(tb)
	out, err := gd.logits.ToHost()
	mustTB(tb, err)
	return out
}

// Q8 fixed-buffer decode must generate the same greedy tokens as the f32 decode.
func TestCUDAQ8GraphDecoderMatchesF32(t *testing.T) {
	m := loadTinyLlama(t)
	gd := buildQ8GraphDecoder(t, m, 64)
	defer gd.free()
	ref := buildResidentLlama(t, m)
	defer ref.free()
	rc, _ := ref.newCaches(64)
	defer func() {
		for _, c := range rc {
			c.Free()
		}
	}()
	prompt := []int32{1, 15043, 29892, 590, 1024, 338}
	var gdLast *tensor.Tensor
	for i, tk := range prompt {
		gdLast = gd.step(t, tk, i)
	}
	refLast := ref.step(t, prompt, rc)
	match := 0
	const steps = 5
	for s := 0; s < steps; s++ {
		gt := int32(argmaxRow(gdLast, 0))
		ft := int32(argmaxRow(refLast, refLast.Shape()[0]-1))
		if gt == ft {
			match++
		}
		t.Logf("step %d: Q8 %d, f32 %d, %s", s, gt, ft, map[bool]string{true: "MATCH", false: "differ"}[gt == ft])
		gdLast = gd.step(t, gt, len(prompt)+s)
		refLast = ref.step(t, []int32{ft}, rc)
	}
	if match < steps-1 {
		t.Fatalf("Q8 fixed-buffer decode diverged: %d/%d match", match, steps)
	}
	t.Logf("Q8 fixed-buffer decode == f32: %d/%d", match, steps)
}

func benchQ8Decode(b *testing.B, prefill, decode int, useGraph bool) {
	m := loadTinyLlama(b)
	gd := buildQ8GraphDecoder(b, m, prefill+decode+2)
	defer gd.free()
	for p := 0; p < prefill; p++ {
		gd.step(b, int32((p*131+1)%m.Config.Vocab), p)
	}
	var tok int32
	if useGraph {
		runtime.LockOSThread()
		mustTB(b, cuda.CaptureBegin())
		gd.forwardBody(b)
		g, err := cuda.CaptureEnd()
		runtime.UnlockOSThread()
		mustTB(b, err)
		gd.graph = g
		gd.pos.Set(prefill)
		gd.emb.EmbedInto([]int32{0}, gd.dx)
		gd.graph.Launch()
		cuda.GraphSync()
		l, _ := gd.logits.ToHost()
		tok = int32(argmaxRow(l, 0))
	} else {
		tok = int32(argmaxRow(gd.step(b, 0, prefill), 0))
	}
	b.ResetTimer()
	for range b.N {
		for d := 0; d < decode; d++ {
			if useGraph {
				gd.pos.Set(prefill + 1 + d)
				gd.emb.EmbedInto([]int32{tok}, gd.dx)
				gd.graph.Launch()
				cuda.GraphSync()
				l, _ := gd.logits.ToHost()
				tok = int32(argmaxRow(l, 0))
			} else {
				tok = int32(argmaxRow(gd.step(b, tok, prefill+1+d), 0))
			}
		}
	}
	b.ReportMetric(float64(decode)*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
}

func BenchmarkTinyLlamaDecodeQ8Fixed_p32(b *testing.B) { benchQ8Decode(b, 32, 64, false) }
func BenchmarkTinyLlamaDecodeQ8Graph_p32(b *testing.B) { benchQ8Decode(b, 32, 64, true) }
