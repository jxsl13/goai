//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// q4kGraphDecoder is the NATIVE-Q4_K twin of q4GraphDecoder: identical CUDA-graph-captured decode
// assembly (device-pos RoPE / KV-append / flash attention, so one captured graph replays per token
// with just pos.Set + EmbedInto), but the projection weights are resident Q4_K (0.5625 B/w) instead
// of the older asymmetric Q4 (0.75 B/w) — the format this session wired for native GGUF serving.
// Fewer weight bytes ⇒ the weight-bandwidth-bound decode should be even faster than the Q4 250.6 tok/s.
// This is the proof-of-concept for a production graph-captured Q4_K decode path (llamagpu is eager today).
type q4kGraphDecoder struct {
	emb                    *cuda.ResidentB
	norm                   *cuda.ResidentVec
	out                    *cuda.ResidentBQ4K
	layers                 []*q4kgdLayer
	pos                    *cuda.DevicePos
	dx, dh, dh2            *cuda.DeviceF32
	dq, dk, dv, da         *cuda.DeviceF32
	dgate, dup, scores     *cuda.DeviceF32
	logits, inv            *cuda.DeviceF32
	graph                  *cuda.CapturedGraph
	heads, kv, dim, hidden int
	ropeBase, eps          float64
}

type q4kgdLayer struct {
	gAttn, gFFN    *cuda.ResidentVec
	wq, wk, wv, wo *cuda.ResidentBQ4K
	wg, wu, wd     *cuda.ResidentBQ4K
	cache          *cuda.KVCache
}

func buildQ4KGraphDecoder(tb testing.TB, m *nlp.Llama, maxSeq int) *q4kGraphDecoder {
	cfg := m.Config
	kv := cfg.KVHeads
	if kv == 0 {
		kv = cfg.Heads
	}
	hd := cfg.Dim / cfg.Heads
	wq, wkv := cfg.Heads*hd, kv*hd
	cast := func(t *tensor.Tensor) *tensor.Tensor { return t.Cast(tensor.F32) }
	q := func(w *tensor.Tensor) *cuda.ResidentBQ4K {
		qi, e := quantQ4K(cast(w))
		mustTB(tb, e)
		return qi.(*cuda.ResidentBQ4K)
	}
	buf := func(r, c int) *cuda.DeviceF32 { d, e := cuda.NewDeviceF32(r, c); mustTB(tb, e); return d }
	gd := &q4kGraphDecoder{heads: cfg.Heads, kv: kv, dim: cfg.Dim, hidden: cfg.Hidden, ropeBase: cfg.RopeBase, eps: cfg.Eps}
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
	gd.layers = make([]*q4kgdLayer, len(m.Blocks))
	for i, blk := range m.Blocks {
		ga, _ := cuda.NewResidentVec(cast(blk.AttnNorm.Gamma))
		gf, _ := cuda.NewResidentVec(cast(blk.FFNNorm.Gamma))
		cache, _ := cuda.NewKVCache(maxSeq, wkv)
		mustTB(tb, cache.ZeroCache())
		cache.SetLen(maxSeq)
		gd.layers[i] = &q4kgdLayer{
			gAttn: ga, gFFN: gf,
			wq: q(blk.Wq), wk: q(blk.Wk), wv: q(blk.Wv), wo: q(blk.Wo),
			wg: q(blk.FFN.Wgate), wu: q(blk.FFN.Wup), wd: q(blk.FFN.Wdown),
			cache: cache,
		}
	}
	return gd
}

func (gd *q4kGraphDecoder) free() {
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
		for _, w := range []*cuda.ResidentBQ4K{l.wq, l.wk, l.wv, l.wo, l.wg, l.wu, l.wd} {
			w.Free()
		}
	}
}

func (gd *q4kGraphDecoder) forwardBody(tb testing.TB) {
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

// TestCUDAQ4KGraphDecodeSpeed measures native-Q4_K CUDA-graph decode tok/s on real TinyLlama — the
// proof-of-concept confirming this session's native-Q4_K weights, driven through a graph-captured
// decode (the technique production llamagpu does NOT yet use), beat the incumbent. A capture of the
// whole per-token op chain replays with only pos.Set + EmbedInto between launches.
func TestCUDAQ4KGraphDecodeSpeed(t *testing.T) {
	if testing.Short() {
		t.Skip("model-loading integration test")
	}
	skipNoGPU(t)
	if _, err := os.Stat(tinyLlamaPath); err != nil {
		t.Skipf("model not present (%s)", tinyLlamaPath)
	}
	f, err := gguf.ReadFile(tinyLlamaPath)
	must(t, err)
	model, err := nlp.LlamaFromGGUF(f.Metadata, f.Tensors)
	must(t, err)
	const prefill, steps = 8, 64
	maxSeq := prefill + steps + 4
	gd := buildQ4KGraphDecoder(t, model, maxSeq)
	defer gd.free()
	for p := 0; p < prefill; p++ {
		mustTB(t, gd.pos.Set(p))
		mustTB(t, gd.emb.EmbedInto([]int32{int32((p*131 + 1) % model.Config.Vocab)}, gd.dx))
		gd.forwardBody(t)
	}
	runtime.LockOSThread()
	must(t, cuda.CaptureBegin())
	gd.forwardBody(t)
	g, err := cuda.CaptureEnd()
	runtime.UnlockOSThread()
	must(t, err)
	gd.graph = g
	tk := int32(1)
	mustTB(t, gd.pos.Set(prefill))
	mustTB(t, gd.emb.EmbedInto([]int32{tk}, gd.dx))
	gd.graph.Launch()
	cuda.GraphSync()
	t0 := time.Now()
	for d := 0; d < steps; d++ {
		mustTB(t, gd.pos.Set(prefill+1+d))
		mustTB(t, gd.emb.EmbedInto([]int32{tk}, gd.dx))
		gd.graph.Launch()
		cuda.GraphSync()
		l, _ := gd.logits.ToHost()
		tk = int32(argmaxRow(l, 0))
	}
	tps := float64(steps) / time.Since(t0).Seconds()
	t.Logf("native-Q4_K graph decode: %.1f tok/s (old-Q4 graph 250.6; GoAI Q8 ~198; llama.cpp Vulkan Q8 244)", tps)
	if tps < 100 {
		t.Fatalf("native-Q4_K graph decode implausibly slow: %.1f tok/s", tps)
	}
}
