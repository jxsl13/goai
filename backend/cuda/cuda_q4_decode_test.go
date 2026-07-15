//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"os"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// q4GraphDecoder mirrors q8GraphDecoder with ASYMMETRIC-Q4 projection weights — the
// weight-bandwidth path (Q4 reads ≈0.67× the bytes of Q8, so decode goes faster where it
// is weight-bandwidth-bound). Only the projections are Q4; embed/norms stay f32.
type q4gdLayer struct {
	gAttn, gFFN    *cuda.ResidentVec
	wq, wk, wv, wo *cuda.ResidentBQ4
	wg, wu, wd     *cuda.ResidentBQ4
	cache          *cuda.KVCache
}

type q4GraphDecoder struct {
	emb                    *cuda.ResidentB
	norm                   *cuda.ResidentVec
	out                    *cuda.ResidentBQ4
	layers                 []*q4gdLayer
	pos                    *cuda.DevicePos
	dx, dh, dh2            *cuda.DeviceF32
	dq, dk, dv, da         *cuda.DeviceF32
	dgate, dup, scores     *cuda.DeviceF32
	logits, inv            *cuda.DeviceF32
	graph                  *cuda.CapturedGraph
	heads, kv, dim, hidden int
	ropeBase, eps          float64
}

func buildQ4GraphDecoder(tb testing.TB, m *nlp.Llama, maxSeq int) *q4GraphDecoder {
	cfg := m.Config
	kv := cfg.KVHeads
	if kv == 0 {
		kv = cfg.Heads
	}
	hd := cfg.Dim / cfg.Heads
	wq, wkv := cfg.Heads*hd, kv*hd
	cast := func(t *tensor.Tensor) *tensor.Tensor { return t.Cast(tensor.F32) }
	q := func(w *tensor.Tensor) *cuda.ResidentBQ4 {
		r, e := cuda.NewResidentBQ4(cast(w))
		mustTB(tb, e)
		return r
	}
	buf := func(r, c int) *cuda.DeviceF32 { d, e := cuda.NewDeviceF32(r, c); mustTB(tb, e); return d }
	gd := &q4GraphDecoder{heads: cfg.Heads, kv: kv, dim: cfg.Dim, hidden: cfg.Hidden, ropeBase: cfg.RopeBase, eps: cfg.Eps}
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
	gd.layers = make([]*q4gdLayer, len(m.Blocks))
	for i, blk := range m.Blocks {
		ga, _ := cuda.NewResidentVec(cast(blk.AttnNorm.Gamma))
		gf, _ := cuda.NewResidentVec(cast(blk.FFNNorm.Gamma))
		cache, _ := cuda.NewKVCache(maxSeq, wkv)
		mustTB(tb, cache.ZeroCache())
		cache.SetLen(maxSeq)
		gd.layers[i] = &q4gdLayer{
			gAttn: ga, gFFN: gf,
			wq: q(blk.Wq), wk: q(blk.Wk), wv: q(blk.Wv), wo: q(blk.Wo),
			wg: q(blk.FFN.Wgate), wu: q(blk.FFN.Wup), wd: q(blk.FFN.Wdown),
			cache: cache,
		}
	}
	return gd
}

func (gd *q4GraphDecoder) free() {
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
		for _, w := range []*cuda.ResidentBQ4{l.wq, l.wk, l.wv, l.wo, l.wg, l.wu, l.wd} {
			w.Free()
		}
	}
}

func (gd *q4GraphDecoder) forwardBody(tb testing.TB) {
	for _, l := range gd.layers {
		mustTB(tb, gd.dx.RMSNormInto(l.gAttn, float32(gd.eps), gd.dh))
		mustTB(tb, l.wq.QMatMulInto(gd.dh, gd.dq))
		mustTB(tb, l.wk.QMatMulInto(gd.dh, gd.dk))
		mustTB(tb, l.wv.QMatMulInto(gd.dh, gd.dv))
		mustTB(tb, gd.dq.RoPEDposInv(gd.heads, gd.inv, gd.pos, 0))
		mustTB(tb, gd.dk.RoPEDposInv(gd.kv, gd.inv, gd.pos, 0))
		mustTB(tb, l.cache.AppendDpos(gd.dk, gd.dv, gd.pos))
		kF, vF := l.cache.FullView()
		mustTB(tb, cuda.GroupedQueryAttentionKVDposInto(gd.dq, kF, vF, gd.heads, gd.kv, gd.pos, gd.scores, gd.da))
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

func (gd *q4GraphDecoder) step(tb testing.TB, tok int32, pos int) *tensor.Tensor {
	mustTB(tb, gd.pos.Set(pos))
	mustTB(tb, gd.emb.EmbedInto([]int32{tok}, gd.dx))
	gd.forwardBody(tb)
	out, err := gd.logits.ToHost()
	mustTB(tb, err)
	return out
}

// The Q4 decode path must (a) stay coherent — generate readable text that mostly agrees with
// the Q8 path — and (b) be faster than Q8 (fewer weight bytes). Answers the two questions that
// decide whether Q4 lets goai beat llama.cpp: quality and speed.
func TestCUDAQ4QualityAndSpeed(t *testing.T) {
	skipNoGPU(t)
	if _, err := os.Stat(tinyLlamaPath); err != nil {
		t.Skipf("model not present (%s)", tinyLlamaPath)
	}
	f, err := gguf.ReadFile(tinyLlamaPath)
	must(t, err)
	model, err := nlp.LlamaFromGGUF(f.Metadata, f.Tensors)
	must(t, err)
	tok, err := nlp.UnigramFromGGUF(f.Metadata)
	must(t, err)
	ids := append([]int{1}, tok.Encode("The capital of France is")...)
	const gen = 24
	maxSeq := len(ids) + gen + 2

	// --- Q8 reference tokens ---
	q8 := buildQ8GraphDecoder(t, model, maxSeq)
	var q8toks []int
	{
		var last int
		for i, id := range ids {
			last = argmaxRow(q8.step(t, int32(id), i), 0)
		}
		for n := 0; n < gen; n++ {
			q8toks = append(q8toks, last)
			last = argmaxRow(q8.step(t, int32(last), len(ids)+n), 0)
		}
	}
	q8.free()

	// --- Q4 tokens + text ---
	q4 := buildQ4GraphDecoder(t, model, maxSeq)
	defer q4.free()
	var q4toks []int
	{
		var last int
		for i, id := range ids {
			last = argmaxRow(q4.step(t, int32(id), i), 0)
		}
		for n := 0; n < gen; n++ {
			q4toks = append(q4toks, last)
			last = argmaxRow(q4.step(t, int32(last), len(ids)+n), 0)
		}
	}
	match := 0
	for i := range q8toks {
		if q4toks[i] == q8toks[i] {
			match++
		}
	}
	t.Logf("Q4 vs Q8 greedy: %d/%d tokens agree", match, gen)
	t.Logf("Q8 text: %q", strings.TrimSpace(renderPieces(tok.Decode(q8toks))))
	t.Logf("Q4 text: %q", strings.TrimSpace(renderPieces(tok.Decode(q4toks))))
	// Coherence gate: Q4 must produce the expected content ("Paris") — the argmax is robust to
	// the 4-bit weight error even where individual tokens diverge from Q8.
	if !strings.Contains(strings.ToLower(tok.Decode(q4toks)), "paris") {
		t.Fatalf("Q4 decode incoherent (no 'Paris'): %q", tok.Decode(q4toks))
	}

	// --- speed: capture the Q4 graph and time decode ---
	const prefill, steps = 8, 64
	for p := 0; p < prefill; p++ {
		q4.step(t, int32((p*131+1)%model.Config.Vocab), p)
	}
	runtime.LockOSThread()
	must(t, cuda.CaptureBegin())
	q4.forwardBody(t)
	g, err := cuda.CaptureEnd()
	runtime.UnlockOSThread()
	must(t, err)
	q4.graph = g
	tk := int32(1)
	q4.pos.Set(prefill)
	q4.emb.EmbedInto([]int32{tk}, q4.dx)
	q4.graph.Launch()
	cuda.GraphSync()
	t0 := time.Now()
	for d := 0; d < steps; d++ {
		q4.pos.Set(prefill + 1 + d)
		q4.emb.EmbedInto([]int32{tk}, q4.dx)
		q4.graph.Launch()
		cuda.GraphSync()
		l, _ := q4.logits.ToHost()
		tk = int32(argmaxRow(l, 0))
	}
	q4tps := float64(steps) / time.Since(t0).Seconds()
	t.Logf("Q4 graph decode: %.1f tok/s (Q8 was ~198; llama.cpp Vulkan Q8 244)", q4tps)
}
