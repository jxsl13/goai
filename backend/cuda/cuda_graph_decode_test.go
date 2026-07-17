//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"runtime"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// graphDecoder runs single-token decode entirely on FIXED, persistent buffers
// with the position in a DevicePos and fixed-size (padded) attention — the
// structurally-constant form that a CUDA graph can capture and replay. This fire
// validates the fixed-buffer forward's correctness (graph capture is the next
// step; the op sequence here is what gets captured).
type gdLayer struct {
	gAttn, gFFN    *cuda.ResidentVec
	wq, wk, wv, wo *cuda.ResidentB
	wg, wu, wd     *cuda.ResidentB
	cache          *cuda.KVCache
}

type graphDecoder struct {
	emb, out               *cuda.ResidentB
	norm                   *cuda.ResidentVec
	layers                 []*gdLayer
	pos                    *cuda.DevicePos
	dx, dh, dh2            *cuda.DeviceF32
	dq, dk, dv, da         *cuda.DeviceF32
	dgate, dup, scores     *cuda.DeviceF32
	logits                 *cuda.DeviceF32
	inv                    *cuda.DeviceF32 // persistent RoPE inv table (capture-safe, no per-call upload)
	graph                  *cuda.CapturedGraph
	heads, kv, dim, hidden int
	ropeBase, eps          float64
}

func buildGraphDecoder(tb testing.TB, m *nlp.Llama, maxSeq int) *graphDecoder {
	cfg := m.Config
	kv := cfg.KVHeads
	if kv == 0 {
		kv = cfg.Heads
	}
	hd := cfg.Dim / cfg.Heads
	wq, wkv := cfg.Heads*hd, kv*hd
	cast := func(t *tensor.Tensor) *tensor.Tensor { return t.Cast(tensor.F32) }
	rb := func(w *tensor.Tensor) *cuda.ResidentB { r, e := cuda.NewResidentB(cast(w)); mustTB(tb, e); return r }
	buf := func(r, c int) *cuda.DeviceF32 { d, e := cuda.NewDeviceF32(r, c); mustTB(tb, e); return d }

	gd := &graphDecoder{
		heads: cfg.Heads, kv: kv, dim: cfg.Dim, hidden: cfg.Hidden,
		ropeBase: cfg.RopeBase, eps: cfg.Eps,
	}
	gd.emb = rb(m.TokEmb)
	gd.norm, _ = cuda.NewResidentVec(cast(m.Norm.Gamma))
	gd.out = rb(m.Out)
	gd.pos, _ = cuda.NewDevicePos()
	gd.dx, gd.dh, gd.dh2 = buf(1, cfg.Dim), buf(1, cfg.Dim), buf(1, cfg.Dim)
	gd.dq, gd.da = buf(1, wq), buf(1, wq)
	gd.dk, gd.dv = buf(1, wkv), buf(1, wkv)
	gd.dgate, gd.dup = buf(1, cfg.Hidden), buf(1, cfg.Hidden)
	gd.scores = buf(1, cfg.Heads*maxSeq)
	gd.logits = buf(1, cfg.Vocab)
	gd.inv, _ = cuda.BuildRoPEInv(hd, cfg.RopeBase)
	gd.layers = make([]*gdLayer, len(m.Blocks))
	for i, blk := range m.Blocks {
		ga, _ := cuda.NewResidentVec(cast(blk.AttnNorm.Gamma))
		gf, _ := cuda.NewResidentVec(cast(blk.FFNNorm.Gamma))
		cache, _ := cuda.NewKVCache(maxSeq, wkv)
		mustTB(tb, cache.ZeroCache())
		cache.SetLen(maxSeq) // fixed-size: attend the whole padded cache
		gd.layers[i] = &gdLayer{
			gAttn: ga, gFFN: gf,
			wq: rb(blk.Wq), wk: rb(blk.Wk), wv: rb(blk.Wv), wo: rb(blk.Wo),
			wg: rb(blk.FFN.Wgate), wu: rb(blk.FFN.Wup), wd: rb(blk.FFN.Wdown),
			cache: cache,
		}
	}
	return gd
}

func (gd *graphDecoder) free() {
	gd.emb.Free()
	gd.norm.Free()
	gd.out.Free()
	gd.pos.Free()
	for _, d := range []*cuda.DeviceF32{gd.dx, gd.dh, gd.dh2, gd.dq, gd.dk, gd.dv, gd.da, gd.dgate, gd.dup, gd.scores, gd.logits, gd.inv} {
		d.Free()
	}
	for _, l := range gd.layers {
		l.gAttn.Free()
		l.gFFN.Free()
		l.cache.Free()
		for _, w := range []*cuda.ResidentB{l.wq, l.wk, l.wv, l.wo, l.wg, l.wu, l.wd} {
			w.Free()
		}
	}
}

// forwardBody is the exact op sequence a graph will capture (everything except
// the token-dependent embed and the logit download). It runs on fixed buffers
// with the position read from gd.pos.
func (gd *graphDecoder) forwardBody(tb testing.TB) {
	for _, l := range gd.layers {
		mustTB(tb, gd.dx.RMSNormInto(l.gAttn, float32(gd.eps), gd.dh))
		mustTB(tb, l.wq.MatMulInto(gd.dh, gd.dq))
		mustTB(tb, l.wk.MatMulInto(gd.dh, gd.dk))
		mustTB(tb, l.wv.MatMulInto(gd.dh, gd.dv))
		mustTB(tb, gd.dq.RoPEDposInv(gd.heads, gd.inv, gd.pos, 0))
		mustTB(tb, gd.dk.RoPEDposInv(gd.kv, gd.inv, gd.pos, 0))
		mustTB(tb, l.cache.AppendDpos(gd.dk, gd.dv, gd.pos))
		kF, vF := l.cache.FullView()
		mustTB(tb, cuda.GroupedQueryAttentionKVDposFlashInto(gd.dq, kF, vF, gd.heads, gd.kv, gd.pos, gd.da))
		mustTB(tb, l.wo.MatMulAccInto(gd.da, gd.dx))
		mustTB(tb, gd.dx.RMSNormInto(l.gFFN, float32(gd.eps), gd.dh2))
		mustTB(tb, l.wg.MatMulInto(gd.dh2, gd.dgate))
		mustTB(tb, l.wu.MatMulInto(gd.dh2, gd.dup))
		mustTB(tb, gd.dgate.SwiGLU(gd.dup))
		mustTB(tb, l.wd.MatMulAccInto(gd.dgate, gd.dx))
	}
	mustTB(tb, gd.dx.RMSNormInto(gd.norm, float32(gd.eps), gd.dh))
	mustTB(tb, gd.out.MatMulInto(gd.dh, gd.logits))
}

// step decodes one token at absolute position pos, returning host logits.
func (gd *graphDecoder) step(tb testing.TB, tok int32, pos int) *tensor.Tensor {
	mustTB(tb, gd.pos.Set(pos))
	mustTB(tb, gd.emb.EmbedInto([]int32{tok}, gd.dx))
	gd.forwardBody(tb)
	out, err := gd.logits.ToHost()
	mustTB(tb, err)
	return out
}

// capture records forwardBody into a CUDA graph; stepGraph replays it per token.
func (gd *graphDecoder) capture(tb testing.TB) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	mustTB(tb, cuda.CaptureBegin())
	gd.forwardBody(tb)
	g, err := cuda.CaptureEnd()
	mustTB(tb, err)
	gd.graph = g
}

func (gd *graphDecoder) stepGraph(tb testing.TB, tok int32, pos int) *tensor.Tensor {
	mustTB(tb, gd.pos.Set(pos))
	mustTB(tb, gd.emb.EmbedInto([]int32{tok}, gd.dx))
	mustTB(tb, gd.graph.Launch())
	mustTB(tb, cuda.GraphSync())
	out, err := gd.logits.ToHost()
	mustTB(tb, err)
	return out
}

// The CUDA-graph replay of the full decode must generate the SAME tokens as the
// direct fixed-buffer decode — the capstone: launch-overhead-free decode.
func TestCUDAGraphDecodeMatchesDirect(t *testing.T) {
	if testing.Short() {
		t.Skip("model-loading integration test; -short = kernel-level gate")
	}
	m := loadTinyLlama(t)
	prompt := []int32{1, 15043, 29892, 590, 1024, 338}
	const gen = 6
	run := func(useGraph bool) []int32 {
		gd := buildGraphDecoder(t, m, 64)
		defer gd.free()
		var last *tensor.Tensor
		for i, tk := range prompt {
			last = gd.step(t, tk, i)
		}
		if useGraph {
			gd.capture(t)
		}
		out := make([]int32, gen)
		for i := 0; i < gen; i++ {
			tok := int32(argmaxRow(last, 0))
			out[i] = tok
			if useGraph {
				last = gd.stepGraph(t, tok, len(prompt)+i)
			} else {
				last = gd.step(t, tok, len(prompt)+i)
			}
		}
		return out
	}
	directToks := run(false)
	graphToks := run(true)
	for i := range directToks {
		if directToks[i] != graphToks[i] {
			t.Fatalf("token %d: graph %d != direct %d (direct %v graph %v)", i, graphToks[i], directToks[i], directToks, graphToks)
		}
	}
	t.Logf("GRAPH decode == direct decode: %v", graphToks)
}

// Graph-replay decode throughput — the capstone measurement.
func benchGraphDecodeReplay(b *testing.B, prefill, decode int) {
	m := loadTinyLlama(b)
	gd := buildGraphDecoder(b, m, prefill+decode+2)
	defer gd.free()
	for p := 0; p < prefill; p++ {
		gd.step(b, int32((p*131+1)%m.Config.Vocab), p)
	}
	gd.capture(b)
	tok := int32(argmaxRow(gd.stepGraph(b, 0, prefill), 0))
	b.ResetTimer()
	for range b.N {
		for d := 0; d < decode; d++ {
			lg := gd.stepGraph(b, tok, prefill+1+d)
			tok = int32(argmaxRow(lg, 0))
		}
	}
	b.ReportMetric(float64(decode)*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
}

func BenchmarkTinyLlamaDecodeGraph_p32(b *testing.B) { benchGraphDecodeReplay(b, 32, 64) }

// The fixed-buffer / device-position / fixed-size-attention decode must generate
// the SAME greedy tokens as the validated f32 KV-cache decode — proving the exact
// op sequence a graph will capture is correct.
func TestCUDAGraphDecoderMatchesF32(t *testing.T) {
	if testing.Short() {
		t.Skip("model-loading integration test; -short = kernel-level gate")
	}
	m := loadTinyLlama(t)
	gd := buildGraphDecoder(t, m, 64)
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
	// feed prompt token-by-token through the graph decoder
	var gdLast *tensor.Tensor
	for i, tk := range prompt {
		gdLast = gd.step(t, tk, i)
	}
	refLast := ref.step(t, prompt, rc) // batched prefill

	pos := len(prompt)
	const steps = 5
	match := 0
	for s := 0; s < steps; s++ {
		gt := int32(argmaxRow(gdLast, 0))
		ft := int32(argmaxRow(refLast, refLast.Shape()[0]-1))
		if gt == ft {
			match++
		}
		t.Logf("step %d: graph-decoder %d, ref %d, %s", s, gt, ft,
			map[bool]string{true: "MATCH", false: "differ"}[gt == ft])
		gdLast = gd.step(t, gt, pos+s)
		refLast = ref.step(t, []int32{ft}, rc)
	}
	if match < steps {
		t.Fatalf("graph decoder diverged: %d/%d tokens match f32", match, steps)
	}
	t.Logf("fixed-buffer decode == f32: %d/%d tokens match", match, steps)
}

// Throughput of the fixed-buffer (pre-graph) decode vs the regular decode — should
// be ≈ equal (both launch-bound); the win comes when this body is graph-captured.
func benchGraphDecoderBody(b *testing.B, prefill, decode int) {
	m := loadTinyLlama(b)
	gd := buildGraphDecoder(b, m, prefill+decode+2)
	defer gd.free()
	for p := 0; p < prefill; p++ {
		gd.step(b, int32((p*131+1)%m.Config.Vocab), p)
	}
	tok := int32(argmaxRow(gd.step(b, 0, prefill), 0))
	b.ResetTimer()
	for range b.N {
		for d := 0; d < decode; d++ {
			lg := gd.step(b, tok, prefill+1+d)
			tok = int32(argmaxRow(lg, 0))
		}
	}
	b.ReportMetric(float64(decode)*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
}

func BenchmarkTinyLlamaDecoderBody_p32(b *testing.B) { benchGraphDecoderBody(b, 32, 64) }
