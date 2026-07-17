//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"runtime"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

type qwenFixedLayer struct {
	an, fn         *cuda.ResidentVec
	wq, wk, wv, wo *cuda.ResidentBQ8
	bq, bk, bv     *cuda.ResidentVec
	wg, wu, wd     *cuda.ResidentBQ8
	cache          *cuda.KVCache
}

// qwenFixed is the Qwen decode on the OPTIMIZED path: persistent fixed buffers +
// device-position + fixed-size padded attention + Q8 + CUDA graph, with the Q/K/V
// AddBias composed into the graph body. Tests that the whole optimized path works
// for a 2nd architecture (only the alloc-path Qwen was tested before).
type qwenFixed struct {
	emb                *cuda.ResidentB
	onorm              *cuda.ResidentVec
	out                *cuda.ResidentBQ8
	layers             []*qwenFixedLayer
	pos                *cuda.DevicePos
	inv                *cuda.DeviceF32
	dx, dh, dh2        *cuda.DeviceF32
	dq, dk, dv, da     *cuda.DeviceF32
	dgate, dup, scores *cuda.DeviceF32
	logits             *cuda.DeviceF32
	graph              *cuda.CapturedGraph
	heads, kv          int
	eps, ropeBase      float64
	tok                *nlp.BPETokenizer
}

func buildQwenFixed(t *testing.T, path string, maxSeq int) *qwenFixed {
	f, err := gguf.ReadFile(path)
	must(t, err)
	mi := func(k string) int {
		switch v := f.Metadata[k].(type) {
		case uint32:
			return int(v)
		case int32:
			return int(v)
		case uint64:
			return int(v)
		case int64:
			return int(v)
		case int:
			return v
		}
		t.Fatalf("meta %s %T", k, f.Metadata[k])
		return 0
	}
	mf := func(k string) float64 {
		switch v := f.Metadata[k].(type) {
		case float32:
			return float64(v)
		case float64:
			return v
		}
		return 0
	}
	dim, nL := mi("qwen2.embedding_length"), mi("qwen2.block_count")
	heads, kv := mi("qwen2.attention.head_count"), mi("qwen2.attention.head_count_kv")
	hd := dim / heads
	wq, wkv := heads*hd, kv*hd
	hidden := mi("qwen2.feed_forward_length")
	q := &qwenFixed{heads: heads, kv: kv, eps: mf("qwen2.attention.layer_norm_rms_epsilon"), ropeBase: mf("qwen2.rope.freq_base")}
	q.tok, err = nlp.BPEFromGGUF(f.Metadata)
	must(t, err)
	ten := func(n string) *tensor.Tensor {
		x := f.Tensors[n]
		if x == nil {
			t.Fatalf("missing %s", n)
		}
		return x
	}
	q8 := func(n string) *cuda.ResidentBQ8 {
		r, e := cuda.NewResidentBQ8(transpose2D(ten(n)))
		must(t, e)
		return r
	}
	vec := func(n string) *cuda.ResidentVec {
		r, e := cuda.NewResidentVec(ten(n).Cast(tensor.F32))
		must(t, e)
		return r
	}
	buf := func(r, c int) *cuda.DeviceF32 { d, e := cuda.NewDeviceF32(r, c); must(t, e); return d }
	q.emb, _ = cuda.NewResidentB(ten("token_embd.weight").Cast(tensor.F32))
	q.onorm = vec("output_norm.weight")
	q.out = q8("output.weight")
	q.pos, _ = cuda.NewDevicePos()
	q.inv, _ = cuda.BuildRoPEInv(hd, q.ropeBase)
	q.dx, q.dh, q.dh2 = buf(1, dim), buf(1, dim), buf(1, dim)
	q.dq, q.da = buf(1, wq), buf(1, wq)
	q.dk, q.dv = buf(1, wkv), buf(1, wkv)
	q.dgate, q.dup = buf(1, hidden), buf(1, hidden)
	q.scores = buf(1, heads*maxSeq)
	q.logits = buf(1, ten("output.weight").Shape()[0])
	q.layers = make([]*qwenFixedLayer, nL)
	for i := 0; i < nL; i++ {
		p := "blk." + itoa(i) + "."
		c, _ := cuda.NewKVCache(maxSeq, wkv)
		must(t, c.ZeroCache())
		c.SetLen(maxSeq)
		q.layers[i] = &qwenFixedLayer{
			an: vec(p + "attn_norm.weight"), fn: vec(p + "ffn_norm.weight"),
			wq: q8(p + "attn_q.weight"), wk: q8(p + "attn_k.weight"), wv: q8(p + "attn_v.weight"), wo: q8(p + "attn_output.weight"),
			bq: vec(p + "attn_q.bias"), bk: vec(p + "attn_k.bias"), bv: vec(p + "attn_v.bias"),
			wg: q8(p + "ffn_gate.weight"), wu: q8(p + "ffn_up.weight"), wd: q8(p + "ffn_down.weight"),
			cache: c,
		}
	}
	return q
}

func (q *qwenFixed) forwardBody(t *testing.T) {
	for _, l := range q.layers {
		must(t, q.dx.RMSNormInto(l.an, float32(q.eps), q.dh))
		must(t, l.wq.QMatMulInto(q.dh, q.dq))
		must(t, l.wk.QMatMulInto(q.dh, q.dk))
		must(t, l.wv.QMatMulInto(q.dh, q.dv))
		must(t, q.dq.AddBias(l.bq)) // Qwen QKV bias (in-place, in the graph body)
		must(t, q.dk.AddBias(l.bk))
		must(t, q.dv.AddBias(l.bv))
		must(t, q.dq.RoPEDposInv(q.heads, q.inv, q.pos, 0))
		must(t, q.dk.RoPEDposInv(q.kv, q.inv, q.pos, 0))
		must(t, l.cache.AppendDpos(q.dk, q.dv, q.pos))
		kF, vF := l.cache.FullView()
		must(t, cuda.GroupedQueryAttentionKVDposFlashInto(q.dq, kF, vF, q.heads, q.kv, q.pos, q.da))
		must(t, l.wo.QMatMulAccInto(q.da, q.dx))
		must(t, q.dx.RMSNormInto(l.fn, float32(q.eps), q.dh2))
		must(t, l.wg.QMatMulInto(q.dh2, q.dgate))
		must(t, l.wu.QMatMulInto(q.dh2, q.dup))
		must(t, q.dgate.SwiGLU(q.dup))
		must(t, l.wd.QMatMulAccInto(q.dgate, q.dx))
	}
	must(t, q.dx.RMSNormInto(q.onorm, float32(q.eps), q.dh))
	must(t, q.out.QMatMulInto(q.dh, q.logits))
}

func (q *qwenFixed) step(t *testing.T, tok int32, pos int) int {
	must(t, q.pos.Set(pos))
	must(t, q.emb.EmbedInto([]int32{tok}, q.dx))
	q.forwardBody(t)
	return q.logits.Argmax()
}

func (q *qwenFixed) capture(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	must(t, cuda.CaptureBegin())
	q.forwardBody(t)
	g, err := cuda.CaptureEnd()
	must(t, err)
	q.graph = g
}

func (q *qwenFixed) stepGraph(t *testing.T, tok int32, pos int) int {
	must(t, q.pos.Set(pos))
	must(t, q.emb.EmbedInto([]int32{tok}, q.dx))
	must(t, q.graph.Launch())
	must(t, cuda.GraphSync())
	return q.logits.Argmax()
}

func (q *qwenFixed) free() {
	q.emb.Free()
	q.onorm.Free()
	q.out.Free()
	q.pos.Free()
	if q.graph != nil {
		q.graph.Free()
	}
	for _, d := range []*cuda.DeviceF32{q.dx, q.dh, q.dh2, q.dq, q.dk, q.dv, q.da, q.dgate, q.dup, q.scores, q.logits, q.inv} {
		d.Free()
	}
	for _, l := range q.layers {
		l.an.Free()
		l.fn.Free()
		l.bq.Free()
		l.bk.Free()
		l.bv.Free()
		l.cache.Free()
		for _, w := range []*cuda.ResidentBQ8{l.wq, l.wk, l.wv, l.wo, l.wg, l.wu, l.wd} {
			w.Free()
		}
	}
}

// The optimized Qwen path (fixed-buffer + device-pos + fixed-size attention + Q8 +
// CUDA graph, with AddBias in the graph body) must generate the SAME greedy tokens
// as the simple alloc-path Qwen — proving the whole optimized stack is correct for
// a 2nd architecture, and measuring Qwen at full speed.
func TestCUDAQwenFixedMatchesAlloc(t *testing.T) {
	if testing.Short() {
		t.Skip("model-loading integration test; -short = kernel-level gate")
	}
	for _, path := range []string{
		qwenPath,
		"../../models/qwen2.5-1.5b-instruct-q8_0.gguf",
		"../../models/qwen2.5-3b-instruct-q8_0.gguf", // 36-layer graph capture; skips if absent
	} {
		t.Run(path, func(t *testing.T) { qwenFixedVsAlloc(t, path) })
	}
}

func qwenFixedVsAlloc(t *testing.T, path string) {
	skipNoGPU(t)
	allocToks := runQwen(t, path) // alloc-path reference (also skips if model absent)

	prompt := "The capital of France is"
	q := buildQwenFixed(t, path, len(allocToks)+16)
	defer q.free()
	ids := q.tok.Encode(prompt)
	var last int
	for i, id := range ids {
		last = q.step(t, int32(id), i)
	}
	q.capture(t)
	fixed := make([]int, 0, len(allocToks))
	t0 := time.Now()
	for n := 0; n < len(allocToks); n++ {
		fixed = append(fixed, last)
		last = q.stepGraph(t, int32(last), len(ids)+n)
	}
	tps := float64(len(fixed)) / time.Since(t0).Seconds()

	for i := range allocToks {
		if fixed[i] != allocToks[i] {
			t.Fatalf("token %d: fixed/graph %d != alloc %d", i, fixed[i], allocToks[i])
		}
	}
	t.Logf("Qwen fixed-buffer+graph == alloc-path (%d tokens); DECODE %.1f tok/s (graph)", len(fixed), tps)
}
