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

const mistral7BPath = "../../models/mistral-7b-instruct-v0.2.Q8_0.gguf"

// fromF32 lifts a [K,N]-f32 quantizer to a QuantTensor quantizer: dequantize the raw
// gguf tensor ([out,in]) and transpose to the input-major [K,N] the f32 constructors take.
func fromF32(f func(*tensor.Tensor) (qProj, error)) func(gguf.QuantTensor) (qProj, error) {
	return func(qt gguf.QuantTensor) (qProj, error) {
		w, err := qt.Dequantize()
		if err != nil {
			return nil, err
		}
		return f(transpose2D(w))
	}
}

// qProj abstracts a GPU-resident quantized projection weight (Q8 or Q4) so ONE decoder
// measures both precisions at 7B scale — the A/B that decides whether Q4's bandwidth
// advantage carries to a production-size, weight-bandwidth-bound model.
type qProj interface {
	QMatMulInto(a, out *cuda.DeviceF32) error
	QMatMulAccInto(a, c *cuda.DeviceF32) error
	Free()
}

type rawLayer struct {
	gAttn, gFFN    *cuda.ResidentVec
	bq, bk, bv     *cuda.ResidentVec // QKV bias (qwen2); nil for the llama family
	wq, wk, wv, wo qProj
	wg, wu, wd     qProj
	cache          *cuda.KVCache    // f32 cache (chain + flash paths)
	cacheF         *cuda.KVCacheF16 // f16 cache (GOAI_CUDA_KV=f16, flash-only)
}

// kvF16 selects the f16 KV cache (half the K/V bytes — the flash kernel's
// bandwidth currency). Opt-in while the A/B runs; flipped by measurement.
func kvF16() bool { return os.Getenv("GOAI_CUDA_KV") == "f16" }

// swigluProj is the optional SwiGLU-epilogue capability of a quantized projection
// (Tw55): out = silu(gate) ⊙ (a·W) in one GEMV launch. Q8 and Q4_K implement it —
// the formats gate/up tensors actually use; others fall back to the 3-op chain.
type swigluProj interface {
	QMatMulSwiGLUInto(a, gate, out *cuda.DeviceF32) error
}

// rawGraphDecoder is the optimized decode path (fixed buffers + device-pos + fixed-size
// attention + CUDA graph + on-device argmax) built DIRECTLY from a gguf.RawFile: each
// weight is dequantized one at a time, re-quantized, uploaded, then released — a 7B
// model never materializes in host memory (f32 7B ≈ 28 GB would not fit; raw Q8 ≈ 6 GB does).
type rawGraphDecoder struct {
	emb                *cuda.ResidentB
	norm               *cuda.ResidentVec
	out                qProj
	layers             []*rawLayer
	pos                *cuda.DevicePos
	inv                *cuda.DeviceF32
	dx, dh, dh2        *cuda.DeviceF32
	dq, dk, dv, da     *cuda.DeviceF32
	dgate, dup, scores *cuda.DeviceF32
	logits             *cuda.DeviceF32
	graph              *cuda.CapturedGraph
	heads, kv          int
	ropeBase, eps      float64
}

func rawMetaInt(tb testing.TB, meta map[string]any, k string) int {
	switch v := meta[k].(type) {
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
	tb.Fatalf("meta %s: unexpected %T", k, meta[k])
	return 0
}

func rawMetaFloat(meta map[string]any, k string, def float64) float64 {
	switch v := meta[k].(type) {
	case float32:
		return float64(v)
	case float64:
		return v
	}
	return def
}

func buildRawGraphDecoder(tb testing.TB, rf *gguf.RawFile, arch string, maxSeq int, quant func(gguf.QuantTensor) (qProj, error)) *rawGraphDecoder {
	deq := func(n string) *tensor.Tensor {
		qt, ok := rf.Tensors[n]
		if !ok {
			tb.Fatalf("missing tensor %s", n)
		}
		t, err := qt.Dequantize()
		mustTB(tb, err)
		return t
	}
	q := func(n string) qProj { // quantizers receive the RAW QuantTensor ([out,in] blocks)
		qt, ok := rf.Tensors[n]
		if !ok {
			tb.Fatalf("missing tensor %s", n)
		}
		r, err := quant(qt)
		mustTB(tb, err)
		return r
	}
	vec := func(n string) *cuda.ResidentVec {
		r, err := cuda.NewResidentVec(deq(n).Cast(tensor.F32))
		mustTB(tb, err)
		return r
	}
	buf := func(r, c int) *cuda.DeviceF32 { d, e := cuda.NewDeviceF32(r, c); mustTB(tb, e); return d }

	dim := rawMetaInt(tb, rf.Metadata, arch+".embedding_length")
	nL := rawMetaInt(tb, rf.Metadata, arch+".block_count")
	heads := rawMetaInt(tb, rf.Metadata, arch+".attention.head_count")
	kv := rawMetaInt(tb, rf.Metadata, arch+".attention.head_count_kv")
	hidden := rawMetaInt(tb, rf.Metadata, arch+".feed_forward_length")
	hd := dim / heads
	wq, wkv := heads*hd, kv*hd

	gd := &rawGraphDecoder{
		heads: heads, kv: kv,
		eps:      rawMetaFloat(rf.Metadata, arch+".attention.layer_norm_rms_epsilon", 1e-5),
		ropeBase: rawMetaFloat(rf.Metadata, arch+".rope.freq_base", 10000),
	}
	var err error
	gd.emb, err = cuda.NewResidentB(deq("token_embd.weight").Cast(tensor.F32))
	mustTB(tb, err)
	gd.norm = vec("output_norm.weight")
	gd.out = q("output.weight")
	gd.pos, err = cuda.NewDevicePos()
	mustTB(tb, err)
	gd.inv, err = cuda.BuildRoPEInv(hd, gd.ropeBase)
	mustTB(tb, err)
	gd.dx, gd.dh, gd.dh2 = buf(1, dim), buf(1, dim), buf(1, dim)
	gd.dq, gd.da = buf(1, wq), buf(1, wq)
	gd.dk, gd.dv = buf(1, wkv), buf(1, wkv)
	gd.dgate, gd.dup = buf(1, hidden), buf(1, hidden)
	gd.scores = buf(1, heads*maxSeq)
	gd.logits = buf(1, rf.Tensors["output.weight"].Shape[0])
	gd.layers = make([]*rawLayer, nL)
	for i := 0; i < nL; i++ {
		p := "blk." + itoa(i) + "."
		var c *cuda.KVCache
		var cf *cuda.KVCacheF16
		if kvF16() {
			cf, err = cuda.NewKVCacheF16(maxSeq, wkv)
			mustTB(tb, err)
			mustTB(tb, cf.ZeroCache())
		} else {
			c, err = cuda.NewKVCache(maxSeq, wkv)
			mustTB(tb, err)
			mustTB(tb, c.ZeroCache())
			c.SetLen(maxSeq)
		}
		l := &rawLayer{
			gAttn: vec(p + "attn_norm.weight"), gFFN: vec(p + "ffn_norm.weight"),
			wq: q(p + "attn_q.weight"), wk: q(p + "attn_k.weight"),
			wv: q(p + "attn_v.weight"), wo: q(p + "attn_output.weight"),
			wg: q(p + "ffn_gate.weight"), wu: q(p + "ffn_up.weight"), wd: q(p + "ffn_down.weight"),
			cache: c, cacheF: cf,
		}
		if _, ok := rf.Tensors[p+"attn_q.bias"]; ok { // qwen2 QKV bias
			l.bq, l.bk, l.bv = vec(p+"attn_q.bias"), vec(p+"attn_k.bias"), vec(p+"attn_v.bias")
		}
		gd.layers[i] = l
	}
	return gd
}

func (gd *rawGraphDecoder) forwardBody(tb testing.TB) {
	for _, l := range gd.layers {
		mustTB(tb, gd.dx.RMSNormInto(l.gAttn, float32(gd.eps), gd.dh))
		mustTB(tb, l.wq.QMatMulInto(gd.dh, gd.dq))
		mustTB(tb, l.wk.QMatMulInto(gd.dh, gd.dk))
		mustTB(tb, l.wv.QMatMulInto(gd.dh, gd.dv))
		if l.bq != nil {
			mustTB(tb, gd.dq.AddBias(l.bq))
			mustTB(tb, gd.dk.AddBias(l.bk))
			mustTB(tb, gd.dv.AddBias(l.bv))
		}
		mustTB(tb, gd.dq.RoPEDposInv(gd.heads, gd.inv, gd.pos, 0))
		mustTB(tb, gd.dk.RoPEDposInv(gd.kv, gd.inv, gd.pos, 0))
		if l.cacheF != nil { // f16 KV (GOAI_CUDA_KV=f16): half the K/V bytes, flash-only
			mustTB(tb, l.cacheF.AppendDpos(gd.dk, gd.dv, gd.pos))
			mustTB(tb, cuda.GroupedQueryAttentionKVF16DposFlashInto(gd.dq, l.cacheF, gd.heads, gd.kv, gd.pos, gd.da))
		} else {
			mustTB(tb, l.cache.AppendDpos(gd.dk, gd.dv, gd.pos))
			kF, vF := l.cache.FullView()
			if os.Getenv("GOAI_CUDA_FUSED_ATTN") == "0" { // A/B toggle: the 3-kernel cuBLAS chain (beaten by flash: -3.5% @ctx160, -26% @ctx2004)
				mustTB(tb, cuda.GroupedQueryAttentionKVDposInto(gd.dq, kF, vF, gd.heads, gd.kv, gd.pos, gd.scores, gd.da))
			} else { // flash decode: GQA K/V-shared split-K + merge — the winner at every context depth
				mustTB(tb, cuda.GroupedQueryAttentionKVDposFlashInto(gd.dq, kF, vF, gd.heads, gd.kv, gd.pos, gd.da))
			}
		}
		mustTB(tb, l.wo.QMatMulAccInto(gd.da, gd.dx))
		mustTB(tb, gd.dx.RMSNormInto(l.gFFN, float32(gd.eps), gd.dh2))
		mustTB(tb, l.wg.QMatMulInto(gd.dh2, gd.dgate))
		if f, ok := l.wu.(swigluProj); ok && os.Getenv("GOAI_CUDA_FFN_FUSE") != "0" {
			// Tw55 fusion: SwiGLU in the up-GEMV epilogue — no separate SwiGLU
			// launch, no dgate round-trip between kernels.
			mustTB(tb, f.QMatMulSwiGLUInto(gd.dh2, gd.dgate, gd.dup))
			mustTB(tb, l.wd.QMatMulAccInto(gd.dup, gd.dx))
		} else {
			mustTB(tb, l.wu.QMatMulInto(gd.dh2, gd.dup))
			mustTB(tb, gd.dgate.SwiGLU(gd.dup))
			mustTB(tb, l.wd.QMatMulAccInto(gd.dgate, gd.dx))
		}
	}
	mustTB(tb, gd.dx.RMSNormInto(gd.norm, float32(gd.eps), gd.dh))
	mustTB(tb, gd.out.QMatMulInto(gd.dh, gd.logits))
}

func (gd *rawGraphDecoder) step(tb testing.TB, tok int32, pos int) int {
	mustTB(tb, gd.pos.Set(pos))
	mustTB(tb, gd.emb.EmbedInto([]int32{tok}, gd.dx))
	gd.forwardBody(tb)
	return gd.logits.Argmax()
}

func (gd *rawGraphDecoder) capture(tb testing.TB) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	mustTB(tb, cuda.CaptureBegin())
	gd.forwardBody(tb)
	g, err := cuda.CaptureEnd()
	mustTB(tb, err)
	gd.graph = g
}

func (gd *rawGraphDecoder) stepGraph(tb testing.TB, tok int32, pos int) int {
	mustTB(tb, gd.pos.Set(pos))
	mustTB(tb, gd.emb.EmbedInto([]int32{tok}, gd.dx))
	mustTB(tb, gd.graph.Launch())
	mustTB(tb, cuda.GraphSync())
	return gd.logits.Argmax()
}

func (gd *rawGraphDecoder) free() {
	gd.emb.Free()
	gd.norm.Free()
	gd.out.Free()
	gd.pos.Free()
	if gd.graph != nil {
		gd.graph.Free()
	}
	for _, d := range []*cuda.DeviceF32{gd.dx, gd.dh, gd.dh2, gd.dq, gd.dk, gd.dv, gd.da, gd.dgate, gd.dup, gd.scores, gd.logits, gd.inv} {
		d.Free()
	}
	for _, l := range gd.layers {
		l.gAttn.Free()
		l.gFFN.Free()
		if l.cache != nil {
			l.cache.Free()
		}
		if l.cacheF != nil {
			l.cacheF.Free()
		}
		for _, b := range []*cuda.ResidentVec{l.bq, l.bk, l.bv} {
			if b != nil {
				b.Free()
			}
		}
		for _, w := range []qProj{l.wq, l.wk, l.wv, l.wo, l.wg, l.wu, l.wd} {
			w.Free()
		}
	}
}

// runRawDecode builds the decoder at one precision, generates greedily, then measures
// graph decode throughput. Returns the generated tokens and tok/s.
func runRawDecode(t *testing.T, rf *gguf.RawFile, arch string, ids []int, gen, maxSeq int, quant func(gguf.QuantTensor) (qProj, error)) ([]int, float64) {
	gd := buildRawGraphDecoder(t, rf, arch, maxSeq, quant)
	defer gd.free()

	var last int
	for i, id := range ids {
		last = gd.step(t, int32(id), i)
	}
	gd.capture(t)
	toks := make([]int, 0, gen)
	for n := 0; n < gen; n++ {
		toks = append(toks, last)
		last = gd.stepGraph(t, int32(last), len(ids)+n)
	}
	// timed window (state already warm; continue decoding in place)
	const steps = 32
	tk := int32(last)
	t0 := time.Now()
	for d := 0; d < steps; d++ {
		tk = int32(gd.stepGraph(t, tk, len(ids)+gen+d))
	}
	tps := float64(steps) / time.Since(t0).Seconds()
	return toks, tps
}

// Mistral-7B-Instruct is the production-scale, weight-bandwidth-bound case for the Q4
// bandwidth lever (PERF-Q4-DECODE showed +23% on TinyLlama-1.1B). At 7B the weights
// dominate decode entirely, so Q4 (≈0.56 B/weight vs Q8's 1.125) should widen the lead —
// this test proves the engine runs a real 7B on 12 GB VRAM at both precisions, that Q4
// stays coherent, and measures the Q8→Q4 speedup at scale.
func TestCUDAMistral7BQ4QualityAndSpeed(t *testing.T) {
	skipNoGPU(t)
	if _, err := os.Stat(mistral7BPath); err != nil {
		t.Skipf("model not present (%s)", mistral7BPath)
	}
	f, err := os.Open(mistral7BPath)
	must(t, err)
	rf, err := gguf.ReadRaw(f)
	f.Close()
	must(t, err)
	// SPM (llama.cpp merge semantics), NOT UnigramFromGGUF: this GGUF carries real
	// rank scores, under which Viterbi fragments the prompt and wrecks generation (§B59).
	tok, err := nlp.SPMFromGGUF(rf.Metadata)
	must(t, err)

	ids := append([]int{1}, tok.Encode("The capital of France is")...)
	const gen = 24
	maxSeq := len(ids) + gen + 40

	q4vsQ8(t, rf, "llama", "Mistral-7B", ids, gen, maxSeq, func(toks []int) string {
		return strings.TrimSpace(tok.Decode(toks))
	})
}

// The Qwen2.5 models are where goai-Q8 trailed llama.cpp-Q8 in absolute terms
// (1.5B: 140 vs 166; 3B: 77 vs 87 tok/s, PERF-SCALEBENCH-2) — if Q4's bandwidth win
// carries here, goai takes the Q4-vs-Q8 lead at 3B. Also exercises the raw-GGUF
// builder's qwen2 path (QKV bias in-graph). Qwen-0.5B is excluded: dim=896 fails the
// Q4 kernel's K%256 constraint (and is the least weight-bound model, where Q4 matters
// least).
func TestCUDAQwenQ4QualityAndSpeed(t *testing.T) {
	skipNoGPU(t)
	for _, tc := range []struct{ path, label string }{
		{"../../models/qwen2.5-1.5b-instruct-q8_0.gguf", "Qwen2.5-1.5B"},
		{"../../models/qwen2.5-3b-instruct-q8_0.gguf", "Qwen2.5-3B"},
	} {
		t.Run(tc.label, func(t *testing.T) {
			if _, err := os.Stat(tc.path); err != nil {
				t.Skipf("model not present (%s)", tc.path)
			}
			f, err := os.Open(tc.path)
			must(t, err)
			rf, err := gguf.ReadRaw(f)
			f.Close()
			must(t, err)
			tok, err := nlp.BPEFromGGUF(rf.Metadata)
			must(t, err)

			ids := tok.Encode("The capital of France is")
			const gen = 24
			maxSeq := len(ids) + gen + 40

			q4vsQ8(t, rf, "qwen2", tc.label, ids, gen, maxSeq, tok.Decode)
		})
	}
}

// q4vsQ8 runs the same model at Q8, asymmetric Q4 and Q4_K, then applies the shared
// gates: every precision stays coherent (the prompt's factual answer appears) and the
// 4-bit paths decode faster than Q8 (fewer weight bytes on a bandwidth-bound GEMV).
func q4vsQ8(t *testing.T, rf *gguf.RawFile, arch, label string, ids []int, gen, maxSeq int, decode func([]int) string) {
	q8 := fromF32(func(w *tensor.Tensor) (qProj, error) { return cuda.NewResidentBQ8(w) })
	q4 := fromF32(func(w *tensor.Tensor) (qProj, error) { return cuda.NewResidentBQ4(w) })

	q8toks, q8tps := runRawDecode(t, rf, arch, ids, gen, maxSeq, q8)
	q4toks, q4tps := runRawDecode(t, rf, arch, ids, gen, maxSeq, q4)
	q4ktoks, q4ktps := runRawDecode(t, rf, arch, ids, gen, maxSeq, fromF32(quantQ4K))

	agree := func(toks []int) int {
		match := 0
		for i := range q8toks {
			if toks[i] == q8toks[i] {
				match++
			}
		}
		return match
	}
	q8text, q4text, q4ktext := decode(q8toks), decode(q4toks), decode(q4ktoks)
	t.Logf("%s Q8 text: %q", label, q8text)
	t.Logf("%s Q4 text: %q", label, q4text)
	t.Logf("%s Q4_K text: %q", label, q4ktext)
	t.Logf("%s greedy agreement vs Q8: Q4 %d/%d | Q4_K %d/%d", label, agree(q4toks), gen, agree(q4ktoks), gen)
	t.Logf("%s DECODE tok/s: Q8 %.1f | Q4 %.1f (%.2fx) | Q4_K %.1f (%.2fx)",
		label, q8tps, q4tps, q4tps/q8tps, q4ktps, q4ktps/q8tps)

	// Quality gates: every precision must answer correctly.
	for _, pr := range []struct{ name, text string }{{"Q8", q8text}, {"Q4", q4text}, {"Q4_K", q4ktext}} {
		if !strings.Contains(strings.ToLower(pr.text), "paris") {
			t.Errorf("%s %s decode incoherent (no 'Paris'): %q", label, pr.name, pr.text)
		}
	}
	// Speed gates: at these scales decode is weight-bandwidth-bound — 4-bit must beat Q8.
	if q4tps <= q8tps {
		t.Errorf("%s Q4 (%.1f tok/s) not faster than Q8 (%.1f tok/s)", label, q4tps, q8tps)
	}
	if q4ktps <= q8tps {
		t.Errorf("%s Q4_K (%.1f tok/s) not faster than Q8 (%.1f tok/s)", label, q4ktps, q8tps)
	}
}
