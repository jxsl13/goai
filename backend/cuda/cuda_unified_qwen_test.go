//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// rawF16Layer is the f16 tensor-core PREFILL twin of rawLayer: projections resident in
// f16, optional qwen2 QKV bias, built straight from a gguf.RawFile (so it covers model
// families nlp.LlamaFromGGUF rejects).
type rawF16Layer struct {
	gAttn, gFFN    *cuda.ResidentVec
	bq, bk, bv     *cuda.ResidentVec // qwen2 QKV bias; nil for the llama family
	wq, wk, wv, wo *cuda.ResidentBF16
	wg, wu, wd     *cuda.ResidentBF16
	heads, kv      int
	ropeBase, eps  float64
}

type rawF16Prefill struct {
	emb    *cuda.ResidentB
	norm   *cuda.ResidentVec
	out    *cuda.ResidentBF16
	layers []*rawF16Layer
	eps    float64
}

func buildRawF16Prefill(tb testing.TB, rf *gguf.RawFile, arch string) *rawF16Prefill {
	deq := func(n string) *tensor.Tensor {
		qt, ok := rf.Tensors[n]
		if !ok {
			tb.Fatalf("missing tensor %s", n)
		}
		t, err := qt.Dequantize()
		mustTB(tb, err)
		return t
	}
	f16 := func(n string) *cuda.ResidentBF16 {
		r, err := cuda.NewResidentBF16(transpose2D(deq(n)))
		mustTB(tb, err)
		return r
	}
	vec := func(n string) *cuda.ResidentVec {
		r, err := cuda.NewResidentVec(deq(n).Cast(tensor.F32))
		mustTB(tb, err)
		return r
	}
	dim := rawMetaInt(tb, rf.Metadata, arch+".embedding_length")
	nL := rawMetaInt(tb, rf.Metadata, arch+".block_count")
	heads := rawMetaInt(tb, rf.Metadata, arch+".attention.head_count")
	kv := rawMetaInt(tb, rf.Metadata, arch+".attention.head_count_kv")
	_ = dim
	p := &rawF16Prefill{eps: rawMetaFloat(rf.Metadata, arch+".attention.layer_norm_rms_epsilon", 1e-5)}
	ropeBase := rawMetaFloat(rf.Metadata, arch+".rope.freq_base", 10000)
	var err error
	p.emb, err = cuda.NewResidentB(deq("token_embd.weight").Cast(tensor.F32))
	mustTB(tb, err)
	p.norm = vec("output_norm.weight")
	p.out = f16("output.weight")
	p.layers = make([]*rawF16Layer, nL)
	for i := 0; i < nL; i++ {
		pre := "blk." + itoa(i) + "."
		l := &rawF16Layer{
			gAttn: vec(pre + "attn_norm.weight"), gFFN: vec(pre + "ffn_norm.weight"),
			wq: f16(pre + "attn_q.weight"), wk: f16(pre + "attn_k.weight"),
			wv: f16(pre + "attn_v.weight"), wo: f16(pre + "attn_output.weight"),
			wg: f16(pre + "ffn_gate.weight"), wu: f16(pre + "ffn_up.weight"), wd: f16(pre + "ffn_down.weight"),
			heads: heads, kv: kv, ropeBase: ropeBase, eps: p.eps,
		}
		if _, ok := rf.Tensors[pre+"attn_q.bias"]; ok {
			l.bq, l.bk, l.bv = vec(pre+"attn_q.bias"), vec(pre+"attn_k.bias"), vec(pre+"attn_v.bias")
		}
		p.layers[i] = l
	}
	return p
}

func (p *rawF16Prefill) free() {
	p.emb.Free()
	p.norm.Free()
	p.out.Free()
	for _, l := range p.layers {
		l.gAttn.Free()
		l.gFFN.Free()
		for _, b := range []*cuda.ResidentVec{l.bq, l.bk, l.bv} {
			if b != nil {
				b.Free()
			}
		}
		for _, w := range []*cuda.ResidentBF16{l.wq, l.wk, l.wv, l.wo, l.wg, l.wu, l.wd} {
			w.Free()
		}
	}
}

// seedForward runs one f16 prefill layer and appends the post-RoPE K/V rows into the
// decode cache — the arch-generic (llama + qwen2) form of the Tw47 handoff.
func (l *rawF16Layer) seedForward(dx *cuda.DeviceF32, cache *cuda.KVCache) (*cuda.DeviceF32, error) {
	rq := backend.RoPEAttrs{Base: l.ropeBase, Heads: l.heads}
	rk := backend.RoPEAttrs{Base: l.ropeBase, Heads: l.kv}
	dh, err := dx.RMSNormTo(l.gAttn, float32(l.eps))
	if err != nil {
		return nil, err
	}
	dq, err := l.wq.MatMulDevice(dh)
	if err != nil {
		return nil, err
	}
	dk, _ := l.wk.MatMulDevice(dh)
	dv, _ := l.wv.MatMulDevice(dh)
	dh.Free()
	if l.bq != nil {
		if err := dq.AddBias(l.bq); err != nil {
			return nil, err
		}
		if err := dk.AddBias(l.bk); err != nil {
			return nil, err
		}
		if err := dv.AddBias(l.bv); err != nil {
			return nil, err
		}
	}
	dq.RoPE(rq)
	dk.RoPE(rk)
	if err := cache.Append(dk, dv); err != nil {
		return nil, err
	}
	da, err := cuda.GroupedQueryAttentionTF32(dq, dk, dv, l.heads, l.kv, true)
	if err != nil {
		return nil, err
	}
	dq.Free()
	dk.Free()
	dv.Free()
	if err := l.wo.MatMulAccInto(da, dx); err != nil {
		return nil, err
	}
	da.Free()
	dh2, _ := dx.RMSNormTo(l.gFFN, float32(l.eps))
	dgate, _ := l.wg.MatMulDevice(dh2)
	dup, _ := l.wu.MatMulDevice(dh2)
	dh2.Free()
	dgate.SwiGLU(dup)
	dup.Free()
	if err := l.wd.MatMulAccInto(dgate, dx); err != nil {
		return nil, err
	}
	dgate.Free()
	return dx, nil
}

// The unified serving path generalized to the 2nd architecture: a Qwen2.5 model's f16
// batched prefill (QKV bias in the f16 stack) seeds the Q4_K decoder's caches. Gates
// mirror Tw47: speed ≥2× over token-by-token prompt processing, and the seeded K rows
// match the decode path's own within the projection-precision budget.
func TestCUDAUnifiedServeQwen(t *testing.T) {
	skipNoGPU(t)
	for _, tc := range []struct{ path, label string }{
		{"../../models/qwen2.5-1.5b-instruct-q8_0.gguf", "Qwen2.5-1.5B"},
		{"../../models/qwen2.5-3b-instruct-q8_0.gguf", "Qwen2.5-3B"},
	} {
		t.Run(tc.label, func(t *testing.T) { unifiedServeQwen(t, tc.path, tc.label) })
	}
}

func unifiedServeQwen(t *testing.T, path, label string) {
	if _, err := os.Stat(path); err != nil {
		t.Skipf("model not present (%s)", path)
	}
	f, err := os.Open(path)
	must(t, err)
	rf, err := gguf.ReadRaw(f)
	f.Close()
	must(t, err)
	tok, err := nlp.BPEFromGGUF(rf.Metadata)
	must(t, err)

	prompt := "France is a country in Western Europe, famous for its art, cuisine and history. " +
		"Millions of tourists visit its museums and monuments every year. The capital of France is"
	ids := tok.Encode(prompt)
	P := len(ids)
	const gen = 16
	maxSeq := P + gen + 8

	q4k := buildRawGraphDecoder(t, rf, "qwen2", maxSeq, fromF32(quantQ4K))
	defer q4k.free()

	// pure decode-path prompt processing (reference timing + reference cache rows)
	t0 := time.Now()
	var last int
	for i, id := range ids {
		last = q4k.step(t, int32(id), i)
	}
	decodePrefill := time.Since(t0)
	var refToks []int
	for n := 0; n < gen; n++ {
		refToks = append(refToks, last)
		last = q4k.step(t, int32(last), P+n)
	}
	q4k.layers[0].cache.SetLen(P)
	refK, err := q4k.layers[0].cache.K().ToHost()
	must(t, err)
	q4k.layers[0].cache.SetLen(maxSeq)

	// f16 batched prefill seeds fresh caches (warmup, reset, timed)
	pf := buildRawF16Prefill(t, rf, "qwen2")
	defer pf.free()
	ids32 := make([]int32, P)
	for i, id := range ids {
		ids32[i] = int32(id)
	}
	seed := func() *cuda.DeviceF32 {
		dx, err := pf.emb.Embed(ids32)
		must(t, err)
		for li, l := range pf.layers {
			dx, err = l.seedForward(dx, q4k.layers[li].cache)
			must(t, err)
		}
		return dx
	}
	reset := func() {
		for _, l := range q4k.layers {
			must(t, l.cache.ZeroCache())
			l.cache.SetLen(0)
		}
	}
	reset()
	seed().Free() // warmup
	must(t, cuda.GraphSync())
	reset()
	t0 = time.Now()
	dx := seed()
	must(t, dx.RMSNorm(pf.norm, float32(pf.eps)))
	dl, err := pf.out.MatMulDevice(dx)
	must(t, err)
	dx.Free()
	lg, err := dl.ToHost()
	must(t, err)
	dl.Free()
	must(t, cuda.GraphSync())
	fastPrefill := time.Since(t0)

	// mechanism gate: seeded rows vs decode-path rows
	q4k.layers[0].cache.SetLen(P)
	seedK, err := q4k.layers[0].cache.K().ToHost()
	must(t, err)
	q4k.layers[0].cache.SetLen(maxSeq)
	var sumAbs, sumDiff float64
	for r := 0; r < P; r++ {
		for c := 0; c < seedK.Shape()[1]; c++ {
			rv := refK.AtF64(r, c)
			sumAbs += mabs(rv)
			sumDiff += mabs(seedK.AtF64(r, c) - rv)
		}
	}
	l1 := sumDiff / sumAbs

	// continue decoding from P for the informational text
	vocab := lg.Shape()[1]
	first, bv := 0, -1e300
	for c := 0; c < vocab; c++ {
		if v := lg.AtF64(P-1, c); v > bv {
			bv, first = v, c
		}
	}
	uniToks := []int{first}
	lastTok := first
	for n := 1; n < gen; n++ {
		lastTok = q4k.step(t, int32(lastTok), P+n-1)
		uniToks = append(uniToks, lastTok)
	}

	t.Logf("%s pure-decode text: %q", label, strings.TrimSpace(tok.Decode(refToks)))
	t.Logf("%s unified text:     %q", label, strings.TrimSpace(tok.Decode(uniToks)))
	t.Logf("prompt P=%d: decode-path prefill %.1f ms | f16 batched %.1f ms (%.1fx)",
		P, float64(decodePrefill.Microseconds())/1000, float64(fastPrefill.Microseconds())/1000,
		float64(decodePrefill)/float64(fastPrefill))
	t.Logf("layer-0 K cache: f16-seeded vs decode-path rows, rel L1 %.4f", l1)

	if float64(decodePrefill) < 2*float64(fastPrefill) {
		t.Errorf("unified prefill not clearly faster: %v vs %v", decodePrefill, fastPrefill)
	}
	if l1 > 0.10 {
		t.Errorf("seeded cache diverges beyond quant budget: rel L1 %.4f", l1)
	}
	if !strings.Contains(strings.ToLower(tok.Decode(uniToks)), "paris") {
		t.Errorf("unified continuation incoherent (no 'Paris'): %q", tok.Decode(uniToks))
	}
}
