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
	wqkv           qProj // fused Q4_K wq|wk|wv (Tw55(b), GOAI_CUDA_QKV_FUSE=1); nil unless fusing
	wg, wu, wd     qProj
	wgu            qProj            // fused Q4_K ffn_gate|ffn_up (GOAI_CUDA_GATEUP_FUSE=1); nil unless fusing
	cache          *cuda.KVCache    // f32 cache (chain + flash paths)
	cacheF         *cuda.KVCacheF16 // f16 cache (GOAI_CUDA_KV=f16, flash-only)
}

// kvF16 selects the f16 KV cache (half the K/V bytes — the flash kernel's
// bandwidth currency). Opt-in while the A/B runs; flipped by measurement.
func kvF16() bool { return os.Getenv("GOAI_CUDA_KV") == "f16" }

// qkvFuse selects the fused-QKV weight path (Tw55(b)): wq|wk|wv concatenated into ONE
// N=(heads+2·kv)·hd Q4_K GEMV instead of three. The floor measurement
// (docs/benchmarking.md) found the GEMV latency-bound at small N — the GQA k/v
// projection (N=256) runs at only 17% of peak vs the q proj's 46% — so folding the
// starved k/v rows into the q launch lifts their occupancy. Opt-in while the A/B runs.
func qkvFuse() bool { return os.Getenv("GOAI_CUDA_QKV_FUSE") == "1" }

// gateUpFuse selects the fused gate+up weight path: ffn_gate|ffn_up concatenated into ONE
// N=2·hidden Q4_K GEMV, then SwiGLU over the two halves. A scientific counterpart to the
// QKV fusion — gate/up are NOT starved (N=5632 = 55% of peak, vs the k/v N=256 = 17%), so
// the occupancy-cliff theory predicts only a small gain here. Opt-in; measured by A/B.
func gateUpFuse() bool { return os.Getenv("GOAI_CUDA_GATEUP_FUSE") == "1" }

// q4kPre selects the pre-decoded-scale Q4_K resident (perf-notes-cuda.md R5) for the projections
// it wins on. Opt-in A/B; bit-exact so decode tokens are identical either way.
func q4kPre() bool { return os.Getenv("GOAI_CUDA_Q4KPRE") == "1" }

// q4kPreProj is true for the projections where the pre-decode's ALU relief beats its +33% bytes
// (measured, docs/perf-notes-cuda.md): k/v (starved small-N, not BW-bound) and ffn_down (large-K).
func q4kPreProj(n string) bool {
	return strings.Contains(n, "attn_k.weight") || strings.Contains(n, "attn_v.weight") || strings.Contains(n, "ffn_down.weight")
}

// stackRows0 concatenates output-major [Ni,K] f32 weight tensors along axis 0 into one
// [ΣNi,K] tensor (row-major, so a plain storage append). Feeding this to the Q4_K
// encoder yields blocks whose first Nq rows are byte-identical to encoding wq alone —
// the fused GEMV is therefore bit-exact per output row vs the three separate GEMVs.
func stackRows0(ts ...*tensor.Tensor) *tensor.Tensor {
	k := ts[0].Shape()[1]
	ntot := 0
	for _, t := range ts {
		ntot += t.Shape()[0]
	}
	out := tensor.New(tensor.F32, tensor.Shape{ntot, k})
	of := out.Storage().F32()
	off := 0
	for _, t := range ts {
		s := t.Contiguous().Storage().F32()
		copy(of[off:off+len(s)], s)
		off += len(s)
	}
	return out
}

// fuseRowsQ4K builds one fused Q4_K projection over the row-stacked weights (QKV or
// gate+up). Only the Q4_K format is fused (the format the A/B and parity tests exercise);
// the caller falls back to separate projections otherwise.
func fuseRowsQ4K(ts ...*tensor.Tensor) (qProj, error) {
	stacked := stackRows0(ts...)
	n, k := stacked.Shape()[0], stacked.Shape()[1]
	blocks, err := gguf.Quantize(stacked, gguf.Q4_K)
	if err != nil {
		return nil, err
	}
	return cuda.NewResidentBQ4KFromBlocks(blocks, k, n)
}

// fuseRowsQ8 is the Q8_0 twin of fuseRowsQ4K: it row-stacks the weights ([ΣN,K]), transposes
// to the in-major [K,ΣN] that NewResidentBQ8 wants, and uploads one fused Q8 projection.
// Q8_0 quantizes each output column's K in independent 32-blocks, so the fused weight's first
// Nq columns are byte-identical to encoding wq alone — bit-exact per output row, only the GEMV
// launch is merged, exactly like the Q4_K case.
func fuseRowsQ8(ts ...*tensor.Tensor) (qProj, error) {
	return cuda.NewResidentBQ8(transpose2D(stackRows0(ts...)))
}

// fuseRows dispatches row-fusion to the decoder's quant format (Tw57): GOAI_CUDA_FUSE_FMT=q8
// fuses as Q8_0, else Q4_K (default, unchanged behavior). Without this a Q8 decoder silently
// requantized its fused QKV/gate-up down to Q4_K — a precision downgrade the separate-GEMV path
// never took. The two knobs (GOAI_CUDA_FUSE_FMT + the model's quant closure) must agree; the
// parity tests pin that.
func fuseRows(ts ...*tensor.Tensor) (qProj, error) {
	if os.Getenv("GOAI_CUDA_FUSE_FMT") == "q8" {
		return fuseRowsQ8(ts...)
	}
	return fuseRowsQ4K(ts...)
}

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
	dqkv               *cuda.DeviceF32 // fused QKV output buffer (Tw55(b)); dq/dk/dv are Views into it
	dgu                *cuda.DeviceF32 // fused gate+up output buffer; dgate/dup are Views into it
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

// unpermuteRawQK reorders a raw llama-arch attn_q/attn_k QuantTensor from GGUF's on-disk
// interleaved rotary row layout back to the HF split-half order GoAI's RoPE expects — the
// B67 fix (ac4709b applied it to nlp.LlamaFromGGUF; this is the raw-decoder twin). A row of
// the [out,in] quant tensor is a fixed-size stripe of blocks, so the un-permute is a lossless
// byte-range reorder: dst row h·hd+q ← src row h·hd+(2i+j), j=q/(hd/2), i=q%(hd/2).
func unpermuteRawQK(tb testing.TB, qt gguf.QuantTensor, heads int) gguf.QuantTensor {
	rows := qt.Shape[0]
	hd := rows / heads
	if heads <= 0 || rows%heads != 0 || hd%2 != 0 || len(qt.Data)%rows != 0 {
		tb.Fatalf("unpermuteRawQK: bad geometry rows=%d heads=%d dataBytes=%d", rows, heads, len(qt.Data))
	}
	rowBytes := len(qt.Data) / rows
	out := make([]byte, len(qt.Data))
	for h := 0; h < heads; h++ {
		for q := 0; q < hd; q++ {
			j, i := q/(hd/2), q%(hd/2)
			src, dst := h*hd+2*i+j, h*hd+q
			copy(out[dst*rowBytes:(dst+1)*rowBytes], qt.Data[src*rowBytes:(src+1)*rowBytes])
		}
	}
	qt.Data = out
	return qt
}

func buildRawGraphDecoder(tb testing.TB, rf *gguf.RawFile, arch string, maxSeq int, quant func(gguf.QuantTensor) (qProj, error)) *rawGraphDecoder {
	// B67: llama-arch GGUFs store attn_q/attn_k rows in the interleaved rotary layout; GoAI's
	// split-half RoPE needs them un-permuted (same transform nlp.LlamaFromGGUF applies since
	// ac4709b — without this the decoder is self-consistent but semantically off-convention,
	// and the f16-seeded serve handoff comparison breaks at rel L1 1.4). qwen2 etc. are
	// NEOX/no-permute on disk and pass through untouched. rf is never mutated (shared by A/Bs).
	tensors := rf.Tensors
	if arch == "llama" {
		heads := rawMetaInt(tb, rf.Metadata, arch+".attention.head_count")
		kvh := rawMetaInt(tb, rf.Metadata, arch+".attention.head_count_kv")
		nL := rawMetaInt(tb, rf.Metadata, arch+".block_count")
		tensors = make(map[string]gguf.QuantTensor, len(rf.Tensors))
		for k, v := range rf.Tensors {
			tensors[k] = v
		}
		for i := 0; i < nL; i++ {
			p := "blk." + itoa(i) + "."
			if qt, ok := tensors[p+"attn_q.weight"]; ok && len(qt.Shape) == 2 {
				tensors[p+"attn_q.weight"] = unpermuteRawQK(tb, qt, heads)
			}
			if qt, ok := tensors[p+"attn_k.weight"]; ok && len(qt.Shape) == 2 {
				tensors[p+"attn_k.weight"] = unpermuteRawQK(tb, qt, kvh)
			}
		}
	}
	deq := func(n string) *tensor.Tensor {
		qt, ok := tensors[n]
		if !ok {
			tb.Fatalf("missing tensor %s", n)
		}
		t, err := qt.Dequantize()
		mustTB(tb, err)
		return t
	}
	q := func(n string) qProj { // quantizers receive the RAW QuantTensor ([out,in] blocks)
		qt, ok := tensors[n]
		if !ok {
			tb.Fatalf("missing tensor %s", n)
		}
		// Q4_K pre-decoded-scale routing (perf-notes-cuda.md R5): the pre-decode wins only where the
		// shape is NOT bandwidth-bound — k/v (starved small-N) and ffn_down (large-K, most scale-decode
		// ALU) — so route ONLY those to ResidentBQ4KPre; q/o/gate/up stay standard (their +33% bytes
		// would lose). GGType 12 == Q4_K. Opt-in A/B (GOAI_CUDA_Q4KPRE=1). Bit-exact → token-identical.
		if q4kPre() && qt.GGType == 12 && q4kPreProj(n) {
			r, err := cuda.NewResidentBQ4KPreFromBlocks(qt.Data, qt.Shape[1], qt.Shape[0])
			mustTB(tb, err)
			return r
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
	gd.da = buf(1, wq)
	if qkvFuse() {
		// One contiguous QKV buffer; dq/dk/dv are zero-copy Views into it so the
		// fused N=wq+2·wkv GEMV writes all three in a single launch (Tw55(b)).
		gd.dqkv = buf(1, wq+2*wkv)
		gd.dq, err = gd.dqkv.View(0, 1, wq)
		mustTB(tb, err)
		gd.dk, err = gd.dqkv.View(wq, 1, wkv)
		mustTB(tb, err)
		gd.dv, err = gd.dqkv.View(wq+wkv, 1, wkv)
		mustTB(tb, err)
	} else {
		gd.dq = buf(1, wq)
		gd.dk, gd.dv = buf(1, wkv), buf(1, wkv)
	}
	if gateUpFuse() {
		gd.dgu = buf(1, 2*hidden)
		gd.dgate, err = gd.dgu.View(0, 1, hidden)
		mustTB(tb, err)
		gd.dup, err = gd.dgu.View(hidden, 1, hidden)
		mustTB(tb, err)
	} else {
		gd.dgate, gd.dup = buf(1, hidden), buf(1, hidden)
	}
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
		_, hasBias := rf.Tensors[p+"attn_q.bias"]
		l := &rawLayer{
			gAttn: vec(p + "attn_norm.weight"), gFFN: vec(p + "ffn_norm.weight"),
			wo: q(p + "attn_output.weight"), wd: q(p + "ffn_down.weight"),
			cache: c, cacheF: cf,
		}
		if qkvFuse() { // fused QKV (Tw55(b)); format follows GOAI_CUDA_FUSE_FMT (Tw57). Bias'd families
			// (qwen2) fuse too (Tw57 slice 2): the bias is additive post-GEMV, and dq/dk/dv are Views into
			// dqkv, so the per-section AddBias below writes the right slice of the fused buffer.
			l.wqkv, err = fuseRows(deq(p+"attn_q.weight"), deq(p+"attn_k.weight"), deq(p+"attn_v.weight"))
			mustTB(tb, err)
		} else {
			l.wq, l.wk, l.wv = q(p+"attn_q.weight"), q(p+"attn_k.weight"), q(p+"attn_v.weight")
		}
		if gateUpFuse() { // fused gate+up (one N=2·hidden GEMV, then SwiGLU over the halves); format per GOAI_CUDA_FUSE_FMT
			l.wgu, err = fuseRows(deq(p+"ffn_gate.weight"), deq(p+"ffn_up.weight"))
			mustTB(tb, err)
		} else {
			l.wg, l.wu = q(p+"ffn_gate.weight"), q(p+"ffn_up.weight")
		}
		if hasBias { // qwen2 QKV bias
			l.bq, l.bk, l.bv = vec(p+"attn_q.bias"), vec(p+"attn_k.bias"), vec(p+"attn_v.bias")
		}
		gd.layers[i] = l
	}
	return gd
}

func (gd *rawGraphDecoder) forwardBody(tb testing.TB) {
	for _, l := range gd.layers {
		mustTB(tb, gd.dx.RMSNormInto(l.gAttn, float32(gd.eps), gd.dh))
		if l.wqkv != nil { // Tw55(b) fused: one N=(heads+2·kv)·hd GEMV; dq/dk/dv View into dqkv
			mustTB(tb, l.wqkv.QMatMulInto(gd.dh, gd.dqkv))
		} else {
			mustTB(tb, l.wq.QMatMulInto(gd.dh, gd.dq))
			mustTB(tb, l.wk.QMatMulInto(gd.dh, gd.dk))
			mustTB(tb, l.wv.QMatMulInto(gd.dh, gd.dv))
		}
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
		if l.wgu != nil { // fused gate+up: one N=2·hidden GEMV → [gate|up], then SwiGLU over the halves
			mustTB(tb, l.wgu.QMatMulInto(gd.dh2, gd.dgu))
			mustTB(tb, gd.dgate.SwiGLU(gd.dup))
			mustTB(tb, l.wd.QMatMulAccInto(gd.dgate, gd.dx))
			continue
		}
		mustTB(tb, l.wg.QMatMulInto(gd.dh2, gd.dgate))
		if f, ok := l.wu.(swigluProj); ok && os.Getenv("GOAI_CUDA_FFN_FUSE") == "1" {
			// Tw55 fusion: SwiGLU in the up-GEMV epilogue — no separate SwiGLU
			// launch, no dgate round-trip between kernels. OPT-IN (default off):
			// token-parity-exact (TestCUDAFFNFuseTokenParity) but measured −0.9%
			// decode @TinyLlama-1.1B (TestCUDAFFNFuseSpeedAB) — the SwiGLU launch
			// is only ~1.8% of the step (PERF-PREFILL-PROFILE), so folding it away
			// can't beat the chain, and the lane-0 epilogue serializes the activation.
			// Kept for correctness + a launch-bound regime; the chain stays default.
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
	// dqkv (fused) owns the QKV memory; dq/dk/dv are Views into it (Free is a no-op).
	// When not fusing, dqkv is nil and dq/dk/dv own separate buffers.
	for _, d := range []*cuda.DeviceF32{gd.dx, gd.dh, gd.dh2, gd.dqkv, gd.dgu, gd.dq, gd.dk, gd.dv, gd.da, gd.dgate, gd.dup, gd.scores, gd.logits, gd.inv} {
		if d != nil {
			d.Free()
		}
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
		for _, w := range []qProj{l.wq, l.wk, l.wv, l.wqkv, l.wo, l.wg, l.wu, l.wgu, l.wd} {
			if w != nil {
				w.Free()
			}
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
	if testing.Short() {
		t.Skip("model-loading integration test; -short = kernel-level gate")
	}
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
	if testing.Short() {
		t.Skip("model-loading integration test; -short = kernel-level gate")
	}
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
