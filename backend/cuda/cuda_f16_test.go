//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// f16 rounds a float32 to the nearest-even float16 value (returned as float32) —
// the host reference for what the GPU conversion kernel must produce.
func f16Round(v float32) float32 {
	x := math.Float32bits(v)
	sign := x & 0x80000000
	em := x & 0x7fffffff
	var h uint16
	switch {
	case em >= 0x47800000: // overflow → inf (nan keeps payload bit)
		if em > 0x7f800000 {
			h = uint16(sign>>16 | 0x7e00)
		} else {
			h = uint16(sign>>16 | 0x7c00)
		}
	case em < 0x38800000: // subnormal/zero
		m := em&0x007fffff | 0x00800000
		shift := 126 - int(em>>23)
		var val, r, s uint32
		if shift < 32 {
			val = m >> uint(shift)
			r = (m >> uint(shift-1)) & 1
			if m&((1<<uint(shift-1))-1) != 0 {
				s = 1
			}
		}
		val += r & (s | (val & 1))
		h = uint16(sign>>16 | val)
	default:
		e := (em>>23 - 112) << 10
		m := (em >> 13) & 0x3ff
		r := (em >> 12) & 1
		var s uint32
		if em&0xfff != 0 {
			s = 1
		}
		h = uint16((sign>>16 | e | m) + (r & (s | (m & 1))))
	}
	// decode back
	sgn := float32(1)
	if h&0x8000 != 0 {
		sgn = -1
	}
	exp := int(h>>10) & 31
	man := int(h & 0x3ff)
	switch exp {
	case 0:
		return sgn * float32(man) * float32(math.Pow(2, -24))
	case 31:
		if man == 0 {
			return sgn * float32(math.Inf(1))
		}
		return float32(math.NaN())
	default:
		return sgn * float32(1+float64(man)/1024) * float32(math.Pow(2, float64(exp-15)))
	}
}

// The f16 tensor-core GEMM must match a host reference computed over the SAME f16
// roundings (weights AND activations rounded to f16, f32 accumulation) — so the
// tolerance covers only summation order, never the f16 quantization itself.
func TestCUDAF16MatMulParity(t *testing.T) {
	skipNoGPU(t)
	t.Setenv("GOAI_CUDA_F16ACC", "0") // this asserts f32-accumulate exactness (maxRel≤1e-4);
	// f16-accumulate (default-on for GeForce) is validated separately at prefill-appropriate tolerance.
	const K, N, M = 512, 48, 3
	rng := rand.New(rand.NewSource(11))
	w := tensor.New(tensor.F32, tensor.Shape{K, N})
	wf := w.Storage().F32()
	for i := range wf {
		wf[i] = float32(rng.NormFloat64())
	}
	a := make([]float32, M*K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}

	rw, err := cuda.NewResidentBF16(w)
	must(t, err)
	defer rw.Free()
	da, err := cuda.NewDeviceF32(M, K)
	must(t, err)
	defer da.Free()
	must(t, da.UploadF32(a))
	dout, err := rw.MatMulDevice(da)
	must(t, err)
	defer dout.Free()
	got, err := dout.ToHost()
	must(t, err)

	var maxRel float64
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			var ref float64
			for k := 0; k < K; k++ {
				ref += float64(f16Round(a[m*K+k])) * float64(f16Round(wf[k*N+n]))
			}
			g := got.AtF64(m, n)
			rel := math.Abs(g-ref) / math.Max(math.Abs(ref), 1e-6)
			if rel > maxRel {
				maxRel = rel
			}
		}
	}
	t.Logf("f16 GemmEx vs f16-rounded host reference: max rel err %.3g", maxRel)
	if maxRel > 1e-4 {
		t.Fatalf("f16 GEMM deviates beyond summation order: %.3g", maxRel)
	}
}

// Prefill A/B: the f32 cuBLAS path vs the f16 tensor-core path on the real TinyLlama
// layer stack. Correctness gate: greedy argmax at every position must match the f32
// path (f16 weight rounding shifts logits ~1e-3 — argmax on real text prompts holds).
func TestCUDAF16PrefillSpeedAndParity(t *testing.T) {
	if testing.Short() {
		t.Skip("model-loading integration test; -short = kernel-level gate")
	}
	skipNoGPU(t)
	if _, err := os.Stat(tinyLlamaPath); err != nil {
		t.Skipf("model not present (%s)", tinyLlamaPath)
	}
	f, err := gguf.ReadFile(tinyLlamaPath)
	must(t, err)
	model, err := nlp.LlamaFromGGUF(f.Metadata, f.Tensors)
	must(t, err)
	cfg := model.Config

	rl := buildResidentLlama(t, model)
	defer rl.free()
	rl16 := buildResidentLlamaF16(t, model)
	defer rl16.free()

	for _, seq := range []int{32, 128} {
		t.Run(fmt.Sprintf("seq%d", seq), func(t *testing.T) { f16PrefillAB(t, rl, rl16, cfg.Vocab, seq) })
	}
}

func f16PrefillAB(t *testing.T, rl *residentLlama, rl16 *residentLlamaF16, vocabSize, seq int) {
	ids := make([]int32, seq)
	for i := range ids {
		ids[i] = int32((i*131 + 1) % vocabSize)
	}

	run32 := func() *tensor.Tensor {
		dx, err := rl.emb.Embed(ids)
		must(t, err)
		for _, l := range rl.layers {
			dx, err = l.forward(dx)
			must(t, err)
		}
		must(t, dx.RMSNorm(rl.norm, float32(rl.eps)))
		dl, err := rl.out.MatMulDevice(dx)
		must(t, err)
		dx.Free()
		out, err := dl.ToHost()
		must(t, err)
		dl.Free()
		return out
	}
	run16 := func() *tensor.Tensor {
		dx, err := rl16.emb.Embed(ids)
		must(t, err)
		for _, l := range rl16.layers {
			dx, err = l.forward(dx)
			must(t, err)
		}
		must(t, dx.RMSNorm(rl16.norm, float32(rl16.eps)))
		dl, err := rl16.out.MatMulDevice(dx)
		must(t, err)
		dx.Free()
		out, err := dl.ToHost()
		must(t, err)
		dl.Free()
		return out
	}

	l32 := run32()
	l16 := run16()
	vocab := l32.Shape()[1]
	argmax := func(lg *tensor.Tensor, r int) int {
		best, bv := 0, math.Inf(-1)
		for c := 0; c < vocab; c++ {
			if v := lg.AtF64(r, c); v > bv {
				bv, best = v, c
			}
		}
		return best
	}
	mism := 0
	for r := 0; r < seq; r++ {
		if argmax(l32, r) != argmax(l16, r) {
			mism++
		}
	}
	t.Logf("f16 vs f32 prefill: %d/%d greedy positions agree", seq-mism, seq)
	if mism > seq/10 { // f16 rounding may flip a few near-ties; >10% means something is wrong
		t.Errorf("f16 prefill diverges: %d/%d positions differ", mism, seq)
	}

	// timed A/B, interleaved (the §T452 lesson), device-side only
	const reps = 5
	var t32, t16 time.Duration
	for i := 0; i < reps; i++ {
		s := time.Now()
		run32()
		t32 += time.Since(s)
		s = time.Now()
		run16()
		t16 += time.Since(s)
	}
	tok32 := float64(seq*reps) / t32.Seconds()
	tok16 := float64(seq*reps) / t16.Seconds()
	t.Logf("prefill seq=%d: f32 %.0f tok/s | f16 tensor-core %.0f tok/s (%.2fx)", seq, tok32, tok16, tok16/tok32)
	if tok16 <= tok32 {
		t.Errorf("f16 prefill (%.0f tok/s) not faster than f32 (%.0f tok/s)", tok16, tok32)
	}
}

// residentLlamaF16 mirrors residentLlama with all projection weights in f16 (embeddings,
// norms and the head stay f32 — the head's [dim,vocab] GEMM is also swapped to f16).
type resLayerF16 struct {
	gAttn, gFFN   *cuda.ResidentVec
	qkv           *cuda.ResidentBF16QKV // fused wq|wk|wv — 1.19x vs three separate GEMMs (BenchmarkQKVProj)
	wo            *cuda.ResidentBF16
	wg, wu, wd    *cuda.ResidentBF16
	heads, kv     int
	ropeBase, eps float64
}

type residentLlamaF16 struct {
	emb    *cuda.ResidentB
	out    *cuda.ResidentBF16
	norm   *cuda.ResidentVec
	layers []*resLayerF16
	eps    float64
}

func buildResidentLlamaF16(tb testing.TB, m *nlp.Llama) *residentLlamaF16 {
	cfg := m.Config
	kv := cfg.KVHeads
	if kv == 0 {
		kv = cfg.Heads
	}
	cast := func(t *tensor.Tensor) *tensor.Tensor { return t.Cast(tensor.F32) }
	mk := func(w *tensor.Tensor) *cuda.ResidentBF16 {
		r, err := cuda.NewResidentBF16(cast(w))
		mustTB(tb, err)
		return r
	}
	rl := &residentLlamaF16{eps: cfg.Eps}
	rl.emb, _ = cuda.NewResidentB(cast(m.TokEmb))
	rl.norm, _ = cuda.NewResidentVec(cast(m.Norm.Gamma))
	rl.out = mk(m.Out)
	rl.layers = make([]*resLayerF16, len(m.Blocks))
	for i, blk := range m.Blocks {
		ga, _ := cuda.NewResidentVec(cast(blk.AttnNorm.Gamma))
		gf, _ := cuda.NewResidentVec(cast(blk.FFNNorm.Gamma))
		qkv, err := cuda.NewResidentBF16QKV(cast(blk.Wq), cast(blk.Wk), cast(blk.Wv))
		mustTB(tb, err)
		rl.layers[i] = &resLayerF16{
			gAttn: ga, gFFN: gf,
			qkv: qkv, wo: mk(blk.Wo),
			wg: mk(blk.FFN.Wgate), wu: mk(blk.FFN.Wup), wd: mk(blk.FFN.Wdown),
			heads: cfg.Heads, kv: kv, ropeBase: cfg.RopeBase, eps: cfg.Eps,
		}
	}
	return rl
}

func (rl *residentLlamaF16) free() {
	rl.emb.Free()
	rl.norm.Free()
	rl.out.Free()
	for _, l := range rl.layers {
		l.gAttn.Free()
		l.gFFN.Free()
		l.qkv.Free()
		for _, w := range []*cuda.ResidentBF16{l.wo, l.wg, l.wu, l.wd} {
			w.Free()
		}
	}
}

// forward consumes dx (frees it) and returns the layer output residual — the f16 twin
// of resLayer.forward.
func (l *resLayerF16) forward(dx *cuda.DeviceF32) (*cuda.DeviceF32, error) {
	rq := backend.RoPEAttrs{Base: l.ropeBase, Heads: l.heads}
	rk := backend.RoPEAttrs{Base: l.ropeBase, Heads: l.kv}
	dh, err := dx.RMSNormTo(l.gAttn, float32(l.eps))
	if err != nil {
		return nil, err
	}
	dq, dk, dv, err := l.qkv.MatMulQKV(dh)
	if err != nil {
		return nil, err
	}
	dh.Free()
	dq.RoPE(rq)
	dk.RoPE(rk)
	// §ROADMAP A2: fused tensor-core attention (nvcc mma.h) — 2.69x the cuBLAS-batched path.
	// hd==64 & seq%16==0 hold for the prefill shapes; fall back otherwise.
	var da *cuda.DeviceF32
	if dq.Cols()/l.heads == 64 && dq.Rows()%16 == 0 {
		da, err = cuda.GroupedQueryAttentionWMMA(dq, dk, dv, l.heads, l.kv)
	} else {
		da, err = cuda.GroupedQueryAttentionTF32(dq, dk, dv, l.heads, l.kv, true)
	}
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
