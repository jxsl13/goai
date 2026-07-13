//go:build vulkan && cgo

package vulkan

import (
	"math"
	"testing"
	"time"
)

// §T385 (ADR-0019 Phase 2 Vulkan INTEGRATION, mirrors Metal §T378/§T379): assemble a FULL
// multi-layer decoder over the Vulkan Recorder — per-layer attention + MLP over per-layer persistent
// KV caches, then final-norm + vocab head → logits, all in ONE Recorder per step. Verified small
// (D=64) against a host float64 reference incl. MLP/gelu/vocab; measured honestly at a realistic
// decode size (D=512) batched vs per-op. Completes [[gpu-ops-all-backends]] parity for the batched
// decode path (Metal §T379 got 3.10× at this size).

type layerW struct {
	wq, wk, wv, wo []float32
	gAttn, gMLP    []float32
	w1             []float32
	w2             []float32
}

type fullModel struct {
	D, H, dk, half, F, V, nLayers, maxLen int
	eps, posDiv, scale                    float32
	layers                                []layerW
	gFinal                                []float32
	wvocab                                []float32
	inv                                   []float32
}

func newFullModel(D, H, nLayers, V, maxLen int) *fullModel {
	dk := D / H
	F := 4 * D
	fill := func(n, seed int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*7+seed*13)%23-11) * 0.02
		}
		return s
	}
	fillG := func(n, seed int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = 0.9 + float32((i+seed)%5)*0.02
		}
		return s
	}
	m := &fullModel{
		D: D, H: H, dk: dk, half: dk / 2, F: F, V: V, nLayers: nLayers, maxLen: maxLen,
		eps: 1e-5, posDiv: 1.0, scale: float32(1.0 / math.Sqrt(float64(dk))),
		gFinal: fillG(D, 99), wvocab: fill(D*V, 77), inv: make([]float32, dk/2),
	}
	for i := range m.inv {
		m.inv[i] = 1.0 / float32(math.Pow(10000, float64(2*i)/float64(dk)))
	}
	for l := range nLayers {
		m.layers = append(m.layers, layerW{
			wq: fill(D*D, l*4+1), wk: fill(D*D, l*4+2), wv: fill(D*D, l*4+3), wo: fill(D*D, l*4+4),
			gAttn: fillG(D, l*2+1), gMLP: fillG(D, l*2+2),
			w1: fill(D*F, l*8+5), w2: fill(F*D, l*8+6),
		})
	}
	return m
}

func rmsnormHost(x []float32, g []float32, eps float32) []float64 {
	D := len(x)
	var ss float64
	for _, xv := range x {
		ss += float64(xv) * float64(xv)
	}
	inv := 1.0 / math.Sqrt(ss/float64(D)+float64(eps))
	o := make([]float64, D)
	for i := range o {
		o[i] = float64(x[i]) * inv * float64(g[i])
	}
	return o
}

func (m *fullModel) hostLogits(x []float32, kC, vC [][]float32, L int) []float32 {
	D, H, dk, F := m.D, m.H, m.dk, m.F
	cur := make([]float32, D)
	copy(cur, x)
	matD := func(in []float64, w []float32, out int) []float64 {
		o := make([]float64, out)
		for j := range out {
			var acc float64
			for p := range in {
				acc += in[p] * float64(w[p*out+j])
			}
			o[j] = acc
		}
		return o
	}
	for l := range m.nLayers {
		lw := m.layers[l]
		xn := rmsnormHost(cur, lw.gAttn, m.eps)
		q, k, v := matD(xn, lw.wq, D), matD(xn, lw.wk, D), matD(xn, lw.wv, D)
		rope := func(t []float64) {
			pos := float64(L) / float64(m.posDiv)
			for h := range H {
				for i := range m.half {
					base := h*dk + i
					theta := float64(m.inv[i])
					c, s := math.Cos(pos*theta), math.Sin(pos*theta)
					lo, hi := t[base], t[base+m.half]
					t[base] = lo*c - hi*s
					t[base+m.half] = hi*c + lo*s
				}
			}
		}
		rope(q)
		rope(k)
		for j := range D {
			kC[l] = append(kC[l], float32(k[j]))
			vC[l] = append(vC[l], float32(v[j]))
		}
		attn := make([]float64, D)
		for h := range H {
			off := h * dk
			sc := make([]float64, L+1)
			mx := math.Inf(-1)
			for j := 0; j <= L; j++ {
				var sd float64
				for d := range dk {
					sd += q[off+d] * float64(kC[l][j*D+off+d])
				}
				sd *= float64(m.scale)
				sc[j] = sd
				if sd > mx {
					mx = sd
				}
			}
			var ls float64
			for j := 0; j <= L; j++ {
				sc[j] = math.Exp(sc[j] - mx)
				ls += sc[j]
			}
			for d := range dk {
				var acc float64
				for j := 0; j <= L; j++ {
					acc += sc[j] * float64(vC[l][j*D+off+d])
				}
				attn[off+d] = acc / ls
			}
		}
		ao := matD(attn, lw.wo, D)
		for j := range D {
			cur[j] = cur[j] + float32(ao[j])
		}
		xn2 := rmsnormHost(cur, lw.gMLP, m.eps)
		hh := matD(xn2, lw.w1, F)
		for j := range hh {
			hh[j] = 0.5 * hh[j] * (1.0 + math.Erf(hh[j]*0.70710678118654752))
		}
		mo := matD(hh, lw.w2, D)
		for j := range D {
			cur[j] = cur[j] + float32(mo[j])
		}
	}
	xf := rmsnormHost(cur, m.gFinal, m.eps)
	logits := matD(xf, m.wvocab, m.V)
	out := make([]float32, m.V)
	for j := range out {
		out[j] = float32(logits[j])
	}
	return out
}

type gpuFullModel struct {
	m                                             *fullModel
	wq, wk, wv, wo, gA, gM, w1, w2                []*DeviceBuffer
	kC, vC                                        []*DeviceBuffer
	dinv, dgFinal, dwvocab                        *DeviceBuffer
	dx, xn, q, k, v, attn, ao, xn2, h, mo, logits *DeviceBuffer
	all                                           []*DeviceBuffer
}

func (m *fullModel) toGPU(t *testing.T) *gpuFullModel {
	g := &gpuFullModel{m: m}
	mk := func(data []float32) *DeviceBuffer {
		b, err := NewDeviceBufferF32(data)
		if err != nil {
			t.Fatal(err)
		}
		g.all = append(g.all, b)
		return b
	}
	D, F := m.D, m.F
	per := func() []*DeviceBuffer { return make([]*DeviceBuffer, m.nLayers) }
	g.wq, g.wk, g.wv, g.wo = per(), per(), per(), per()
	g.gA, g.gM, g.w1, g.w2 = per(), per(), per(), per()
	g.kC, g.vC = per(), per()
	for l := range m.nLayers {
		lw := m.layers[l]
		g.wq[l], g.wk[l], g.wv[l], g.wo[l] = mk(lw.wq), mk(lw.wk), mk(lw.wv), mk(lw.wo)
		g.gA[l], g.gM[l] = mk(lw.gAttn), mk(lw.gMLP)
		g.w1[l], g.w2[l] = mk(lw.w1), mk(lw.w2)
		g.kC[l] = mk(make([]float32, m.maxLen*D))
		g.vC[l] = mk(make([]float32, m.maxLen*D))
	}
	g.dinv, g.dgFinal, g.dwvocab = mk(m.inv), mk(m.gFinal), mk(m.wvocab)
	g.dx, g.xn = mk(make([]float32, D)), mk(make([]float32, D))
	g.q, g.k, g.v = mk(make([]float32, D)), mk(make([]float32, D)), mk(make([]float32, D))
	g.attn, g.ao, g.xn2 = mk(make([]float32, D)), mk(make([]float32, D)), mk(make([]float32, D))
	g.h, g.mo = mk(make([]float32, F)), mk(make([]float32, D))
	g.logits = mk(make([]float32, m.V))
	return g
}

func (g *gpuFullModel) release() {
	for _, b := range g.all {
		b.Release()
	}
}

func (g *gpuFullModel) stepOps(L int) []func(r *Recorder) error {
	m := g.m
	D, H, dk, F := m.D, m.H, m.dk, m.F
	var ops []func(r *Recorder) error
	for l := range m.nLayers {
		wq, wk, wv, wo := g.wq[l], g.wk[l], g.wv[l], g.wo[l]
		gA, gM, w1, w2 := g.gA[l], g.gM[l], g.w1[l], g.w2[l]
		kC, vC := g.kC[l], g.vC[l]
		ops = append(ops,
			func(r *Recorder) error { return r.RMSNorm(g.dx, gA, g.xn, 1, D, m.eps) },
			func(r *Recorder) error { return r.MatMul(g.xn, wq, g.q, 1, D, D) },
			func(r *Recorder) error { return r.MatMul(g.xn, wk, g.k, 1, D, D) },
			func(r *Recorder) error { return r.MatMul(g.xn, wv, g.v, 1, D, D) },
			func(r *Recorder) error { return r.RoPE(g.q, g.dinv, g.q, 1, D, H, dk, m.half, L, m.posDiv) },
			func(r *Recorder) error { return r.RoPE(g.k, g.dinv, g.k, 1, D, H, dk, m.half, L, m.posDiv) },
			func(r *Recorder) error { return r.Blit(g.k, 0, kC, L*D, D) },
			func(r *Recorder) error { return r.Blit(g.v, 0, vC, L*D, D) },
			func(r *Recorder) error { return r.MHA(g.q, kC, vC, g.attn, 1, L+1, D, H, H, dk, 1, 0, m.scale) },
			func(r *Recorder) error { return r.MatMul(g.attn, wo, g.ao, 1, D, D) },
			func(r *Recorder) error { return r.Binary(g.dx, g.ao, g.dx, 0) },
			func(r *Recorder) error { return r.RMSNorm(g.dx, gM, g.xn2, 1, D, m.eps) },
			func(r *Recorder) error { return r.MatMul(g.xn2, w1, g.h, 1, D, F) },
			func(r *Recorder) error { return r.Unary(g.h, g.h, unaryGELU) },
			func(r *Recorder) error { return r.MatMul(g.h, w2, g.mo, 1, F, D) },
			func(r *Recorder) error { return r.Binary(g.dx, g.mo, g.dx, 0) },
		)
	}
	ops = append(ops,
		func(r *Recorder) error { return r.RMSNorm(g.dx, g.dgFinal, g.xn, 1, D, m.eps) },
		func(r *Recorder) error { return r.MatMul(g.xn, g.dwvocab, g.logits, 1, D, m.V) },
	)
	return ops
}

func TestVulkanFullModelDecodeCorrectness(t *testing.T) {
	if !Available() {
		t.Skip("vulkan: no compute-capable device (§V4)")
	}
	const D, H, nLayers, V, maxLen, steps = 64, 4, 2, 48, 16, 4
	m := newFullModel(D, H, nLayers, V, maxLen)
	g := m.toGPU(t)
	defer g.release()

	kHost := make([][]float32, nLayers)
	vHost := make([][]float32, nLayers)

	for step := range steps {
		x := make([]float32, D)
		for i := range x {
			x[i] = float32((i+step*5)%11-5) * 0.07
		}
		if err := g.dx.UploadF32(x); err != nil {
			t.Fatal(err)
		}
		r, err := NewRecorder()
		if err != nil {
			t.Fatal(err)
		}
		for _, op := range g.stepOps(step) {
			if err := op(r); err != nil {
				t.Fatal(err)
			}
		}
		if err := r.Finish(); err != nil {
			t.Fatal(err)
		}
		r.Free()
		got := make([]float32, V)
		if err := g.logits.DownloadF32(got); err != nil {
			t.Fatal(err)
		}
		want := m.hostLogits(x, kHost, vHost, step)
		for i := range want {
			if math.Abs(float64(got[i]-want[i])) > 2e-3*math.Max(1, math.Abs(float64(want[i]))) {
				t.Fatalf("step %d logit[%d]: gpu %v vs host %v", step, i, got[i], want[i])
			}
		}
	}
	t.Logf("vulkan full %d-layer decoder, D=%d V=%d: Recorder logits match host float64 across %d steps", nLayers, D, V, steps)
}

// §T386: embed-via-blit closes the token→logits→token loop FULLY ON-DEVICE on Vulkan too — a step's
// input is the token's embedding row, just a blit from a resident embedding table [V,D]. Drives a
// real autoregressive loop (next token = argmax) and verifies GPU logits match the host reference.
func TestVulkanFullModelDecodeEmbedViaBlit(t *testing.T) {
	if !Available() {
		t.Skip("vulkan: no compute-capable device (§V4)")
	}
	const D, H, nLayers, V, maxLen, steps = 64, 4, 2, 40, 16, 5
	m := newFullModel(D, H, nLayers, V, maxLen)
	g := m.toGPU(t)
	defer g.release()

	embHost := make([]float32, V*D)
	for i := range embHost {
		embHost[i] = float32((i*3)%23-11) * 0.03
	}
	emb, err := NewDeviceBufferF32(embHost)
	if err != nil {
		t.Fatal(err)
	}
	defer emb.Release()

	kHost := make([][]float32, nLayers)
	vHost := make([][]float32, nLayers)
	tok := 7

	argmax := func(v []float32) int {
		best, bi := v[0], 0
		for i, x := range v {
			if x > best {
				best, bi = x, i
			}
		}
		return bi
	}

	for step := range steps {
		r, err := NewRecorder()
		if err != nil {
			t.Fatal(err)
		}
		if err := r.Blit(emb, tok*D, g.dx, 0, D); err != nil { // on-device embed
			t.Fatal(err)
		}
		for _, op := range g.stepOps(step) {
			if err := op(r); err != nil {
				t.Fatal(err)
			}
		}
		if err := r.Finish(); err != nil {
			t.Fatal(err)
		}
		r.Free()
		got := make([]float32, V)
		if err := g.logits.DownloadF32(got); err != nil {
			t.Fatal(err)
		}
		x := make([]float32, D)
		copy(x, embHost[tok*D:(tok+1)*D])
		want := m.hostLogits(x, kHost, vHost, step)
		for i := range want {
			if math.Abs(float64(got[i]-want[i])) > 2e-3*math.Max(1, math.Abs(float64(want[i]))) {
				t.Fatalf("step %d tok %d logit[%d]: gpu %v vs host %v", step, tok, i, got[i], want[i])
			}
		}
		tok = argmax(got)
	}
	t.Logf("vulkan autoregressive decode via on-device embed-blit, %d steps: GPU logits match host each step", steps)
}

func TestVulkanFullModelDecodeBench(t *testing.T) {
	if !Available() {
		t.Skip("vulkan: no compute-capable device (§V4)")
	}
	const D, H, nLayers, V, maxLen = 512, 8, 6, 1024, 64
	m := newFullModel(D, H, nLayers, V, maxLen)
	g := m.toGPU(t)
	defer g.release()

	x := make([]float32, D)
	for i := range x {
		x[i] = float32(i%13-6) * 0.05
	}
	if err := g.dx.UploadF32(x); err != nil {
		t.Fatal(err)
	}
	const L = 32
	ops := g.stepOps(L)
	batched := func() {
		r, err := NewRecorder()
		if err != nil {
			t.Fatal(err)
		}
		for _, op := range ops {
			if err := op(r); err != nil {
				t.Fatal(err)
			}
		}
		if err := r.Finish(); err != nil {
			t.Fatal(err)
		}
		r.Free()
	}
	perOp := func() {
		for _, op := range ops {
			r, err := NewRecorder()
			if err != nil {
				t.Fatal(err)
			}
			if err := op(r); err != nil {
				t.Fatal(err)
			}
			if err := r.Finish(); err != nil {
				t.Fatal(err)
			}
			r.Free()
		}
	}
	timeit := func(fn func()) float64 {
		fn()
		const N = 30
		s := time.Now()
		for range N {
			fn()
		}
		return float64(time.Since(s).Microseconds()) / N / 1000
	}
	po := timeit(perOp)
	ba := timeit(batched)
	t.Logf("vulkan full decode step D=%d H=%d %d-layer V=%d @L=%d (%d ops): per-op %.3f ms | batched %.3f ms | speedup %.2fx",
		D, H, nLayers, V, L, len(ops), po, ba, po/ba)
}
