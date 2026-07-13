//go:build vulkan && cgo

package vulkan

import (
	"math"
	"testing"
	"time"
)

// §T390 (ADR-0019 Phase 2, mirrors Metal §T389): the full LLAMA-style decode assembly on Vulkan —
// RMSNorm + RoPE + GROUPED-QUERY attention (kvHeads < heads) + SwiGLU FFN — completing both-backends
// parity for the real nlp.Llama architecture. Verified vs host float64, then measured batched vs per-op.

type llamaModel struct {
	D, H, KVH, dk, half, kvDim, hidden, V, nLayers, maxLen int
	eps, posDiv, scale                                     float32
	wq, wk, wv, wo                                         [][]float32
	gAttn, gFFN                                            [][]float32
	wGate, wUp, wDown                                      [][]float32
	gFinal, wvocab, inv                                    []float32
}

func newLlamaModel(D, H, KVH, hidden, V, nLayers, maxLen int) *llamaModel {
	dk := D / H
	kvDim := KVH * dk
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
	m := &llamaModel{
		D: D, H: H, KVH: KVH, dk: dk, half: dk / 2, kvDim: kvDim, hidden: hidden, V: V,
		nLayers: nLayers, maxLen: maxLen, eps: 1e-5, posDiv: 1.0,
		scale:  float32(1.0 / math.Sqrt(float64(dk))),
		gFinal: fillG(D, 99), wvocab: fill(D*V, 77), inv: make([]float32, dk/2),
	}
	for i := range m.inv {
		m.inv[i] = 1.0 / float32(math.Pow(10000, float64(2*i)/float64(dk)))
	}
	for l := range nLayers {
		m.wq = append(m.wq, fill(D*D, l*9+1))
		m.wk = append(m.wk, fill(D*kvDim, l*9+2))
		m.wv = append(m.wv, fill(D*kvDim, l*9+3))
		m.wo = append(m.wo, fill(D*D, l*9+4))
		m.gAttn = append(m.gAttn, fillG(D, l*2+1))
		m.gFFN = append(m.gFFN, fillG(D, l*2+2))
		m.wGate = append(m.wGate, fill(D*hidden, l*9+5))
		m.wUp = append(m.wUp, fill(D*hidden, l*9+6))
		m.wDown = append(m.wDown, fill(hidden*D, l*9+7))
	}
	return m
}

func rmsHost(x []float32, g []float32, eps float32) []float64 {
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

func matvec(in []float64, w []float32, out int) []float64 {
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

func (m *llamaModel) hostLogits(x []float32, kC, vC [][]float32, L int) []float32 {
	D, H, dk, kvDim, hidden := m.D, m.H, m.dk, m.kvDim, m.hidden
	rep := H / m.KVH
	cur := make([]float32, D)
	copy(cur, x)
	for l := range m.nLayers {
		xn := rmsHost(cur, m.gAttn[l], m.eps)
		q := matvec(xn, m.wq[l], D)
		k := matvec(xn, m.wk[l], kvDim)
		v := matvec(xn, m.wv[l], kvDim)
		ropeT := func(t []float64, heads int) {
			pos := float64(L) / float64(m.posDiv)
			for h := range heads {
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
		ropeT(q, H)
		ropeT(k, m.KVH)
		for j := range kvDim {
			kC[l] = append(kC[l], float32(k[j]))
			vC[l] = append(vC[l], float32(v[j]))
		}
		attn := make([]float64, D)
		for h := range H {
			qOff := h * dk
			kvOff := (h / rep) * dk
			sc := make([]float64, L+1)
			mx := math.Inf(-1)
			for j := 0; j <= L; j++ {
				var sd float64
				for d := range dk {
					sd += q[qOff+d] * float64(kC[l][j*kvDim+kvOff+d])
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
					acc += sc[j] * float64(vC[l][j*kvDim+kvOff+d])
				}
				attn[qOff+d] = acc / ls
			}
		}
		ao := matvec(attn, m.wo[l], D)
		for j := range D {
			cur[j] += float32(ao[j])
		}
		xn2 := rmsHost(cur, m.gFFN[l], m.eps)
		gate := matvec(xn2, m.wGate[l], hidden)
		up := matvec(xn2, m.wUp[l], hidden)
		hh := make([]float64, hidden)
		for j := range hidden {
			silu := gate[j] / (1.0 + math.Exp(-gate[j]))
			hh[j] = silu * up[j]
		}
		mo := matvec(hh, m.wDown[l], D)
		for j := range D {
			cur[j] += float32(mo[j])
		}
	}
	xf := rmsHost(cur, m.gFinal, m.eps)
	logits := matvec(xf, m.wvocab, m.V)
	out := make([]float32, m.V)
	for j := range out {
		out[j] = float32(logits[j])
	}
	return out
}

type gpuLlama struct {
	m                                                *llamaModel
	wq, wk, wv, wo, gA, gF, wG, wU, wD, kC, vC       []*DeviceBuffer
	dinv, dgFinal, dwvocab                           *DeviceBuffer
	dx, xn, q, k, v, attn, ao, xn2, gate, up, mo, lg *DeviceBuffer
	all                                              []*DeviceBuffer
}

func (m *llamaModel) toGPU(t *testing.T) *gpuLlama {
	g := &gpuLlama{m: m}
	mk := func(data []float32) *DeviceBuffer {
		b, err := NewDeviceBufferF32(data)
		if err != nil {
			t.Fatal(err)
		}
		g.all = append(g.all, b)
		return b
	}
	D, kvDim, hidden := m.D, m.kvDim, m.hidden
	for l := range m.nLayers {
		g.wq = append(g.wq, mk(m.wq[l]))
		g.wk = append(g.wk, mk(m.wk[l]))
		g.wv = append(g.wv, mk(m.wv[l]))
		g.wo = append(g.wo, mk(m.wo[l]))
		g.gA = append(g.gA, mk(m.gAttn[l]))
		g.gF = append(g.gF, mk(m.gFFN[l]))
		g.wG = append(g.wG, mk(m.wGate[l]))
		g.wU = append(g.wU, mk(m.wUp[l]))
		g.wD = append(g.wD, mk(m.wDown[l]))
		g.kC = append(g.kC, mk(make([]float32, m.maxLen*kvDim)))
		g.vC = append(g.vC, mk(make([]float32, m.maxLen*kvDim)))
	}
	g.dinv, g.dgFinal, g.dwvocab = mk(m.inv), mk(m.gFinal), mk(m.wvocab)
	g.dx, g.xn = mk(make([]float32, D)), mk(make([]float32, D))
	g.q = mk(make([]float32, D))
	g.k, g.v = mk(make([]float32, kvDim)), mk(make([]float32, kvDim))
	g.attn, g.ao, g.xn2 = mk(make([]float32, D)), mk(make([]float32, D)), mk(make([]float32, D))
	g.gate, g.up = mk(make([]float32, hidden)), mk(make([]float32, hidden))
	g.mo, g.lg = mk(make([]float32, D)), mk(make([]float32, m.V))
	return g
}

func (g *gpuLlama) release() {
	for _, b := range g.all {
		b.Release()
	}
}

func (g *gpuLlama) stepOps(L int) []func(r *Recorder) error {
	m := g.m
	D, H, KVH, dk, kvDim, hidden := m.D, m.H, m.KVH, m.dk, m.kvDim, m.hidden
	var ops []func(r *Recorder) error
	for l := range m.nLayers {
		wq, wk, wv, wo := g.wq[l], g.wk[l], g.wv[l], g.wo[l]
		gA, gF := g.gA[l], g.gF[l]
		wG, wU, wD, kC, vC := g.wG[l], g.wU[l], g.wD[l], g.kC[l], g.vC[l]
		ops = append(ops,
			func(r *Recorder) error { return r.RMSNorm(g.dx, gA, g.xn, 1, D, m.eps) },
			func(r *Recorder) error { return r.MatMul(g.xn, wq, g.q, 1, D, D) },
			func(r *Recorder) error { return r.MatMul(g.xn, wk, g.k, 1, D, kvDim) },
			func(r *Recorder) error { return r.MatMul(g.xn, wv, g.v, 1, D, kvDim) },
			func(r *Recorder) error { return r.RoPE(g.q, g.dinv, g.q, 1, D, H, dk, m.half, L, m.posDiv) },
			func(r *Recorder) error { return r.RoPE(g.k, g.dinv, g.k, 1, kvDim, KVH, dk, m.half, L, m.posDiv) },
			func(r *Recorder) error { return r.Blit(g.k, 0, kC, L*kvDim, kvDim) },
			func(r *Recorder) error { return r.Blit(g.v, 0, vC, L*kvDim, kvDim) },
			func(r *Recorder) error { return r.MHA(g.q, kC, vC, g.attn, 1, L+1, D, H, KVH, dk, 1, 0, m.scale) },
			func(r *Recorder) error { return r.MatMul(g.attn, wo, g.ao, 1, D, D) },
			func(r *Recorder) error { return r.Binary(g.dx, g.ao, g.dx, 0) },
			func(r *Recorder) error { return r.RMSNorm(g.dx, gF, g.xn2, 1, D, m.eps) },
			func(r *Recorder) error { return r.MatMul(g.xn2, wG, g.gate, 1, D, hidden) },
			func(r *Recorder) error { return r.Unary(g.gate, g.gate, unarySiLU) },
			func(r *Recorder) error { return r.MatMul(g.xn2, wU, g.up, 1, D, hidden) },
			func(r *Recorder) error { return r.Binary(g.gate, g.up, g.gate, 2) },
			func(r *Recorder) error { return r.MatMul(g.gate, wD, g.mo, 1, hidden, D) },
			func(r *Recorder) error { return r.Binary(g.dx, g.mo, g.dx, 0) },
		)
	}
	ops = append(ops,
		func(r *Recorder) error { return r.RMSNorm(g.dx, g.dgFinal, g.xn, 1, D, m.eps) },
		func(r *Recorder) error { return r.MatMul(g.xn, g.dwvocab, g.lg, 1, D, m.V) },
	)
	return ops
}

func TestVulkanLlamaDecodeCorrectness(t *testing.T) {
	if !Available() {
		t.Skip("vulkan: no compute-capable device (§V4)")
	}
	const D, H, KVH, hidden, V, nLayers, maxLen, steps = 64, 8, 2, 176, 48, 2, 16, 4
	m := newLlamaModel(D, H, KVH, hidden, V, nLayers, maxLen)
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
		if err := g.lg.DownloadF32(got); err != nil {
			t.Fatal(err)
		}
		want := m.hostLogits(x, kHost, vHost, step)
		for i := range want {
			if math.Abs(float64(got[i]-want[i])) > 2e-3*math.Max(1, math.Abs(float64(want[i]))) {
				t.Fatalf("step %d logit[%d]: gpu %v vs host %v", step, i, got[i], want[i])
			}
		}
	}
	t.Logf("vulkan Llama-style decoder (GQA %d:%d, SwiGLU), %d-layer D=%d: Recorder logits match host across %d steps", H, KVH, nLayers, D, steps)
}

func TestVulkanLlamaDecodeBench(t *testing.T) {
	if !Available() {
		t.Skip("vulkan: no compute-capable device (§V4)")
	}
	const D, H, KVH, hidden, V, nLayers, maxLen = 512, 8, 2, 1376, 1024, 6, 64
	m := newLlamaModel(D, H, KVH, hidden, V, nLayers, maxLen)
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
	run := func(perOp bool) {
		if perOp {
			for _, op := range ops {
				r, _ := NewRecorder()
				_ = op(r)
				_ = r.Finish()
				r.Free()
			}
			return
		}
		r, _ := NewRecorder()
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
	timeit := func(perOp bool) float64 {
		run(perOp)
		const N = 30
		s := time.Now()
		for range N {
			run(perOp)
		}
		return float64(time.Since(s).Microseconds()) / N / 1000
	}
	po := timeit(true)
	ba := timeit(false)
	t.Logf("vulkan Llama decode step D=%d GQA%d:%d %d-layer V=%d @L=%d (%d ops): per-op %.3f ms | batched %.3f ms | speedup %.2fx",
		D, H, KVH, nLayers, V, L, len(ops), po, ba, po/ba)
}
