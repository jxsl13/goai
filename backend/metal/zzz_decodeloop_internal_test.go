//go:build darwin && cgo

package metal

import (
	"math"
	"testing"
	"time"
)

// §T378 (ADR-0019 Phase 2 INTEGRATION): assemble a full attention-block decode step over a
// PERSISTENT device-resident KV cache and run it for several steps in the Recorder — the novel
// cross-step state management (cache grows each step, RoPE posOffset advances, decode-attn sees
// the growing cache, k/v appended via blit) that the whole batched-decode path hinges on. The MLP
// block (matmul→gelu→matmul) is already covered by §T374 + the cross-validated kernels, so this
// isolates the genuinely-new integration risk. Verified against a host float64 reference.
//
// Per step, given token hidden x[1,D] and cache length L:
//   xn=rmsnorm(x,gAttn); q=xn·Wq; k=xn·Wk; v=xn·Wv; q=rope(q,L); k=rope(k,L);
//   blit k→kCache[L]; blit v→vCache[L]; attn=mha(q,kCache,vCache,sq=1,sk=L+1,causal);
//   ao=attn·Wo; xout=x+ao   → return xout, L++.

type decodeModel struct {
	D, H, dk, half, maxLen int
	eps, posDiv            float32
	scale                  float32
	wq, wk, wv, wo         []float32 // [D,D] each
	gAttn                  []float32 // [D]
	inv                    []float32 // [half]
}

func newDecodeModel(D, H, maxLen int) *decodeModel {
	dk := D / H
	m := &decodeModel{
		D: D, H: H, dk: dk, half: dk / 2, maxLen: maxLen,
		eps: 1e-5, posDiv: 1.0,
		scale: float32(1.0 / math.Sqrt(float64(dk))),
		wq:    make([]float32, D*D), wk: make([]float32, D*D),
		wv: make([]float32, D*D), wo: make([]float32, D*D),
		gAttn: make([]float32, D), inv: make([]float32, dk/2),
	}
	fill := func(s []float32, seed int) {
		for i := range s {
			s[i] = float32((i+seed)%17-8) * 0.03
		}
	}
	fill(m.wq, 1)
	fill(m.wk, 2)
	fill(m.wv, 3)
	fill(m.wo, 4)
	for i := range m.gAttn {
		m.gAttn[i] = 0.9 + float32(i%5)*0.02
	}
	for i := range m.inv {
		m.inv[i] = 1.0 / float32(math.Pow(10000, float64(2*i)/float64(dk)))
	}
	return m
}

// hostDecodeStep runs one attention-block step on the host (float64) with growing host caches.
func (m *decodeModel) hostDecodeStep(x []float32, kCache, vCache *[]float32, L int) []float32 {
	D, H, dk := m.D, m.H, m.dk
	// rmsnorm
	var ss float64
	for _, xv := range x {
		ss += float64(xv) * float64(xv)
	}
	inv := 1.0 / math.Sqrt(ss/float64(D)+float64(m.eps))
	xn := make([]float64, D)
	for i := range xn {
		xn[i] = float64(x[i]) * inv * float64(m.gAttn[i])
	}
	matvec := func(w []float32) []float64 { // [1,D]·[D,D]
		o := make([]float64, D)
		for j := range D {
			var acc float64
			for p := range D {
				acc += xn[p] * float64(w[p*D+j])
			}
			o[j] = acc
		}
		return o
	}
	q, k, v := matvec(m.wq), matvec(m.wk), matvec(m.wv)
	// rope on q,k (posOffset=L)
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
	// append k,v to caches
	for j := range D {
		*kCache = append(*kCache, float32(k[j]))
		*vCache = append(*vCache, float32(v[j]))
	}
	// decode attention: q[1,D] vs caches[0..L] (L+1 keys), per head
	attn := make([]float64, D)
	for h := range H {
		off := h * dk
		scores := make([]float64, L+1)
		mx := math.Inf(-1)
		for j := 0; j <= L; j++ {
			var sdot float64
			for d := range dk {
				sdot += q[off+d] * float64((*kCache)[j*D+off+d])
			}
			sdot *= float64(m.scale)
			scores[j] = sdot
			if sdot > mx {
				mx = sdot
			}
		}
		var l float64
		for j := 0; j <= L; j++ {
			scores[j] = math.Exp(scores[j] - mx)
			l += scores[j]
		}
		for d := range dk {
			var acc float64
			for j := 0; j <= L; j++ {
				acc += scores[j] * float64((*vCache)[j*D+off+d])
			}
			attn[off+d] = acc / l
		}
	}
	// ao = attn·Wo, xout = x + ao
	xout := make([]float32, D)
	for j := range D {
		var acc float64
		for p := range D {
			acc += attn[p] * float64(m.wo[p*D+j])
		}
		xout[j] = x[j] + float32(acc)
	}
	return xout
}

func TestDecodeLoopKVCache(t *testing.T) {
	if !Available() {
		t.Skip("metal: no gpu device — skipped")
	}
	const D, H, maxLen, steps = 32, 4, 16, 5
	m := newDecodeModel(D, H, maxLen)
	dk := m.dk

	// device weights (constant across steps)
	dwq, _ := NewDeviceBufferF32(m.wq)
	dwk, _ := NewDeviceBufferF32(m.wk)
	dwv, _ := NewDeviceBufferF32(m.wv)
	dwo, _ := NewDeviceBufferF32(m.wo)
	dg, _ := NewDeviceBufferF32(m.gAttn)
	dinv, _ := NewDeviceBufferF32(m.inv)
	// persistent device KV caches (sized to max; only [0..L] rows valid)
	kCache, _ := NewDeviceBufferF32(make([]float32, maxLen*D))
	vCache, _ := NewDeviceBufferF32(make([]float32, maxLen*D))
	// per-step scratch
	dx, _ := NewDeviceBufferF32(make([]float32, D))
	xn, _ := NewDeviceBufferF32(make([]float32, D))
	q, _ := NewDeviceBufferF32(make([]float32, D))
	k, _ := NewDeviceBufferF32(make([]float32, D))
	v, _ := NewDeviceBufferF32(make([]float32, D))
	attn, _ := NewDeviceBufferF32(make([]float32, D))
	ao, _ := NewDeviceBufferF32(make([]float32, D))
	xout, _ := NewDeviceBufferF32(make([]float32, D))
	bufs := []*DeviceBuffer{dwq, dwk, dwv, dwo, dg, dinv, kCache, vCache, dx, xn, q, k, v, attn, ao, xout}
	defer func() {
		for _, b := range bufs {
			b.Release()
		}
	}()

	// stepOps returns the ordered record-mode ops for one attention-block decode step at cache
	// length L (11 ops: norm→q/k/v→rope q/k→append k/v→decode-attn→o-proj→resid).
	stepOps := func(L int) []func(r *Recorder) error {
		return []func(r *Recorder) error{
			func(r *Recorder) error { return r.RMSNorm(dx, dg, xn, 1, D, m.eps) },
			func(r *Recorder) error { return r.MatMul(xn, dwq, q, 1, D, D) },
			func(r *Recorder) error { return r.MatMul(xn, dwk, k, 1, D, D) },
			func(r *Recorder) error { return r.MatMul(xn, dwv, v, 1, D, D) },
			func(r *Recorder) error { return r.RoPE(q, dinv, q, 1, D, H, dk, m.half, L, m.posDiv) },
			func(r *Recorder) error { return r.RoPE(k, dinv, k, 1, D, H, dk, m.half, L, m.posDiv) },
			func(r *Recorder) error { return r.Blit(k, 0, kCache, L*D, D) },
			func(r *Recorder) error { return r.Blit(v, 0, vCache, L*D, D) },
			func(r *Recorder) error { return r.MHA(q, kCache, vCache, attn, 1, L+1, D, H, H, dk, 1, 0, m.scale) },
			func(r *Recorder) error { return r.MatMul(attn, dwo, ao, 1, D, D) },
			func(r *Recorder) error { return r.Binary(dx, ao, xout, 0) },
		}
	}
	// batched: one Recorder for the whole step.
	recordStep := func(_ *Recorder, L int) error {
		r, err := NewRecorder()
		if err != nil {
			return err
		}
		for _, op := range stepOps(L) {
			if err := op(r); err != nil {
				return err
			}
		}
		if err := r.Finish(); err != nil {
			return err
		}
		r.Free()
		return nil
	}

	// host reference caches
	hostK := make([]float32, 0, maxLen*D)
	hostV := make([]float32, 0, maxLen*D)

	for step := range steps {
		// new token hidden
		x := make([]float32, D)
		for i := range x {
			x[i] = float32((i+step*3)%13-6) * 0.08
		}
		if err := dx.UploadF32(x); err != nil {
			t.Fatal(err)
		}
		// GPU batched step (recordStep manages its own Recorder)
		if err := recordStep(nil, step); err != nil {
			t.Fatal(err)
		}
		gotOut := make([]float32, D)
		if err := xout.DownloadF32(gotOut); err != nil {
			t.Fatal(err)
		}
		// host reference (also grows hostK/hostV)
		wantOut := m.hostDecodeStep(x, &hostK, &hostV, step)
		for i := range wantOut {
			if math.Abs(float64(gotOut[i]-wantOut[i])) > 1e-3*math.Max(1, math.Abs(float64(wantOut[i]))) {
				t.Fatalf("step %d out[%d]: gpu %v vs host %v", step, i, gotOut[i], wantOut[i])
			}
		}
		// each token is fresh input; the KV cache keeps growing, which is what's being tested.
	}
	t.Logf("decode loop over %d steps, D=%d H=%d: GPU Recorder matches host float64 across a growing KV cache", steps, D, H)

	// A/B (§T374 pattern): the assembled step, batched (1 cmd-buffer) vs per-op (11 cmd-buffers),
	// at a fixed cache length — the honest per-step decode headroom the integration recovers.
	L := steps
	batched := func() {
		r, err := NewRecorder()
		if err != nil {
			t.Fatal(err)
		}
		for _, op := range stepOps(L) {
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
		for _, op := range stepOps(L) {
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
		const N = 100
		s := time.Now()
		for range N {
			fn()
		}
		return float64(time.Since(s).Microseconds()) / N / 1000
	}
	po := timeit(perOp)
	ba := timeit(batched)
	t.Logf("assembled attn-block step (11 ops) @L=%d: per-op %.3f ms | batched %.3f ms | speedup %.2fx",
		L, po, ba, po/ba)
}
