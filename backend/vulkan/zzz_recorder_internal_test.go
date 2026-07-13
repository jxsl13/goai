//go:build vulkan && cgo

package vulkan

import (
	"math"
	"testing"
)

// §T382 (ADR-0019 Phase 2 Vulkan port foundation, mirrors Metal §T370): a chain of record-mode ops
// over device buffers in ONE Vulkan command buffer produces the same result as running them
// separately — proving the Recorder primitive (device-buffer chaining, one submit/waitIdle,
// explicit barriers standing in for Metal's automatic hazard tracking).
func TestVulkanRecorderUnaryChain(t *testing.T) {
	if !Available() {
		t.Skip("vulkan: no compute-capable device (§V4)")
	}
	const n = 4096
	src := make([]float32, n)
	for i := range src {
		src[i] = float32((i%200)-100) * 0.05
	}
	a, err := NewDeviceBufferF32(src)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Release()
	b, _ := NewDeviceBufferF32(make([]float32, n))
	defer b.Release()
	c, _ := NewDeviceBufferF32(make([]float32, n))
	defer c.Release()

	rec, err := NewRecorder()
	if err != nil {
		t.Fatal(err)
	}
	// chain: b = relu(a); c = exp(b); b = neg(c)  → final in b
	if err := rec.Unary(a, b, unaryReLU); err != nil {
		t.Fatal(err)
	}
	if err := rec.Unary(b, c, unaryExp); err != nil {
		t.Fatal(err)
	}
	if err := rec.Unary(c, b, unaryNeg); err != nil {
		t.Fatal(err)
	}
	if err := rec.Finish(); err != nil {
		t.Fatal(err)
	}
	rec.Free()

	got := make([]float32, n)
	if err := b.DownloadF32(got); err != nil {
		t.Fatal(err)
	}
	for i := range src {
		want := -math.Exp(math.Max(float64(src[i]), 0))
		if math.Abs(float64(got[i])-want) > 1e-4*math.Max(1, math.Abs(want)) {
			t.Fatalf("[%d]: chained %v vs want %v", i, got[i], want)
		}
	}
}

// §T383: record-mode matmul + residual add (binary) over device buffers — d = (A·B) + C — matches
// the host reference. Covers the Vulkan Recorder's matmul and binary ops chained with a barrier.
func TestVulkanRecorderMatMulAdd(t *testing.T) {
	if !Available() {
		t.Skip("vulkan: no compute-capable device (§V4)")
	}
	const m, k, n = 12, 20, 8
	aSrc := make([]float32, m*k)
	bSrc := make([]float32, k*n)
	cSrc := make([]float32, m*n) // residual
	for i := range aSrc {
		aSrc[i] = float32((i%13)-6) * 0.1
	}
	for i := range bSrc {
		bSrc[i] = float32((i%11)-5) * 0.07
	}
	for i := range cSrc {
		cSrc[i] = float32((i%7)-3) * 0.2
	}
	a, err := NewDeviceBufferF32(aSrc)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Release()
	b, _ := NewDeviceBufferF32(bSrc)
	defer b.Release()
	cbuf, _ := NewDeviceBufferF32(cSrc)
	defer cbuf.Release()
	prod, _ := NewDeviceBufferF32(make([]float32, m*n)) // A·B
	defer prod.Release()
	d, _ := NewDeviceBufferF32(make([]float32, m*n)) // (A·B)+C
	defer d.Release()

	rec, err := NewRecorder()
	if err != nil {
		t.Fatal(err)
	}
	if err := rec.MatMul(a, b, prod, m, k, n); err != nil {
		t.Fatal(err)
	}
	if err := rec.Binary(prod, cbuf, d, 0 /*add*/); err != nil {
		t.Fatal(err)
	}
	if err := rec.Finish(); err != nil {
		t.Fatal(err)
	}
	rec.Free()

	got := make([]float32, m*n)
	if err := d.DownloadF32(got); err != nil {
		t.Fatal(err)
	}
	for i := range m {
		for j := range n {
			var acc float64
			for p := range k {
				acc += float64(aSrc[i*k+p]) * float64(bSrc[p*n+j])
			}
			want := acc + float64(cSrc[i*n+j])
			if math.Abs(float64(got[i*n+j])-want) > 1e-4*math.Max(1, math.Abs(want)) {
				t.Fatalf("[%d,%d]: %v vs want %v", i, j, got[i*n+j], want)
			}
		}
	}
}

// §T383: record-mode matmul + rmsnorm over device buffers — e = rmsnorm(A·B)·gamma — the decode
// block shape (project→norm), matching the host inv=rsqrt(mean(x²)+eps) reference.
func TestVulkanRecorderRMSNormChain(t *testing.T) {
	if !Available() {
		t.Skip("vulkan: no compute-capable device (§V4)")
	}
	const m, k, n = 8, 16, 24
	const eps = 1e-5
	aSrc := make([]float32, m*k)
	bSrc := make([]float32, k*n)
	gSrc := make([]float32, n)
	for i := range aSrc {
		aSrc[i] = float32((i%13)-6) * 0.1
	}
	for i := range bSrc {
		bSrc[i] = float32((i%11)-5) * 0.07
	}
	for i := range gSrc {
		gSrc[i] = 0.5 + float32(i%5)*0.1
	}
	a, err := NewDeviceBufferF32(aSrc)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Release()
	b, _ := NewDeviceBufferF32(bSrc)
	defer b.Release()
	g, _ := NewDeviceBufferF32(gSrc)
	defer g.Release()
	c, _ := NewDeviceBufferF32(make([]float32, m*n))
	defer c.Release()
	e, _ := NewDeviceBufferF32(make([]float32, m*n))
	defer e.Release()

	rec, err := NewRecorder()
	if err != nil {
		t.Fatal(err)
	}
	if err := rec.MatMul(a, b, c, m, k, n); err != nil {
		t.Fatal(err)
	}
	if err := rec.RMSNorm(c, g, e, m, n, eps); err != nil {
		t.Fatal(err)
	}
	if err := rec.Finish(); err != nil {
		t.Fatal(err)
	}
	rec.Free()

	got := make([]float32, m*n)
	if err := e.DownloadF32(got); err != nil {
		t.Fatal(err)
	}
	for i := range m {
		row := make([]float64, n)
		var ss float64
		for j := range n {
			var acc float64
			for p := range k {
				acc += float64(aSrc[i*k+p]) * float64(bSrc[p*n+j])
			}
			row[j] = acc
			ss += acc * acc
		}
		inv := 1.0 / math.Sqrt(ss/float64(n)+eps)
		for j := range n {
			want := row[j] * inv * float64(gSrc[j])
			if math.Abs(float64(got[i*n+j])-want) > 1e-4*math.Max(1, math.Abs(want)) {
				t.Fatalf("[%d,%d]: %v vs want %v", i, j, got[i*n+j], want)
			}
		}
	}
}

// §T383: record-mode RoPE over device buffers matches the host rotate-half reference (decode
// shape: seq=1 at absolute position posOffset = KV-cache length).
func TestVulkanRecorderRoPE(t *testing.T) {
	if !Available() {
		t.Skip("vulkan: no compute-capable device (§V4)")
	}
	const seq, heads, hd = 1, 2, 8
	const half = hd / 2
	const width = heads * hd
	const posOffset = 5
	const posDiv = float32(1)
	qSrc := make([]float32, seq*width)
	invSrc := make([]float32, half)
	for i := range qSrc {
		qSrc[i] = float32((i%9)-4) * 0.1
	}
	for i := range invSrc {
		invSrc[i] = 1.0 / float32(math.Pow(10000, float64(2*i)/float64(hd)))
	}
	q, err := NewDeviceBufferF32(qSrc)
	if err != nil {
		t.Fatal(err)
	}
	defer q.Release()
	inv, _ := NewDeviceBufferF32(invSrc)
	defer inv.Release()
	o, _ := NewDeviceBufferF32(make([]float32, seq*width))
	defer o.Release()

	rec, err := NewRecorder()
	if err != nil {
		t.Fatal(err)
	}
	if err := rec.RoPE(q, inv, o, seq, width, heads, hd, half, posOffset, posDiv); err != nil {
		t.Fatal(err)
	}
	if err := rec.Finish(); err != nil {
		t.Fatal(err)
	}
	rec.Free()

	got := make([]float32, seq*width)
	if err := o.DownloadF32(got); err != nil {
		t.Fatal(err)
	}
	for p := range seq {
		pos := float64(p+posOffset) / float64(posDiv)
		for h := range heads {
			for i := range half {
				base := p*width + h*hd + i
				theta := float64(invSrc[i])
				c, s := math.Cos(pos*theta), math.Sin(pos*theta)
				qi, qih := float64(qSrc[base]), float64(qSrc[base+half])
				w0 := qi*c - qih*s
				w1 := qih*c + qi*s
				if math.Abs(float64(got[base])-w0) > 1e-4*math.Max(1, math.Abs(w0)) {
					t.Fatalf("p=%d h=%d i=%d lo: %v vs %v", p, h, i, got[base], w0)
				}
				if math.Abs(float64(got[base+half])-w1) > 1e-4*math.Max(1, math.Abs(w1)) {
					t.Fatalf("p=%d h=%d i=%d hi: %v vs %v", p, h, i, got[base+half], w1)
				}
			}
		}
	}
}

// §T388: the Llama SwiGLU FFN — SiLU(x·Wgate) ⊙ (x·Wup) · Wdown — assembled from existing Recorder
// ops (matmul + unary SiLU + binary mul + matmul), cross-validated against a host reference. The
// FFN style the real nlp Llama uses; with RMSNorm + RoPE + decode-attn the Recorder covers a Llama block.
func TestVulkanRecorderSwiGLU(t *testing.T) {
	if !Available() {
		t.Skip("vulkan: no compute-capable device (§V4)")
	}
	const dim, hidden = 32, 88
	xSrc := make([]float32, dim)
	wgSrc := make([]float32, dim*hidden)
	wuSrc := make([]float32, dim*hidden)
	wdSrc := make([]float32, hidden*dim)
	for i := range xSrc {
		xSrc[i] = float32((i%9)-4) * 0.1
	}
	for i := range wgSrc {
		wgSrc[i] = float32((i%13)-6) * 0.03
		wuSrc[i] = float32((i%11)-5) * 0.04
	}
	for i := range wdSrc {
		wdSrc[i] = float32((i%7)-3) * 0.05
	}
	x, err := NewDeviceBufferF32(xSrc)
	if err != nil {
		t.Fatal(err)
	}
	defer x.Release()
	wg, _ := NewDeviceBufferF32(wgSrc)
	defer wg.Release()
	wu, _ := NewDeviceBufferF32(wuSrc)
	defer wu.Release()
	wd, _ := NewDeviceBufferF32(wdSrc)
	defer wd.Release()
	gate, _ := NewDeviceBufferF32(make([]float32, hidden))
	defer gate.Release()
	up, _ := NewDeviceBufferF32(make([]float32, hidden))
	defer up.Release()
	out, _ := NewDeviceBufferF32(make([]float32, dim))
	defer out.Release()

	rec, err := NewRecorder()
	if err != nil {
		t.Fatal(err)
	}
	if err := rec.MatMul(x, wg, gate, 1, dim, hidden); err != nil {
		t.Fatal(err)
	}
	if err := rec.Unary(gate, gate, unarySiLU); err != nil {
		t.Fatal(err)
	}
	if err := rec.MatMul(x, wu, up, 1, dim, hidden); err != nil {
		t.Fatal(err)
	}
	if err := rec.Binary(gate, up, gate, 2); err != nil {
		t.Fatal(err)
	}
	if err := rec.MatMul(gate, wd, out, 1, hidden, dim); err != nil {
		t.Fatal(err)
	}
	if err := rec.Finish(); err != nil {
		t.Fatal(err)
	}
	rec.Free()

	got := make([]float32, dim)
	if err := out.DownloadF32(got); err != nil {
		t.Fatal(err)
	}
	h := make([]float64, hidden)
	for j := range hidden {
		var g, u float64
		for p := range dim {
			g += float64(xSrc[p]) * float64(wgSrc[p*hidden+j])
			u += float64(xSrc[p]) * float64(wuSrc[p*hidden+j])
		}
		silu := g / (1.0 + math.Exp(-g))
		h[j] = silu * u
	}
	for j := range dim {
		var acc float64
		for p := range hidden {
			acc += h[p] * float64(wdSrc[p*dim+j])
		}
		if math.Abs(float64(got[j])-acc) > 1e-3*math.Max(1, math.Abs(acc)) {
			t.Fatalf("[%d]: %v vs want %v", j, got[j], acc)
		}
	}
}

// §T387: record-mode LayerNorm chains after a matmul over device buffers — e = layernorm(A·B)·γ+β
// — the GPT-2-style block boundary (rmsnorm is the Llama variant).
func TestVulkanRecorderLayerNormChain(t *testing.T) {
	if !Available() {
		t.Skip("vulkan: no compute-capable device (§V4)")
	}
	const m, k, n = 8, 16, 24
	const eps = 1e-5
	aSrc := make([]float32, m*k)
	bSrc := make([]float32, k*n)
	gSrc := make([]float32, n)
	betaSrc := make([]float32, n)
	for i := range aSrc {
		aSrc[i] = float32((i%13)-6) * 0.1
	}
	for i := range bSrc {
		bSrc[i] = float32((i%11)-5) * 0.07
	}
	for i := range gSrc {
		gSrc[i] = 0.5 + float32(i%5)*0.1
		betaSrc[i] = float32((i%3)-1) * 0.2
	}
	a, err := NewDeviceBufferF32(aSrc)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Release()
	b, _ := NewDeviceBufferF32(bSrc)
	defer b.Release()
	g, _ := NewDeviceBufferF32(gSrc)
	defer g.Release()
	beta, _ := NewDeviceBufferF32(betaSrc)
	defer beta.Release()
	c, _ := NewDeviceBufferF32(make([]float32, m*n))
	defer c.Release()
	e, _ := NewDeviceBufferF32(make([]float32, m*n))
	defer e.Release()

	rec, err := NewRecorder()
	if err != nil {
		t.Fatal(err)
	}
	if err := rec.MatMul(a, b, c, m, k, n); err != nil {
		t.Fatal(err)
	}
	if err := rec.LayerNorm(c, g, beta, e, m, n, eps); err != nil {
		t.Fatal(err)
	}
	if err := rec.Finish(); err != nil {
		t.Fatal(err)
	}
	rec.Free()

	got := make([]float32, m*n)
	if err := e.DownloadF32(got); err != nil {
		t.Fatal(err)
	}
	for i := range m {
		row := make([]float64, n)
		var mean float64
		for j := range n {
			var acc float64
			for p := range k {
				acc += float64(aSrc[i*k+p]) * float64(bSrc[p*n+j])
			}
			row[j] = acc
			mean += acc
		}
		mean /= float64(n)
		var variance float64
		for j := range n {
			d := row[j] - mean
			variance += d * d
		}
		variance /= float64(n)
		inv := 1.0 / math.Sqrt(variance+eps)
		for j := range n {
			want := (row[j]-mean)*inv*float64(gSrc[j]) + float64(betaSrc[j])
			if math.Abs(float64(got[i*n+j])-want) > 1e-3*math.Max(1, math.Abs(want)) {
				t.Fatalf("[%d,%d]: %v vs want %v", i, j, got[i*n+j], want)
			}
		}
	}
}

// §T384: record-mode MHA with separate query/key lengths — the decode-against-KV-cache shape
// (1 query vs L cached keys) — matches a host softmax reference. The Vulkan mha shader separates
// sq/sk like the Metal gMHA kernel.
func TestVulkanRecorderDecodeAttn(t *testing.T) {
	if !Available() {
		t.Skip("vulkan: no compute-capable device (§V4)")
	}
	const sq, kvLen, heads, dk = 1, 6, 2, 8
	const dm = heads * dk
	const kvHeads = heads
	scale := float32(1.0 / math.Sqrt(float64(dk)))
	qSrc := make([]float32, sq*dm)
	kSrc := make([]float32, kvLen*kvHeads*dk)
	vSrc := make([]float32, kvLen*kvHeads*dk)
	for i := range qSrc {
		qSrc[i] = float32((i%17)-8) * 0.05
	}
	for i := range kSrc {
		kSrc[i] = float32((i%13)-6) * 0.06
		vSrc[i] = float32((i%11)-5) * 0.07
	}
	q, err := NewDeviceBufferF32(qSrc)
	if err != nil {
		t.Fatal(err)
	}
	defer q.Release()
	k, _ := NewDeviceBufferF32(kSrc)
	defer k.Release()
	v, _ := NewDeviceBufferF32(vSrc)
	defer v.Release()
	o, _ := NewDeviceBufferF32(make([]float32, sq*dm))
	defer o.Release()

	rec, err := NewRecorder()
	if err != nil {
		t.Fatal(err)
	}
	if err := rec.MHA(q, k, v, o, sq, kvLen, dm, heads, kvHeads, dk, 1, 0, scale); err != nil {
		t.Fatal(err)
	}
	if err := rec.Finish(); err != nil {
		t.Fatal(err)
	}
	rec.Free()

	got := make([]float32, sq*dm)
	if err := o.DownloadF32(got); err != nil {
		t.Fatal(err)
	}
	dkv := kvHeads * dk
	for h := range heads {
		qOff, kvOff := h*dk, h*dk
		scores := make([]float64, kvLen)
		mx := math.Inf(-1)
		for j := range kvLen {
			var s float64
			for d := range dk {
				s += float64(qSrc[qOff+d]) * float64(kSrc[j*dkv+kvOff+d])
			}
			s *= float64(scale)
			scores[j] = s
			if s > mx {
				mx = s
			}
		}
		var l float64
		for j := range kvLen {
			scores[j] = math.Exp(scores[j] - mx)
			l += scores[j]
		}
		for d := range dk {
			var acc float64
			for j := range kvLen {
				acc += scores[j] * float64(vSrc[j*dkv+kvOff+d])
			}
			want := acc / l
			g := float64(got[qOff+d])
			if math.IsNaN(g) || math.Abs(g-want) > 1e-4*math.Max(1, math.Abs(want)) {
				t.Fatalf("h=%d d=%d: %v vs want %v", h, d, g, want)
			}
		}
	}
}

// §T384: the KV-cache APPEND pattern end-to-end in the Vulkan Recorder — compute k=xn·Wk (matmul),
// then blit that [1,dkv] row into cache[cacheLen] — over device buffers in one command buffer.
// The appended row must equal the host matmul and other cache rows stay untouched.
func TestVulkanRecorderKVAppend(t *testing.T) {
	if !Available() {
		t.Skip("vulkan: no compute-capable device (§V4)")
	}
	const D, dkv, maxLen = 16, 12, 8
	const cacheLen = 3
	xnSrc := make([]float32, D)
	wkSrc := make([]float32, D*dkv)
	for i := range xnSrc {
		xnSrc[i] = float32((i%7)-3) * 0.1
	}
	for i := range wkSrc {
		wkSrc[i] = float32((i%9)-4) * 0.05
	}
	cacheSrc := make([]float32, maxLen*dkv)
	for i := range cacheSrc {
		cacheSrc[i] = -99.0
	}
	xn, err := NewDeviceBufferF32(xnSrc)
	if err != nil {
		t.Fatal(err)
	}
	defer xn.Release()
	wk, _ := NewDeviceBufferF32(wkSrc)
	defer wk.Release()
	kNew, _ := NewDeviceBufferF32(make([]float32, dkv))
	defer kNew.Release()
	cache, _ := NewDeviceBufferF32(cacheSrc)
	defer cache.Release()

	rec, err := NewRecorder()
	if err != nil {
		t.Fatal(err)
	}
	if err := rec.MatMul(xn, wk, kNew, 1, D, dkv); err != nil {
		t.Fatal(err)
	}
	if err := rec.Blit(kNew, 0, cache, cacheLen*dkv, dkv); err != nil {
		t.Fatal(err)
	}
	if err := rec.Finish(); err != nil {
		t.Fatal(err)
	}
	rec.Free()

	got := make([]float32, maxLen*dkv)
	if err := cache.DownloadF32(got); err != nil {
		t.Fatal(err)
	}
	for j := range dkv {
		var acc float64
		for p := range D {
			acc += float64(xnSrc[p]) * float64(wkSrc[p*dkv+j])
		}
		g := float64(got[cacheLen*dkv+j])
		if math.Abs(g-acc) > 1e-4*math.Max(1, math.Abs(acc)) {
			t.Fatalf("appended k[%d]: %v vs want %v", j, g, acc)
		}
	}
	for row := range maxLen {
		if row == cacheLen {
			continue
		}
		for j := range dkv {
			if got[row*dkv+j] != -99.0 {
				t.Fatalf("row %d col %d clobbered: %v", row, j, got[row*dkv+j])
			}
		}
	}
}
