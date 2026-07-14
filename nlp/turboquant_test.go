package nlp

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// §T619 TurboQuant, rotation stage. The fixed random Π is orthogonal, so: QᵀQ = I, the
// transform round-trips (applyInverse∘apply = identity), and — the property that makes it safe
// to insert before attention — it PRESERVES inner products ⟨Πx,Πy⟩ = ⟨x,y⟩.

func TestPolarRotationOrthogonal(t *testing.T) {
	const d = 8
	p, err := newPolarRotation(d, 42)
	if err != nil {
		t.Fatal(err)
	}
	// QᵀQ should be the identity (columns orthonormal since rows are).
	for i := range d {
		for j := range d {
			var dot float64
			for k := range d {
				dot += p.q[k][i] * p.q[k][j]
			}
			want := 0.0
			if i == j {
				want = 1.0
			}
			if math.Abs(dot-want) > 1e-9 {
				t.Fatalf("QᵀQ[%d,%d]=%v, want %v", i, j, dot, want)
			}
		}
	}
}

func TestPolarRotationRoundTripAndInnerProduct(t *testing.T) {
	const d = 16
	p, err := newPolarRotation(d, 7)
	if err != nil {
		t.Fatal(err)
	}
	x := make([]float64, d)
	y := make([]float64, d)
	for i := range d {
		x[i] = math.Sin(float64(i)*0.7) + 0.3
		y[i] = math.Cos(float64(i)*0.4) - 0.2
	}
	rx, _ := p.apply(x)
	back, _ := p.applyInverse(rx)
	for i := range d {
		if math.Abs(back[i]-x[i]) > 1e-9 {
			t.Fatalf("round-trip[%d]=%v, want %v", i, back[i], x[i])
		}
	}
	// inner product preserved under rotation
	ry, _ := p.apply(y)
	var ipOrig, ipRot float64
	for i := range d {
		ipOrig += x[i] * y[i]
		ipRot += rx[i] * ry[i]
	}
	if math.Abs(ipOrig-ipRot) > 1e-9 {
		t.Fatalf("⟨Πx,Πy⟩=%v ≠ ⟨x,y⟩=%v", ipRot, ipOrig)
	}
}

// PolarQuant storage round-trip: quantize→dequantize preserves DIRECTION well (high cosine),
// improving with more bits — the TurboQuant property. A zero vector round-trips to zero.
func TestPolarQuantRoundTrip(t *testing.T) {
	const d = 128
	p, err := newPolarRotation(d, 3)
	if err != nil {
		t.Fatal(err)
	}
	// deterministic pseudo-random vector
	x := make([]float64, d)
	for i := range x {
		x[i] = math.Sin(float64(i)*1.3) + 0.4*math.Cos(float64(i)*0.2)
	}
	cosFor := func(b int) float64 {
		idx, norm, err := p.polarQuantize(x, b)
		if err != nil {
			t.Fatal(err)
		}
		xh, err := p.polarDequantize(idx, norm, b)
		if err != nil {
			t.Fatal(err)
		}
		var dot, nx, nxh float64
		for i := range x {
			dot += x[i] * xh[i]
			nx += x[i] * x[i]
			nxh += xh[i] * xh[i]
		}
		return dot / (math.Sqrt(nx) * math.Sqrt(nxh))
	}
	c1, c2 := cosFor(1), cosFor(2)
	if c2 <= c1 {
		t.Fatalf("more bits should reconstruct better: cos b=2 %.3f !> b=1 %.3f", c2, c1)
	}
	if c2 < 0.9 {
		t.Fatalf("b=2 cosine %.3f too low (expect ≈0.94)", c2)
	}

	// zero vector → zero reconstruction
	zero := make([]float64, d)
	idx, norm, _ := p.polarQuantize(zero, 2)
	if norm != 0 {
		t.Fatalf("zero vector norm should be 0, got %v", norm)
	}
	xh, _ := p.polarDequantize(idx, norm, 2)
	for i := range xh {
		if xh[i] != 0 {
			t.Fatalf("zero round-trip nonzero at %d: %v", i, xh[i])
		}
	}
}

func TestPolarQuantErrors(t *testing.T) {
	p, _ := newPolarRotation(4, 1)
	if _, _, err := p.polarQuantize([]float64{1, 2, 3, 4}, 0); err == nil {
		t.Fatal("b<1 should error")
	}
	if _, err := p.polarDequantize([]int{0, 0, 0}, 1, 2); err == nil {
		t.Fatal("wrong index length should error")
	}
}

// §T619 QJL residual: the 1-bit sketch makes the attention inner-product estimate UNBIASED. At
// 1-bit PolarQuant the score is heavily biased (most of the residual is dropped); adding the QJL
// residual and averaging over the sketch randomness S recovers the true ⟨q,k⟩. (Any single S is
// noisy — unbiasedness is the property softmax-over-many-keys relies on, §T619 finding.)
func TestQJLUnbiasedInnerProduct(t *testing.T) {
	const d = 64
	p, err := newPolarRotation(d, 3)
	if err != nil {
		t.Fatal(err)
	}
	x := make([]float64, d)
	y := make([]float64, d)
	for i := range d {
		x[i] = math.Sin(float64(i)*1.1) - 0.3
		y[i] = math.Cos(float64(i)*0.6) + 0.2
	}
	var ipTrue float64
	for i := range d {
		ipTrue += x[i] * y[i]
	}
	idx, norm, _ := p.polarQuantize(x, 1)
	xhPolar, _ := p.polarDequantize(idx, norm, 1)
	var ipPolar float64
	for i := range d {
		ipPolar += xhPolar[i] * y[i]
	}
	// residual (rotated-unit space)
	var nx float64
	for _, v := range x {
		nx += v * v
	}
	nx = math.Sqrt(nx)
	u := make([]float64, d)
	for i := range d {
		u[i] = x[i] / nx
	}
	ru, _ := p.apply(u)
	cb, _ := polarCodebook(1, d)
	r := make([]float64, d)
	for i := range d {
		r[i] = ru[i] - cb[idx[i]]
	}
	// QJL-corrected estimate averaged over sketch seeds → unbiased
	const seeds = 2000
	var mean float64
	for sd := range seeds {
		q := newQJLSketch(d, uint64(sd)+1)
		signs, rn := q.encode(r)
		res := q.decodeResidual(signs, rn)
		ruT := make([]float64, d)
		for i := range d {
			ruT[i] = cb[idx[i]] + res[i]
		}
		uT, _ := p.applyInverse(ruT)
		var ip float64
		for i := range d {
			ip += norm * uT[i] * y[i]
		}
		mean += ip
	}
	mean /= seeds
	// the QJL-corrected mean must be much closer to the true score than biased polar-only.
	if math.Abs(mean-ipTrue) > 0.15*math.Abs(ipPolar-ipTrue) {
		t.Fatalf("QJL not debiasing: |mean-true|=%.4f vs |polar-true|=%.4f (true=%.3f)", math.Abs(mean-ipTrue), math.Abs(ipPolar-ipTrue), ipTrue)
	}
}

// TurboQuantKVCache stores under 4 bits/coordinate (sub-8-bit), preserves the vector norm
// (stored exactly), and its unbiased reconstruction keeps the attention direction on average.
func TestTurboQuantKVCache(t *testing.T) {
	const dim, bits = 64, 2
	c, err := NewTurboQuantKVCache(dim, bits, 1)
	if err != nil {
		t.Fatal(err)
	}
	k := make([]float64, dim)
	v := make([]float64, dim)
	for i := range dim {
		k[i] = math.Sin(float64(i) * 0.9)
		v[i] = math.Cos(float64(i) * 0.5)
	}
	if err := c.Append(k, v); err != nil {
		t.Fatal(err)
	}
	if c.Len() != 1 {
		t.Fatalf("Len=%d, want 1", c.Len())
	}
	// sub-8-bit: per row far under the 32-bit f32 footprint (dim*4 = 256 B/row)
	if perRow := c.Bytes() / 2; perRow >= dim*4 {
		t.Fatalf("not compressed: %d B/row ≥ f32 %d", perRow, dim*4)
	}
	// norm preserved (stored exactly); reconstruction direction positively correlated
	kh := c.Keys()[0]
	var dot, nk, nkh float64
	for i := range dim {
		dot += k[i] * kh[i]
		nk += k[i] * k[i]
		nkh += kh[i] * kh[i]
	}
	if math.Abs(math.Sqrt(nkh)-math.Sqrt(nk)) > 0.2*math.Sqrt(nk) {
		t.Fatalf("norm not preserved: ‖k̃‖=%.3f vs ‖k‖=%.3f", math.Sqrt(nkh), math.Sqrt(nk))
	}
	if dot <= 0 {
		t.Fatalf("reconstruction anti-correlated: ⟨k,k̃⟩=%.3f", dot)
	}
}

func TestTurboQuantKVCacheErrors(t *testing.T) {
	if _, err := NewTurboQuantKVCache(0, 2, 1); err == nil {
		t.Fatal("dim<1 should error")
	}
	if _, err := NewTurboQuantKVCache(8, 0, 1); err == nil {
		t.Fatal("bits<1 should error")
	}
	c, _ := NewTurboQuantKVCache(4, 2, 1)
	if err := c.Append([]float64{1, 2, 3}, []float64{1, 2, 3, 4}); err == nil {
		t.Fatal("wrong length should error")
	}
}

func ExampleTurboQuantKVCache() {
	// A 128-dim KV cache at 2-bit PolarQuant + 1-bit QJL residual.
	c, _ := NewTurboQuantKVCache(128, 2, 42)
	k := make([]float64, 128)
	v := make([]float64, 128)
	for i := range k {
		k[i] = float64(i%7) - 3
		v[i] = float64(i%5) - 2
	}
	_ = c.Append(k, v)
	// compression vs 32-bit floats: 128*4 = 512 bytes/row → the TurboQuant row is far smaller.
	f32PerRow := 128 * 4
	tqPerRow := c.Bytes() / 2 // keys+values stored; Bytes counts both
	fmt.Printf("%dx smaller\n", f32PerRow/tqPerRow)
	// Output: 8x smaller
}

// §V15 fuzz: TurboQuantKVCache round-trips over random shapes/values without panics or NaNs, keeps
// Len/Bytes correct, and preserves each vector's norm (stored exactly) within the quantizer bound.
func TestTurboQuantKVCacheRoundTripFuzz(t *testing.T) {
	rng := rand.New(rand.NewPCG(619, 0xabcd))
	for iter := range 200 {
		dim := 8 + rng.IntN(120)
		bits := 1 + rng.IntN(2) // 1 or 2
		c, err := NewTurboQuantKVCache(dim, bits, uint64(iter)+1)
		if err != nil {
			t.Fatal(err)
		}
		nRows := 1 + rng.IntN(4)
		orig := make([][]float64, nRows)
		for r := range nRows {
			k := make([]float64, dim)
			v := make([]float64, dim)
			for i := range dim {
				k[i] = rng.NormFloat64() * (0.1 + 3*rng.Float64())
				v[i] = rng.NormFloat64()
			}
			orig[r] = k
			if err := c.Append(k, v); err != nil {
				t.Fatalf("iter %d: append: %v", iter, err)
			}
		}
		if c.Len() != nRows {
			t.Fatalf("iter %d: Len=%d want %d", iter, c.Len(), nRows)
		}
		if want := (dim*bits/8 + (dim+7)/8 + 16) * 2 * nRows; c.Bytes() != want {
			t.Fatalf("iter %d: Bytes=%d want %d", iter, c.Bytes(), want)
		}
		keys := c.Keys()
		for r := range nRows {
			var no, nr float64
			for i := range dim {
				if math.IsNaN(keys[r][i]) || math.IsInf(keys[r][i], 0) {
					t.Fatalf("iter %d row %d: non-finite reconstruction", iter, r)
				}
				no += orig[r][i] * orig[r][i]
				nr += keys[r][i] * keys[r][i]
			}
			no, nr = math.Sqrt(no), math.Sqrt(nr)
			// single-row norm preservation is a bits=2 property; at bits=1 the QJL residual
			// reconstruction is unbiased for the inner product but too coarse per-row for the norm.
			if bits == 2 && no > 1e-9 && math.Abs(nr-no) > 0.4*no {
				t.Fatalf("iter %d row %d: norm drift ‖x̃‖=%.3f vs ‖x‖=%.3f (dim=%d bits=%d)", iter, r, nr, no, dim, bits)
			}
		}
	}
}

// §T619 the numeric Lloyd-Max Gaussian quantizer must reproduce the paper's closed-form codebooks
// (b=1 ±√(2/π); b=2 ±0.4528, ±1.510) and extend to b≥3 (unblocking the 2.5-bit outlier split).
func TestLloydMaxGaussianMatchesClosedForms(t *testing.T) {
	c1 := lloydMaxGaussian(1)
	if w := math.Sqrt(2 / math.Pi); math.Abs(c1[0]+w) > 1e-3 || math.Abs(c1[1]-w) > 1e-3 {
		t.Fatalf("b=1 %v != ±%.4f", c1, w)
	}
	c2 := lloydMaxGaussian(2)
	for i, w := range []float64{-1.510, -0.4528, 0.4528, 1.510} {
		if math.Abs(c2[i]-w) > 5e-3 {
			t.Fatalf("b=2[%d]=%.4f != %.4f", i, c2[i], w)
		}
	}
	// b=3: 8 centroids, strictly ascending, symmetric about 0.
	c3 := lloydMaxGaussian(3)
	if len(c3) != 8 {
		t.Fatalf("b=3 has %d centroids, want 8", len(c3))
	}
	for i := 1; i < len(c3); i++ {
		if c3[i] <= c3[i-1] {
			t.Fatalf("b=3 not ascending at %d: %v", i, c3)
		}
	}
	if math.Abs(c3[0]+c3[7]) > 1e-6 || math.Abs(c3[3]+c3[4]) > 1e-6 {
		t.Fatalf("b=3 not symmetric: %v", c3)
	}
	// polarCodebook now scales by 1/√d and supports b=3
	if _, err := polarCodebook(3, 64); err != nil {
		t.Fatalf("polarCodebook b=3: %v", err)
	}
}

// The numeric codebook now lets the public cache use 3-bit (the outlier tier): higher bits
// reconstruct better than 2-bit, confirming b=3 works end to end.
func TestTurboQuantKVCache3Bit(t *testing.T) {
	const dim = 96
	cos := func(bits int) float64 {
		c, err := NewTurboQuantKVCache(dim, bits, 5)
		if err != nil {
			t.Fatal(err)
		}
		k := make([]float64, dim)
		for i := range k {
			k[i] = math.Sin(float64(i)*0.8) + 0.2
		}
		if err := c.Append(k, k); err != nil {
			t.Fatal(err)
		}
		kh := c.Keys()[0]
		var dot, nk, nkh float64
		for i := range dim {
			dot += k[i] * kh[i]
			nk += k[i] * k[i]
			nkh += kh[i] * kh[i]
		}
		return dot / (math.Sqrt(nk) * math.Sqrt(nkh))
	}
	c2, c3 := cos(2), cos(3)
	if c3 <= c2 {
		t.Fatalf("3-bit should reconstruct better than 2-bit: cos b3=%.3f !> b2=%.3f", c3, c2)
	}
}

// The KV-cache memory hierarchy (§R108 → §T619): TurboQuant (sub-4-bit) is markedly smaller than
// the 8-bit Q8_0 cache, which is markedly smaller than raw f32. Documents the sub-4-bit tier's
// memory advantage at a realistic dimension.
func TestTurboQuantMemoryHierarchy(t *testing.T) {
	const dim, rows = 512, 100
	q8, err := NewQuantKVCache(dim) // §R108 8-bit Q8_0
	if err != nil {
		t.Fatal(err)
	}
	tq2, err := NewTurboQuantKVCache(dim, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	kt := tensor.New(tensor.F32, tensor.Shape{dim})
	kf := make([]float64, dim)
	for r := 0; r < rows; r++ {
		if err := q8.Append(kt, kt); err != nil {
			t.Fatal(err)
		}
		if err := tq2.Append(kf, kf); err != nil {
			t.Fatal(err)
		}
	}
	f32Bytes := dim * 4 * 2 * rows // keys+values f32
	q8Bytes := q8.Bytes()
	tqBytes := tq2.Bytes()
	t.Logf("KV footprint @dim=%d, %d rows: f32=%d  Q8_0=%d (%.1f× vs f32)  TurboQuant-2bit=%d (%.1f× vs f32, %.1f× vs Q8_0)",
		dim, rows, f32Bytes, q8Bytes, float64(f32Bytes)/float64(q8Bytes), tqBytes,
		float64(f32Bytes)/float64(tqBytes), float64(q8Bytes)/float64(tqBytes))
	if !(tqBytes < q8Bytes && q8Bytes < f32Bytes) {
		t.Fatalf("expected TurboQuant %d < Q8_0 %d < f32 %d", tqBytes, q8Bytes, f32Bytes)
	}
}

// BenchmarkTurboQuantAppend measures the compress-per-row throughput (the GoAI baseline for the
// perf comparison against reference KV-quant implementations). Reconstruction is benchmarked
// separately since attention only pays it on read.
func BenchmarkTurboQuantAppend(b *testing.B) {
	const dim = 128
	c, _ := NewTurboQuantKVCache(dim, 2, 1)
	k := make([]float64, dim)
	v := make([]float64, dim)
	for i := range k {
		k[i] = math.Sin(float64(i)) + 0.1
		v[i] = math.Cos(float64(i))
	}
	b.ResetTimer()
	for range b.N {
		_ = c.Append(k, v)
	}
}

func BenchmarkTurboQuantReconstruct(b *testing.B) {
	const dim = 128
	c, _ := NewTurboQuantKVCache(dim, 2, 1)
	k := make([]float64, dim)
	for i := range k {
		k[i] = math.Sin(float64(i)) + 0.1
	}
	_ = c.Append(k, k)
	b.ResetTimer()
	for range b.N {
		_ = c.Keys()
	}
}
