package gguf

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// q4kSuperRange returns the super-block value range hi−lo (with the offset clamped
// non-positive, as the encoder does) — the span Q4_K's 4-bit affine grid covers, so the
// per-element error is bounded relative to it.
func q4kSuperRange(x []float64, sb int) float64 {
	lo, hi := x[sb*qkK], x[sb*qkK]
	for i := 1; i < qkK; i++ {
		v := x[sb*qkK+i]
		lo, hi = math.Min(lo, v), math.Max(hi, v)
	}
	if lo > 0 {
		lo = 0
	}
	return hi - lo
}

// §V15 (§R100): Q4_K dequant(quant(x)) stays within the affine 4-bit bound. The finest
// sub-block step is R/15 (its 6-bit scale pins to 63, R = super-block range); a coarser
// sub-block's scale AND min are each 6-bit-quantized (scale-of-scale + min-of-min), widening
// the worst element error — empirically ≈ R/15, so R/12 is a sound per-element bound.
func TestQuantizeQ4_KRoundTrip(t *testing.T) {
	const n = 3 * qkK
	xf := randF32Block(n, 21)
	x := tensor.FromFloat64(tensor.Shape{n}, xf)
	data, err := Quantize(x, Q4_K)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != n/qkK*q4kBlockSize {
		t.Fatalf("Q4_K size = %d, want %d", len(data), n/qkK*q4kBlockSize)
	}
	deq, err := Dequantize(data, Q4_K, n)
	if err != nil {
		t.Fatal(err)
	}
	df := deq.Storage().F32()
	for sb := range n / qkK {
		R := q4kSuperRange(xf, sb)
		// R/12 for the 4-bit grid step + amax·f16 slack for the d/dmin·min offset rounding
		// (a constant block has R=0 but its offset still carries f16 error ∝ magnitude).
		bound := R/12 + q6kSuperAmax(xf, sb)*0.002
		for i := range qkK {
			k := sb*qkK + i
			if e := math.Abs(float64(df[k]) - xf[k]); e > bound {
				t.Errorf("Q4_K super-block %d [%d]: error %g > bound %g", sb, i, e, bound)
			}
		}
	}
}

// §V15: the affine offset is genuine. Q4_K's reconstruction y = d·sc·nibble − dmin·min has
// a non-negative subtracted min (dmin,min6 ≥ 0), so the floor is ≤ 0 — the format targets
// zero-crossing weights, spending its 16 codes on the ACTUAL [lo,hi] span rather than the
// symmetric [−amax,amax] of Q4_0. On asymmetric zero-crossing data (values in ≈[−0.3,2.7])
// Q4_K therefore reconstructs strictly better than Q4_0, which wastes codes on the empty
// negative tail. (An all-positive block cannot lift its floor above 0 — a real format limit,
// not an encoder bug; ggml's make_qkx2_quants clamps min non-positive identically.)
func TestQuantizeQ4_KAffineOffset(t *testing.T) {
	xf := make([]float64, qkK)
	for i := range xf {
		xf[i] = 1.2 + 1.5*math.Sin(float64(i)*0.3) // ≈[−0.3, 2.7], crosses zero asymmetrically
	}
	x := tensor.FromFloat64(tensor.Shape{qkK}, xf)
	maxErrOf := func(qt QuantType) float64 {
		data, _ := Quantize(x, qt)
		deq, _ := Dequantize(data, qt, qkK)
		df := deq.Storage().F32()
		var e float64
		for i := range qkK {
			e = math.Max(e, math.Abs(float64(df[i])-xf[i]))
		}
		return e
	}
	q4k, q40 := maxErrOf(Q4_K), maxErrOf(Q4_0)
	// span ≈3, Q4_K's affine grid step ≈3/15=0.2 → error ≲0.12.
	if q4k > 0.12 {
		t.Errorf("Q4_K affine: max error %g too high (offset not captured?)", q4k)
	}
	if q4k >= q40 {
		t.Errorf("Q4_K (%g) should beat symmetric Q4_0 (%g) on asymmetric zero-crossing data", q4k, q40)
	}
}

// §V15: Q4_K is a real 4-bit format — mean reconstruction error is a small fraction of the
// signal even though it is 2 bits coarser than Q6_K.
func TestQuantizeQ4_KAccurate(t *testing.T) {
	const n = 4 * qkK
	xf := randF32Block(n, 22)
	x := tensor.FromFloat64(tensor.Shape{n}, xf)
	data, _ := Quantize(x, Q4_K)
	deq, _ := Dequantize(data, Q4_K, n)
	df := deq.Storage().F32()
	var sumErr, sumAbs float64
	for i := range n {
		sumErr += math.Abs(float64(df[i]) - xf[i])
		sumAbs += math.Abs(xf[i])
	}
	if rel := sumErr / sumAbs; rel > 0.06 {
		t.Errorf("Q4_K mean relative error %.4f too high", rel)
	}
}

// §V15: dequantize→requantize does not drift beyond one quant step. The affine (scale,min)
// is data-dependent, so byte-idempotence is too strong; the invariant is a bounded fixed
// neighbourhood (R/12), not a diverging map.
func TestQuantizeQ4_KStable(t *testing.T) {
	const n = 2 * qkK
	x := tensor.FromFloat64(tensor.Shape{n}, randF32Block(n, 23))
	b1, _ := Quantize(x, Q4_K)
	deq1, _ := Dequantize(b1, Q4_K, n)
	b2, _ := Quantize(deq1, Q4_K)
	deq2, _ := Dequantize(b2, Q4_K, n)
	d1, d2 := deq1.Storage().F32(), deq2.Storage().F32()
	for sb := range n / qkK {
		d1f := cloneF32(d1)
		R := q4kSuperRange(d1f, sb)
		bound := R/12 + q6kSuperAmax(d1f, sb)*0.002
		for i := range qkK {
			k := sb*qkK + i
			if e := math.Abs(float64(d1[k] - d2[k])); e > bound {
				t.Errorf("Q4_K requant drift at [%d]: %v vs %v (bound %g)", k, d1[k], d2[k], bound)
			}
		}
	}
}

// §V15: numel not a multiple of the 256-element super-block is rejected, not silently
// tail-truncated.
func TestQuantizeQ4_KRejectsMisaligned(t *testing.T) {
	x := tensor.FromFloat64(tensor.Shape{200}, randF32Block(200, 24))
	if _, err := Quantize(x, Q4_K); err == nil {
		t.Error("Q4_K accepted numel 200 (not a multiple of 256)")
	}
}

// getScaleMinK4 ⁻¹: putScaleMinK4 packs 8 six-bit scales + 8 six-bit mins so getScaleMinK4
// recovers them exactly — the notoriously fiddly 6-bit splice is bit-exact (§R100).
func TestQ4KScaleMinPackRoundTrip(t *testing.T) {
	var sc, mn [q4kSubs]byte
	for j := range q4kSubs {
		sc[j] = byte((j*11 + 7) & 63) // spread across 0..63, exercises high 2 bits
		mn[j] = byte((j*23 + 5) & 63)
	}
	packed := make([]byte, 12)
	putScaleMinK4(&sc, &mn, packed)
	for j := range q4kSubs {
		gs, gm := getScaleMinK4(j, packed)
		if gs != sc[j] || gm != mn[j] {
			t.Errorf("sub-block %d: got (sc=%d,min=%d), want (%d,%d)", j, gs, gm, sc[j], mn[j])
		}
	}
}

// §V15 / E2E: QMatMul over a Q4_K-quantized weight approximates the full-precision matmul —
// Quantize produces QMatMul-compatible Q4_K bytes, dequantized one 256-aligned row at a time.
func TestQuantizeQ4_KMatMul(t *testing.T) {
	const m, k, nOut = 2, qkK, 3
	x := tensor.FromFloat64(tensor.Shape{m, k}, randF32Block(m*k, 25))
	wf := randF32Block(nOut*k, 26)
	w := tensor.FromFloat64(tensor.Shape{nOut, k}, wf)

	qw, err := Quantize(w, Q4_K)
	if err != nil {
		t.Fatal(err)
	}
	got, err := QMatMul(x, qw, Q4_K, nOut, k)
	if err != nil {
		t.Fatal(err)
	}
	for mi := range m {
		for ni := range nOut {
			var want float64
			for ki := range k {
				want += x.AtF64(mi, ki) * wf[ni*k+ki]
			}
			if e := math.Abs(got.AtF64(mi, ni) - want); e > 0.06*math.Max(1, math.Abs(want)) {
				t.Errorf("Q4_K QMatMul[%d,%d] = %g, full-precision %g (err %g)", mi, ni, got.AtF64(mi, ni), want, e)
			}
		}
	}
}

// FuzzQuantizeQ4_K: any 256-element block quantizes and dequantizes within the bound and
// never panics (§V15). R/8 gives adversarial headroom over the ~R/15 typical worst case.
// The amax/120 term is the 6-bit MIN granularity (§B48): dmin = maxMin/63, so a constant
// sub-block (scale 0 — the nibble cannot compensate) carries an inherent error of up to
// dmin/2 ≈ amax/126 even from an OPTIMAL encoder; /120 leaves slack for the f16 dmin.
// The fuzzer found exactly that shape (one outlier min raises dmin, a flat sub-block
// eats the granularity): error 0.469 vs optimal ceiling 0.496 — encoder optimal, the
// old amax·0.002 bound was structurally 4× too tight.
func FuzzQuantizeQ4_K(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4})
	f.Add([]byte("q4k-fuzz-seed"))
	f.Fuzz(func(t *testing.T, data []byte) {
		vals := make([]float64, qkK)
		for i := range vals {
			if i < len(data) {
				vals[i] = float64(int8(data[i])) * 0.5
			}
		}
		x := tensor.FromFloat64(tensor.Shape{qkK}, vals)
		b, err := Quantize(x, Q4_K)
		if err != nil {
			t.Fatal(err)
		}
		deq, err := Dequantize(b, Q4_K, qkK)
		if err != nil {
			t.Fatal(err)
		}
		R := q4kSuperRange(vals, 0)
		bound := R/8 + q6kSuperAmax(vals, 0)/120 + 1e-9
		df := deq.Storage().F32()
		for i := range qkK {
			if e := math.Abs(float64(df[i]) - vals[i]); e > bound {
				t.Fatalf("[%d]: error %g > bound %g (R=%g)", i, e, bound, R)
			}
		}
	})
}

// A Q4_K super-block stores 256 values in 144 bytes (~4.5 bits each) — the dominant modern
// GGUF weight format — reading back within a small affine-quant error.
func ExampleQuantize_q4K() {
	w := tensor.FromFloat64(tensor.Shape{qkK}, randF32Block(qkK, 0))
	data, _ := Quantize(w, Q4_K)
	deq, _ := Dequantize(data, Q4_K, qkK)
	var maxErr, R float64
	R = q4kSuperRange(cloneTensorF64(w), 0)
	for i := range qkK {
		maxErr = math.Max(maxErr, math.Abs(float64(deq.Storage().F32()[i])-w.AtF64(i)))
	}
	fmt.Println(len(data), "bytes for", w.Numel(), "values; within R/12:", maxErr < R/12)
	// Output: 144 bytes for 256 values; within R/12: true
}

// cloneTensorF64 materializes a 1-D tensor's values as []float64 for the range helper.
func cloneTensorF64(t *tensor.Tensor) []float64 {
	out := make([]float64, t.Numel())
	for i := range out {
		out[i] = t.AtF64(i)
	}
	return out
}
