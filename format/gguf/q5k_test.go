package gguf

import (
	"encoding/binary"
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// §V15 (§R102): Q5_K dequant(quant(x)) stays within the affine 5-bit bound. Two error terms:
// the 5-bit grid step R/31 (R = super-block range) → ≈R/62 per element, bounded by R/32; and
// the per-sub-block MIN (offset), which is itself 6-bit-quantized with step dmin ≈ amax/63, so
// the offset error reaches ≈ amax/126 on every element of a sub-block whose min differs from
// the super-block max min (worst for a clustered, large-magnitude, tiny-range block). The
// amax·0.015 term covers that min-quantization (≈amax/126) plus f16 rounding of d/dmin.
func TestQuantizeQ5_KRoundTrip(t *testing.T) {
	const n = 3 * qkK
	xf := randF32Block(n, 31)
	x := tensor.FromFloat64(tensor.Shape{n}, xf)
	data, err := Quantize(x, Q5_K)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != n/qkK*q5kBlockSize {
		t.Fatalf("Q5_K size = %d, want %d", len(data), n/qkK*q5kBlockSize)
	}
	deq, err := Dequantize(data, Q5_K, n)
	if err != nil {
		t.Fatal(err)
	}
	df := deq.Storage().F32()
	for sb := range n / qkK {
		R := q4kSuperRange(xf, sb)
		bound := R/32 + q6kSuperAmax(xf, sb)*0.015
		for i := range qkK {
			k := sb*qkK + i
			if e := math.Abs(float64(df[k]) - xf[k]); e > bound {
				t.Errorf("Q5_K super-block %d [%d]: error %g > bound %g", sb, i, e, bound)
			}
		}
	}
}

// §V15 (§R102): the qh high-bit plane is decoded correctly. Hand-build a block (independent of
// the encoder) with a known high bit set and check dequantQ5_K reconstructs y = d·sc6·q5 −
// dmin·min6 with q5 = nibble + 16·(qh bit) — anchors the READ path (esp. the 5th bit) to the
// spec formula, not just to a self-consistent round-trip.
func TestQ5KGoldenHighBit(t *testing.T) {
	raw := make([]byte, q5kBlockSize)
	d, dmin := float32(0.5), float32(0.25)
	binary.LittleEndian.PutUint16(raw[0:], f32ToF16(d))
	binary.LittleEndian.PutUint16(raw[2:], f32ToF16(dmin))
	var sc, mn [q4kSubs]byte
	for j := range q4kSubs {
		sc[j] = byte(9 + j*4) // all < 64
		mn[j] = byte(3 + j*2)
	}
	putScaleMinK4(&sc, &mn, raw[4:16])
	qh := raw[16:48]
	qs := raw[48:176]
	// pair 0, element 0: low nibble 5, HIGH BIT set (u1=bit0) → q5 = 5+16 = 21
	//                    high nibble 3, high bit CLEAR (u2=bit1)  → q5 = 3
	qs[0] = 0x35 // high nibble 3, low nibble 5
	qh[0] = 0x01 // bit0 set (u1 for pair 0 low), bit1 clear
	got, _ := dequantQ5_K(tensor.Shape{qkK}, raw)
	df := got.Storage().F32()
	want0 := d*float32(sc[0])*21 - dmin*float32(mn[0]) // is=0
	want32 := d*float32(sc[1])*3 - dmin*float32(mn[1]) // is=1, no high bit
	if math.Abs(float64(df[0]-want0)) > 1e-5 {
		t.Errorf("elem0 (high bit) %v want %v", df[0], want0)
	}
	if math.Abs(float64(df[32]-want32)) > 1e-5 {
		t.Errorf("elem32 (no high bit) %v want %v", df[32], want32)
	}
}

// §V15: Q5_K is a genuine 5-bit format — mean reconstruction error is well below Q4_K's (the
// extra bit halves the grid step).
func TestQuantizeQ5_KAccurate(t *testing.T) {
	const n = 4 * qkK
	xf := randF32Block(n, 32)
	x := tensor.FromFloat64(tensor.Shape{n}, xf)
	data, _ := Quantize(x, Q5_K)
	deq, _ := Dequantize(data, Q5_K, n)
	df := deq.Storage().F32()
	var sumErr, sumAbs float64
	for i := range n {
		sumErr += math.Abs(float64(df[i]) - xf[i])
		sumAbs += math.Abs(xf[i])
	}
	if rel := sumErr / sumAbs; rel > 0.035 {
		t.Errorf("Q5_K mean relative error %.4f too high", rel)
	}
}

// §V15: dequantize→requantize does not drift beyond one quant step (data-dependent affine
// scale makes byte-idempotence too strong; the invariant is a bounded neighbourhood).
func TestQuantizeQ5_KStable(t *testing.T) {
	const n = 2 * qkK
	x := tensor.FromFloat64(tensor.Shape{n}, randF32Block(n, 33))
	b1, _ := Quantize(x, Q5_K)
	deq1, _ := Dequantize(b1, Q5_K, n)
	b2, _ := Quantize(deq1, Q5_K)
	deq2, _ := Dequantize(b2, Q5_K, n)
	d1, d2 := deq1.Storage().F32(), deq2.Storage().F32()
	for sb := range n / qkK {
		d1f := cloneF32(d1)
		R := q4kSuperRange(d1f, sb)
		bound := R/32 + q6kSuperAmax(d1f, sb)*0.015
		for i := range qkK {
			k := sb*qkK + i
			if e := math.Abs(float64(d1[k] - d2[k])); e > bound {
				t.Errorf("Q5_K requant drift at [%d]: %v vs %v (bound %g)", k, d1[k], d2[k], bound)
			}
		}
	}
}

// §V15: numel not a multiple of the 256-element super-block is rejected, not silently
// tail-truncated.
func TestQuantizeQ5_KRejectsMisaligned(t *testing.T) {
	x := tensor.FromFloat64(tensor.Shape{200}, randF32Block(200, 34))
	if _, err := Quantize(x, Q5_K); err == nil {
		t.Error("Q5_K accepted numel 200 (not a multiple of 256)")
	}
}

// §V15: Q5_K is more accurate than Q4_K on the same data (5 bits vs 4) — confirms the high-bit
// plane genuinely adds resolution rather than being ignored.
func TestQuantizeQ5_KBeatsQ4_K(t *testing.T) {
	const n = 2 * qkK
	xf := randF32Block(n, 35)
	x := tensor.FromFloat64(tensor.Shape{n}, xf)
	errOf := func(qt QuantType) float64 {
		data, _ := Quantize(x, qt)
		deq, _ := Dequantize(data, qt, n)
		df := deq.Storage().F32()
		var e float64
		for i := range n {
			e += math.Abs(float64(df[i]) - xf[i])
		}
		return e
	}
	if q5, q4 := errOf(Q5_K), errOf(Q4_K); q5 >= q4 {
		t.Errorf("Q5_K total error %g should be < Q4_K %g", q5, q4)
	}
}

// §V15 / E2E: QMatMul over a Q5_K-quantized weight approximates the full-precision matmul,
// dequantized one 256-aligned row at a time.
func TestQuantizeQ5_KMatMul(t *testing.T) {
	const m, k, nOut = 2, qkK, 3
	x := tensor.FromFloat64(tensor.Shape{m, k}, randF32Block(m*k, 36))
	wf := randF32Block(nOut*k, 37)
	w := tensor.FromFloat64(tensor.Shape{nOut, k}, wf)

	qw, err := Quantize(w, Q5_K)
	if err != nil {
		t.Fatal(err)
	}
	got, err := QMatMul(x, qw, Q5_K, nOut, k)
	if err != nil {
		t.Fatal(err)
	}
	for mi := range m {
		for ni := range nOut {
			var want float64
			for ki := range k {
				want += x.AtF64(mi, ki) * wf[ni*k+ki]
			}
			if e := math.Abs(got.AtF64(mi, ni) - want); e > 0.03*math.Max(1, math.Abs(want)) {
				t.Errorf("Q5_K QMatMul[%d,%d] = %g, full-precision %g (err %g)", mi, ni, got.AtF64(mi, ni), want, e)
			}
		}
	}
}

// FuzzQuantizeQ5_K: any 256-element block quantizes and dequantizes within the R/16 bound and
// never panics (§V15). R/16 gives adversarial headroom over the ~R/61 typical worst case.
func FuzzQuantizeQ5_K(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4})
	f.Add([]byte("q5k-fuzz-seed"))
	f.Fuzz(func(t *testing.T, data []byte) {
		vals := make([]float64, qkK)
		for i := range vals {
			if i < len(data) {
				vals[i] = float64(int8(data[i])) * 0.5
			}
		}
		x := tensor.FromFloat64(tensor.Shape{qkK}, vals)
		b, err := Quantize(x, Q5_K)
		if err != nil {
			t.Fatal(err)
		}
		deq, err := Dequantize(b, Q5_K, qkK)
		if err != nil {
			t.Fatal(err)
		}
		R := q4kSuperRange(vals, 0)
		bound := R/16 + q6kSuperAmax(vals, 0)*0.015 + 1e-9
		df := deq.Storage().F32()
		for i := range qkK {
			if e := math.Abs(float64(df[i]) - vals[i]); e > bound {
				t.Fatalf("[%d]: error %g > bound %g (R=%g)", i, e, bound, R)
			}
		}
	})
}

// A Q5_K super-block stores 256 values in 176 bytes (~5.5 bits each) — the Q5_K_M weight
// format — reading back within a small affine-quant error.
func ExampleQuantize_q5K() {
	w := tensor.FromFloat64(tensor.Shape{qkK}, randF32Block(qkK, 0))
	data, _ := Quantize(w, Q5_K)
	deq, _ := Dequantize(data, Q5_K, qkK)
	var maxErr, R float64
	R = q4kSuperRange(cloneTensorF64(w), 0)
	for i := range qkK {
		maxErr = math.Max(maxErr, math.Abs(float64(deq.Storage().F32()[i])-w.AtF64(i)))
	}
	fmt.Println(len(data), "bytes for", w.Numel(), "values; within R/32:", maxErr < R/32)
	// Output: 176 bytes for 256 values; within R/32: true
}
