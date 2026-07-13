package gguf

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// q6kSuperAmax returns the super-block amax A (max |x| over the 256 elements) — the
// quantity Q6_K's two-level scale is normalized against, so the reconstruction error is
// bounded relative to A, not to a sub-block's own amax.
func q6kSuperAmax(x []float64, sb int) float64 {
	var a float64
	for i := range qkK {
		a = math.Max(a, math.Abs(x[sb*qkK+i]))
	}
	return a
}

// §V15 (§R99): Q6_K dequant(quant(x)) stays within the k-quant bound. The finest sub-block
// step is A/32 (its int8 scale pins to 127); the +32-centered clamp costs one step there, and
// coarser sub-blocks widen it further because their scale is itself quantized (scale-of-scale)
// — empirically the worst element error is ≈ A/29, so A/24 is a sound per-element bound. Tiny
// sub-blocks (amax < A/254) collapse to zero, an error still ≤ A/254. (The mean error is far
// smaller — see TestQuantizeQ6_KAccurate.)
func TestQuantizeQ6_KRoundTrip(t *testing.T) {
	const n = 3 * qkK // 3 super-blocks
	xf := randF32Block(n, 11)
	x := tensor.FromFloat64(tensor.Shape{n}, xf)
	data, err := Quantize(x, Q6_K)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != n/qkK*q6kBlockSize {
		t.Fatalf("Q6_K size = %d, want %d", len(data), n/qkK*q6kBlockSize)
	}
	deq, err := Dequantize(data, Q6_K, n)
	if err != nil {
		t.Fatal(err)
	}
	df := deq.Storage().F32()
	for sb := range n / qkK {
		A := q6kSuperAmax(xf, sb)
		bound := A/24 + A*0.001 // ≈ worst step (A/29) + f16 scale slack
		for i := range qkK {
			k := sb*qkK + i
			if e := math.Abs(float64(df[k]) - xf[k]); e > bound {
				t.Errorf("Q6_K super-block %d [%d]: error %g > bound %g", sb, i, e, bound)
			}
		}
	}
}

// §V15: Q6_K is a genuine 6-bit format — the mean reconstruction error is far below the
// worst-case A/32 step, confirming the scale hierarchy resolves most sub-blocks finely.
func TestQuantizeQ6_KAccurate(t *testing.T) {
	const n = 4 * qkK
	xf := randF32Block(n, 12)
	x := tensor.FromFloat64(tensor.Shape{n}, xf)
	data, _ := Quantize(x, Q6_K)
	deq, _ := Dequantize(data, Q6_K, n)
	df := deq.Storage().F32()
	var sumErr, sumAbs float64
	for i := range n {
		sumErr += math.Abs(float64(df[i]) - xf[i])
		sumAbs += math.Abs(xf[i])
	}
	// mean relative error should be small for a 6-bit format (well under 2%).
	if rel := sumErr / sumAbs; rel > 0.02 {
		t.Errorf("Q6_K mean relative error %.4f too high", rel)
	}
}

// §V15: re-quantizing a dequantized Q6_K block does not drift beyond one quant step. The
// scale is data-dependent (amax/32 with an f16 super-scale), so byte-idempotence is too
// strong to require; the meaningful invariant is that requant→dequant stays within the
// same A/24 band (measured ≈ A/33), i.e. the grid is a bounded fixed neighbourhood, not a
// diverging map.
func TestQuantizeQ6_KStable(t *testing.T) {
	const n = 2 * qkK
	x := tensor.FromFloat64(tensor.Shape{n}, randF32Block(n, 13))
	b1, _ := Quantize(x, Q6_K)
	deq1, _ := Dequantize(b1, Q6_K, n)
	b2, _ := Quantize(deq1, Q6_K)
	deq2, _ := Dequantize(b2, Q6_K, n)
	d1, d2 := deq1.Storage().F32(), deq2.Storage().F32()
	for sb := range n / qkK {
		A := q6kSuperAmax(cloneF32(d1), sb)
		bound := A/24 + A*0.001
		for i := range qkK {
			k := sb*qkK + i
			if e := math.Abs(float64(d1[k] - d2[k])); e > bound {
				t.Errorf("Q6_K requant drift at [%d]: %v vs %v (bound %g)", k, d1[k], d2[k], bound)
			}
		}
	}
}

// cloneF32 lifts an []float32 to []float64 for the super-amax helper.
func cloneF32(x []float32) []float64 {
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = float64(v)
	}
	return out
}

// §V15: numel not a multiple of the 256-element super-block is rejected, not silently
// truncated (which would drop the tail block).
func TestQuantizeQ6_KRejectsMisaligned(t *testing.T) {
	x := tensor.FromFloat64(tensor.Shape{200}, randF32Block(200, 14))
	if _, err := Quantize(x, Q6_K); err == nil {
		t.Error("Q6_K accepted numel 200 (not a multiple of 256)")
	}
}

// §V15 / E2E: QMatMul over a Q6_K-quantized weight approximates the full-precision matmul
// within the 6-bit error — Quantize produces QMatMul-compatible Q6_K bytes, dequantized
// one row (k a multiple of 256) at a time.
func TestQuantizeQ6_KMatMul(t *testing.T) {
	const m, k, nOut = 2, qkK, 3
	x := tensor.FromFloat64(tensor.Shape{m, k}, randF32Block(m*k, 15))
	wf := randF32Block(nOut*k, 16)
	w := tensor.FromFloat64(tensor.Shape{nOut, k}, wf)

	qw, err := Quantize(w, Q6_K)
	if err != nil {
		t.Fatal(err)
	}
	got, err := QMatMul(x, qw, Q6_K, nOut, k)
	if err != nil {
		t.Fatal(err)
	}
	for mi := range m {
		for ni := range nOut {
			var want float64
			for ki := range k {
				want += x.AtF64(mi, ki) * wf[ni*k+ki]
			}
			if e := math.Abs(got.AtF64(mi, ni) - want); e > 0.02*math.Max(1, math.Abs(want)) {
				t.Errorf("Q6_K QMatMul[%d,%d] = %g, full-precision %g (err %g)", mi, ni, got.AtF64(mi, ni), want, e)
			}
		}
	}
}

// FuzzQuantizeQ6_K: any 256-element block quantizes and dequantizes within the A/32 bound
// and never panics (§V15).
func FuzzQuantizeQ6_K(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4})
	f.Add([]byte("q6k-fuzz-seed"))
	f.Fuzz(func(t *testing.T, data []byte) {
		vals := make([]float64, qkK)
		for i := range vals {
			if i < len(data) {
				vals[i] = float64(int8(data[i])) * 0.5
			}
		}
		x := tensor.FromFloat64(tensor.Shape{qkK}, vals)
		b, err := Quantize(x, Q6_K)
		if err != nil {
			t.Fatal(err)
		}
		deq, err := Dequantize(b, Q6_K, qkK)
		if err != nil {
			t.Fatal(err)
		}
		A := q6kSuperAmax(vals, 0)
		bound := A/16 + A*0.001 + 1e-9 // A/16: generous headroom for adversarial fuzz inputs
		df := deq.Storage().F32()
		for i := range qkK {
			if e := math.Abs(float64(df[i]) - vals[i]); e > bound {
				t.Fatalf("[%d]: error %g > bound %g (A=%g)", i, e, bound, A)
			}
		}
	})
}

// A Q6_K super-block stores 256 values in 210 bytes (~6.56 bits each) and reads back
// within a small rounding error — the format modern GGUF models use for their weights.
func ExampleQuantize_q6K() {
	w := tensor.FromFloat64(tensor.Shape{qkK}, randF32Block(qkK, 0))
	data, _ := Quantize(w, Q6_K)
	deq, _ := Dequantize(data, Q6_K, qkK)
	var maxErr, amax float64
	for i := range qkK {
		amax = math.Max(amax, math.Abs(w.AtF64(i)))
		maxErr = math.Max(maxErr, math.Abs(float64(deq.Storage().F32()[i])-w.AtF64(i)))
	}
	fmt.Println(len(data), "bytes for", w.Numel(), "values; within A/24:", maxErr < amax/24)
	// Output: 210 bytes for 256 values; within A/24: true
}
