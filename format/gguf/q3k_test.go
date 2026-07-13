package gguf

import (
	"encoding/binary"
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// §R103: packScalesQ3_K is the bit-exact inverse of the ggml aux/kmask scale unpack — the 16
// six-bit scales survive a pack→unpack cycle (the notoriously fiddly splice that stores each
// scale's low nibble and 2 high bits in separate byte regions).
func TestQ3KScalePackRoundTrip(t *testing.T) {
	var sc [16]byte
	for i := range sc {
		sc[i] = byte((i*7 + 5) & 63) // spread across 0..63, exercises the high 2 bits
	}
	packed := make([]byte, 12)
	packScalesQ3_K(&sc, packed)
	got := q3kUnpackScales(packed)
	for i := range sc {
		if got[i] != sc[i] {
			t.Errorf("scale %d: got %d, want %d", i, got[i], sc[i])
		}
	}
}

// §V15 (§R103): golden decode — hand-build a block (independent of the encoder) and check
// dequantQ3_K against the spec formula y = d·(sc6−32)·(q3−4), q3 = 2 low bits | (hmask bit)<<2,
// with the INVERTED hmask (bit set → the +4 high bit; bit clear → −4). Anchors the READ path
// (scale unpack + shift/is indexing + hmask arithmetic) to the definition, not a self-round-trip.
func TestQ3KGoldenDecode(t *testing.T) {
	raw := make([]byte, q3kBlockSize)
	dAll := float32(0.5)
	var sc [16]byte
	for i := range sc {
		sc[i] = byte(40 + i) // sc−32 = 8..23 (positive signed scales)
	}
	packScalesQ3_K(&sc, raw[96:108])
	binary.LittleEndian.PutUint16(raw[108:], f32ToF16(dAll))
	hm, qs := raw[0:32], raw[32:96]
	// element 0 (nblock0,j0,g0,l0): qs[0] bits0-1 = 3, hmask[0] bit0 SET → q3 = 3|(1<<2) = 7, val 3
	qs[0] = 0x03
	hm[0] = 0x01
	// element 32 (nblock0,j1,g0,l0): qs[0] bits2-3 = 1, hmask[0] bit1 CLEAR → q3 = 1, val −3
	qs[0] |= 0x01 << 2
	got, _ := dequantQ3_K(tensor.Shape{qkK}, raw)
	df := got.Storage().F32()
	want0 := dAll * float32(int(sc[0])-32) * 3     // is=0 → sc[0]
	want32 := dAll * float32(int(sc[2])-32) * (-3) // element 32 is (nblock0,j1,g0) → is=2
	if math.Abs(float64(df[0]-want0)) > 1e-5 {
		t.Errorf("elem0 (high bit set) %v, want %v", df[0], want0)
	}
	if math.Abs(float64(df[32]-want32)) > 1e-5 {
		t.Errorf("elem32 (high bit clear) %v, want %v", df[32], want32)
	}
}

// §V15 (§R103): Q3_K dequant(quant(x)) stays within the symmetric 3-bit bound. The quant is
// q3−4 ∈ [−4,3] over 8 levels, so the per-sub-block step is ≈ amax_sub/4 and the worst element
// error (incl. the 6-bit scale-of-scale coarsening of quieter sub-blocks) is ≈ A/6 (A =
// super-block amax); A/5 is a sound bound. The amax·f16 slack covers d's f16 rounding.
func TestQuantizeQ3_KRoundTrip(t *testing.T) {
	const n = 3 * qkK
	xf := randF32Block(n, 41)
	x := tensor.FromFloat64(tensor.Shape{n}, xf)
	data, err := Quantize(x, Q3_K)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != n/qkK*q3kBlockSize {
		t.Fatalf("Q3_K size = %d, want %d", len(data), n/qkK*q3kBlockSize)
	}
	deq, err := Dequantize(data, Q3_K, n)
	if err != nil {
		t.Fatal(err)
	}
	df := deq.Storage().F32()
	for sb := range n / qkK {
		A := q6kSuperAmax(xf, sb)
		bound := A/5 + A*0.005
		for i := range qkK {
			k := sb*qkK + i
			if e := math.Abs(float64(df[k]) - xf[k]); e > bound {
				t.Errorf("Q3_K super-block %d [%d]: error %g > bound %g", sb, i, e, bound)
			}
		}
	}
}

// §V15: Q3_K is a genuine (coarse) 3-bit format — mean relative error is a few×Q4_K's but
// bounded well under the ~16 % worst-case single element.
func TestQuantizeQ3_KAccurate(t *testing.T) {
	const n = 4 * qkK
	xf := randF32Block(n, 42)
	x := tensor.FromFloat64(tensor.Shape{n}, xf)
	data, _ := Quantize(x, Q3_K)
	deq, _ := Dequantize(data, Q3_K, n)
	df := deq.Storage().F32()
	var sumErr, sumAbs float64
	for i := range n {
		sumErr += math.Abs(float64(df[i]) - xf[i])
		sumAbs += math.Abs(xf[i])
	}
	if rel := sumErr / sumAbs; rel > 0.13 {
		t.Errorf("Q3_K mean relative error %.4f too high", rel)
	}
}

// §V15: dequantize→requantize does not drift beyond one quant step (data-dependent scale makes
// byte-idempotence too strong; the invariant is a bounded neighbourhood).
func TestQuantizeQ3_KStable(t *testing.T) {
	const n = 2 * qkK
	x := tensor.FromFloat64(tensor.Shape{n}, randF32Block(n, 43))
	b1, _ := Quantize(x, Q3_K)
	deq1, _ := Dequantize(b1, Q3_K, n)
	b2, _ := Quantize(deq1, Q3_K)
	deq2, _ := Dequantize(b2, Q3_K, n)
	d1, d2 := deq1.Storage().F32(), deq2.Storage().F32()
	for sb := range n / qkK {
		A := q6kSuperAmax(cloneF32(d1), sb)
		bound := A/5 + A*0.005
		for i := range qkK {
			k := sb*qkK + i
			if e := math.Abs(float64(d1[k] - d2[k])); e > bound {
				t.Errorf("Q3_K requant drift at [%d]: %v vs %v (bound %g)", k, d1[k], d2[k], bound)
			}
		}
	}
}

// §V15: numel not a multiple of the 256-element super-block is rejected, not tail-truncated.
func TestQuantizeQ3_KRejectsMisaligned(t *testing.T) {
	x := tensor.FromFloat64(tensor.Shape{200}, randF32Block(200, 44))
	if _, err := Quantize(x, Q3_K); err == nil {
		t.Error("Q3_K accepted numel 200 (not a multiple of 256)")
	}
}

// §V15 / E2E: QMatMul over a Q3_K-quantized weight approximates the full-precision matmul
// (coarse 3-bit → looser tolerance), dequantized one 256-aligned row at a time.
func TestQuantizeQ3_KMatMul(t *testing.T) {
	const m, k, nOut = 2, qkK, 3
	x := tensor.FromFloat64(tensor.Shape{m, k}, randF32Block(m*k, 45))
	wf := randF32Block(nOut*k, 46)
	w := tensor.FromFloat64(tensor.Shape{nOut, k}, wf)

	qw, err := Quantize(w, Q3_K)
	if err != nil {
		t.Fatal(err)
	}
	got, err := QMatMul(x, qw, Q3_K, nOut, k)
	if err != nil {
		t.Fatal(err)
	}
	for mi := range m {
		for ni := range nOut {
			var want float64
			for ki := range k {
				want += x.AtF64(mi, ki) * wf[ni*k+ki]
			}
			if e := math.Abs(got.AtF64(mi, ni) - want); e > 0.15*math.Max(1, math.Abs(want)) {
				t.Errorf("Q3_K QMatMul[%d,%d] = %g, full-precision %g (err %g)", mi, ni, got.AtF64(mi, ni), want, e)
			}
		}
	}
}

// FuzzQuantizeQ3_K: any 256-element block quantizes and dequantizes within the A/4 bound and
// never panics (§V15). A/4 gives adversarial headroom over the ~A/6 typical worst case.
func FuzzQuantizeQ3_K(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4})
	f.Add([]byte("q3k-fuzz-seed"))
	f.Fuzz(func(t *testing.T, data []byte) {
		vals := make([]float64, qkK)
		for i := range vals {
			if i < len(data) {
				vals[i] = float64(int8(data[i])) * 0.5
			}
		}
		x := tensor.FromFloat64(tensor.Shape{qkK}, vals)
		b, err := Quantize(x, Q3_K)
		if err != nil {
			t.Fatal(err)
		}
		deq, err := Dequantize(b, Q3_K, qkK)
		if err != nil {
			t.Fatal(err)
		}
		A := q6kSuperAmax(vals, 0)
		bound := A/4 + A*0.005 + 1e-9
		df := deq.Storage().F32()
		for i := range qkK {
			if e := math.Abs(float64(df[i]) - vals[i]); e > bound {
				t.Fatalf("[%d]: error %g > bound %g (A=%g)", i, e, bound, A)
			}
		}
	})
}

// A Q3_K super-block stores 256 values in 110 bytes (~3.44 bits each) — the low-bit format the
// Q3_K_M/Q3_K_L mixes use to run large models on limited hardware.
func ExampleQuantize_q3K() {
	w := tensor.FromFloat64(tensor.Shape{qkK}, randF32Block(qkK, 0))
	data, _ := Quantize(w, Q3_K)
	deq, _ := Dequantize(data, Q3_K, qkK)
	var maxErr, amax float64
	for i := range qkK {
		amax = math.Max(amax, math.Abs(w.AtF64(i)))
		maxErr = math.Max(maxErr, math.Abs(float64(deq.Storage().F32()[i])-w.AtF64(i)))
	}
	fmt.Println(len(data), "bytes for", w.Numel(), "values; within A/5:", maxErr < amax/5)
	// Output: 110 bytes for 256 values; within A/5: true
}
