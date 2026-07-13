package gguf

import (
	"encoding/binary"
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// §V15 (§R104): golden decode — hand-build a block (independent of the encoder) and check
// dequantQ2_K against the spec formula y = d·(sc&0xF)·q2 − dmin·(sc>>4), q2 = (q>>shift)&3 ∈
// [0,3]. Anchors the READ path (nibble scale/min split, shift/is indexing) to the definition.
func TestQ2KGoldenDecode(t *testing.T) {
	raw := make([]byte, q2kBlockSize)
	d, dmin := float32(0.5), float32(0.25)
	binary.LittleEndian.PutUint16(raw[80:], f32ToF16(d))
	binary.LittleEndian.PutUint16(raw[82:], f32ToF16(dmin))
	sc, qs := raw[0:16], raw[16:80]
	for i := range sc {
		sc[i] = byte(5+i) | byte(3+i)<<4 // low nibble = 4-bit scale, high nibble = 4-bit min
	}
	// element 0 (nblock0,j0,g0,l0): qs[0] bits0-1 = 3, is=0
	qs[0] = 0x03
	// element 32 (nblock0,j1,g0,l0): qs[0] bits2-3 = 2, is=2
	qs[0] |= 0x02 << 2
	got, _ := dequantQ2_K(tensor.Shape{qkK}, raw)
	df := got.Storage().F32()
	want0 := d*float32(sc[0]&0xF)*3 - dmin*float32(sc[0]>>4)
	want32 := d*float32(sc[2]&0xF)*2 - dmin*float32(sc[2]>>4)
	if math.Abs(float64(df[0]-want0)) > 1e-5 {
		t.Errorf("elem0 %v, want %v", df[0], want0)
	}
	if math.Abs(float64(df[32]-want32)) > 1e-5 {
		t.Errorf("elem32 %v, want %v", df[32], want32)
	}
}

// §V15 (§R104): Q2_K dequant(quant(x)) stays within the 2-bit affine bound. The quant q2 ∈
// [0,3] over 4 levels → per-sub-block step ≈ R/3 and worst element error ≈ R/6 (R = super-block
// range); the amax·0.05 term covers the 4-bit min (offset) quantization (dmin ≈ amax/15) and
// f16 rounding. R/4 gives margin over the measured ~R/6 worst.
func TestQuantizeQ2_KRoundTrip(t *testing.T) {
	const n = 3 * qkK
	xf := randF32Block(n, 51)
	x := tensor.FromFloat64(tensor.Shape{n}, xf)
	data, err := Quantize(x, Q2_K)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != n/qkK*q2kBlockSize {
		t.Fatalf("Q2_K size = %d, want %d", len(data), n/qkK*q2kBlockSize)
	}
	deq, err := Dequantize(data, Q2_K, n)
	if err != nil {
		t.Fatal(err)
	}
	df := deq.Storage().F32()
	for sb := range n / qkK {
		R := q4kSuperRange(xf, sb)
		bound := R/4 + q6kSuperAmax(xf, sb)*0.05
		for i := range qkK {
			k := sb*qkK + i
			if e := math.Abs(float64(df[k]) - xf[k]); e > bound {
				t.Errorf("Q2_K super-block %d [%d]: error %g > bound %g", sb, i, e, bound)
			}
		}
	}
}

// §V15: Q2_K is a genuine (very coarse) 2-bit format — mean relative error is large but
// bounded; this pins that it stays a small-ish fraction (not garbage).
func TestQuantizeQ2_KAccurate(t *testing.T) {
	const n = 4 * qkK
	xf := randF32Block(n, 52)
	x := tensor.FromFloat64(tensor.Shape{n}, xf)
	data, _ := Quantize(x, Q2_K)
	deq, _ := Dequantize(data, Q2_K, n)
	df := deq.Storage().F32()
	var sumErr, sumAbs float64
	for i := range n {
		sumErr += math.Abs(float64(df[i]) - xf[i])
		sumAbs += math.Abs(xf[i])
	}
	if rel := sumErr / sumAbs; rel > 0.25 {
		t.Errorf("Q2_K mean relative error %.4f too high", rel)
	}
}

// §V15: dequantize→requantize does not drift beyond one quant step (data-dependent affine
// scale makes byte-idempotence too strong; the invariant is a bounded neighbourhood).
func TestQuantizeQ2_KStable(t *testing.T) {
	const n = 2 * qkK
	x := tensor.FromFloat64(tensor.Shape{n}, randF32Block(n, 53))
	b1, _ := Quantize(x, Q2_K)
	deq1, _ := Dequantize(b1, Q2_K, n)
	b2, _ := Quantize(deq1, Q2_K)
	deq2, _ := Dequantize(b2, Q2_K, n)
	d1, d2 := deq1.Storage().F32(), deq2.Storage().F32()
	for sb := range n / qkK {
		d1f := cloneF32(d1)
		R := q4kSuperRange(d1f, sb)
		bound := R/4 + q6kSuperAmax(d1f, sb)*0.05
		for i := range qkK {
			k := sb*qkK + i
			if e := math.Abs(float64(d1[k] - d2[k])); e > bound {
				t.Errorf("Q2_K requant drift at [%d]: %v vs %v (bound %g)", k, d1[k], d2[k], bound)
			}
		}
	}
}

// §V15: numel not a multiple of the 256-element super-block is rejected, not tail-truncated.
func TestQuantizeQ2_KRejectsMisaligned(t *testing.T) {
	x := tensor.FromFloat64(tensor.Shape{200}, randF32Block(200, 54))
	if _, err := Quantize(x, Q2_K); err == nil {
		t.Error("Q2_K accepted numel 200 (not a multiple of 256)")
	}
}

// §V15 / E2E: QMatMul over a Q2_K-quantized weight approximates the full-precision matmul
// (very coarse 2-bit → loose tolerance), dequantized one 256-aligned row at a time.
func TestQuantizeQ2_KMatMul(t *testing.T) {
	const m, k, nOut = 2, qkK, 3
	x := tensor.FromFloat64(tensor.Shape{m, k}, randF32Block(m*k, 55))
	wf := randF32Block(nOut*k, 56)
	w := tensor.FromFloat64(tensor.Shape{nOut, k}, wf)

	qw, err := Quantize(w, Q2_K)
	if err != nil {
		t.Fatal(err)
	}
	got, err := QMatMul(x, qw, Q2_K, nOut, k)
	if err != nil {
		t.Fatal(err)
	}
	for mi := range m {
		for ni := range nOut {
			var want float64
			for ki := range k {
				want += x.AtF64(mi, ki) * wf[ni*k+ki]
			}
			if e := math.Abs(got.AtF64(mi, ni) - want); e > 0.35*math.Max(1, math.Abs(want)) {
				t.Errorf("Q2_K QMatMul[%d,%d] = %g, full-precision %g (err %g)", mi, ni, got.AtF64(mi, ni), want, e)
			}
		}
	}
}

// FuzzQuantizeQ2_K: any 256-element block quantizes and dequantizes within the R/3 bound and
// never panics (§V15). R/3 gives adversarial headroom over the ~R/6 typical worst case.
func FuzzQuantizeQ2_K(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4})
	f.Add([]byte("q2k-fuzz-seed"))
	f.Fuzz(func(t *testing.T, data []byte) {
		vals := make([]float64, qkK)
		for i := range vals {
			if i < len(data) {
				vals[i] = float64(int8(data[i])) * 0.5
			}
		}
		x := tensor.FromFloat64(tensor.Shape{qkK}, vals)
		b, err := Quantize(x, Q2_K)
		if err != nil {
			t.Fatal(err)
		}
		deq, err := Dequantize(b, Q2_K, qkK)
		if err != nil {
			t.Fatal(err)
		}
		R := q4kSuperRange(vals, 0)
		bound := R/3 + q6kSuperAmax(vals, 0)*0.05 + 1e-9
		df := deq.Storage().F32()
		for i := range qkK {
			if e := math.Abs(float64(df[i]) - vals[i]); e > bound {
				t.Fatalf("[%d]: error %g > bound %g (R=%g)", i, e, bound, R)
			}
		}
	})
}

// A Q2_K super-block stores 256 values in 84 bytes (~2.63 bits each) — the smallest mainstream
// quant, letting the Q2_K mix fit very large models on tight hardware (at a real accuracy cost).
func ExampleQuantize_q2K() {
	w := tensor.FromFloat64(tensor.Shape{qkK}, randF32Block(qkK, 0))
	data, _ := Quantize(w, Q2_K)
	deq, _ := Dequantize(data, Q2_K, qkK)
	var maxErr, R float64
	R = q4kSuperRange(cloneTensorF64(w), 0)
	for i := range qkK {
		maxErr = math.Max(maxErr, math.Abs(float64(deq.Storage().F32()[i])-w.AtF64(i)))
	}
	fmt.Println(len(data), "bytes for", w.Numel(), "values; within R/4:", maxErr < R/4)
	// Output: 84 bytes for 256 values; within R/4: true
}
