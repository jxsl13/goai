package gguf

import (
	"bytes"
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// §R94: the f32→f16 encoder inverts f16ToF32 on f16-representable values and matches
// the canonical bit patterns; overflow saturates to inf.
func TestF32ToF16(t *testing.T) {
	cases := []struct {
		f    float32
		bits uint16
	}{
		{0, 0x0000}, {1, 0x3C00}, {-1, 0xBC00}, {0.5, 0x3800}, {2, 0x4000},
		{-2, 0xC000}, {65504, 0x7BFF}, {131072, 0x7C00}, // 2^17 overflows → inf
	}
	for _, c := range cases {
		if got := f32ToF16(c.f); got != c.bits {
			t.Errorf("f32ToF16(%v) = %#04x, want %#04x", c.f, got, c.bits)
		}
	}
	// round-trip: encoding then decoding an f16-representable value is exact
	for _, v := range []float32{0.15625, -3.5, 1024, 0.0001220703125, 100.5} {
		if back := f16ToF32(f32ToF16(v)); back != v {
			t.Errorf("f16 round-trip %v → %v", v, back)
		}
	}
}

func randF32Block(n, seed int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = math.Sin(float64(i*7+seed)*0.31+0.2) * 3
	}
	return x
}

// §V15 (§R94): Q8_0 quantize→dequantize stays within the block rounding bound
// (|err| ≤ d/2 ≈ amax/254) per 32-block.
func TestQuantizeQ8_0RoundTrip(t *testing.T) {
	const n = 96 // 3 blocks
	x := tensor.FromFloat64(tensor.Shape{n}, randF32Block(n, 1))
	data, err := Quantize(x, Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	deq, err := Dequantize(data, Q8_0, n)
	if err != nil {
		t.Fatal(err)
	}
	df := deq.Storage().F32()
	for b := range n / blockElems {
		var amax float64
		for i := range blockElems {
			amax = math.Max(amax, math.Abs(x.AtF64(b*blockElems+i)))
		}
		bound := amax/254 + amax*0.001 // rounding + f16 scale slack
		for i := range blockElems {
			k := b*blockElems + i
			if e := math.Abs(float64(df[k]) - x.AtF64(k)); e > bound {
				t.Errorf("Q8_0 block %d [%d]: error %g > bound %g", b, i, e, bound)
			}
		}
	}
}

// §V15 (§R94): Q4_0 quantize→dequantize stays within its (coarser) 4-bit bound.
func TestQuantizeQ4_0RoundTrip(t *testing.T) {
	const n = 64
	x := tensor.FromFloat64(tensor.Shape{n}, randF32Block(n, 2))
	data, err := Quantize(x, Q4_0)
	if err != nil {
		t.Fatal(err)
	}
	deq, _ := Dequantize(data, Q4_0, n)
	df := deq.Storage().F32()
	for b := range n / blockElems {
		var amax float64
		for i := range blockElems {
			amax = math.Max(amax, math.Abs(x.AtF64(b*blockElems+i)))
		}
		// Q4_0 is asymmetric (represents (nibble−8)∈[−8,7], range [−0.875·amax, amax]),
		// so an opposite-sign extreme adds up to ~amax/8 beyond the half-step → amax/4.
		bound := amax / 4
		for i := range blockElems {
			k := b*blockElems + i
			if e := math.Abs(float64(df[k]) - x.AtF64(k)); e > bound {
				t.Errorf("Q4_0 block %d [%d]: error %g > bound %g", b, i, e, bound)
			}
		}
	}
}

// §V15: values already on the quantization grid quantize and dequantize back exactly.
// The Q8_0 grid step is amax/127, so the values are step·q with integer q and one q at
// ±127 to pin amax (step is f16-exact, making the stored scale exact too).
func TestQuantizeOnGridExact(t *testing.T) {
	const step = 0.25 // f16-exact scale
	vals := make([]float64, blockElems)
	for i := range vals {
		vals[i] = step * float64(127-4*i) // q = 127,123,…,3 (|q| ≤ 127, max = 127)
	}
	x := tensor.FromFloat64(tensor.Shape{blockElems}, vals)
	data, _ := Quantize(x, Q8_0)
	deq, _ := Dequantize(data, Q8_0, blockElems)
	for i := range blockElems {
		if math.Abs(float64(deq.Storage().F32()[i])-vals[i]) > 1e-9 {
			t.Errorf("on-grid Q8_0 [%d]: %v != %v", i, deq.Storage().F32()[i], vals[i])
		}
	}
}

// §V15: re-quantizing a dequantized block yields byte-identical output — the grid is
// stable (idempotent), the strongest round-trip invariant.
func TestQuantizeIdempotent(t *testing.T) {
	for _, qt := range []QuantType{Q8_0, Q4_0} {
		x := tensor.FromFloat64(tensor.Shape{64}, randF32Block(64, 3))
		b1, _ := Quantize(x, qt)
		deq, _ := Dequantize(b1, qt, 64)
		b2, _ := Quantize(deq, qt)
		if !bytes.Equal(b1, b2) {
			t.Errorf("quant type %d not idempotent under dequant→requant", qt)
		}
	}
}

// §V15 / E2E: QMatMul over a freshly quantized weight approximates the full-precision
// matmul within the quantization error — Quantize produces QMatMul-compatible bytes.
func TestQuantizeQMatMul(t *testing.T) {
	const m, k, nOut = 2, 64, 3
	x := tensor.FromFloat64(tensor.Shape{m, k}, randF32Block(m*k, 4))
	wf := randF32Block(nOut*k, 5)
	w := tensor.FromFloat64(tensor.Shape{nOut, k}, wf)

	qw, err := Quantize(w, Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	got, err := QMatMul(x, qw, Q8_0, nOut, k)
	if err != nil {
		t.Fatal(err)
	}
	for mi := range m {
		for ni := range nOut {
			var want float64
			for ki := range k {
				want += x.AtF64(mi, ki) * wf[ni*k+ki]
			}
			if e := math.Abs(got.AtF64(mi, ni) - want); e > 0.05*math.Max(1, math.Abs(want)) {
				t.Errorf("QMatMul[%d,%d] = %g, full-precision %g (err %g)", mi, ni, got.AtF64(mi, ni), want, e)
			}
		}
	}
}

// FuzzQuantizeRoundTrip: any float block quantizes and dequantizes within the Q8_0
// rounding bound, and never panics (§V15).
func FuzzQuantizeRoundTrip(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4})
	f.Add([]byte("quantize-fuzz-seed"))
	f.Fuzz(func(t *testing.T, data []byte) {
		vals := make([]float64, blockElems)
		for i := range vals {
			if i < len(data) {
				vals[i] = float64(int8(data[i])) * 0.7
			}
		}
		x := tensor.FromFloat64(tensor.Shape{blockElems}, vals)
		b, err := Quantize(x, Q8_0)
		if err != nil {
			t.Fatal(err)
		}
		deq, err := Dequantize(b, Q8_0, blockElems)
		if err != nil {
			t.Fatal(err)
		}
		var amax float64
		for _, v := range vals {
			amax = math.Max(amax, math.Abs(v))
		}
		bound := amax/254 + amax*0.001 + 1e-9
		for i := range blockElems {
			if e := math.Abs(float64(deq.Storage().F32()[i]) - vals[i]); e > bound {
				t.Fatalf("[%d]: error %g > bound %g", i, e, bound)
			}
		}
	})
}

// A weight tensor can be quantized to the ggml Q8_0 layout and read back, trading a
// small rounding error for 4× smaller storage.
func ExampleQuantize() {
	w := tensor.FromFloat64(tensor.Shape{32}, randF32Block(32, 0))
	data, _ := Quantize(w, Q8_0)
	deq, _ := Dequantize(data, Q8_0, 32)
	fmt.Println(len(data), "bytes for", w.Numel(), "values; dequant close:",
		math.Abs(float64(deq.Storage().F32()[0])-w.AtF64(0)) < 0.05)
	// Output: 34 bytes for 32 values; dequant close: true
}
