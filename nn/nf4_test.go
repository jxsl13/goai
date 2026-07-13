package nn_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// §V16 tier-1: the NF4 codebook matches the bitsandbytes reference exactly and has
// the defining structure (sorted, zero at index 7, ±1 endpoints).
func TestNF4Codebook(t *testing.T) {
	want := []float64{
		-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
		-0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
		0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
		0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
	}
	for i, w := range want {
		if nn.NF4Code(i) != w {
			t.Errorf("NF4Code(%d) = %v, want %v", i, nn.NF4Code(i), w)
		}
	}
	if nn.NF4Code(7) != 0.0 {
		t.Error("index 7 must be exactly 0.0")
	}
	if nn.NF4Code(0) != -1.0 || nn.NF4Code(15) != 1.0 {
		t.Error("endpoints must be ±1.0")
	}
	for i := 1; i < 16; i++ { // strictly ascending
		if nn.NF4Code(i) <= nn.NF4Code(i-1) {
			t.Errorf("codebook not sorted at %d", i)
		}
	}
}

// maxHalfGap is the largest quantization error a block-normalized value can incur
// (half the widest gap between adjacent codes).
func maxHalfGap() float64 {
	var g float64
	for i := 1; i < 16; i++ {
		if d := nn.NF4Code(i) - nn.NF4Code(i-1); d > g {
			g = d
		}
	}
	return g / 2
}

// §V15: NF4 quantize→dequantize snaps each value to absmax·nearest-code within the
// half-gap bound, and re-quantizing the dequantized values is idempotent (same
// packed indices and scales).
func TestNF4RoundTrip(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	w := make([]float64, 200)
	for i := range w {
		w[i] = rng.NormFloat64() * 0.5
	}
	for _, bs := range []int{1, 16, 64, 200} {
		packed, absmax := nn.QuantizeNF4(w, bs)
		hat := nn.DequantizeNF4(packed, absmax, bs, len(w))

		bound := maxHalfGap()
		for i := range w {
			am := absmax[i/bs]
			if math.Abs(w[i]-hat[i]) > am*bound+1e-12 {
				t.Errorf("bs=%d w[%d]=%v hat=%v exceeds bound %v", bs, i, w[i], hat[i], am*bound)
			}
		}
		// idempotent: re-quantizing the reconstruction reproduces packed + scales
		p2, a2 := nn.QuantizeNF4(hat, bs)
		for i := range packed {
			if p2[i] != packed[i] {
				t.Fatalf("bs=%d not idempotent: packed[%d] %d != %d", bs, i, p2[i], packed[i])
			}
		}
		for i := range absmax {
			if math.Abs(a2[i]-absmax[i]) > 1e-12 {
				t.Fatalf("bs=%d absmax[%d] %v != %v", bs, i, a2[i], absmax[i])
			}
		}
	}
}

// §V15 fuzz: no hostile input panics; round-trip stays within the half-gap bound
// and is idempotent for any weights and block size.
func FuzzNF4RoundTrip(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4, 5, 6, 7, 8}, uint8(4))
	f.Fuzz(func(t *testing.T, data []byte, bsRaw uint8) {
		if len(data) == 0 {
			return
		}
		w := make([]float64, len(data))
		for i, b := range data {
			w[i] = (float64(b) - 128) / 40 // spread around 0
		}
		bs := int(bsRaw)%64 + 1
		packed, absmax := nn.QuantizeNF4(w, bs)
		hat := nn.DequantizeNF4(packed, absmax, bs, len(w))
		bound := maxHalfGap()
		for i := range w {
			if math.Abs(w[i]-hat[i]) > absmax[i/bs]*bound+1e-9 {
				t.Fatalf("w[%d]=%v hat=%v exceeds bound", i, w[i], hat[i])
			}
		}
		p2, _ := nn.QuantizeNF4(hat, bs)
		for i := range packed {
			if p2[i] != packed[i] {
				t.Fatalf("not idempotent at %d", i)
			}
		}
	})
}

// QLoRA: with B=0 the adapter is a no-op, so the forward equals the (NF4-quantized)
// base matmul, and only the adapter receives gradients — the base is frozen.
func TestQLoRALinear(t *testing.T) {
	w := tensor.FromFloat64(tensor.Shape{4, 3}, []float64{
		0.5, -0.5, 1.0, -1.0, 0.1, 0.0, 0.25, -0.3, 0.7, -0.7, 0.4, -0.2,
	})
	q, err := nn.NewQLoRA(w, 6 /*blockSize*/, 2 /*rank*/, 4 /*alpha*/, 1)
	if err != nil {
		t.Fatal(err)
	}
	x := tensor.FromFloat64(tensor.Shape{2, 4}, []float64{1, 0, -1, 0.5, 0.3, 2, -0.3, 1})

	y, err := q.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	// B=0 → y == x · dequant(base)
	base, err := backend.Execute(backend.NewContext(), backend.OpMatMul, []*tensor.Tensor{x, q.Base(tensor.F64)}, nil)
	if err != nil {
		t.Fatal(err)
	}
	for k := range y.Numel() {
		idx := tensor.Unravel(k, y.Shape())
		if math.Abs(y.AtF64(idx...)-base[0].AtF64(idx...)) > 1e-12 {
			t.Errorf("QLoRA[%d] = %v, want base %v (B=0 no-op)", k, y.AtF64(idx...), base[0].AtF64(idx...))
		}
	}
	// only A, B are trainable
	if ps := q.Params(); len(ps) != 2 || ps[0] != q.A || ps[1] != q.B {
		t.Error("QLoRA must expose only the adapter A, B as params")
	}
}

// NF4 stores each weight in 4 bits: eight weights pack into four bytes plus one
// block scale, and dequantizing snaps each value to absmax·nearest-code. Here the
// block absmax is 1.0, so the value 1.0 comes back exactly.
func ExampleQuantizeNF4() {
	w := []float64{0.5, -0.5, 1.0, -1.0, 0.1, 0.0, 0.25, -0.3}
	packed, absmax := nn.QuantizeNF4(w, 8)
	back := nn.DequantizeNF4(packed, absmax, 8, len(w))
	fmt.Printf("%d weights -> %d bytes + 1 scale; w_hat[2]=%.3f\n", len(w), len(packed), back[2])
	// Output: 8 weights -> 4 bytes + 1 scale; w_hat[2]=1.000
}

// §V15: double-quantizing the NF4 scales round-trips within 8-bit precision and is
// stable under a second pass (offset re-centering makes it near-, not exactly,
// idempotent).
func TestNF4DoubleQuantRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	scales := make([]float64, 500)
	for i := range scales {
		scales[i] = math.Abs(rng.NormFloat64())*0.5 + 0.01 // positive, absmax-like
	}
	for _, bs := range []int{64, 256, 500} {
		codes, am2, off := nn.QuantizeScalesNF4(scales, bs)
		hat := nn.DequantizeScalesNF4(codes, am2, off, bs, len(scales))
		for i := range scales {
			if bound := am2[i/bs]/127 + 1e-12; math.Abs(scales[i]-hat[i]) > bound {
				t.Errorf("bs=%d scale[%d]=%v hat=%v exceeds %v", bs, i, scales[i], hat[i], bound)
			}
		}
		c2, a2, o2 := nn.QuantizeScalesNF4(hat, bs) // second pass barely drifts
		hat2 := nn.DequantizeScalesNF4(c2, a2, o2, bs, len(scales))
		for i := range hat {
			if math.Abs(hat[i]-hat2[i]) > am2[i/bs]/127+1e-9 {
				t.Errorf("bs=%d scale-quant not stable at %d", bs, i)
			}
		}
	}
}

// §V15 fuzz: no input panics; the scale round-trip stays within the 8-bit bound.
func FuzzNF4DoubleQuant(f *testing.F) {
	f.Add([]byte{10, 40, 90, 5, 200}, uint16(256))
	f.Fuzz(func(t *testing.T, data []byte, bsRaw uint16) {
		if len(data) == 0 {
			return
		}
		scales := make([]float64, len(data))
		for i, b := range data {
			scales[i] = float64(b)/50 + 0.001 // positive
		}
		bs := int(bsRaw)%512 + 1
		codes, am2, off := nn.QuantizeScalesNF4(scales, bs)
		hat := nn.DequantizeScalesNF4(codes, am2, off, bs, len(scales))
		for i := range scales {
			if math.Abs(scales[i]-hat[i]) > am2[i/bs]/127+1e-9 {
				t.Fatalf("scale[%d] exceeds bound", i)
			}
		}
	})
}

// End-to-end: NF4 weight quantization then double-quantizing its scales adds only a
// small extra error on top of plain NF4, while shrinking the scale storage ~8×.
func TestNF4DoubleQuantEndToEnd(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	w := make([]float64, 640) // 10 blocks of 64
	for i := range w {
		w[i] = rng.NormFloat64()
	}
	packed, absmax := nn.QuantizeNF4(w, 64) // 10 fp64 scales
	codes, am2, off := nn.QuantizeScalesNF4(absmax, 256)
	if len(codes) != 10 || len(am2) != 1 {
		t.Fatalf("sizes: codes %d am2 %d, want 10 / 1", len(codes), len(am2))
	}
	absmaxHat := nn.DequantizeScalesNF4(codes, am2, off, 256, len(absmax))

	whatPlain := nn.DequantizeNF4(packed, absmax, 64, len(w))
	whatDQ := nn.DequantizeNF4(packed, absmaxHat, 64, len(w))
	var maxExtra float64
	for i := range w {
		if d := math.Abs(whatDQ[i] - whatPlain[i]); d > maxExtra {
			maxExtra = d
		}
	}
	if maxExtra > 0.1 {
		t.Errorf("double-quant added %v extra error (too large)", maxExtra)
	}
}

// Double quantization (QLoRA §3) shrinks the NF4 block scales from fp64 to ~1 byte
// each: 300 scales become 300 one-byte codes plus a couple of second-level
// constants, versus 300×8 bytes of fp64.
func ExampleQuantizeScalesNF4() {
	scales := make([]float64, 300)
	for i := range scales {
		scales[i] = 0.5 + float64(i%7)*0.01
	}
	codes, am2, _ := nn.QuantizeScalesNF4(scales, 256)
	fmt.Printf("%d scales -> %d code bytes + %d block constants\n", len(scales), len(codes), len(am2))
	// Output: 300 scales -> 300 code bytes + 2 block constants
}
