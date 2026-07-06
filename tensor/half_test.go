package tensor

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

type halfGolden struct {
	X        []float64 `json:"x"`
	F16Bits  []uint16  `json:"f16_bits"`
	F16Back  []float64 `json:"f16_back"`
	BF16Bits []uint16  `json:"bf16_bits"`
	BF16Back []float64 `json:"bf16_back"`
}

func loadHalf(t *testing.T) halfGolden {
	t.Helper()
	raw, err := os.ReadFile("testdata/half.json")
	if err != nil {
		t.Fatalf("read golden (run make golden): %v", err)
	}
	var g halfGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

// §V1: f32→f16 bits match numpy.float16 exactly (round-to-nearest-even), and the
// f16→f32 round-trip matches numpy's widening.
func TestF16BitParity(t *testing.T) {
	g := loadHalf(t)
	for i, x := range g.X {
		got := f32ToF16(float32(x))
		if got != g.F16Bits[i] {
			t.Errorf("f32ToF16(%g): bits 0x%04X, want 0x%04X", x, got, g.F16Bits[i])
		}
		back := float64(f16ToF32(g.F16Bits[i]))
		if back != g.F16Back[i] {
			t.Errorf("f16ToF32(0x%04X) = %g, want %g", g.F16Bits[i], back, g.F16Back[i])
		}
	}
}

// §V1: f32→bf16 bits match torch.bfloat16 exactly (round-to-nearest-even, NOT
// truncation), and bf16→f32 is the exact left-shift widening.
func TestBF16BitParity(t *testing.T) {
	g := loadHalf(t)
	if len(g.BF16Bits) == 0 {
		t.Skip("bf16 golden absent (torch missing at generation)")
	}
	for i, x := range g.X {
		got := f32ToBF16(float32(x))
		if got != g.BF16Bits[i] {
			t.Errorf("f32ToBF16(%g): bits 0x%04X, want 0x%04X", x, got, g.BF16Bits[i])
		}
		back := float64(bf16ToF32(g.BF16Bits[i]))
		if back != g.BF16Back[i] {
			t.Errorf("bf16ToF32(0x%04X) = %g, want %g", g.BF16Bits[i], back, g.BF16Back[i])
		}
	}
}

// bf16 must round-to-nearest-even, distinguishable from truncation. 1.0009766 has
// f32 bits 0x3F808000; truncation → 0x3F80 (1.0), RNE → 0x3F81 (rounds up).
func TestBF16RoundNotTruncate(t *testing.T) {
	// pick a value whose low 16 bits are exactly 0x8000 with an odd-ish successor
	f := math.Float32frombits(0x3F808001) // just above the tie → rounds up
	if got := f32ToBF16(f); got != 0x3F81 {
		t.Errorf("RNE up: got 0x%04X want 0x3F81", got)
	}
	// exact tie 0x3F808000: RNE to even → 0x3F80 (even mantissa), not 0x3F81
	tie := math.Float32frombits(0x3F808000)
	if got := f32ToBF16(tie); got != 0x3F80 {
		t.Errorf("RNE tie-to-even: got 0x%04X want 0x3F80", got)
	}
}

// bf16 preserves NaN as NaN (never turns it into ±Inf).
func TestBF16NaNPreserved(t *testing.T) {
	nan := float32(math.NaN())
	bits := f32ToBF16(nan)
	got := bf16ToF32(bits)
	if !math.IsNaN(float64(got)) {
		t.Errorf("bf16 NaN not preserved: 0x%04X → %g", bits, got)
	}
}

// §V1: tensor-level Cast to F16/BF16 goes through the same conversion and is
// reversible within the format's precision.
func TestCastHalfRoundTrip(t *testing.T) {
	g := loadHalf(t)
	src := FromFloat64(Shape{len(g.X)}, g.X)
	for _, dt := range []Dtype{F16, BF16} {
		if dt == BF16 && len(g.BF16Bits) == 0 {
			continue
		}
		h := src.Cast(dt)
		if h.Dtype() != dt {
			t.Fatalf("cast dtype = %v, want %v", h.Dtype(), dt)
		}
		want := g.F16Back
		if dt == BF16 {
			want = g.BF16Back
		}
		for i := range g.X {
			if v := h.AtF64(i); v != want[i] {
				t.Errorf("%v cast[%d] = %g, want %g", dt, i, v, want[i])
			}
		}
	}
}
