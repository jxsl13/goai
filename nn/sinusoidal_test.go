package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §V16 tier-1 (arXiv:1706.03762 §3.5, pos=0): every angle is 0, so row 0 is [sin 0, cos 0, …] =
// [0, 1, 0, 1, …].
func TestSinusoidalRow0(t *testing.T) {
	pe, err := nn.SinusoidalPositionalEncoding(3, 6, 10000, tensor.F64)
	if err != nil {
		t.Fatal(err)
	}
	for j := 0; j < 6; j++ {
		want := 0.0
		if j%2 == 1 {
			want = 1.0 // cos 0
		}
		if math.Abs(pe.AtF64(0, j)-want) > 1e-12 {
			t.Errorf("PE[0,%d] = %g, want %g", j, pe.AtF64(0, j), want)
		}
	}
}

// §V16 tier-1: PE(pos,2i)=sin(pos/base^(2i/d)), PE(pos,2i+1)=cos(…). Verified against a hand
// computation for d=4 (frequencies 1 and 1/100) at pos=1.
func TestSinusoidalParity(t *testing.T) {
	pe, err := nn.SinusoidalPositionalEncoding(2, 4, 10000, tensor.F64)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{math.Sin(1), math.Cos(1), math.Sin(0.01), math.Cos(0.01)}
	for j, w := range want {
		if math.Abs(pe.AtF64(1, j)-w) > 1e-12 {
			t.Errorf("PE[1,%d] = %.12g, want %.12g", j, pe.AtF64(1, j), w)
		}
	}
}

// §V16 tier-1 (interleaved sin/cos share a frequency): for every position and pair, the even (sin)
// and odd (cos) features satisfy sin²+cos²=1 — proving 2i and 2i+1 use the SAME frequency and are
// interleaved, exactly as the paper's equation.
func TestSinusoidalPairingInterleaved(t *testing.T) {
	pe, _ := nn.SinusoidalPositionalEncoding(20, 16, 10000, tensor.F64)
	for pos := 0; pos < 20; pos++ {
		for i := 0; i < 8; i++ {
			s, c := pe.AtF64(pos, 2*i), pe.AtF64(pos, 2*i+1)
			if math.Abs(s*s+c*c-1) > 1e-12 {
				t.Errorf("pair (%d,%d) at pos %d: sin²+cos²=%g, want 1", 2*i, 2*i+1, pos, s*s+c*c)
			}
		}
	}
}

// §V16 tier-1 (bounded): all entries lie in [−1,1].
func TestSinusoidalBounded(t *testing.T) {
	pe, _ := nn.SinusoidalPositionalEncoding(50, 32, 10000, tensor.F64)
	for i := range pe.Numel() {
		v := pe.AtF64(tensor.Unravel(i, pe.Shape())...)
		if v < -1-1e-12 || v > 1+1e-12 {
			t.Fatalf("entry %g out of [−1,1]", v)
		}
	}
}

// §V2: shape and the base≤0 default (10000).
func TestSinusoidalShapeAndBaseDefault(t *testing.T) {
	pe, _ := nn.SinusoidalPositionalEncoding(7, 8, 10000, tensor.F64)
	if !pe.Shape().Equal(tensor.Shape{7, 8}) {
		t.Errorf("shape %v, want [7 8]", pe.Shape())
	}
	def, _ := nn.SinusoidalPositionalEncoding(7, 8, 0, tensor.F64) // base≤0 → 10000
	for i := range pe.Numel() {
		c := tensor.Unravel(i, pe.Shape())
		if pe.AtF64(c...) != def.AtF64(c...) {
			t.Fatalf("base=0 default differs from base=10000 at %v", c)
		}
	}
}

func TestSinusoidalErrors(t *testing.T) {
	if _, err := nn.SinusoidalPositionalEncoding(4, 5, 10000, tensor.F64); err == nil {
		t.Error("odd dModel: want error")
	}
	if _, err := nn.SinusoidalPositionalEncoding(0, 4, 10000, tensor.F64); err == nil {
		t.Error("seqLen 0: want error")
	}
	if _, err := nn.SinusoidalPositionalEncoding(4, 0, 10000, tensor.F64); err == nil {
		t.Error("dModel 0: want error")
	}
}

func ExampleSinusoidalPositionalEncoding() {
	pe, _ := nn.SinusoidalPositionalEncoding(4, 8, 10000, tensor.F64)
	// Row 0 (position 0) is [sin 0, cos 0, …] = alternating 0, 1.
	fmt.Printf("%.0f %.0f %.0f %.0f\n", pe.AtF64(0, 0), pe.AtF64(0, 1), pe.AtF64(0, 2), pe.AtF64(0, 3))
	// Output:
	// 0 1 0 1
}

// §V16 tier-1 (§R230, channel permutation): the concat layout is exactly a permutation of the
// interleaved table — concat[:,i]=interleaved[:,2i] (sines) and concat[:,half+i]=interleaved[:,2i+1]
// (cosines). This proves the concat variant is equivalent to the already-verified interleaved one.
func TestSinusoidalConcatIsPermutation(t *testing.T) {
	const seqLen, dModel = 12, 16
	half := dModel / 2
	inter, err := nn.SinusoidalPositionalEncoding(seqLen, dModel, 10000, tensor.F64)
	if err != nil {
		t.Fatal(err)
	}
	con, err := nn.SinusoidalPositionalEncodingConcat(seqLen, dModel, 10000, tensor.F64)
	if err != nil {
		t.Fatal(err)
	}
	for pos := 0; pos < seqLen; pos++ {
		for i := 0; i < half; i++ {
			if math.Abs(con.AtF64(pos, i)-inter.AtF64(pos, 2*i)) > 1e-15 {
				t.Errorf("concat[%d,%d] != interleaved[%d,%d] (sines)", pos, i, pos, 2*i)
			}
			if math.Abs(con.AtF64(pos, half+i)-inter.AtF64(pos, 2*i+1)) > 1e-15 {
				t.Errorf("concat[%d,%d] != interleaved[%d,%d] (cosines)", pos, half+i, pos, 2*i+1)
			}
		}
	}
}

// §V16 tier-1 (concat row 0): pos 0 ⇒ first half all sin(0)=0, second half all cos(0)=1.
func TestSinusoidalConcatRow0(t *testing.T) {
	pe, _ := nn.SinusoidalPositionalEncodingConcat(3, 6, 10000, tensor.F64)
	for j := 0; j < 3; j++ {
		if pe.AtF64(0, j) != 0 {
			t.Errorf("concat[0,%d]=%g, want 0 (sine)", j, pe.AtF64(0, j))
		}
		if pe.AtF64(0, 3+j) != 1 {
			t.Errorf("concat[0,%d]=%g, want 1 (cosine)", 3+j, pe.AtF64(0, 3+j))
		}
	}
}

// §V2 (bounds + errors).
func TestSinusoidalConcatBoundedAndErrors(t *testing.T) {
	pe, _ := nn.SinusoidalPositionalEncodingConcat(30, 8, 10000, tensor.F64)
	for i := range pe.Numel() {
		if v := pe.AtF64(tensor.Unravel(i, pe.Shape())...); v < -1-1e-12 || v > 1+1e-12 {
			t.Fatalf("entry %g out of [−1,1]", v)
		}
	}
	if _, err := nn.SinusoidalPositionalEncodingConcat(4, 5, 10000, tensor.F64); err == nil {
		t.Error("odd dModel: want error")
	}
	if _, err := nn.SinusoidalPositionalEncodingConcat(0, 4, 10000, tensor.F64); err == nil {
		t.Error("seqLen 0: want error")
	}
}

func ExampleSinusoidalPositionalEncodingConcat() {
	pe, _ := nn.SinusoidalPositionalEncodingConcat(4, 8, 10000, tensor.F64)
	// Row 0 = [sin 0 ×4, cos 0 ×4] = [0 0 0 0 1 1 1 1].
	for j := 0; j < 8; j++ {
		fmt.Printf("%.0f ", pe.AtF64(0, j))
	}
	fmt.Println()
	// Output:
	// 0 0 0 0 1 1 1 1
}

// BenchmarkSinusoidalPE covers both the interleaved and concatenated layouts at a representative
// transformer size (seqLen=512, dModel=512 → 256 frequency terms per row). The per-frequency
// wavelength 1/base^(2i/dModel) is invariant across positions, so a naive per-(pos,i) math.Pow
// re-evaluates it seqLen× redundantly — this benchmark guards the loop-invariant hoist.
func BenchmarkSinusoidalPE(b *testing.B) {
	const seqLen, dModel = 512, 512
	b.Run("interleaved/f64", func(b *testing.B) {
		b.ReportAllocs()
		for range b.N {
			if _, err := nn.SinusoidalPositionalEncoding(seqLen, dModel, 10000, tensor.F64); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("interleaved/f32", func(b *testing.B) {
		b.ReportAllocs()
		for range b.N {
			if _, err := nn.SinusoidalPositionalEncoding(seqLen, dModel, 10000, tensor.F32); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("concat/f64", func(b *testing.B) {
		b.ReportAllocs()
		for range b.N {
			if _, err := nn.SinusoidalPositionalEncodingConcat(seqLen, dModel, 10000, tensor.F64); err != nil {
				b.Fatal(err)
			}
		}
	})
}
