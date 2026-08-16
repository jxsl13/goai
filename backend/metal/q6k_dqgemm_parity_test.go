//go:build darwin && cgo

package metal

import (
	"math"
	"testing"
)

// The dequant+GEMM prompt-processing path now serves Q6_K as well as Q4_K, because TinyLlama
// Q4_K_M is a MIXED file: 21 of its tensors are Q6_K and those include ffn_down on 10 of 22 layers.
// This checks the Q6_K expansion against the cooperative Q6_K kernel on the same weights.
func TestQ6KDequantGemmMatchesCooperative(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	for _, c := range []struct{ M, K, N int }{
		{32, 256, 32},
		{32, 512, 40}, // N not a multiple of 32 — partial tile
		{64, 2048, 64},
		{24, 768, 33},
	} {
		nb := c.K / 256
		raw := make([]byte, c.N*nb*210)
		for i := range raw {
			raw[i] = byte(i*29 + 13)
		}
		// The block scale d is a half at offset 208. A pseudo-random byte pair lands in the
		// NaN/Inf exponent range for some blocks, which makes BOTH paths produce NaN and the
		// comparison meaningless — write a valid small positive scale instead.
		for b := 0; b+210 <= len(raw); b += 210 {
			raw[b+208], raw[b+209] = 0x00, 0x34 // half 0.25, little-endian
		}
		x := make([]float32, c.M*c.K)
		for i := range x {
			x[i] = float32(math.Sin(float64(i)*0.19)) * 0.4
		}
		rw, err := Backend{}.UploadQuant(raw, 14, c.N, c.K)
		if err != nil {
			t.Skip(err)
		}
		rq := rw.(*ResidentQWeight)
		xb, _ := NewDeviceBufferF32(x)
		ob, _ := NewDeviceBufferF32(make([]float32, c.M*c.N))
		run := func() []float32 {
			r, err := NewRecorder()
			if err != nil {
				t.Fatal(err)
			}
			if err := r.QMatMulResident(xb, rq, ob, c.M); err != nil {
				t.Fatal(err)
			}
			r.Commit()
			r.Wait()
			r.Free()
			o := make([]float32, c.M*c.N)
			if err := ob.DownloadF32(o); err != nil {
				t.Fatal(err)
			}
			return o
		}
		SetQ4KDequantGemm(true)
		got := run()
		// Reference is the SCALAR kernel, not the cooperative one: the cooperative Q6_K kernel
		// accumulates partial sums per nibble group and multiplies by the scales afterwards,
		// while this path and the scalar kernel both dequantize per element. Comparing against
		// the cooperative form measures that regrouping rather than this kernel's correctness.
		SetQ4KDequantGemm(false)
		SetQ6KCooperative(false)
		want := run()
		SetQ6KCooperative(true)
		SetQ4KDequantGemm(true)
		var maxRel float64
		for i := range want {
			d := math.Abs(float64(got[i] - want[i]))
			if math.IsNaN(d) {
				t.Fatalf("M=%d K=%d N=%d: NaN at %d (got %v want %v)", c.M, c.K, c.N, i, got[i], want[i])
			}
			den := math.Max(1, math.Abs(float64(want[i])))
			if r := d / den; r > maxRel {
				maxRel = r
			}
		}
		// 1.5e-3: MPS GEMM and the scalar kernel sum the K terms in different orders, and the
		// difference grows with K (1.5e-4 at K=256, 9.5e-4 at K=2048). The EXPANSION itself is
		// verified exactly by TestQ6KDequantExact; this test guards the wiring and the shapes.
		if maxRel > 1.5e-3 {
			t.Errorf("M=%d K=%d N=%d: Q6_K dequant+GEMM vs cooperative max relative %.3e", c.M, c.K, c.N, maxRel)
		} else {
			t.Logf("M=%d K=%d N=%d: max relative %.3e", c.M, c.K, c.N, maxRel)
		}
		rq.Close()
		xb.Release()
		ob.Release()
	}
}
