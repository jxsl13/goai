package gguf

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// QMatMul takes a fused single-token path when m==1: it folds the per-block dequant
// straight into the dot product instead of materializing the weight row. The claim on
// that path is that it is bit-for-bit the general path, and NOTHING held it — the
// package's QMatMul tests all compare against a float reference at 1e-5, which a
// reassociated accumulation passes comfortably.
//
// This gate uses the production code as its own oracle rather than a frozen copy: run
// the SAME activation row as m==1 (fused) and as row 0 of an m==2 call (general path).
// A hand-written reference would drift; this cannot, and it costs nothing to keep true.
func TestQMatMulFusedDecodeMatchesGeneralPathExactly(t *testing.T) {
	rng := rand.New(rand.NewSource(20260728))
	// k must be a multiple of the 32-element block. Cover one block, several blocks,
	// and a k large enough that the general path's scratch row leaves L1.
	for _, k := range []int{32, 64, 256, 4096} {
		for _, n := range []int{1, 5, 32} {
			// Wide exponent spread, so any change in accumulation order rounds
			// differently and is caught (a uniform [0,1) fill would hide it).
			w := tensor.New(tensor.F32, tensor.Shape{n, k})
			ws := w.Storage().F32()
			for i := range ws {
				ws[i] = float32(rng.NormFloat64() * math.Pow(2, float64(rng.Intn(9)-4)))
			}
			for _, qt := range []QuantType{Q8_0, Q4_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K} {
				if k%256 != 0 && qt != Q8_0 && qt != Q4_0 {
					continue // K-quants use a 256-element superblock
				}
				qb, err := Quantize(w, qt)
				if err != nil {
					t.Fatalf("quantize qt=%d k=%d: %v", qt, k, err)
				}
				x2 := tensor.New(tensor.F32, tensor.Shape{2, k})
				xs := x2.Storage().F32()
				for i := range k {
					v := float32(rng.NormFloat64() * math.Pow(2, float64(rng.Intn(9)-4)))
					xs[i], xs[k+i] = v, v // both rows identical; row 0 is the comparand
				}
				x1 := tensor.New(tensor.F32, tensor.Shape{1, k})
				copy(x1.Storage().F32(), xs[:k])

				fused, err := QMatMul(x1, qb, qt, n, k)
				if err != nil {
					t.Fatalf("fused qt=%d k=%d n=%d: %v", qt, k, n, err)
				}
				general, err := QMatMul(x2, qb, qt, n, k)
				if err != nil {
					t.Fatalf("general qt=%d k=%d n=%d: %v", qt, k, n, err)
				}
				gf, ff := general.Storage().F32(), fused.Storage().F32()
				// Q4_K's row dot is the tolerance-gated asm kernel on the simd build
				// (f32-vector accumulation vs the scalar f64) → within a tight ulp
				// tolerance, not bit-identical. Every other format stays exact.
				q4kTol := qt == Q4_K && q4kDotIsAsm
				for ni := range n {
					if q4kTol {
						if d := math.Abs(float64(ff[ni] - gf[ni])); d > 2e-3*math.Abs(float64(gf[ni]))+2e-3 {
							t.Fatalf("qt=Q4_K(asm) k=%d n=%d out[%d]: %v vs %v (|Δ|=%g)", k, n, ni, ff[ni], gf[ni], d)
						}
						continue
					}
					if math.Float32bits(ff[ni]) != math.Float32bits(gf[ni]) {
						t.Fatalf("qt=%d k=%d n=%d out[%d]: m==1 %v (%#x) != m==2 row 0 %v (%#x)"+
							" — the single-token path is not bit-identical to the general one",
							qt, k, n, ni, ff[ni], math.Float32bits(ff[ni]), gf[ni], math.Float32bits(gf[ni]))
					}
				}
			}
		}
	}
}
