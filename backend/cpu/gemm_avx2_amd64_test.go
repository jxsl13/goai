//go:build amd64 && goexperiment.simd

package cpu

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestGemmF32Tile6x16AVX2 checks the raw asm microkernel against a naive triple loop, including a
// non-multiple K and a strided (unpacked) B/C.
func TestGemmF32Tile6x16AVX2(t *testing.T) {
	const k, lda, ldb, ldc = 13, 20, 40, 48 // k not a multiple of anything; padded strides
	a := make([]float32, 6*lda)
	b := make([]float32, k*ldb)
	c := make([]float32, 6*ldc)
	for i := range a {
		a[i] = float32((i*7)%11) - 5
	}
	for i := range b {
		b[i] = float32((i*5)%13)*0.3 - 1
	}
	gemmF32Tile6x16AVX2(&a[0], &b[0], &c[0], k, lda, ldb, ldc)
	for r := 0; r < 6; r++ {
		for j := 0; j < 16; j++ {
			var want float32
			for p := 0; p < k; p++ {
				want += a[r*lda+p] * b[p*ldb+j]
			}
			if d := math.Abs(float64(c[r*ldc+j] - want)); d > 1e-3 {
				t.Fatalf("C[%d,%d]=%v want %v (Δ%.2e)", r, j, c[r*ldc+j], want, d)
			}
		}
	}
}

// TestGemmF32Asm6x16KcBlocked exercises the kc-blocked path (k > gemmAsmKC and k·n large) against a
// naive f64 reference — the smaller-shape tests stay in the single-pass branch.
func TestGemmF32Asm6x16KcBlocked(t *testing.T) {
	be, _ := backend.Get(backend.CPU)
	ctx := backend.NewContext().WithBackend(be)
	const m, k, n = 18, 300, 8000 // k>128, k*n=2.4M>2M → kc-blocked
	a := tensor.Randn(tensor.F32, 7, tensor.Shape{m, k})
	b := tensor.Randn(tensor.F32, 8, tensor.Shape{k, n})
	got, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{a, b}, nil)
	if err != nil {
		t.Fatal(err)
	}
	af, bf, cf := a.Storage().F32(), b.Storage().F32(), got[0].Storage().F32()
	for _, ij := range [][2]int{{0, 0}, {5, 17}, {11, 7999}, {17, 4000}, {6, 16}, {12, 15}, {0, 7999}} {
		i, j := ij[0], ij[1]
		var want float64
		for p := 0; p < k; p++ {
			want += float64(af[i*k+p]) * float64(bf[p*n+j])
		}
		tol := 5e-5 * math.Max(1, math.Abs(want))
		if d := math.Abs(float64(cf[i*n+j]) - want); d > tol {
			t.Fatalf("C[%d,%d]=%v want %v (Δ%.2e > %.2e)", i, j, cf[i*n+j], want, d, tol)
		}
	}
}

// TestGemmF32Asm6x16Edges exercises the driver across shapes that hit the m%6 and n%16 remainder
// fallbacks, comparing OpMatMul (which routes here) against a naive f64-accumulated reference.
func TestGemmF32Asm6x16Edges(t *testing.T) {
	be, _ := backend.Get(backend.CPU)
	ctx := backend.NewContext().WithBackend(be)
	for _, sh := range [][3]int{{6, 8, 16}, {13, 9, 20}, {19, 7, 33}, {6, 1, 16}, {25, 11, 48}} {
		m, k, n := sh[0], sh[1], sh[2]
		a := tensor.Randn(tensor.F32, 1, tensor.Shape{m, k})
		b := tensor.Randn(tensor.F32, 2, tensor.Shape{k, n})
		got, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{a, b}, nil)
		if err != nil {
			t.Fatalf("%v: %v", sh, err)
		}
		af, bf, cf := a.Storage().F32(), b.Storage().F32(), got[0].Storage().F32()
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				var want float64
				for p := 0; p < k; p++ {
					want += float64(af[i*k+p]) * float64(bf[p*n+j])
				}
				tol := 5e-5 * math.Max(1, math.Abs(want))
				if d := math.Abs(float64(cf[i*n+j]) - want); d > tol {
					t.Fatalf("shape %v C[%d,%d]=%v want %v (Δ%.2e > %.2e)", sh, i, j, cf[i*n+j], want, d, tol)
				}
			}
		}
	}
}
