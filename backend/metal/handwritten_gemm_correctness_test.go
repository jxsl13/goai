//go:build darwin && cgo

package metal

// TestHandwrittenGEMMIsCorrect verifies that sg_gemm_f16 computes a matmul at all.
//
// It was measured for four commits before this test existed. The timing probe allocates PRIVATE
// buffers and never reads the result, so it would have reported a fast wrong kernel indefinitely —
// and every tile-sweep conclusion drawn from it (BN=32 optimum, register blocking harmful, 0.51x of
// MPS) rests on the arithmetic being right. Checking it late is a process failure; leaving it
// unchecked would have been a worse one.
//
// Against an f64 CPU reference, with shapes that exercise partial tiles in both dimensions
// (BM=64, BN=32, so M=96 and N=48 leave remainders):
//
//	M= 64 K=  64 N= 32   maxRel 5.513e-03
//	M= 64 K=2048 N= 64   maxRel 3.714e-02
//	M= 96 K= 128 N= 48   maxRel 8.905e-03
//	M=512 K= 256 N=128   maxRel 1.674e-02
//
// No NaN, and the remainder tiles are right. The error scale is f16 input precision rather than a
// defect: the kernel converts A and B to half, so each term carries ~1e-3 relative error, and the
// sum of K such terms grows with K — which is exactly the ordering seen (K=2048 worst, K=64 best).
// The tolerance is set at 5e-2 to accept that while still catching a genuine indexing error, which
// would produce order-1 relative error rather than order-1e-2.
import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestHandwrittenGEMMIsCorrect(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	rng := rand.New(rand.NewSource(9))
	for _, c := range []struct{ m, k, n int }{{64, 64, 32}, {64, 2048, 64}, {96, 128, 48}, {512, 256, 128}} {
		a := make([]float32, c.m*c.k)
		b := make([]float32, c.k*c.n)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		for i := range b {
			b[i] = float32(rng.NormFloat64())
		}
		got, err := CheckSGGemm(a, b, c.m, c.k, c.n)
		if err != nil {
			t.Fatal(err)
		}
		maxRel, bad := 0.0, 0
		for i := 0; i < c.m; i++ {
			for j := 0; j < c.n; j++ {
				var ref float64
				for kk := 0; kk < c.k; kk++ {
					ref += float64(a[i*c.k+kk]) * float64(b[kk*c.n+j])
				}
				g := float64(got[i*c.n+j])
				if math.IsNaN(g) {
					bad++
					continue
				}
				if r := math.Abs(g-ref) / math.Max(1, math.Abs(ref)); r > maxRel {
					maxRel = r
				}
			}
		}
		fmt.Printf("GCHK M=%3d K=%4d N=%3d  maxRel=%.3e  NaN=%d\n", c.m, c.k, c.n, maxRel, bad)
		if maxRel > 5e-2 || bad > 0 {
			t.Errorf("M=%d K=%d N=%d: hand-written GEMM is WRONG (maxRel %.3e, NaN %d)", c.m, c.k, c.n, maxRel, bad)
		}
	}
}
