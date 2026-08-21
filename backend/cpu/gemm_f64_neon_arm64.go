//go:build arm64 && goexperiment.simd

package cpu

import "sync"

// gemmF64Tile4x8Neon accumulates one 4x8 tile into C. The reduction over k is
// ascending and uses one fused FMLA chain per output, matching arm64's scalar
// FMADDD contraction exactly. k must be at least one.
//
//go:noescape
func gemmF64Tile4x8Neon(a, b, c *float64, k, lda, ldb, ldc int)

var f64GemmPackScratch = sync.Pool{New: func() any { b := make([]float64, 0); return &b }}

func getF64GemmPack(n int) *[]float64 {
	bp := f64GemmPackScratch.Get().(*[]float64)
	if cap(*bp) < n {
		*bp = make([]float64, n)
	} else {
		*bp = (*bp)[:n]
	}
	return bp
}

func putF64GemmPack(bp *[]float64) { f64GemmPackScratch.Put(bp) }

func gemmF64Band(A, B, C []float64, loRow, hiRow, k, n int) {
	gemmF64BandPortable(A, B, C, loRow, hiRow, k, n)
}

// gemmF64Full owns the complete operation, so B is packed exactly once before
// its shared panel buffer is consumed by every parallel row tile.
func gemmF64Full(A, B, C []float64, m, k, n int) bool {
	if m < 4 || k == 0 || n < 8 {
		return false
	}
	panels := n / 8
	packP := getF64GemmPack(panels * k * 8)
	defer putF64GemmPack(packP)
	packed := *packP
	parallelWork(panels, k*8, func(lo, hi int) {
		packBPanelsF64(B, packed, lo, hi, k, n)
	})

	fullRows := m - m%4
	rowTiles := fullRows / 4
	parallelWork(rowTiles, 4*k*n, func(lo, hi int) {
		for tile := lo; tile < hi; tile++ {
			i := tile * 4
			for panel := 0; panel < panels; panel++ {
				j := panel * 8
				gemmF64Tile4x8Neon(&A[i*k], &packed[panel*k*8], &C[i*n+j], k, k, 8, n)
			}
			gemmF64FourRowTail(A, B, C, i, k, n, panels*8)
		}
	})
	if fullRows < m {
		gemmF64BandPortable(A, B, C, fullRows, m, k, n)
	}
	return true
}

func packBPanelsF64(B, packed []float64, loPanel, hiPanel, k, n int) {
	for panel := loPanel; panel < hiPanel; panel++ {
		src := panel * 8
		dst := packed[panel*k*8 : (panel+1)*k*8]
		for p := 0; p < k; p++ {
			copy(dst[p*8:p*8+8], B[p*n+src:p*n+src+8])
		}
	}
}

func gemmF64FourRowTail(A, B, C []float64, i, k, n, jLo int) {
	if jLo == n {
		return
	}
	a0 := A[(i+0)*k : (i+1)*k]
	a1 := A[(i+1)*k : (i+2)*k]
	a2 := A[(i+2)*k : (i+3)*k]
	a3 := A[(i+3)*k : (i+4)*k]
	c0 := C[(i+0)*n : (i+1)*n]
	c1 := C[(i+1)*n : (i+2)*n]
	c2 := C[(i+2)*n : (i+3)*n]
	c3 := C[(i+3)*n : (i+4)*n]
	for j := jLo; j < n; j++ {
		v0, v1, v2, v3 := c0[j], c1[j], c2[j], c3[j]
		bo := j
		for p := 0; p < k; p++ {
			bv := B[bo]
			v0 += a0[p] * bv
			v1 += a1[p] * bv
			v2 += a2[p] * bv
			v3 += a3[p] * bv
			bo += n
		}
		c0[j], c1[j], c2[j], c3[j] = v0, v1, v2, v3
	}
}
