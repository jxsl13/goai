//go:build arm64 && goexperiment.simd

package cpu

import "sync"

// gemmF64Tile4x8Neon accumulates one 4x8 tile into C. The reduction over k is
// ascending and uses one fused FMLA chain per output, matching arm64's scalar
// FMADDD contraction exactly. k must be at least one.
//
//go:noescape
func gemmF64Tile4x8Neon(a, b, c *float64, k, lda, ldb, ldc int)

// f64GemmPackScratch is not cleared on get: packBPanelsF64 overwrites every
// element before the NEON tile reads it.
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

// gemmF64Band packs one L2-sized column block per parallel row band. Packing
// inside the band duplicates one read of B per worker, but keeps scratch near
// 256 KiB and is amortized over every 4-row tile owned by that worker.
func gemmF64Band(A, B, C []float64, loRow, hiRow, k, n int) {
	if k == 0 || loRow >= hiRow || n == 0 {
		return
	}
	if hiRow-loRow < 4 || n < 8 {
		gemmF64BandPortable(A, B, C, loRow, hiRow, k, n)
		return
	}

	const l2Target = 256 << 10
	nc := n
	if k*n*8 > l2Target {
		nc = l2Target / (8 * k) / 8 * 8
		if nc < 8 {
			nc = 8
		}
	}
	packCols := nc
	if packCols > n-n%8 {
		packCols = n - n%8
	}
	packP := getF64GemmPack(k * packCols)
	defer putF64GemmPack(packP)

	for jb := 0; jb < n; jb += nc {
		jHi := min(jb+nc, n)
		nv8 := jHi - (jHi-jb)%8
		packed := (*packP)[:(nv8-jb)*k]
		packBPanelsF64(B, packed, jb, nv8, k, n)
		gemmF64BandNeonCols(A, B, packed, C, loRow, hiRow, k, n, jb, nv8, jHi)
	}
}

// packBPanelsF64 copies [jLo,jHi) in 8-column panels, producing
// [panel][k][8]. Reads are strided once; every tile then streams packed B.
func packBPanelsF64(B, packed []float64, jLo, jHi, k, n int) {
	panels := (jHi - jLo) / 8
	for panel := 0; panel < panels; panel++ {
		src := jLo + panel*8
		dst := packed[panel*k*8 : (panel+1)*k*8]
		for p := 0; p < k; p++ {
			copy(dst[p*8:p*8+8], B[p*n+src:p*n+src+8])
		}
	}
}

func gemmF64BandNeonCols(A, B, packed, C []float64, loRow, hiRow, k, n, jLo, nv8, jHi int) {
	panels := (nv8 - jLo) / 8
	i := loRow
	for ; i+3 < hiRow; i += 4 {
		for panel := 0; panel < panels; panel++ {
			j := jLo + panel*8
			gemmF64Tile4x8Neon(&A[i*k], &packed[panel*k*8], &C[i*n+j], k, k, 8, n)
		}
		if nv8 < jHi {
			a0 := A[(i+0)*k : (i+1)*k]
			a1 := A[(i+1)*k : (i+2)*k]
			a2 := A[(i+2)*k : (i+3)*k]
			a3 := A[(i+3)*k : (i+4)*k]
			c0 := C[(i+0)*n : (i+1)*n]
			c1 := C[(i+1)*n : (i+2)*n]
			c2 := C[(i+2)*n : (i+3)*n]
			c3 := C[(i+3)*n : (i+4)*n]
			for j := nv8; j < jHi; j++ {
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
	}
	for ; i < hiRow; i++ {
		ar := A[i*k : (i+1)*k]
		ci := C[i*n : (i+1)*n]
		for j := jLo; j < jHi; j++ {
			v := ci[j]
			bo := j
			for p := 0; p < k; p++ {
				v += ar[p] * B[bo]
				bo += n
			}
			ci[j] = v
		}
	}
}
