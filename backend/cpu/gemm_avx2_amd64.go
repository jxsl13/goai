//go:build amd64 && goexperiment.simd

package cpu

// gemmF32Tile6x16AVX2 / ...Acc (gemm_avx2_amd64.s) compute C[6][16] = / += Σ_p A[r][p]·B[p][j] with 12
// YMM accumulators — the hand-asm register tile that lifts f32 GEMM past the archsimd allocator
// ceiling. The overwrite form zeros the accumulators; the Acc form loads C first (for kc-blocking).
func gemmF32Tile6x16AVX2(a, b, c *float32, k, lda, ldb, ldc int)
func gemmF32Tile6x16AVX2Acc(a, b, c *float32, k, lda, ldb, ldc int)

// gemmAsmKC bounds the k-block so the reused B slab (kc×n) stays roughly L2-resident.
const gemmAsmKC = 128

// gemmAsmF32Ok reports whether the AVX2-asm 6×16 path applies (needs FMA and a full 6-row tile).
func gemmAsmF32Ok(m, n int) bool { return gemmHasFMA && m >= 6 && n >= 16 }

// gemmF32Asm6x16 computes C[m,n] = A[m,k]·B[k,n] (row-major, unpacked) using the 6×16 asm microkernel
// for the bulk and the archsimd band kernel for the m%6 / n%16 edges. C is OVERWRITTEN (not +=), like
// the archsimd path, and every region is disjoint so the two kernels never touch the same C element.
// Grained over 6-row TILES (with column splitting when there are fewer tiles than workers) so every
// core runs full B-reuse tiles.
func gemmF32Asm6x16(A, B, C []float32, m, k, n int) {
	m6 := m - m%6
	n16 := n &^ 15
	if m6 > 0 && n16 > 0 {
		rowTiles := m6 / 6
		// kc-blocking (slab OUTER so each B[p0:p0+kc,:] stays cache-hot across the parallel row sweep)
		// only pays once B[k,n] outgrows the shared L3 and threads contend re-reading it — below that
		// its extra C load/store traffic is a net loss (measured 512/1024 slower, 2048 +30%). So block
		// only for the memory-bound regime; small/mid GEMMs take a single overwrite pass.
		if k > gemmAsmKC && k*n > 2_000_000 {
			for p0 := 0; p0 < k; p0 += gemmAsmKC {
				kc := gemmAsmKC
				if p0+kc > k {
					kc = k - p0
				}
				first := p0 == 0
				parallelWork(rowTiles, 6*kc*n16, func(loT, hiT int) {
					for t := loT; t < hiT; t++ {
						i := t * 6
						aBase := &A[i*k+p0]
						for j := 0; j < n16; j += 16 {
							if first {
								gemmF32Tile6x16AVX2(aBase, &B[p0*n+j], &C[i*n+j], kc, k, n, n)
							} else {
								gemmF32Tile6x16AVX2Acc(aBase, &B[p0*n+j], &C[i*n+j], kc, k, n, n)
							}
						}
					}
				})
			}
		} else {
			parallelWork(rowTiles, 6*k*n16, func(loT, hiT int) {
				for t := loT; t < hiT; t++ {
					i := t * 6
					aBase := &A[i*k]
					for j := 0; j < n16; j += 16 {
						gemmF32Tile6x16AVX2(aBase, &B[j], &C[i*n+j], k, k, n, n)
					}
				}
			})
		}
	}
	// edge B: columns [n16,n) over every row.
	if n16 < n {
		gemmF32BandDirectCols(A, B, C, 0, m, k, n, n16, n)
	}
	// edge C: rows [m6,m) over the full-width columns [0,n16).
	if m6 < m && n16 > 0 {
		gemmF32BandDirectCols(A, B, C, m6, m, k, n, 0, n16)
	}
}
