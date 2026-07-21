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
		// kc-blocking + B-panel packing (slab OUTER so each packed B[p0:p0+kc,:] slab stays cache-hot
		// across the parallel row sweep) only pays once B[k,n] outgrows the shared L3 and threads
		// contend re-reading it — below that the extra C load/store + pack traffic is a net loss
		// (measured: 512/1024 slower, 2048 +16%, 4096 within ~1.2x of numpy). So block+pack only in the
		// memory-bound regime; small/mid GEMMs take a single overwrite pass.
		//
		// Packing DOUBLES B traffic (write-to-bp + read) to buy a sequential compute-read, so it only
		// pays when that win outweighs the pack write:
		//   - k≥1024: below this the pack write isn't amortized even for large m — small-k/large-n takes
		//     single-pass (measured 256×512×4096 232→297, 512×512×4096 250→280, 256×512×8192 201→254).
		//   - then (rowTiles≥12 i.e. m≳72) OR (k≥4096 where the stream win dominates regardless of m):
		//     medium-m at moderate k — batched/speculative decode (m=8,32) — takes single-pass
		//     (8×2048×2048 9.5→12.8, 32×2048×2048 32→43), while the large-k down-proj 32×5632×2048 stays
		//     packed (23 vs 21). Every designed-for large-n×large-k shape (k≥2048, m≥512) stays packed.
		if k >= 1024 && k*n > 2_000_000 && (rowTiles >= 12 || k >= 4096) {
			nb := n16 / 16
			bp := make([]float32, gemmAsmKC*n16) // packed B slab, reused across k-slabs (stride kc*16 per block)
			for p0 := 0; p0 < k; p0 += gemmAsmKC {
				kc := gemmAsmKC
				if p0+kc > k {
					kc = k - p0
				}
				first := p0 == 0
				// pack B[p0:p0+kc, :] into contiguous [kc][16] panels so the microkernel streams B
				// sequentially (ldb=16) instead of striding by n every k-step.
				parallelWork(nb, kc*16, func(lo, hi int) {
					for blk := lo; blk < hi; blk++ {
						j0, dst := blk*16, bp[blk*kc*16:]
						for p := 0; p < kc; p++ {
							src := (p0 + p) * n
							copy(dst[p*16:p*16+16], B[src+j0:src+j0+16])
						}
					}
				})
				// Grain over (rowTile × colBlock) UNITS, not just rowTiles: medium-m shapes have
				// rowTiles ≪ cores (m=32→5, m=8→1) and a rowTile-only split leaves most cores idle
				// (measured 8×2048×2048 9.5, 32×2048×2048 32 GFLOP/s). w=t*nb+blk keeps the same
				// row-major order, so a contiguous worker range sees identical A-panel/packed-B reuse
				// as before for large m, while small m now fills every core. C tiles stay disjoint.
				parallelWork(rowTiles*nb, 6*kc*16, func(lo, hi int) {
					for w := lo; w < hi; w++ {
						t, blk := w/nb, w%nb
						aBase := &A[t*6*k+p0]
						if first {
							gemmF32Tile6x16AVX2(aBase, &bp[blk*kc*16], &C[t*6*n+blk*16], kc, k, 16, n)
						} else {
							gemmF32Tile6x16AVX2Acc(aBase, &bp[blk*kc*16], &C[t*6*n+blk*16], kc, k, 16, n)
						}
					}
				})
			}
		} else {
			// Single pass (cache-resident B): parallelize over 16-col BLOCKS, not 6-row tiles, so each
			// worker reuses a small B-column slice from L2 rather than every worker re-reading all of B
			// from L3. Each (col-block, row-tile) writes a disjoint C tile. +4% @512/1024.
			nb := n16 / 16
			parallelWork(nb, rowTiles*6*k, func(loB, hiB int) {
				for cb := loB; cb < hiB; cb++ {
					j := cb * 16
					for t := 0; t < rowTiles; t++ {
						i := t * 6
						gemmF32Tile6x16AVX2(&A[i*k], &B[j], &C[i*n+j], k, k, n, n)
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
