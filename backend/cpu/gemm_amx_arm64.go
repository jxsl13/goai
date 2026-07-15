//go:build darwin && goexperiment.simd

package cpu

// Raw Apple-AMX F32 GEMM driver (ADR-0027, Path B — pure Go, no cgo). The
// dense 32-aligned interior runs the 32×32 AMX tile kernel in
// gemm_amx_arm64.s (see that file for the encoding + preemption-safety
// story); the m%32 / n%32 tails reuse the ADR-0026 NEON band kernels. Both
// operands are pre-packed so the tile streams its A columns and B rows with
// 256-byte quad loads:
//
//	apack[tile][p*32+r] = A[(tile*32+r)*k + p]  (A columns, transposed)
//	bpack[tile][p*32+j] = B[p*n + tile*32 + j]  (B rows, like the NEON pack)
//
// Numerics: fma32 accumulates f32-native like the NEON tile and Accelerate —
// each C element sums its k products in ascending p order in one fused chain,
// so the ADR-0021 tolerance contract (rtol 2e-3 vs the f64 ref) applies
// unchanged. The default (non-simd) build never sees this file.
//
// Gating: darwin-only at build time (macOS is the only OS whose kernel
// context-switches AMX state; on Linux/Asahi the ops trap) and M1/M2/M3-only
// at run time (M4 replaced AMX with SME) — otherwise gemmF32AMX stays nil and
// gemmF32 serves everything with NEON/Accelerate.

import (
	"runtime"
	"strings"
	"sync/atomic"
	"syscall"
	"unsafe"
)

// gemmF32TileAMX32x32 computes one dense 32×32 store-once tile on the AMX
// unit; apack/bpack are 256-byte-aligned packed panels (see gemm_amx_arm64.s).
// k must be ≥ 1.
//
//go:noescape
func gemmF32TileAMX32x32(apack, bpack, c *float32, k, ldc int)

func init() {
	if amxSupported() {
		gemmF32AMX = gemmF32AMXCompute
	}
}

// amxSupported reports whether the chip carries the AMX coprocessor. There is
// no sysctl feature bit for it (it is undocumented), so gate on the brand
// string: every M1/M2/M3-family core has it, M4+ moved to SME. Conservative
// by construction — unknown chips fall back to the NEON/Accelerate paths.
func amxSupported() bool {
	brand, err := syscall.Sysctl("machdep.cpu.brand_string")
	if err != nil {
		return false
	}
	return strings.Contains(brand, "Apple M1") ||
		strings.Contains(brand, "Apple M2") ||
		strings.Contains(brand, "Apple M3")
}

// gemmF32AMXCompute computes C[m,n] = A·B on the AMX units, f32-native
// (ADR-0021 tolerance). Returns false — punting to the caller's next path —
// when the shape has no 32×32 interior to amortize the packing passes.
func gemmF32AMXCompute(A, B, C []float32, m, k, n int) bool {
	if m < 32 || n < 32 {
		return false
	}
	mi, ni := m&^31, n&^31
	rowT, colT := mi/32, ni/32

	apP := getF32Raw(rowT*k*32 + 64)
	bpP := getF32Raw(colT*k*32 + 64)
	ap := alignedF32(*apP, rowT*k*32)
	bp := alignedF32(*bpP, colT*k*32)

	// Supertile grouping: workers pull g×g GROUPS of tiles off an atomic
	// counter. The grouping is L2 blocking — a g×g group touches g apack +
	// g bpack panels for g² tiles, cutting panel re-streaming ×g on the
	// shapes whose packed operands outgrow L2 (512×2048×8192: +~55% measured
	// at g=4→8 over ungrouped) — and the dynamic pull keeps the asymmetric
	// P/E clusters load-balanced (same rationale as the NEON path). g
	// shrinks for small tile grids so there are always enough groups.
	// Worker count: every core issues to its cluster's AMX unit (M2 Pro: 2
	// P-AMX + 1 E-AMX); a single core cannot saturate one unit, so the sweep
	// (gemm_amx_bench_test.go) climbs monotonically to GOMAXPROCS — 12
	// workers beat 8 by ~4% and 4 by ~12% at 1024³.
	tiles := rowT * colT
	workers := min(max(runtime.GOMAXPROCS(0), 1), tiles)
	gRows, gCols := 1, 1
	if colT >= 8*rowT && rowT*k*128 <= 6<<20 && colT >= 3*workers {
		// Very wide grids (n ≫ m, e.g. vocab projections): take FULL-height
		// column strips — the whole apack stays L2-resident and every bpack
		// panel is streamed exactly once.
		gRows, gCols = rowT, max(colT/(3*workers), 1)
	} else {
		for _, cand := range []int{8, 4, 2} {
			if tiles >= cand*cand*3*workers {
				gRows, gCols = cand, cand
				break
			}
		}
	}
	sgCols := (colT + gCols - 1) / gCols
	supers := ((rowT + gRows - 1) / gRows) * sgCols

	// ONE pool pass for pack + compute: every worker first pulls packing
	// items (t < rowT: transpose an A tile panel; else: copy a B panel —
	// contiguous reads, 32-strided / contiguous writes), then spin-waits the
	// stragglers out (pool wake/park barriers cost more than the short spin
	// — the T657 lesson) and pulls supertiles.
	packItems := int64(rowT + colT)
	var packNext, packDone, next atomic.Int64
	parallelWork(workers, max(tiles/workers, 1)*32*k*32, func(_, _ int) {
		for {
			t := int(packNext.Add(1)) - 1
			if t >= int(packItems) {
				break
			}
			if t < rowT {
				dst := ap[t*k*32 : (t+1)*k*32]
				for r := 0; r < 32; r++ {
					row := A[(t*32+r)*k : (t*32+r+1)*k]
					for p, v := range row {
						dst[p*32+r] = v
					}
				}
			} else {
				panel := t - rowT
				dst := bp[panel*k*32 : (panel+1)*k*32]
				src := panel * 32
				for p := 0; p < k; p++ {
					copy(dst[p*32:p*32+32], B[p*n+src:p*n+src+32])
				}
			}
			packDone.Add(1)
		}
		for packDone.Load() < packItems { // in-pass barrier: pack ⇒ tiles
		}
		for {
			s := int(next.Add(1)) - 1
			if s >= supers {
				return
			}
			gr, gc := s/sgCols, s%sgCols
			for rt := gr * gRows; rt < min((gr+1)*gRows, rowT); rt++ {
				for ct := gc * gCols; ct < min((gc+1)*gCols, colT); ct++ {
					gemmF32TileAMX32x32(&ap[rt*k*32], &bp[ct*k*32], &C[(rt*32)*n+ct*32], k, n)
				}
			}
		}
	})

	putF32(apP)
	putF32(bpP)

	// Tails ride the NEON band kernels (f32-native, same tolerance contract).
	if ni < n { // right column strip, rows [0,mi)
		parallelWork((mi+3)/4, 4*k*(n-ni), func(lo, hi int) {
			gemmF32BandNeonCols(A, B, nil, C, lo*4, min(hi*4, mi), k, n, ni, n)
		})
	}
	if mi < m { // bottom strip (≤31 rows), all columns
		blocks := (n + 255) / 256
		parallelWork(blocks, (m-mi)*k*256, func(lo, hi int) {
			gemmF32BandNeonCols(A, B, nil, C, mi, m, k, n, lo*256, min(hi*256, n))
		})
	}
	return true
}

// alignedF32 returns the 256-byte-aligned length-n window of buf (which must
// have n+64 capacity headroom). AMX quad loads move 256 bytes and pair loads
// demand 128-byte alignment; Go's allocator only guarantees 16.
func alignedF32(buf []float32, n int) []float32 {
	off := int((256 - uintptr(unsafe.Pointer(unsafe.SliceData(buf)))&255) & 255 >> 2)
	return buf[off : off+n : off+n]
}
