//go:build arm64 && goexperiment.simd

package classic

import "github.com/jxsl13/goai/internal/simd"

const rbfColumnSIMD = true

// rbfColumnBand preserves the scalar squared-distance accumulation and batches
// only the completed nonnegative distances through the two-lane NEON exp leaf.
func rbfColumnBand(dst, xi []float64, rows [][]float64, gamma float64) {
	rows = rows[:len(dst)]
	for t, row := range rows {
		var distance float64
		for i := range xi {
			d := xi[i] - row[i]
			distance += d * d
		}
		dst[t] = distance
	}
	simd.ExpScaledF64(dst, dst, -gamma)
}
