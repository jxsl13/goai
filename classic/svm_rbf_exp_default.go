//go:build !(arm64 && goexperiment.simd)

package classic

import "math"

const rbfColumnSIMD = false

// rbfColumnBand is the portable semantic twin of the ARM64 SIMD implementation.
// The compile-time false route keeps the established fused kernel path in use.
func rbfColumnBand(dst, xi []float64, rows [][]float64, gamma float64) {
	rows = rows[:len(dst)]
	for t, row := range rows {
		var distance float64
		for i := range xi {
			d := xi[i] - row[i]
			distance += d * d
		}
		dst[t] = math.Exp(-gamma * distance)
	}
}
