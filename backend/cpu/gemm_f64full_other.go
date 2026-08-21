//go:build !(arm64 && goexperiment.simd)

package cpu

func gemmF64Full(_, _, _ []float64, _, _, _ int) bool { return false }
