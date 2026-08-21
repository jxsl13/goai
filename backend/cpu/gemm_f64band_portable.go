//go:build !(goexperiment.simd && (amd64 || arm64))

package cpu

func gemmF64Band(A, B, C []float64, loRow, hiRow, k, n int) {
	gemmF64BandPortable(A, B, C, loRow, hiRow, k, n)
}
