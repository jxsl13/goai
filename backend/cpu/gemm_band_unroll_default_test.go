//go:build !(amd64 && goexperiment.simd)

package cpu

func gemmF64BandUnderTest(A, B, C []float64, loRow, hiRow, k, n int) {
	gemmF64Band(A, B, C, loRow, hiRow, k, n)
}

func gemmF32BandUnderTest(A, B []float32, acc []float64, loRow, hiRow, k, n int) {
	gemmF32Band(A, B, acc, loRow, hiRow, k, n)
}
