//go:build arm64 || (amd64 && goexperiment.simd)

package nlp_test

const q8DecodeUsesSIMDTolerance = true
