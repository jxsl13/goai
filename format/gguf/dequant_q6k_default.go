//go:build !arm64

package gguf

func dequantQ6_KIntoArch(dst []float32, raw []byte) {
	dequantQ6_KIntoScalar(dst, raw)
}
