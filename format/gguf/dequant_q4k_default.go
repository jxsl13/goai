//go:build !arm64

package gguf

func dequantQ4_KIntoArch(dst []float32, raw []byte) {
	dequantQ4_KIntoScalar(dst, raw)
}
