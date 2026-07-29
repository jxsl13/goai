//go:build !(amd64 && goexperiment.simd)

package gguf

const q4kDotIsAsm = false
