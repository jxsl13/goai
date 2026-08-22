//go:build amd64 || arm64

package gguf

import "unsafe"

// iq2xsCodeWords exposes the contiguous little-endian uint16 code plane on
// architectures whose native byte order and unaligned-load support match the
// IQ2_XS wire layout. scratch is retained in the signature for the portable
// implementation and disappears after inlining here.
func iq2xsCodeWords(block []byte, scratch *[32]uint16) *[32]uint16 {
	_ = scratch
	_ = block[65]
	return (*[32]uint16)(unsafe.Pointer(&block[2]))
}
