//go:build !amd64 && !arm64

package gguf

import "encoding/binary"

// iq2xsCodeWords keeps byte order and alignment portable on architectures
// where the IQ2_XS wire layout cannot be viewed directly as native uint16s.
func iq2xsCodeWords(block []byte, scratch *[32]uint16) *[32]uint16 {
	for i := range scratch {
		//perfscan:ignore PS4001 big-endian/alignment-safe fallback cannot use a native same-layout bulk copy
		scratch[i] = binary.LittleEndian.Uint16(block[2+i*2:])
	}
	return scratch
}
