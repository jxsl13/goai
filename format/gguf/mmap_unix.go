//go:build darwin || dragonfly || freebsd || linux || netbsd || openbsd

package gguf

import (
	"os"
	"syscall"
)

// mmapFileReadOnly maps one regular file without moving its descriptor offset. A mapping failure is
// a transparent performance miss: ReadFile falls back to the portable buffered reader.
func mmapFileReadOnly(f *os.File, size int) ([]byte, bool) {
	b, err := syscall.Mmap(int(f.Fd()), 0, size, syscall.PROT_READ, syscall.MAP_SHARED)
	return b, err == nil
}

func munmapFile(b []byte) error { return syscall.Munmap(b) }
