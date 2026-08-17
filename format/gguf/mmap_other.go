//go:build !darwin && !dragonfly && !freebsd && !linux && !netbsd && !openbsd

package gguf

import "os"

func mmapFileReadOnly(*os.File, int) ([]byte, bool) { return nil, false }
func munmapFile([]byte) error                       { return nil }
