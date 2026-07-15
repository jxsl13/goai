package npy

import (
	"io"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// Perf guard for the writeData bulk-encode path (docs/perf-notes-lowlevel.md).
func BenchmarkSaveF32_1M(b *testing.B) {
	const n = 1 << 20
	t := tensor.New(tensor.F32, tensor.Shape{n})
	s := t.Storage().F32()
	for i := range s {
		s[i] = float32(i)
	}
	b.SetBytes(4 * n)
	b.ResetTimer()
	for range b.N {
		if err := Save(io.Discard, t); err != nil {
			b.Fatal(err)
		}
	}
}
