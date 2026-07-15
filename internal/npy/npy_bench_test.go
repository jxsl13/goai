package npy

import (
	"bytes"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// Perf guards for the bulk byte<->float conversion loops in Read/Write
// (docs/perf-notes-lowlevel.md). 1M elements: big enough that header parsing
// noise vanishes and the per-element loop dominates.

func benchTensor(dt tensor.Dtype, n int) *tensor.Tensor {
	t := tensor.New(dt, tensor.Shape{n})
	switch dt {
	case tensor.F32:
		s := t.Storage().F32()
		for i := range s {
			s[i] = float32(i)
		}
	case tensor.F64:
		s := t.Storage().F64()
		for i := range s {
			s[i] = float64(i)
		}
	}
	return t
}

func benchRead(b *testing.B, dt tensor.Dtype) {
	const n = 1 << 20
	var buf bytes.Buffer
	if err := Write(&buf, benchTensor(dt, n)); err != nil {
		b.Fatal(err)
	}
	data := buf.Bytes()
	b.SetBytes(int64(n) * int64(dt.Size()))
	b.ResetTimer()
	for range b.N {
		if _, err := Read(bytes.NewReader(data)); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkReadF32_1M(b *testing.B) { benchRead(b, tensor.F32) }
func BenchmarkReadF64_1M(b *testing.B) { benchRead(b, tensor.F64) }

func benchWrite(b *testing.B, dt tensor.Dtype) {
	const n = 1 << 20
	t := benchTensor(dt, n)
	b.SetBytes(int64(n) * int64(dt.Size()))
	b.ResetTimer()
	for range b.N {
		var buf bytes.Buffer
		buf.Grow(n*dt.Size() + 128)
		if err := Write(&buf, t); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkWriteF32_1M(b *testing.B) { benchWrite(b, tensor.F32) }
func BenchmarkWriteF64_1M(b *testing.B) { benchWrite(b, tensor.F64) }
