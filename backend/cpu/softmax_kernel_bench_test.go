package cpu

import "testing"

var softmaxRowMaxSink float32

func BenchmarkRowMaxF32_2048(b *testing.B) {
	x := randF32(2048, 71)
	b.SetBytes(int64(len(x) * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		softmaxRowMaxSink = rowMaxF32(x)
	}
}

func BenchmarkScaleRowF32_2048(b *testing.B) {
	x := randF32(2048, 72)
	b.SetBytes(int64(len(x) * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		scaleRowF32(x, 0.99999994)
	}
}

func BenchmarkAxpbRowF32_2048(b *testing.B) {
	x := randF32(2048, 73)
	b.SetBytes(int64(len(x) * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		axpbRowF32(x, 0.99999994, 0.0000001)
	}
}
