package nlp

import (
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// The benchmarks below exist to keep a recorded negative perf verdict (T846)
// re-checkable rather than taken on faith.
//
// [TokenSampler] takes a []float64, so every decode step materialises the whole
// vocabulary as a host slice via rowLogits before a single token is chosen. That
// reads like an obvious serving-hot-path waste — a 256k-entry vocabulary is a
// 2 MB allocation per token, thrown away immediately.
//
// It measures as noise. rowLogits already has typed fast paths, so the cost is
// allocation and memory traffic, not per-element dispatch (note F64 and F32 come
// out the same: nothing is saved by skipping the widening conversion). Against a
// real decode step the ratio is ~0.02% at a 1000-entry vocabulary and ~0.1% at
// Gemma's 256k, because a forward pass through the model dwarfs one linear pass
// over its output row.
//
// So the []float64 signature is NOT worth breaking for performance. If it is
// ever changed, change it for API reasons — a streaming or device-resident
// sampler that never needs the full row — and re-run these to confirm the
// arithmetic still holds on the hardware of the day.
func benchRowLogits(b *testing.B, vocab int, dt tensor.Dtype) {
	t := tensor.Zeros(dt, tensor.Shape{1, vocab})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = rowLogits(t)
	}
}

// BenchmarkRowLogits1kF32 matches the vocabulary of the Generate benchmarks in
// this package, so the two are directly comparable.
func BenchmarkRowLogits1kF32(b *testing.B) { benchRowLogits(b, 1000, tensor.F32) }

// BenchmarkRowLogits32kF32 is Llama-scale.
func BenchmarkRowLogits32kF32(b *testing.B) { benchRowLogits(b, 32000, tensor.F32) }

// BenchmarkRowLogits256kF32 is Gemma-scale — the worst case in the loadable set.
func BenchmarkRowLogits256kF32(b *testing.B) { benchRowLogits(b, 256000, tensor.F32) }

// BenchmarkRowLogits256kF64 pins that the widening conversion is not the cost:
// it runs level with the F32 arm, which only copies.
func BenchmarkRowLogits256kF64(b *testing.B) { benchRowLogits(b, 256000, tensor.F64) }
