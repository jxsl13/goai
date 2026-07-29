package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkRetentionRecurrent covers the O(1)-state recurrent form. Only the chunkwise
// form had a benchmark, so nothing could validate a change to this one.
func BenchmarkRetentionRecurrent(b *testing.B) {
	const seq, d = 512, 128
	mk := func(seed float64) *tensor.Tensor {
		t := tensor.New(tensor.F64, tensor.Shape{seq, d})
		s := t.Storage().F64()
		for i := range s {
			s[i] = math.Sin(seed + 0.013*float64(i))
		}
		return t
	}
	q, k, v := mk(1), mk(2), mk(3)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := nn.RetentionRecurrent(q, k, v, 0.96); err != nil {
			b.Fatal(err)
		}
	}
}

// TestRetentionRecurrentBitIdentical pins the output bit-for-bit across shapes covering
// every remainder class of a 4-way unroll on d_v, including d_v below 4.
func TestRetentionRecurrentBitIdentical(t *testing.T) {
	for _, g := range retRecGolden {
		seq, d := int(g[0]), int(g[1])
		mk := func(seed float64) *tensor.Tensor {
			x := tensor.New(tensor.F64, tensor.Shape{seq, d})
			s := x.Storage().F64()
			for i := range s {
				s[i] = math.Sin(seed + 1.7*float64(i))
			}
			return x
		}
		out, err := nn.RetentionRecurrent(mk(1), mk(2), mk(3), 0.9)
		if err != nil {
			t.Fatalf("seq=%d d=%d: %v", seq, d, err)
		}
		h := uint64(14695981039346656037)
		for i := range out.Numel() {
			bits := math.Float64bits(out.AtF64(tensor.Unravel(i, out.Shape())...))
			for s := 0; s < 64; s += 8 {
				h = (h ^ ((bits >> s) & 0xff)) * 1099511628211
			}
		}
		if uint64(g[2]) != h {
			t.Fatalf("seq=%d d=%d: checksum %d, want %d", seq, d, h, uint64(g[2]))
		}
	}
}
