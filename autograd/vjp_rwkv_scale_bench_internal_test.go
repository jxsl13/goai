package autograd

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// The WKV backward's COMPLEXITY, benchmarked as a shape sweep rather than asserted in a comment.
//
// It was O(seq^2) per channel until T1002 and is now O(seq), and the only way to see that is to
// vary seq at a fixed d and read the ratios. Before: 248µs, 2681µs, 10335µs at 64/256/512, which
// is 3.85x per doubling — the quadratic 4x. After: 101µs, 320µs, 580µs, which is 1.81x per
// doubling. A change that reintroduced any per-(t,i) work would show here as the ratio climbing
// back toward 4 long before it showed anywhere else.
func benchWKVSeq(b *testing.B, seq, d int) {
	vjp := vjps[backend.OpWKV]
	k, v, w, u, g := wkvInputs(seq, d, tensor.F64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := vjp(nil, []*tensor.Tensor{k, v, w, u}, nil, nil, g); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkWKVScale64(b *testing.B)  { benchWKVSeq(b, 64, 64) }
func BenchmarkWKVScale128(b *testing.B) { benchWKVSeq(b, 128, 64) }
func BenchmarkWKVScale256(b *testing.B) { benchWKVSeq(b, 256, 64) }
func BenchmarkWKVScale512(b *testing.B) { benchWKVSeq(b, 512, 64) }
