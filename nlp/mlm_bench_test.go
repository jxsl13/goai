package nlp_test

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// BenchmarkMLMMaskExcluding covers the per-token loop in MLMMaskExcluding, which had no benchmark
// at all. It sweeps the number of protected ids because that is what selects the shape of the
// membership test — including 0, which is the path MLMMask itself takes (it delegates here with a
// nil specialIDs, and probing a nil map still costs a call per token).
//
// maskProb is the reference 0.15, so the RNG draws that follow a non-protected token happen at a
// realistic rate rather than on every iteration.
func BenchmarkMLMMaskExcluding(b *testing.B) {
	const seqLen, vocab = 512, 32000
	tokens := make([]int, seqLen)
	for i := range tokens {
		tokens[i] = (i * 7919) % vocab
	}
	// Protect ids the sequence actually contains, so membership genuinely fires.
	for _, n := range []int{0, 4, 8} {
		special := make([]int, n)
		for i := range special {
			special[i] = tokens[i*13%seqLen]
		}
		b.Run(fmt.Sprintf("special=%d", n), func(b *testing.B) {
			rng := rand.New(rand.NewPCG(11, 23))
			b.ReportAllocs()
			for range b.N {
				nlp.MLMMaskExcluding(tokens, 0.15, 103, vocab, special, rng)
			}
		})
	}
}
