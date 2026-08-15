//go:build darwin && cgo

package llamagpu_test

import (
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// TestLongContextDecodeGap records the largest open performance gap in this repo, because the
// headline tg64 figure hides it completely.
//
// Short-context decode is ~0.97x of llama.cpp and looks nearly settled. Long-context decode is not.
// Measured live on an M2 Pro, TinyLlama-1.1B Q4_K_M, 32 tokens generated after a prefill:
//
//	ctx     goai      llama.cpp   ratio
//	   8   153.75        158.39   0.97x
//	 512    64.16       ~161.6    0.40x
//	1024    41.35       ~161      0.26x
//	1536    29.94       ~161      0.19x
//
// llama.cpp's figures are derived from `llama-bench -pg N,32` in the same session: pp512+tg32 runs
// 544 tokens at 1267.70 t/s (0.4291 s) of which prefill is 512/2215.99 = 0.2311 s, leaving 32 tokens
// in 0.1980 s. Its decode is essentially FLAT in context — 161 t/s at ctx=512 against tg128's
// 158.39 — which is what the arithmetic predicts, since a 1.1B model's KV traffic at ctx=1536
// (~69 MB/token) is small next to its ~599 MB of weights.
//
// Ours collapses instead, and not because of the KV traffic: at ctx=1536 a token takes ~33.4 ms
// where the weights account for ~3.3 ms and the KV read for ~0.4 ms.
//
// Two measurements narrow it down:
//
//  1. TestDecodeLeaveOneOut's synthetic chain, built from the same ops and shapes, runs 7585.9us at
//     ctx=1536 — 131.8 tok/s. The real decoder manages 29.94. So the real path is 4.4x slower than
//     a faithful model of itself, and the missing time is NOT in the ops the chain contains.
//  2. Tracing dispatch sizes for one decode step shows every op class identical between ctx=8 and
//     ctx=1536 except attention, whose element count goes 0.86M -> 138.5M. So the context-dependent
//     cost is entirely attention, as expected — but the call counts are ~2x what a 22-layer model
//     should issue (44 mha, 90 rmsnorm, 262 qmatmul for one step), which is unexplained and is the
//     first thing to chase.
//
// Split-K attention (TestSplitKDecodeLongContext) buys 1.07x here. That is real but small against a
// 5.4x gap, which confirms the problem is not the attention kernel's efficiency.
//
// The cause is therefore still UNKNOWN. Ruled out so far: attention kernel efficiency, KV bandwidth,
// the ops contained in the modelled chain, and (now) any doubling of issued work.
//
// This test is a REGRESSION ANCHOR, not a benchmark: it records the ratios so the gap cannot be
// forgotten behind the flattering short-context number, and fails only if long-context decode gets
// materially worse than recorded.
func TestLongContextDecodeGap(t *testing.T) {
	if !metal.Available() {
		t.Skip("no metal")
	}
	f, _ := os.Open(os.Getenv("GOAI_TINYLLAMA_GGUF"))
	if f == nil {
		t.Skip("no model")
	}
	defer f.Close()
	raw, _ := gguf.ReadRaw(f)
	qm, _ := nlp.QuantLlamaFromGGUF(raw.Metadata, raw.Tensors)
	defer qm.Close()
	dec, err := llamagpu.NewQuant(qm)
	if err != nil {
		t.Skip(err)
	}
	defer dec.Release()

	for _, c := range []struct {
		plen  int
		floor float64 // recorded tok/s, minus generous headroom
	}{{8, 100}, {512, 40}, {1536, 18}} {
		p := make([]int, c.plen)
		for i := range p {
			p[i] = 1 + i%2000
		}
		dec.StepNLast(p, 0)
		best := 0.0
		for range 2 {
			st := time.Now()
			if _, e := dec.Generate(p, 32, nlp.Greedy()); e != nil {
				t.Fatal(e)
			}
			if v := 32 / time.Since(st).Seconds(); v > best {
				best = v
			}
		}
		fmt.Printf("LONGCTX ctx=%5d  %7.2f tok/s\n", c.plen, best)
		if best < c.floor {
			t.Errorf("ctx=%d decode %.2f tok/s below recorded floor %.0f", c.plen, best, c.floor)
		}
	}
}
