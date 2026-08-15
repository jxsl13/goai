package nlp_test

import (
	"fmt"
	"testing"
	"time"

	"github.com/jxsl13/goai/nlp"
)

// TestBPEThroughputGuard is an order-of-magnitude tripwire on BPE encode/decode throughput.
//
// The win it protects is documented on BenchmarkGPT2Encode: GoAI's pure-Go BPE beats tiktoken's
// Rust core end-to-end (T882). But a Benchmark never fails CI, so a regression that halved
// throughput would pass every gate silently — the same hole that let a 1600x SVC fit regression
// through the classic suite (see TestClassicFitTimeGuard).
//
// Floors are ~1/3 of what an M2 Pro measures (46.7 MB/s encode, 960 MB/s decode), which is loose
// enough for slower CI hardware and a loaded machine. This is NOT a claim about the tiktoken margin:
// verifying that needs the companion internal/benchcompare/tokenizer_compare.py with tiktoken
// installed. It only catches the case where the Go side falls off a cliff.
func TestBPEThroughputGuard(t *testing.T) {
	tk, err := nlp.LoadGPT2("testdata/gpt2_ranks.txt")
	if err != nil {
		t.Skipf("gpt2 ranks unavailable (run make golden): %v", err)
	}
	text := t882Corpus()
	mb := float64(len(text)) / 1e6

	best := 0.0
	var toks []int
	for i := 0; i < 3; i++ {
		st := time.Now()
		toks = tk.Encode(text)
		if r := mb / time.Since(st).Seconds(); r > best {
			best = r
		}
	}
	fmt.Printf("BPEGUARD encode %.1f MB/s (floor 15)  tokens=%d\n", best, len(toks))
	if best < 15 {
		t.Errorf("BPE encode %.1f MB/s is below the 15 MB/s floor — an order-of-magnitude "+
			"regression, not machine noise", best)
	}

	bestD := 0.0
	for i := 0; i < 3; i++ {
		st := time.Now()
		_ = tk.Decode(toks)
		if r := mb / time.Since(st).Seconds(); r > bestD {
			bestD = r
		}
	}
	fmt.Printf("BPEGUARD decode %.1f MB/s (floor 250)\n", bestD)
	if bestD < 250 {
		t.Errorf("BPE decode %.1f MB/s is below the 250 MB/s floor", bestD)
	}
}
