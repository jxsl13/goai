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
//
// The assertion is mutation-verified: raising the encode floor to an unreachable 500 MB/s makes this
// test fail, so the check fires rather than merely existing. The token count is printed as a second
// signal — 237,208 is the value that matched tiktoken bit-for-bit in T882, so a change in it means
// the tokenizer's OUTPUT moved and not just its speed. A throughput guard that ignored correctness
// would pass a tokenizer that got fast by being wrong.
func TestBPEThroughputGuard(t *testing.T) {
	tk, err := nlp.LoadGPT2("testdata/gpt2_ranks.txt")
	if err != nil {
		t.Skipf("gpt2 ranks unavailable (run make golden): %v", err)
	}
	text := t882Corpus()
	mb := float64(len(text)) / 1e6

	const encodeFloor = 15.0
	best := 0.0
	var toks []int
	for i := 0; i < 3; i++ {
		st := time.Now()
		toks = tk.Encode(text)
		if r := mb / time.Since(st).Seconds(); r > best {
			best = r
		}
	}
	fmt.Printf("BPEGUARD encode %.1f MB/s (floor %.0f)  tokens=%d\n", best, encodeFloor, len(toks))

	// CORRECTNESS FIRST, and unconditionally: 237208 is the count that matched tiktoken
	// bit-for-bit in T882. This used to be printed and never asserted, so the one part of this
	// test that belongs in CI was not actually a gate. A tokenizer that got fast by being wrong
	// would have sailed through.
	if len(toks) != 237208 {
		t.Errorf("encoded %d tokens, want 237208 — the tokenizer's OUTPUT moved, not just its speed", len(toks))
	}

	// THROUGHPUT ONLY ON A DEV BOX. The floors are ~1/3 of an M2 Pro, which this file claimed was
	// "loose enough for slower CI hardware" — measured wrong: GitHub's shared runners report as
	// little as 1.3 MB/s encode and 20.2 MB/s decode, 36x and 12x under the floors. On hardware
	// that variable the guard cannot tell a regression from a busy neighbour, which is the same
	// dev-box/runner split ci.yml documents ("-short on runners").
	if testing.Short() {
		t.Skip("throughput floors are calibrated for a dev box; runners are too variable")
	}
	if best < encodeFloor {
		t.Errorf("BPE encode %.1f MB/s is below the %.0f MB/s floor — an order-of-magnitude "+
			"regression, not machine noise", best, encodeFloor)
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
