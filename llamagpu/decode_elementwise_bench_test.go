//go:build darwin && cgo

package llamagpu_test

import (
	"fmt"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// TestDecodeThroughputTinyLlamaShape guards the fix for the largest single defect found in the
// decode path: the elementwise ops ran over the WHOLE context-sized buffer for a single decoded row.
//
// Decoder buffers are allocated for ctx rows so one Decoder serves both prefill and decode. Every
// recorded op takes a row count — RMSNorm, RoPE, MHA, AddBias — except Binary, which derived its
// length from the buffer. So each residual add touched ctx*dim = 1024*2048 = 2,097,152 elements and
// each SwiGLU ctx*hidden = 5,767,168, for one row of real work. A 22-layer decode issued ~46.8 adds
// and ~23.4 SwiGLUs per token: about 233M elements, roughly 2.8 GB of traffic per token, against
// ~520 MB of actual weight reads.
//
// How it was found, because the route matters: a GPU-vs-wall split showed 17.5 of 20.3 ms/token was
// GPU-busy, while measured matmuls (4.1 ms for 154 projections over 520 MB, at 132 GB/s) plus the
// isolated small-kernel costs came to ~5 ms. The missing 13 ms was only located by logging the
// ACTUAL element counts the recorder was called with, rather than the shapes they were assumed to
// have. Every earlier estimate used the assumed shapes and was wrong by three orders of magnitude.
//
// Interleaved A/B, 3 alternations, non-overlapping: 48.7 -> 138.0 tok/s (2.83x). The generated token
// stream is bit-identical over 64 tokens from an 8-token prompt, exercising prefill (rows>1) and
// decode (rows=1); bounding only skips buffer tail elements that no consumer reads.
//
// The floor is loose on purpose — absolute throughput is machine- and thermal-dependent, and this
// exists to catch a return to the unbounded form (which halves throughput and worse), not to pin a
// number.
func TestDecodeThroughputTinyLlamaShape(t *testing.T) {
	if !metal.Available() {
		t.Skip("no metal")
	}
	if testing.Short() {
		t.Skip("builds a 22-layer model")
	}
	cfg := nlp.LlamaConfig{Vocab: 32000, Ctx: 1024, Dim: 2048, Heads: 32, KVHeads: 4, Layers: 22, Hidden: 5632, Eps: 1e-5, RopeBase: 10000}
	m, err := nlp.NewLlama(cfg, 7)
	if err != nil {
		t.Fatal(err)
	}
	qm, err := nlp.QuantizeLlama(m, gguf.Q4_K)
	if err != nil {
		t.Fatal(err)
	}
	defer qm.Close()
	dec, err := llamagpu.NewQuant(qm)
	if err != nil {
		t.Skip(err)
	}
	defer dec.Release()

	prompt := []int{1, 2, 3, 4}
	if _, err := dec.Generate(prompt, 8, nlp.Greedy()); err != nil {
		t.Fatal(err)
	}
	const genN = 32
	best := 0.0
	for range 3 {
		start := time.Now()
		if _, err := dec.Generate(prompt, genN, nlp.Greedy()); err != nil {
			t.Fatal(err)
		}
		if v := float64(genN) / time.Since(start).Seconds(); v > best {
			best = v
		}
	}
	fmt.Printf("decode TinyLlama-shape Q4_K: %.1f tok/s\n", best)
	if best < 90 {
		t.Errorf("%.1f tok/s — decode has regressed; the unbounded-elementwise form measured 48.7 and the bounded form 138.0", best)
	}
}
