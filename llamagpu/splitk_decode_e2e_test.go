//go:build darwin && cgo

package llamagpu_test

// TestSplitKDecodeLongContext is the end-to-end anchor for split-K (flash-decoding) attention, and
// records that its benefit is confined to long context.
//
// TestDecodeLeaveOneOut put attention at 31.5% of a decode token — the only large recoverable share,
// since the weight matmuls are already at the memory ceiling. The single-pass kernel dispatches only
// `heads` threadgroups of 32 threads, about 1024 threads for the whole GPU, so it cannot hide its own
// memory latency. Split-K gives each (head, key-chunk) pair its own threadgroup and merges the
// per-chunk online-softmax partials in a second pass.
//
// Leaf effect (M2 Pro, 32 heads over 4 KV heads, dk 64):
//
//	sk= 128  50.65us -> 49.74us  1.02x
//	sk= 256  65.84us -> 56.70us  1.16x
//	sk= 512  84.61us -> 60.37us  1.40x
//	sk=1024 129.51us -> 99.01us  1.31x
//	sk=2048 280.83us -> 170.66us 1.65x
//
// End to end, generating 32 tokens after a prefill of the given length:
//
//	ctx=   8  153.75 -> 150.84 tok/s  0.981x   <- CONTROL: sk<128, split-K never fires
//	ctx= 512   59.96 ->  64.16 tok/s  1.070x
//	ctx=1024   38.70 ->  41.35 tok/s  1.069x
//	ctx=1536   27.76 ->  29.94 tok/s  1.079x
//
// The ctx=8 row is the control rather than a result: the gate requires sk>=128, so both arms run
// identical code there and the 0.981x reads this machine's noise (~2%). The three long-context
// readings cluster at 1.07x, comfortably above it.
//
// Short-prompt generation sees NOTHING, which is why the first end-to-end attempt measured 0.989x
// and looked like a failure: generating from an 8-token prompt runs at sk 8..136, where the leaf
// gain is 1.02x. A leaf win only reaches the model at the shapes the model actually runs.
//
// Worth noting what this regime costs in absolute terms: decode falls 153 -> 60 -> 38.7 -> 27.8
// tok/s as context grows, so attention over a long KV cache dominates there far more than in the
// short-context tg64 benchmark usually quoted.
//
// The generated token stream is IDENTICAL in both arms at every context length. The kernel is not
// bit-exact with the single-pass one (maxRel ~2e-05, from merging chunk partials in a different
// order), so identical tokens is the meaningful check, not identical logits.
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

func TestSplitKDecodeLongContext(t *testing.T) {
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

	for _, plen := range []int{8, 512, 1024, 1536} {
		prompt := make([]int, plen)
		for i := range prompt {
			prompt[i] = 1 + i%2000
		}
		n := 32
		var tps [2]float64
		var tok [2]int
		for arm := 0; arm < 2; arm++ {
			metal.SetSplitKDecode(arm == 1)
			dec.StepNLast(prompt, 0)
			best := 0.0
			last := 0
			for range 3 {
				st := time.Now()
				out, e := dec.Generate(prompt, n, nlp.Greedy())
				if e != nil {
					t.Fatal(e)
				}
				if v := float64(n) / time.Since(st).Seconds(); v > best {
					best = v
				}
				last = out[len(out)-1]
			}
			tps[arm], tok[arm] = best, last
		}
		if tok[0] != tok[1] {
			t.Errorf("ctx=%d: split-K changed the generated token (%d vs %d)", plen, tok[0], tok[1])
		}
		fmt.Printf("SKDEC ctx=%5d single=%7.2f splitk=%7.2f tok/s %.3fx  lastTok %d/%d %s\n",
			plen, tps[0], tps[1], tps[1]/tps[0], tok[0], tok[1],
			map[bool]string{true: "SAME", false: "DIFFER"}[tok[0] == tok[1]])
	}
	metal.SetSplitKDecode(true)
}
