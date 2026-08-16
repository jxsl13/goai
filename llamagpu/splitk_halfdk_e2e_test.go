//go:build darwin && cgo

package llamagpu_test

// TestSplitKHalfDKDecode is the end-to-end anchor for splitting dk across lane PAIRS in split-K
// attention — the inner-loop restructuring that TestDecodeAttnRoofline and TestDecodeAttnOccupancy
// pointed at after bandwidth, compute and more-threadgroups were each excluded.
//
// The old pass-1 kernel held q[64]+acc[64] per thread. Lanes 2i and 2i+1 now cooperate on one key,
// each owning 32 dims, so the arrays halve; 16 keys are in flight per iteration instead of 32, and
// each key costs one extra simd_shuffle_xor to finish its dot across the pair. Both lanes of a pair
// see the same s, so the online-softmax state stays consistent and the final merge runs over lanes
// of matching parity (widths 2..16).
//
// Leaf, 32 heads over 4 KV heads, dk 64:
//
//	sk= 512   57.41us ->  21.57us  2.66x
//	sk=1024  100.42us ->  33.60us  2.99x
//	sk=2048  170.38us ->  58.33us  2.92x
//
// Occupancy only moved 384 -> 448 threads/threadgroup, so this is NOT mainly an occupancy win as
// predicted: a lane pair now reads two contiguous 32-dim halves of one key instead of 32 lanes each
// striding a whole 64-dim key, which coalesces far better. The register argument found the right
// change for partly the wrong reason, and that is worth knowing before tuning it further.
//
// Decode-only end to end, interleaved, two runs:
//
//	ctx=   8  176.18 -> 179.17  1.017x  |  176.14 -> 181.31  1.029x
//	ctx= 512  149.68 -> 170.99  1.142x  |  151.95 -> 175.83  1.157x
//	ctx=1536  118.37 -> 156.47  1.322x  |  117.22 -> 154.06  1.314x
//
// Against llama.cpp measured in the SAME thermal window (pp512 2216.85, pp1536 2076.19,
// pp512+tg32 1361.69, pp1536+tg32 1670.55, so decode-after-prefill is 189.9 and 161.0 t/s):
// ctx=512 0.93x and ctx=1536 0.96x, up from 0.90x and 0.73x.
//
// Thermal caution recorded because it nearly produced a false headline: llama.cpp's tg128 measured
// 158.39 earlier in the session, 195.59 on a cooled machine, and 175.58 +/- 23.69 minutes later.
// Comparing our fresh number against its throttled one briefly showed 1.13x "beating" llama.cpp.
// Only same-window measurement of both sides means anything here.
//
// FOLLOW-UP: the coalescing post-mortem above predicted that splitting dk further would help again,
// and it does. A QUAD variant (lanes 4i..4i+3, 16 dims each, 8 keys in flight) measures 1.17x/1.05x/
// 1.18x over this pair variant at sk=512/1024/2048 and is now the default. End to end it is only
// ~1.0-1.5% at long context — inside the noise floor — because attention is a much smaller share of
// a token once the 2.9x above is in. metal.SetSplitKQuadDK(false) falls back to the pair variant,
// and the numbers in this file were measured against the pre-pair kernel, so they still describe
// what this change bought.
//
// The argmax token is identical in both arms at every context length.
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

func TestSplitKHalfDKDecode(t *testing.T) {
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
	dec, _ := llamagpu.NewQuant(qm)
	defer dec.Release()
	for _, plen := range []int{8, 512, 1536} {
		p := make([]int, plen)
		for i := range p {
			p[i] = 1 + i%2000
		}
		dec.StepNLast(p, 0)
		var tps [2]float64
		var tok [2]int
		for arm := 0; arm < 2; arm++ {
			metal.SetSplitKHalfDK(arm == 1)
			for i := 0; i < 3; i++ {
				dec.Step(5, plen+i)
			}
			const n = 12
			var wall float64
			last := 0
			for i := 0; i < n; i++ {
				st := time.Now()
				out, e := dec.Step(5, plen+3+i)
				if e != nil {
					t.Fatal(e)
				}
				wall += time.Since(st).Seconds() * 1e3
				mx := float32(-1e30)
				for j, v := range out {
					if v > mx {
						mx, last = v, j
					}
				}
			}
			tps[arm], tok[arm] = 1000/(wall/n), last
		}
		if tok[0] != tok[1] {
			t.Errorf("ctx=%d: half-dk split-K changed the predicted token (%d vs %d)", plen, tok[0], tok[1])
		}
		fmt.Printf("HE2E ctx=%5d full=%7.2f half=%7.2f tok/s %.3fx  argmax %d/%d %s\n",
			plen, tps[0], tps[1], tps[1]/tps[0], tok[0], tok[1],
			map[bool]string{true: "SAME", false: "DIFFER"}[tok[0] == tok[1]])
	}
	metal.SetSplitKHalfDK(false)
}
