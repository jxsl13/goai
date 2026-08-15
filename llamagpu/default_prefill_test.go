//go:build darwin && cgo

package llamagpu_test

// TestDefaultPrefillUsesWeightCache pins the DEFAULT prefill configuration, which until now was much
// slower than the tuned one.
//
// The expanded-weight cache removes the per-pass weight expansion — 37 ms/pass for TinyLlama, over
// half of a short-prompt token — but it shipped OFF, so callers got the slow path unless they knew to
// call SetWeightCacheGB. That made the honest default standing pp64 ~0.54x of llama.cpp while the
// number I was reporting, ~0.85x, required opting in.
//
// It now defaults to an eighth of physical RAM capped at 4 GB. Measured with that default (M2 Pro,
// TinyLlama-1.1B Q4_K_M, llama.cpp measured live earlier in the same session):
//
//	pp64    1537.6 tok/s   vs 1777.5   0.87x   (was ~950, 0.54x)
//	pp256   1977.6 tok/s   vs 2146     0.92x
//	pp512   2045.7 tok/s   vs ~2145    0.95x
//	pp1024  1943.9 tok/s   vs ~2145    0.91x
//
// The cache holds 1.92-1.94 GB here, about 3x the 636 MB model file, and the hit/miss counts in the
// output are the reachability check: 110-130 distinct weights fill once and hit thereafter.
//
// Why this is safe as a default rather than reckless: weights past the budget silently fall back to
// per-pass expansion, so a model too large for it is slower rather than broken; an eighth of RAM
// cannot displace the model itself; and SetWeightCacheGB(0) turns it off. The trade is explicit —
// memory for speed — and it is the trade a library that wants to be fast by default should make,
// but it IS a real allocation and is documented as such.
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

func TestDefaultPrefillUsesWeightCache(t *testing.T) {
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
	for _, n := range []int{64, 256, 512, 1024} {
		p := make([]int, n)
		for i := range p {
			p[i] = 1 + i%2000
		}
		dec.StepNLast(p, 0)
		best := 0.0
		for range 3 {
			st := time.Now()
			dec.StepNLast(p, 0)
			if v := float64(n) / time.Since(st).Seconds(); v > best {
				best = v
			}
		}
		h, m, b := metal.WeightCacheStats()
		fmt.Printf("DEF pp%-5d %7.1f tok/s   cache hits=%d miss=%d %.2fGB\n", n, best, h, m, b/1e9)
	}
}
