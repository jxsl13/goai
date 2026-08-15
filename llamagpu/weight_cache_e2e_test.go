//go:build darwin && cgo

package llamagpu_test

// TestWeightCachePrefill is the end-to-end anchor for the persistent expanded-weight cache.
//
// Fitting whole-model prefill cost as fixed + marginal*n over n=64..512 (TinyLlama-1.1B Q4_K_M,
// M2 Pro) isolates what short prompts are actually paying for:
//
//	expand-then-GEMM   fixed 37.03 ms/pass   marginal 0.4691 ms/token
//	cooperative        fixed  0.00 ms/pass   marginal 2.7109 ms/token
//
// The cooperative arm's fixed term is EXACTLY zero, which is what identifies the whole fixed cost as
// the weight expansion rather than launch overhead, sampling or the logits download — and GPU time
// was 97% of wall at every length, so it is GPU work, not host time. At n=64 it is ~54% of the pass.
// llama.cpp's equivalent fixed term is ~6.6 ms, which is the entire short-prompt gap.
//
// The expansion is redone every pass over weights that never change, so it is cached. Interleaved
// arms, two runs:
//
//	n=  64   959.8 -> 1233.7  1.285x   |   953.1 -> 1224.5  1.285x
//	n= 256  1613.3 -> 1784.8  1.106x   |  1627.7 -> 1751.6  1.076x
//	n= 512  1842.9 -> 1887.3  1.024x   |  1834.3 -> 1806.4  0.985x
//	n=1024  1897.5 -> 1927.3  1.016x   |  1919.3 -> 1951.5  1.017x
//
// Only n=64 and n=256 are real. n=512 straddles 1.0, and at n=1024 the cache is INERT (hits=0,
// misses=0 — M is past the cap) so both arms run identical code and the 1.016x/1.017x is a direct
// reading of this machine's noise floor. An earlier sweep made that explicit: with both arms
// identical at n=768 the ratio still came out 0.962x, i.e. ~4% noise, which is why the 384/512
// readings are not claimed. The cap is kept at 512 because four readings there average +1.4% with
// no evidence of harm, not because a win was demonstrated.
//
// The argmax token is identical in both arms at every length. Hit/miss counts are asserted rather
// than printed only: 100 distinct weights fill on the first pass and hit thereafter, which is also
// the reachability check — a cache that silently never fired would show hits=0 and an unchanged time.
//
// COST: ~1.69 GB to hold 100 Q4_K weights in f16, about 2.7x the 636 MB model file. That is why the
// cache is OFF by default; this test enables it explicitly.
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

func TestWeightCachePrefill(t *testing.T) {
	if !metal.Available() {
		t.Skip("no metal")
	}
	f, err := os.Open(os.Getenv("GOAI_TINYLLAMA_GGUF"))
	if err != nil {
		t.Skip(err)
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

	for _, n := range []int{64, 256, 512, 1024} {
		p := make([]int, n)
		for j := range p {
			p[j] = 1 + j%2000
		}
		var tps [2]float64
		var arg [2]int
		for arm := 0; arm < 2; arm++ {
			if arm == 1 {
				metal.SetWeightCacheGB(4.0)
			} else {
				metal.SetWeightCacheGB(0)
			}
			dec.StepNLast(p, 0) // warm; also fills the cache on arm 1
			best, am := 0.0, 0
			for range 3 {
				st := time.Now()
				out, e := dec.StepNLast(p, 0)
				if e != nil {
					t.Fatal(e)
				}
				if v := float64(n) / time.Since(st).Seconds(); v > best {
					best = v
				}
				mx := float32(-1e30)
				for i, v := range out {
					if v > mx {
						mx, am = v, i
					}
				}
			}
			tps[arm], arg[arm] = best, am
		}
		h, m, b := metal.WeightCacheStats()
		if arg[0] != arg[1] {
			t.Errorf("n=%d: cache changed the predicted token (%d vs %d)", n, arg[0], arg[1])
		}
		if n <= 512 && (m == 0 || h == 0) {
			t.Errorf("n=%d: cache never fired (hits=%d misses=%d) — the path under test was not reached", n, h, m)
		}
		fmt.Printf("CACHE n=%4d off=%7.1f on=%7.1f tok/s %.3fx  argmax %d/%d %s  hits=%d miss=%d %.2f GB\n",
			n, tps[0], tps[1], tps[1]/tps[0], arg[0], arg[1],
			map[bool]string{true: "SAME", false: "DIFFER"}[arg[0] == arg[1]], h, m, b/1e9)
	}
	metal.SetWeightCacheGB(0)
}
