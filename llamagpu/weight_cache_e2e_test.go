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
// The expansion is redone every pass over weights that never change, so it is cached. Q4_K alone got
// fixed cost from 39.96 to 22.91 ms; adding Q6_K takes it to 9.15 ms, close to llama.cpp's ~6.6.
//
// Q6_K matters far more than its share suggests: it is 16.9% of TinyLlama's weights but was 16.5 of
// those 22.9 ms, because its dequantization runs ~3.6x slower per element than Q4_K's (89 vs 24 us
// per million elements). Forcing Q6_K off the expansion path measured that directly — fixed cost
// fell 22.91 -> 6.40 ms while marginal rose 0.4833 -> 0.8383.
//
// Interleaved arms, three runs:
//
//	n=  64   959.4 -> 1509.8  1.574x  |  949.9 -> 1529.4  1.610x  |  942.3 -> 1479.6  1.570x
//	n= 256  1634.3 -> 1863.9  1.141x  | 1591.0 -> 1887.7  1.186x  | 1596.8 -> 1871.6  1.172x
//	n= 512  1767.7 -> 1943.8  1.100x  | 1793.0 -> 1932.7  1.078x  | 1802.4 -> 1929.2  1.070x
//	n=1024  1950.6 -> 1882.6  0.965x  | 1929.9 -> 1925.8  0.998x  | 1931.4 -> 1931.7  1.000x
//
// n=1024 is the control, not a result: the cache is INERT there (hits=0, misses=0 — M is past the
// cap), so both arms run identical code and the spread reads this machine's noise directly. It has
// been as bad as 0.962x between two identical arms, which is why nothing inside ~4% is claimed
// anywhere in this file. The 512 cap was re-probed after Q6_K joined and 768/1024/1536 came out
// 1.026x/0.980x/1.000x — all inside that band, so 512 is where a win stops being demonstrable
// rather than where harm begins.
//
// The argmax token is identical in both arms at every length. Hit/miss counts are asserted rather
// than printed only: 100 distinct weights fill on the first pass and hit thereafter, which is also
// the reachability check — a cache that silently never fired would show hits=0 and an unchanged time.
//
// COST: ~1.92 GB to hold 110 Q4_K and Q6_K weights in f16, about 3x the 636 MB model file. That is
// why the cache is OFF by default; this test enables it explicitly. Note how cheap Q6_K was to add:
// +0.23 GB removed 13.8 ms of the fixed cost, because the win is not paying for the expansion each
// pass rather than making the expansion smaller.
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
