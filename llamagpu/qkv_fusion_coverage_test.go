//go:build darwin && cgo

package llamagpu_test

// TestQKVFusionCoverage records why the fused-QKV path (SPEC T613) covers only half of TinyLlama's
// layers, which is worth ~8.9% of prefill at M=64.
//
// qfused concatenates the RAW QUANTIZED BYTES of q, k and v into one [dim, dim+2*kvDim] weight, so
// it is valid only when all three share a quant type; otherwise it returns nil and the block keeps
// three separate matmuls. On tinyllama-1.1b-chat Q4_K_M:
//
//	layers where q/k/v share a quant type: 12
//	layers where they differ:              10   (q=Q4_K, k=Q4_K, v=Q6_K)
//
// Q4_K_M is a mixed file by construction — llama.cpp promotes attn_v (and ffn_down) to Q6_K on
// selected layers — so 10 of 22 blocks run the unfused chain.
//
// The cost, from TestPrefillLeaveOneOut: a fused-QKV chain measures 36.13 ms at M=64 against the
// real decoder's 39.34, i.e. ~8.9%; at M=512 it is ~2.8%. And the reason it hurts most at small M is
// shape, not FLOPs — a 2048x256 k or v projection runs at 7.4% of peak at M=64 against gate/up's
// 67.3%, so k and v are 2.3% of a layer's FLOPs but ~15% of its matmul time.
//
// A partial fix exists and is not implemented here: in every differing layer q and k DO share a type
// (both Q4_K), so q|k could fuse into one N=2304 matmul and leave v separate, removing one of the
// two narrow GEMMs. It is not a one-line change — the fused layout is a single q‖k‖v buffer that
// RoPE and attention index with a element offset (qElemOff), and a partial fusion changes those
// offsets — so it needs the plumbing done deliberately rather than squeezed in.
//
// This test is the coverage guard: if a future model or requantisation changes the mix, the counts
// move and the ~8.9% estimate above no longer applies.
import (
	"fmt"
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/jxsl13/goai/format/gguf"
)

func TestQKVFusionCoverage(t *testing.T) {
	f, _ := os.Open(os.Getenv("GOAI_TINYLLAMA_GGUF"))
	if f == nil {
		t.Skip("no model")
	}
	defer f.Close()
	raw, _ := gguf.ReadRaw(f)
	byName := map[string]uint32{}
	for n, ti := range raw.Tensors {
		byName[n] = ti.GGType
	}
	agree, disagree := 0, 0
	var sample []string
	keys := make([]string, 0, len(byName))
	for k := range byName {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if !strings.HasSuffix(k, "attn_q.weight") {
			continue
		}
		base := strings.TrimSuffix(k, "attn_q.weight")
		q, okq := byName[base+"attn_q.weight"]
		kk, okk := byName[base+"attn_k.weight"]
		v, okv := byName[base+"attn_v.weight"]
		if !okq || !okk || !okv {
			continue
		}
		if q == kk && kk == v {
			agree++
		} else {
			disagree++
			if len(sample) < 4 {
				sample = append(sample, fmt.Sprintf("%s q=%d k=%d v=%d", base, q, kk, v))
			}
		}
	}
	fmt.Printf("QKVTYPE layers where q/k/v share a quant type: %d   differ: %d\n", agree, disagree)
	for _, s := range sample {
		fmt.Printf("QKVTYPE   %s\n", s)
	}
}
