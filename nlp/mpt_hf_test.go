package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/format/npy"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
)

// TestMPTFromHF is the forward-parity anchor for the MPT converter (§V16): a real
// transformers MptForCausalLM must reproduce its logits after MPTFromHF loads it. MPT is the
// library's first ALiBi model, so this test doubles as the check that GoAI's OpMHA ALiBi
// slopes and bias convention match MosaicML's — the slopes are the geometric sequence
// 2^(−8·(h+1)/n) (alibi_bias_max=8) for the golden's power-of-two head count, and the
// key-index vs relative-distance bias forms differ only by a softmax-invariant per-row
// constant, so the attention weights coincide exactly.
func TestMPTFromHF(t *testing.T) {
	ts, _, err := safetensors.LoadFile("testdata/mpt_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	model, err := nlp.MPTFromHF(ts, nlp.MPTConfig{Heads: 4, Eps: 1e-5, Ctx: 32})
	if err != nil {
		t.Fatalf("MPTFromHF: %v", err)
	}
	golden, err := npy.LoadFile("testdata/mpt_hf_logits.npy")
	if err != nil {
		t.Fatal(err)
	}
	got, err := model.Forward(backend.NewContext(), []int{3, 7, 1, 9, 4, 2, 8})
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if got.Shape()[0] != golden.Shape()[0] || got.Shape()[1] != golden.Shape()[1] {
		t.Fatalf("shape %v != golden %v", got.Shape(), golden.Shape())
	}
	var maxAbs float64
	for i := range golden.Shape()[0] {
		for j := range golden.Shape()[1] {
			if d := math.Abs(got.AtF64(i, j) - golden.AtF64(i, j)); d > maxAbs {
				maxAbs = d
			}
		}
	}
	t.Logf("MPT max abs logit diff vs transformers: %.3e", maxAbs)
	if maxAbs > 2e-3 {
		t.Fatalf("MPT diverges from transformers: %.3e", maxAbs)
	}
}
