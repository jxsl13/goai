package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// TestLlamaFromHFSharded is the end-to-end smoke for sharded checkpoints
// (safetensors.LoadSharded): the golden sharded fixture is a real transformers
// LlamaForCausalLM split into 5 shards by save_pretrained (llama-shaped on
// purpose — geometry mirrors the llama_hf_gqa golden), so the merged map feeds
// LlamaFromHF unchanged and the forward pass must produce logits identical to
// the same model loaded from its single-file export. The fixture lives in
// format/safetensors/testdata (shared rather than duplicated; regenerate with
// scripts/golden_sharded_safetensors.py).
func TestLlamaFromHFSharded(t *testing.T) {
	cfg := nlp.LlamaConfig{Heads: 4, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, Ctx: 32}
	tokens := []int{1, 4, 7, 2, 9, 0, 3}

	shardedTS, _, err := safetensors.LoadSharded("../format/safetensors/testdata/sharded/model.safetensors.index.json")
	if err != nil {
		t.Fatalf("LoadSharded (run scripts/golden_sharded_safetensors.py): %v", err)
	}
	singleTS, _, err := safetensors.LoadFile("../format/safetensors/testdata/sharded_single.safetensors")
	if err != nil {
		t.Fatalf("LoadFile single: %v", err)
	}

	forward := func(ts map[string]*tensor.Tensor) *tensor.Tensor {
		t.Helper()
		model, err := nlp.LlamaFromHF(ts, cfg)
		if err != nil {
			t.Fatalf("LlamaFromHF: %v", err)
		}
		out, err := model.Forward(backend.NewContext(), tokens)
		if err != nil {
			t.Fatalf("forward: %v", err)
		}
		return out
	}
	got, want := forward(shardedTS), forward(singleTS)
	if !got.Shape().Equal(want.Shape()) {
		t.Fatalf("logit shape %v != %v", got.Shape(), want.Shape())
	}
	var l1 float64
	for i := range want.Shape()[0] {
		for j := range want.Shape()[1] {
			if got.AtF64(i, j) != want.AtF64(i, j) {
				t.Fatalf("logit (%d,%d): sharded %v != single %v", i, j, got.AtF64(i, j), want.AtF64(i, j))
			}
			l1 += math.Abs(want.AtF64(i, j))
		}
	}
	if l1 == 0 {
		t.Fatal("all-zero logits — the smoke exercised nothing")
	}
}
