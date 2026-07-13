package nlp_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §V16 tier-1 (§R96): NEFTune plugs into training via Embed → nn.NEFTune →
// ForwardFromEmbed — the noise perturbs the logits (vs a clean forward) and gradients
// still flow to the token embedding, so a model can be fine-tuned with noisy embeddings.
func TestNEFTuneTrainingIntegration(t *testing.T) {
	m := tinyLlama(t)
	tokens := []int{1, 3, 2, 5}

	clean, err := m.Forward(backend.NewContext(), tokens)
	if err != nil {
		t.Fatal(err)
	}

	tape := autograd.NewTape()
	emb, err := m.Embed(tape.Context(), tokens)
	if err != nil {
		t.Fatal(err)
	}
	noisy, err := nn.NEFTune(tape.Context(), emb, 5, rand.New(rand.NewPCG(7, 11)))
	if err != nil {
		t.Fatal(err)
	}
	logits, err := m.ForwardFromEmbed(tape.Context(), noisy)
	if err != nil {
		t.Fatal(err)
	}

	// the noise changed the output relative to a clean forward
	var diff float64
	for i := range logits.Numel() {
		idx := tensor.Unravel(i, logits.Shape())
		diff = math.Max(diff, math.Abs(logits.AtF64(idx...)-clean.AtF64(idx...)))
	}
	if diff == 0 {
		t.Error("NEFTune noise did not change the logits")
	}

	// gradients still reach the token embedding through the noisy path
	loss, err := backend.Execute(tape.Context(), backend.OpMean, []*tensor.Tensor{logits}, backend.ReduceAttrs{})
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	if tape.Grad(m.TokEmb) == nil {
		t.Error("token embedding got no gradient through the NEFTune training path")
	}
}
