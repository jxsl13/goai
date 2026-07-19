package nlp_test

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
)

func ExampleLayerSkipTrainConfig_DropoutRates() {
	cfg := nlp.LayerSkipTrainConfig{MaxDropout: 0.2}
	rates, err := cfg.DropoutRates(5)
	if err != nil {
		panic(err)
	}
	fmt.Printf("first=%.3f middle=%.3f last=%.3f\n", rates[0], rates[2], rates[4])
	// Output: first=0.000 middle=0.083 last=0.200
}

func ExampleLayerSkipTrainConfig_ExitWeights() {
	cfg := nlp.LayerSkipTrainConfig{ExitScale: 0.25}
	weights, err := cfg.ExitWeights(5)
	if err != nil {
		panic(err)
	}
	fmt.Printf("%.3f %.3f %.3f %.3f %.3f\n", weights[0], weights[1], weights[2], weights[3], weights[4])
	// Output: 0.000 0.031 0.094 0.188 0.688
}

func ExampleLlama_LayerSkipLoss() {
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 8, Ctx: 8, Dim: 8, Heads: 2, Layers: 3, Hidden: 12, Eps: 1e-5,
	}, 4)
	if err != nil {
		panic(err)
	}
	targets := layerSkipTargets(2, 3, 1)
	loss, err := m.LayerSkipLoss(backend.NewContext(), []int{1, 2, 3}, targets, nlp.LayerSkipTrainConfig{
		MaxDropout: 0.1, ExitScale: 0.1, Seed: 7,
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(loss.Shape(), loss.AtF64() > 0)
	// Output: () true
}
