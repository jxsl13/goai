package nlp_test

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

func ExampleLlama_offloadPlan() {
	model, _ := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 11, Ctx: 8, Dim: 8, Heads: 2, KVHeads: 1, Layers: 2, Hidden: 16, Eps: 1e-5,
	}, 1)
	plan, _ := backend.PlanOffload(backend.OffloadConfig{
		LayerBytes: []int64{1 << 20, 1 << 20},
		Budgets:    map[backend.Name]int64{backend.CPU: 1 << 20},
		Prefer:     []backend.Name{backend.CPU},
		Fallback:   backend.Ref,
	})
	model.LayerBackends = plan.Layers
	logits, _ := model.Forward(backend.NewContext(), []int{2, 5})
	fmt.Println(logits.Shape(), plan.OnFallback)
	// Output: (2, 11) 1
}
