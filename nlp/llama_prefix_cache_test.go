package nlp_test

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
)

func ExampleLlamaPrefixCache() {
	m, _ := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 8, Ctx: 16, Dim: 8, Heads: 2, KVHeads: 1,
		Layers: 2, Hidden: 16, Eps: 1e-5,
	}, 7)
	cache, _ := nlp.NewLlamaPrefixCache(m)
	ctx := backend.NewContext()

	_, firstReuse, _ := cache.Prefill(ctx, []int{1, 2, 3, 4})
	logits, secondReuse, _ := cache.Prefill(ctx, []int{1, 2, 3, 4, 5, 6})
	fmt.Println(firstReuse, secondReuse, logits.Shape())
	// Output: 0 4 (1, 8)
}
