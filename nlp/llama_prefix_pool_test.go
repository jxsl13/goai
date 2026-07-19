package nlp_test

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
)

func ExampleLlamaPrefixPool() {
	m, _ := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 8, Ctx: 16, Dim: 8, Heads: 2, KVHeads: 1,
		Layers: 2, Hidden: 16, Eps: 1e-5,
	}, 7)
	pool, _ := nlp.NewLlamaPrefixPool(m, 2)
	ctx := backend.NewContext()
	a := []int{1, 2, 3, 4}
	b := []int{1, 2, 5, 6}

	_, first, _ := pool.Prefill(ctx, "team", a)
	_, second, _ := pool.Prefill(ctx, "team", b)
	logits, again, _ := pool.Prefill(ctx, "team", a)
	fmt.Println(first, second, again, pool.Len(), logits.Shape())
	// Output: 0 2 4 2 (1, 8)
}
