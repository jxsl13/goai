package nlp

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

var prefillAppendBenchOut *tensor.Tensor

// Shared-prefix workload: a 96-token document/system prefix and an 8-token new
// turn. Prefix setup is intentionally outside the append timer, as it is in APC.
func BenchmarkLlamaPrefillAppendSharedPrefix(b *testing.B) {
	m, err := NewLlama(LlamaConfig{
		Vocab: 128, Ctx: 128, Dim: 16, Heads: 4, KVHeads: 2,
		Layers: 3, Hidden: 32, Eps: 1e-5, RopeBase: 10000,
	}, 17)
	if err != nil {
		b.Fatal(err)
	}
	const prefixLen, suffixLen = 96, 8
	full := make([]int, prefixLen+suffixLen)
	for i := range full {
		full[i] = (i*17 + 3) % m.Config.Vocab
	}
	ctx := (&backend.Context{Backend: backend.Reference()})

	b.Run("full_recompute", func(b *testing.B) {
		b.ReportAllocs()
		for range b.N {
			out, err := m.Prefill(ctx, m.NewCache(), full)
			if err != nil {
				b.Fatal(err)
			}
			prefillAppendBenchOut = out
		}
	})

	b.Run("reuse_96_append_8", func(b *testing.B) {
		cache := m.NewCache()
		if _, err := m.Prefill(ctx, cache, full[:prefixLen]); err != nil {
			b.Fatal(err)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for range b.N {
			out, err := m.PrefillAppend(ctx, cache, full[prefixLen:])
			if err != nil {
				b.Fatal(err)
			}
			prefillAppendBenchOut = out
			b.StopTimer()
			if err := cache.truncate(prefixLen); err != nil {
				b.Fatal(err)
			}
			b.StartTimer()
		}
	})
}
