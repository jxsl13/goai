//go:build darwin && cgo

package llamagpu

import (
	"math"
	"slices"
	"testing"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
)

func TestMetalQuantPrefillGateUpRouteExact(t *testing.T) {
	if !metal.Available() {
		t.Skip("Metal unavailable")
	}
	metal.SetWeightCacheGB(0)
	metal.SetWeightCacheGB(1)
	t.Cleanup(func() {
		metal.SetWeightCacheGB(0)
		metal.SetWeightCacheGB(4)
	})
	cfg := nlp.LlamaConfig{
		Vocab: 256, Ctx: 96, Dim: 256, Heads: 4, KVHeads: 1, Layers: 1,
		Hidden: 512, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 73)
	if err != nil {
		t.Fatal(err)
	}
	qm, err := nlp.QuantizeLlama(m, gguf.Q4_K)
	if err != nil {
		t.Fatal(err)
	}
	defer qm.Close()

	for _, rows := range []int{24, 64} {
		t.Run(prefillGateUpRowsName(rows), func(t *testing.T) {
			control, err := newQuantMetalWithFeatures(qm, false, true, false)
			if err != nil {
				t.Fatal(err)
			}
			defer control.Release()
			candidate, err := newQuantMetalWithFeatures(qm, false, true, true)
			if err != nil {
				t.Fatal(err)
			}
			defer candidate.Release()
			if control.blocks[0].wGUPrefill != nil {
				t.Fatal("control unexpectedly contains grouped prefill weight")
			}
			if candidate.blocks[0].wGUPrefill == nil || candidate.blocks[0].wG == nil || candidate.blocks[0].wU == nil {
				t.Fatal("candidate must retain separate decode weights and add one grouped prefill weight")
			}
			prompt := make([]int, rows)
			for i := range prompt {
				prompt[i] = 1 + (i*29)%200
			}
			want, err := control.StepNLast(prompt, 0)
			if err != nil {
				t.Fatal(err)
			}
			got, err := candidate.StepNLast(prompt, 0)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got, want) {
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("logit %d = %08x, want %08x", i, math.Float32bits(got[i]), math.Float32bits(want[i]))
					}
				}
			}
		})
	}
}

func prefillGateUpRowsName(rows int) string {
	if rows == 24 {
		return "rows24"
	}
	return "rows64"
}
