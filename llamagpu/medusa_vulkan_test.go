//go:build darwin && cgo && vulkan

package llamagpu_test

import (
	"testing"

	"github.com/jxsl13/goai/backend/vulkan"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// §T446: the batched-Medusa loop is backend-agnostic core code — the ε=1 collapse onto plain
// greedy Generate must hold on the Vulkan decoder exactly as on Metal.
func TestMedusaGenerateGPTVulkanAllRejectIsGreedy(t *testing.T) {
	if !vulkan.Available() {
		t.Skip("vulkan: no device")
	}
	cfg := nlp.GPTConfig{Vocab: 48, Ctx: 64, Dim: 64, Heads: 4, Layers: 2, Eps: 1e-5}
	m := randGPT(t, cfg)
	heads, err := nlp.NewMedusaHeads(3, cfg.Dim, cfg.Vocab, 9)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := llamagpu.NewGPTVulkan(m)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Release()
	prompt := []int{5, 12, 3}
	const maxNew = 24
	got, stats, err := llamagpu.MedusaGenerate(dec, heads, prompt, maxNew, 1.0, 1e9)
	if err != nil {
		t.Fatal(err)
	}
	ref, err := llamagpu.NewGPTVulkan(m)
	if err != nil {
		t.Fatal(err)
	}
	defer ref.Release()
	want, err := ref.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token[%d]: medusa %d != plain %d", i, got[i], want[i])
		}
	}
	if stats.Accepted != 0 {
		t.Fatalf("ε=1 accepted %d", stats.Accepted)
	}
}
