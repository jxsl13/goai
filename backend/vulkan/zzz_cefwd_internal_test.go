//go:build vulkan && cgo

package vulkan

import (
	"math"
	"os"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// §T403: the new vulkan OpCrossEntropy + OpEmbed forwards == the reference (parity with metal
// §T401/T402; both were silent CPU fallbacks on vulkan too).
func TestVulkanCEEmbedForwardCrossReference(t *testing.T) {
	if os.Getenv("GOAI_VULKAN_SOFTWARE_ICD") != "" {
		// Known value divergence on software ICDs (Mesa lavapipe), found by the
		// §T600 CI lane; green on MoltenVK. Root-cause investigation: SPEC §T603.
		t.Skip("skipped: known divergence on software Vulkan ICDs (lavapipe) — tracked as SPEC T603")
	}
	if !Available() {
		t.Skip("vulkan: no compute-capable device (§V4)")
	}
	vctx := backend.NewContext().WithBackend(Backend{})
	rctx := backend.NewContext().WithBackend(backend.Reference())
	rand := func(n, seed int) *tensor.Tensor {
		x := tensor.New(tensor.F32, tensor.Shape{n})
		s := x.Storage().F32()
		for i := range s {
			s[i] = float32((i*7+seed*13)%23-11) * 0.02
		}
		return x
	}
	resh := func(x *tensor.Tensor, r, c int) *tensor.Tensor {
		o, err := x.Reshape(tensor.Shape{r, c})
		if err != nil {
			t.Fatal(err)
		}
		return o
	}

	// crossentropy forward
	for _, sh := range [][2]int{{4, 16}, {8, 40}, {17, 4096}} {
		b, c := sh[0], sh[1]
		logits := resh(rand(b*c, 3), b, c)
		tg := tensor.New(tensor.F32, tensor.Shape{b})
		for i := 0; i < b; i++ {
			tg.SetF64(float64((i*3)%c), i)
		}
		g, _ := backend.Execute(vctx, backend.OpCrossEntropy, []*tensor.Tensor{logits, tg}, nil)
		w, _ := backend.Execute(rctx, backend.OpCrossEntropy, []*tensor.Tensor{logits, tg}, nil)
		if math.Abs(g[0].AtF64()-w[0].AtF64()) > 1e-4*math.Max(1, math.Abs(w[0].AtF64())) {
			t.Fatalf("CE [%dx%d]: vk %v vs ref %v", b, c, g[0].AtF64(), w[0].AtF64())
		}
	}

	// embed forward
	const n, d, m = 4096, 512, 256
	table := resh(rand(n*d, 8), n, d)
	idx := tensor.New(tensor.F32, tensor.Shape{m})
	for i := 0; i < m; i++ {
		idx.SetF64(float64((i*7)%n), i)
	}
	g, _ := backend.Execute(vctx, backend.OpEmbed, []*tensor.Tensor{table, idx}, nil)
	w, _ := backend.Execute(rctx, backend.OpEmbed, []*tensor.Tensor{table, idx}, nil)
	gs, ws := g[0].Storage().F32(), w[0].Storage().F32()
	for i := range ws {
		if gs[i] != ws[i] {
			t.Fatalf("embed [%d]: vk %v vs ref %v", i, gs[i], ws[i])
		}
	}
}
