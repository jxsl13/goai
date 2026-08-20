//go:build vulkan && cgo

package vulkan_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/vulkan"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// skipNoGPU implements §V4: missing accel → skip WITH log, never silent pass.
func skipNoGPU(t *testing.T) {
	t.Helper()
	if !vulkan.Available() {
		t.Skip("vulkan: no compute-capable Vulkan device — skipping (§V4)")
	}
}

func TestVulkanMemoryProber(t *testing.T) {
	skipNoGPU(t)
	var p backend.MemoryProber = vulkan.Backend{}
	bytes, ok := p.AvailableMemory()
	budgets := backend.ProbeBudgets(backend.Vulkan)
	if !ok {
		if _, present := budgets[backend.Vulkan]; present {
			t.Fatalf("unknown direct budget must be omitted, got %d", budgets[backend.Vulkan])
		}
		t.Log("VK_EXT_memory_budget unavailable; ProbeBudgets safely omits Vulkan")
		return
	}
	if bytes <= 0 || budgets[backend.Vulkan] <= 0 {
		t.Fatalf("Vulkan budget direct=%d, ProbeBudgets=%d", bytes, budgets[backend.Vulkan])
	}
}

// crossTol is the §V11 tolerance for GPU f32 GEMM: the tiled matmul.comp shader
// accumulates in f32 (reordered vs the f64 reference), so rtol scales with the
// reduction length: rtol(K) = 2.5e-6·√K.
//
// The √K FORM is dictated by f32 accumulation error (~O(√K) growth); the CONSTANT
// was raised 1e-6 → 2.5e-6 (§B45/§T624 backprop) after measuring the vulkan
// matmul.comp path vs the f64 ref across realistic hidden dims. The old 1e-6
// constant only had headroom for K≤512 (it was NEVER exercised above 512, matmul
// cross-ref topped out at 5×7 / 256×512) and FAILED at model-scale K — measured
// maxRel (vulkan f32 vs ref f64, worst of 8 seed pairs, m=n=256):
//
//	K= 128  3.7e-6   old-tol(1e-6) 1.13e-5   new-tol(2.5e-6) 2.83e-5
//	K= 512  1.5e-5   old-tol 2.26e-5   new-tol 5.66e-5
//	K= 768  2.6e-5   old-tol 2.77e-5   new-tol 6.93e-5   (old GRAZED: 1.07× margin)
//	K=1024  2.7e-5   old-tol 3.20e-5   new-tol 8.00e-5
//	K=2048  4.9e-5   old-tol 4.53e-5   new-tol 1.13e-4   (old FAILED: tol<err)
//	K=4096  8.3e-5   old-tol 6.40e-5   new-tol 1.60e-4   (old FAILED: tol<err)
//
// The vulkan tiled-shader error profile is nearly identical to metal's MPS GEMM
// (§B45) despite the different accumulation order, so the SAME 2.5e-6 constant
// applies. It keeps a ~1.9–7.6× margin over the worst measured f32 error at every
// tested K (thinnest 1.93× at K=4096), yet stays ≈100× below the ≥1e-2 error a
// REAL matmul bug produces, so it still catches bugs (non-vacuousness assert below).
func crossTol(k int) float64 { return 2.5e-6 * math.Sqrt(float64(k)) }

// §V3/§V11: the Vulkan compute matmul matches the Pure-Go reference within the
// K-scaled tolerance across shapes — guards the shader's row-major indexing and
// the ceil(N/16)×ceil(M/16) dispatch mapping (§R44).
func TestVulkanCrossReference(t *testing.T) {
	skipNoGPU(t)
	vb, ok := backend.Get(backend.Vulkan)
	if !ok {
		t.Fatal("vulkan available but not registered")
	}
	ref, _ := backend.Get(backend.Ref)

	shapes := []struct{ m, k, n int }{
		{1, 1, 1}, {2, 3, 4}, {16, 16, 16}, {17, 15, 33}, {128, 128, 128}, {256, 512, 128},
		// Model-scale reduction dims (hidden sizes): the matmul cross-ref was
		// previously untested above K=512, where f32 accumulation error grows
		// ~√K and the old 1e-6·√K tol grazed (K=768) then failed (K≥2048) —
		// the §B45/§T624 latent tol bug. Shapes match the crossTol calibration
		// measurements documented above.
		{256, 768, 256}, {512, 1024, 512}, {512, 2048, 512}, {512, 4096, 512},
	}
	for _, s := range shapes {
		a := bench.RandF32(tensor.Shape{s.m, s.k}, 1)
		b := bench.RandF32(tensor.Shape{s.k, s.n}, 2)
		gv, err := backend.Execute(backend.NewContext().WithBackend(vb), backend.OpMatMul, []*tensor.Tensor{a, b}, nil)
		if err != nil {
			t.Fatalf("vulkan %v: %v", s, err)
		}
		gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpMatMul, []*tensor.Tensor{a, b}, nil)
		if err != nil {
			t.Fatal(err)
		}
		rtol := crossTol(s.k)
		for i := range gv[0].Numel() {
			idx := tensor.Unravel(i, gv[0].Shape())
			g, r := gv[0].AtF64(idx...), gr[0].AtF64(idx...)
			if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r)) {
				t.Fatalf("shape %v [%d]: vulkan %v vs ref %v (rtol %g)", s, i, g, r, rtol)
			}
		}

		// Non-vacuousness (§B45/§T624): confirm crossTol would CATCH a real bug
		// at this K. A genuine matmul error yields ≥1e-2 rel deviation — perturb
		// one element by 1e-2·|r| and assert it exceeds the (loosened) tolerance,
		// so the wider constant still fails on real regressions, not just noise.
		if n := gv[0].Numel(); n > 0 {
			idx := tensor.Unravel(n/2, gv[0].Shape())
			r := gr[0].AtF64(idx...)
			bug := 1e-2 * math.Max(1, math.Abs(r))
			if bug <= rtol*math.Max(1, math.Abs(r)) {
				t.Fatalf("shape %v: crossTol %g too loose — a 1e-2 rel bug (Δ=%g) would slip through", s, rtol, bug)
			}
		}
	}
}

// A non-square product not aligned to the 16×16 workgroup is the sharpest check
// that the dispatch overhang mask (row>=M||col>=N → return) and row-major
// indexing are correct.
func TestVulkanRectangularUnaligned(t *testing.T) {
	skipNoGPU(t)
	vb, _ := backend.Get(backend.Vulkan)
	ref, _ := backend.Get(backend.Ref)
	a := bench.RandF32(tensor.Shape{7, 13}, 5)
	b := bench.RandF32(tensor.Shape{13, 3}, 6)
	gv, err := backend.Execute(backend.NewContext().WithBackend(vb), backend.OpMatMul, []*tensor.Tensor{a, b}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !gv[0].Shape().Equal(tensor.Shape{7, 3}) {
		t.Fatalf("result shape %v, want [7 3]", gv[0].Shape())
	}
	gr, _ := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpMatMul, []*tensor.Tensor{a, b}, nil)
	for i := range gv[0].Numel() {
		idx := tensor.Unravel(i, gv[0].Shape())
		if math.Abs(gv[0].AtF64(idx...)-gr[0].AtF64(idx...)) > crossTol(13)*math.Max(1, math.Abs(gr[0].AtF64(idx...))) {
			t.Fatalf("rect [%d]: vulkan %v vs ref %v", i, gv[0].AtF64(idx...), gr[0].AtF64(idx...))
		}
	}
}

// §T46: with the vulkan backend built in and a device present, backend.Default()
// AUTO-SELECTS it — the user writes no backend-selection code, yet matmuls run on
// the GPU (vulkan is ahead of cpu in the default preference order).
func TestVulkanAutoSelected(t *testing.T) {
	skipNoGPU(t)
	if got := backend.Default().Name(); got != backend.Vulkan {
		t.Errorf("backend.Default() = %q, want vulkan (auto-selected by preference)", got)
	}
}

// §V3/§V11: Vulkan fused attention (OpMHA) matches the Pure-Go reference within an
// f32 tolerance, across heads, GQA, causal, KV-cache (sq<sk) and attn-scale (§T89).
func TestVulkanMHACrossReference(t *testing.T) {
	skipNoGPU(t)
	vb, _ := backend.Get(backend.Vulkan)
	ref, _ := backend.Get(backend.Ref)

	cases := []struct {
		name              string
		sq, sk, heads, kv int
		dk                int
		causal            bool
		scale             float64
	}{
		{"mha", 4, 4, 2, 2, 4, false, 1},
		{"causal", 6, 6, 4, 4, 4, true, 1},
		{"gqa", 5, 5, 4, 2, 8, true, 1},
		{"mqa", 8, 8, 8, 1, 4, false, 1},
		{"kvcache", 3, 7, 2, 2, 8, true, 1},
		{"attnscale", 6, 6, 4, 4, 8, true, 1.2},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dm := c.heads * c.dk
			dkv := c.kv * c.dk
			q := bench.RandF32(tensor.Shape{c.sq, dm}, 1)
			k := bench.RandF32(tensor.Shape{c.sk, dkv}, 2)
			v := bench.RandF32(tensor.Shape{c.sk, dkv}, 3)
			attrs := backend.AttnAttrs{Heads: c.heads, KVHeads: c.kv, Causal: c.causal, Scale: c.scale}
			ins := []*tensor.Tensor{q, k, v}
			gv, err := backend.Execute(backend.NewContext().WithBackend(vb), backend.OpMHA, ins, attrs)
			if err != nil {
				t.Fatalf("vulkan mha: %v", err)
			}
			gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpMHA, ins, attrs)
			if err != nil {
				t.Fatal(err)
			}
			rtol := crossTol(c.dk+c.sk) + 1e-6
			for i := range gv[0].Numel() {
				idx := tensor.Unravel(i, gv[0].Shape())
				g, r := gv[0].AtF64(idx...), gr[0].AtF64(idx...)
				if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r))+1e-5 {
					t.Fatalf("%s [%d]: vulkan %v vs ref %v (rtol %g)", c.name, i, g, r, rtol)
				}
			}
		})
	}
}

// §V3/§V11: Vulkan FlashAttention-2 forward (OpFlashAttn) matches BOTH the reference
// flashattn AND the reference naive attention (OpMHA) within an f32 tolerance — the
// online-softmax tiling is exact, differing only by float reassociation — across
// heads, causal masking and per-head dims (§T110).
func TestVulkanFlashAttnCrossReference(t *testing.T) {
	skipNoGPU(t)
	vb, _ := backend.Get(backend.Vulkan)
	ref, _ := backend.Get(backend.Ref)

	cases := []struct {
		name           string
		seq, heads, kv int
		dk             int
		causal         bool
	}{
		{"flash", 4, 2, 2, 4, false},
		{"causal", 6, 4, 4, 4, true},
		{"big", 8, 8, 8, 16, true},
		{"onehead", 5, 1, 1, 8, false},
		{"wide", 10, 2, 2, 32, true},
		{"gqa", 8, 4, 2, 8, true},   // grouped-query
		{"mqa", 7, 6, 1, 4, true},   // multi-query
		{"gqa3", 9, 6, 3, 4, false}, // grouped-query, non-causal
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dm, dkv := c.heads*c.dk, c.kv*c.dk
			q := bench.RandF32(tensor.Shape{c.seq, dm}, 1)
			k := bench.RandF32(tensor.Shape{c.seq, dkv}, 2)
			v := bench.RandF32(tensor.Shape{c.seq, dkv}, 3)
			ins := []*tensor.Tensor{q, k, v}
			attrs := backend.AttnAttrs{Heads: c.heads, KVHeads: c.kv, Causal: c.causal}
			gv, err := backend.Execute(backend.NewContext().WithBackend(vb), backend.OpFlashAttn, ins, attrs)
			if err != nil {
				t.Fatalf("vulkan flashattn: %v", err)
			}
			gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpFlashAttn, ins, attrs)
			if err != nil {
				t.Fatal(err)
			}
			gmha, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpMHA, ins, attrs)
			if err != nil {
				t.Fatal(err)
			}
			rtol := crossTol(c.dk+c.seq) + 1e-6
			atol := 1e-5
			for i := range gv[0].Numel() {
				idx := tensor.Unravel(i, gv[0].Shape())
				g, r, mh := gv[0].AtF64(idx...), gr[0].AtF64(idx...), gmha[0].AtF64(idx...)
				if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r))+atol {
					t.Fatalf("%s [%d]: vulkan-flash %v vs ref-flash %v (rtol %g)", c.name, i, g, r, rtol)
				}
				if math.Abs(g-mh) > rtol*math.Max(1, math.Abs(mh))+atol {
					t.Fatalf("%s [%d]: vulkan-flash %v vs ref-mha %v (not exact attention)", c.name, i, g, mh)
				}
			}
		})
	}
}

// §V3/§V11: Vulkan fused attention with SLIDING-WINDOW (§T115, §R62) matches the
// reference — each query attends only its `window` most recent keys — across window
// sizes, heads, GQA and causal/non-causal, including a window wider than the sequence
// (≡ full attention), the KV-cache case (sq<sk) and window==1.
func TestVulkanMHAWindowCrossReference(t *testing.T) {
	skipNoGPU(t)
	vb, _ := backend.Get(backend.Vulkan)
	ref, _ := backend.Get(backend.Ref)

	cases := []struct {
		name              string
		sq, sk, heads, kv int
		dk, window        int
		causal            bool
	}{
		{"swa", 8, 8, 2, 2, 4, 3, true},
		{"swa-noncausal", 8, 8, 4, 4, 4, 2, false},
		{"swa-gqa", 8, 8, 4, 2, 8, 4, true},
		{"window>=seq", 6, 6, 2, 2, 4, 100, true},
		{"swa-kvcache", 3, 9, 2, 2, 8, 4, true},
		{"window1", 7, 7, 2, 2, 4, 1, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dm := c.heads * c.dk
			dkv := c.kv * c.dk
			q := bench.RandF32(tensor.Shape{c.sq, dm}, 1)
			k := bench.RandF32(tensor.Shape{c.sk, dkv}, 2)
			v := bench.RandF32(tensor.Shape{c.sk, dkv}, 3)
			attrs := backend.AttnAttrs{Heads: c.heads, KVHeads: c.kv, Causal: c.causal, Window: c.window}
			ins := []*tensor.Tensor{q, k, v}
			gv, err := backend.Execute(backend.NewContext().WithBackend(vb), backend.OpMHA, ins, attrs)
			if err != nil {
				t.Fatalf("vulkan mha window: %v", err)
			}
			gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpMHA, ins, attrs)
			if err != nil {
				t.Fatal(err)
			}
			rtol := crossTol(c.dk+c.window) + 1e-6
			for i := range gv[0].Numel() {
				idx := tensor.Unravel(i, gv[0].Shape())
				g, r := gv[0].AtF64(idx...), gr[0].AtF64(idx...)
				if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r))+1e-5 {
					t.Fatalf("%s [%d]: vulkan %v vs ref %v (rtol %g)", c.name, i, g, r, rtol)
				}
			}
		})
	}
}

// §I4: attention features the vulkan kernel does not implement (ALiBi) fall back to
// the reference and still match it bit-for-bit. Sliding window now runs on the GPU
// (§T115), covered by TestVulkanMHAWindowCrossReference.
func TestVulkanMHAFallbackFeatures(t *testing.T) {
	skipNoGPU(t)
	vb, _ := backend.Get(backend.Vulkan)
	ref, _ := backend.Get(backend.Ref)
	q := bench.RandF32(tensor.Shape{6, 8}, 1)
	k := bench.RandF32(tensor.Shape{6, 8}, 2)
	v := bench.RandF32(tensor.Shape{6, 8}, 3)
	ins := []*tensor.Tensor{q, k, v}
	for _, attrs := range []backend.AttnAttrs{
		{Heads: 2, Causal: true, ALiBi: true},
	} {
		gv, err := backend.Execute(backend.NewContext().WithBackend(vb), backend.OpMHA, ins, attrs)
		if err != nil {
			t.Fatalf("vulkan fallback %+v: %v", attrs, err)
		}
		gr, _ := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpMHA, ins, attrs)
		for i := range gv[0].Numel() {
			idx := tensor.Unravel(i, gv[0].Shape())
			if math.Abs(gv[0].AtF64(idx...)-gr[0].AtF64(idx...)) > 1e-12 {
				t.Fatalf("fallback %+v [%d]: vulkan != ref", attrs, i)
			}
		}
	}
}

// §V3/§V11: the Vulkan SDPA backward (OpMHABackward) matches the reference
// dQ/dK/dV within an f32 tolerance. Shared dK/dV accumulate via float atomics
// (VK_EXT_shader_atomic_float), whose non-deterministic order widens the tolerance
// with seq (§T90).
func TestVulkanMHABackwardCrossReference(t *testing.T) {
	skipNoGPU(t)
	vb, _ := backend.Get(backend.Vulkan)
	ref, _ := backend.Get(backend.Ref)

	cases := []struct {
		name               string
		seq, heads, kv, dk int
		causal             bool
		scale              float64
		window             int
	}{
		{"mha", 4, 2, 2, 4, false, 1, 0},
		{"causal", 6, 4, 4, 4, true, 1, 0},
		{"gqa", 6, 4, 2, 8, true, 1, 0},
		{"mqa", 8, 8, 1, 4, false, 1, 0},
		{"attnscale", 6, 4, 4, 8, true, 1.2, 0},
		{"swa", 8, 2, 2, 4, true, 1, 3},     // sliding-window backward (§T129)
		{"swa-gqa", 8, 4, 2, 8, true, 1, 4}, // grouped-query sliding window
		{"swa-window1", 7, 2, 2, 4, true, 1, 1},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dm := c.heads * c.dk
			dkv := c.kv * c.dk
			q := bench.RandF32(tensor.Shape{c.seq, dm}, 1)
			k := bench.RandF32(tensor.Shape{c.seq, dkv}, 2)
			v := bench.RandF32(tensor.Shape{c.seq, dkv}, 3)
			g := bench.RandF32(tensor.Shape{c.seq, dm}, 4)
			attrs := backend.AttnAttrs{Heads: c.heads, KVHeads: c.kv, Causal: c.causal, Scale: c.scale, Window: c.window}
			ins := []*tensor.Tensor{q, k, v, g}
			gv, err := backend.Execute(backend.NewContext().WithBackend(vb), backend.OpMHABackward, ins, attrs)
			if err != nil {
				t.Fatalf("vulkan mha-backward: %v", err)
			}
			gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpMHABackward, ins, attrs)
			if err != nil {
				t.Fatal(err)
			}
			rtol := crossTol(c.seq*c.dk) + 1e-5
			names := []string{"dQ", "dK", "dV"}
			for o := range gv {
				for i := range gv[o].Numel() {
					idx := tensor.Unravel(i, gv[o].Shape())
					gg, rr := gv[o].AtF64(idx...), gr[o].AtF64(idx...)
					if math.Abs(gg-rr) > rtol*math.Max(1, math.Abs(rr))+2e-4 {
						t.Fatalf("%s %s[%d]: vulkan %v vs ref %v (rtol %g)", c.name, names[o], i, gg, rr, rtol)
					}
				}
			}
		})
	}
}

// End-to-end GPU training (§T90): a full attention forward+backward on a
// vulkan-backed tape produces the same dQ/dK/dV as a reference-backed tape — the
// mha VJP dispatches OpMHABackward, so on a vulkan tape the backward runs on the
// GPU (when float atomics are available; else it falls back to ref and still
// matches).
func TestVulkanMHATrainingMatchesRef(t *testing.T) {
	skipNoGPU(t)
	vb, _ := backend.Get(backend.Vulkan)
	ref, _ := backend.Get(backend.Ref)
	q := bench.RandF32(tensor.Shape{6, 16}, 1)
	k := bench.RandF32(tensor.Shape{6, 16}, 2)
	v := bench.RandF32(tensor.Shape{6, 16}, 3)
	attrs := backend.AttnAttrs{Heads: 4, Causal: true}

	grads := func(be backend.Backend) [3]*tensor.Tensor {
		tp := autograd.NewTapeOn(be)
		out, err := backend.Execute(tp.Context(), backend.OpMHA, []*tensor.Tensor{q, k, v}, attrs)
		if err != nil {
			t.Fatal(err)
		}
		if err := tp.Backward(out[0]); err != nil {
			t.Fatal(err)
		}
		return [3]*tensor.Tensor{tp.Grad(q), tp.Grad(k), tp.Grad(v)}
	}
	gv, gr := grads(vb), grads(ref)
	names := []string{"dQ", "dK", "dV"}
	for o := range gv {
		if gv[o] == nil || gr[o] == nil {
			t.Fatalf("%s: nil gradient", names[o])
		}
		for i := range gv[o].Numel() {
			idx := tensor.Unravel(i, gv[o].Shape())
			if math.Abs(gv[o].AtF64(idx...)-gr[o].AtF64(idx...)) > 1e-3*math.Max(1, math.Abs(gr[o].AtF64(idx...)))+3e-4 {
				t.Fatalf("%s[%d]: vulkan-tape %v vs ref-tape %v", names[o], i, gv[o].AtF64(idx...), gr[o].AtF64(idx...))
			}
		}
	}
}

// §V3/§V11: Vulkan 2-D convolution matches the Pure-Go reference within an f32
// tolerance, across stride, padding, multi-channel/filter, ±bias (§T91).
func TestVulkanConv2DCrossReference(t *testing.T) {
	skipNoGPU(t)
	vb, _ := backend.Get(backend.Vulkan)
	ref, _ := backend.Get(backend.Ref)

	cases := []struct {
		name                   string
		n, c, h, wd, f, kh, kw int
		stride, pad            int
		bias                   bool
		big                    bool // §T624: model-scale case
	}{
		{"basic", 1, 1, 5, 5, 1, 3, 3, 1, 0, false, false},
		{"pad", 2, 3, 6, 6, 4, 3, 3, 1, 1, true, false},
		{"stride2", 1, 3, 8, 8, 5, 3, 3, 2, 1, true, false},
		{"1x1", 2, 4, 5, 5, 6, 1, 1, 1, 0, true, false},
		{"nonsquare", 1, 2, 7, 5, 3, 2, 3, 2, 1, false, false},
		// §T624: realistic CNN feature-map scales through the fused conv_igemm
		// (live) path — the tiny cases above never exercised a large C·KH·KW
		// reduction, the same coverage gap that hid §B45 (matmul) and §B56 (MHA).
		// deep28 has the largest reduction (k=576) so it is the crossTol stress
		// case; conv1x1 is a realistic bottleneck (k=256). Kept small (f64 ref is
		// O(N·F·HW·C·K²)) to stay well under the §B55 wall-time budget.
		{"fmap32", 4, 16, 32, 32, 32, 3, 3, 1, 1, true, true},    // reduction k=144
		{"deep28", 2, 64, 28, 28, 32, 3, 3, 1, 1, true, true},    // reduction k=576 (max)
		{"conv1x1", 8, 256, 14, 14, 128, 1, 1, 1, 0, true, true}, // reduction k=256
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			x := bench.RandF32(tensor.Shape{c.n, c.c, c.h, c.wd}, 1)
			w := bench.RandF32(tensor.Shape{c.f, c.c, c.kh, c.kw}, 2)
			ins := []*tensor.Tensor{x, w}
			if c.bias {
				ins = append(ins, bench.RandF32(tensor.Shape{c.f}, 3))
			}
			attrs := backend.ConvAttrs{Stride: c.stride, Pad: c.pad}
			gv, err := backend.Execute(backend.NewContext().WithBackend(vb), backend.OpConv2D, ins, attrs)
			if err != nil {
				t.Fatalf("vulkan conv2d: %v", err)
			}
			gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpConv2D, ins, attrs)
			if err != nil {
				t.Fatal(err)
			}
			if !gv[0].Shape().Equal(gr[0].Shape()) {
				t.Fatalf("shape %v vs ref %v", gv[0].Shape(), gr[0].Shape())
			}
			rtol := crossTol(c.c*c.kh*c.kw) + 1e-6
			var maxRel float64
			for i := range gv[0].Numel() {
				idx := tensor.Unravel(i, gv[0].Shape())
				g, r := gv[0].AtF64(idx...), gr[0].AtF64(idx...)
				rel := math.Abs(g-r) / math.Max(1, math.Abs(r))
				if rel > maxRel {
					maxRel = rel
				}
				if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r))+1e-5 {
					t.Fatalf("%s [%d]: vulkan %v vs ref %v (rtol %g)", c.name, i, g, r, rtol)
				}
			}
			// §T624: log the f32-vs-f64 error at scale so the margin is visible.
			if c.big {
				t.Logf("§T624 %s: k=%d rtol=%g maxRel=%g margin=%.2fx",
					c.name, c.c*c.kh*c.kw, rtol, maxRel, rtol/math.Max(maxRel, 1e-30))
			}
		})
	}
	// §T624 non-vacuousness: a gross conv error (1e-2, ≈100× the largest legit
	// f32 deviation) must still exceed the loosest tol used here (largest
	// reduction k=576) — the test can actually fail.
	grossTol := crossTol(64*3*3) + 1e-6
	if 1e-2 <= grossTol*1+1e-5 {
		t.Fatalf("§T624 vacuous: gross 1e-2 error within conv tol %g", grossTol)
	}
}

// §V3/§V11: the Vulkan conv2d backward (OpConv2DBackward) matches the reference
// dX/dW/dBias within an f32 tolerance. Shared dX/dW/dBias accumulate via float
// atomics (VK_EXT_shader_atomic_float), so the tolerance scales with contributions
// (§T102) — completes GPU parity for conv training (metal §T101).
func TestVulkanConv2DBackwardCrossReference(t *testing.T) {
	skipNoGPU(t)
	vb, _ := backend.Get(backend.Vulkan)
	ref, _ := backend.Get(backend.Ref)

	cases := []struct {
		name                   string
		n, c, h, wd, f, kh, kw int
		stride, pad            int
	}{
		{"basic", 1, 1, 5, 5, 1, 3, 3, 1, 0},
		{"pad", 2, 3, 6, 6, 4, 3, 3, 1, 1},
		{"stride2", 1, 3, 8, 8, 5, 3, 3, 2, 1},
		{"1x1", 2, 4, 5, 5, 6, 1, 1, 1, 0},
		{"nonsquare", 1, 2, 7, 5, 3, 2, 3, 2, 1},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			ho := (c.h+2*c.pad-c.kh)/c.stride + 1
			wo := (c.wd+2*c.pad-c.kw)/c.stride + 1
			x := bench.RandF32(tensor.Shape{c.n, c.c, c.h, c.wd}, 1)
			w := bench.RandF32(tensor.Shape{c.f, c.c, c.kh, c.kw}, 2)
			dO := bench.RandF32(tensor.Shape{c.n, c.f, ho, wo}, 3)
			ins := []*tensor.Tensor{x, w, dO}
			attrs := backend.ConvAttrs{Stride: c.stride, Pad: c.pad}
			gv, err := backend.Execute(backend.NewContext().WithBackend(vb), backend.OpConv2DBackward, ins, attrs)
			if err != nil {
				t.Fatalf("vulkan conv2d-backward: %v", err)
			}
			gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpConv2DBackward, ins, attrs)
			if err != nil {
				t.Fatal(err)
			}
			rtol := crossTol(c.n*ho*wo) + 1e-5
			names := []string{"dX", "dW", "dBias"}
			for o := range gv {
				for i := range gv[o].Numel() {
					idx := tensor.Unravel(i, gv[o].Shape())
					g, r := gv[o].AtF64(idx...), gr[o].AtF64(idx...)
					if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r))+2e-4 {
						t.Fatalf("%s %s[%d]: vulkan %v vs ref %v (rtol %g)", c.name, names[o], i, g, r, rtol)
					}
				}
			}
		})
	}
}

// Fallback: ops the vulkan backend does not serve route to the reference (§I4).
func TestVulkanFallback(t *testing.T) {
	skipNoGPU(t)
	vb, _ := backend.Get(backend.Vulkan)
	x := bench.RandF32(tensor.Shape{8}, 3)
	out, err := backend.Execute(backend.NewContext().WithBackend(vb), backend.OpExp, []*tensor.Tensor{x}, nil)
	if err != nil {
		t.Fatalf("fallback exp failed: %v", err)
	}
	if out[0].Numel() != 8 {
		t.Fatal("fallback result wrong")
	}
}

func benchMatMulOn(b *testing.B, name backend.Name, sz int) {
	be, ok := backend.Get(name)
	if !ok {
		b.Skipf("%s not available", name)
	}
	x := bench.RandF32(tensor.Shape{sz, sz}, 1)
	y := bench.RandF32(tensor.Shape{sz, sz}, 2)
	ctx := backend.NewContext().WithBackend(be)
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{x, y}, nil); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportMetric(2*float64(sz)*float64(sz)*float64(sz)/float64(b.Elapsed().Nanoseconds())*float64(b.N), "GFLOP/s")
}

// §C3 gate benchmarks: vulkan vs the ceiling-optimized Pure-Go cpu backend.
func BenchmarkMatMulF32_512_vulkan(b *testing.B)  { benchMatMulOn(b, backend.Vulkan, 512) }
func BenchmarkMatMulF32_512_cpu(b *testing.B)     { benchMatMulOn(b, backend.CPU, 512) }
func BenchmarkMatMulF32_1024_vulkan(b *testing.B) { benchMatMulOn(b, backend.Vulkan, 1024) }
func BenchmarkMatMulF32_1024_cpu(b *testing.B)    { benchMatMulOn(b, backend.CPU, 1024) }

// TestVulkanRetentionCrossReference V-CROSSes the Vulkan RetNet retention forward against the f64
// CPU reference OpRetention (§T173, the twin of Metal §T172): both compute the parallel form
// (QKᵀ⊙D)V, agreeing within crossTol (§V3/§V11), on MoltenVK.
func TestVulkanRetentionCrossReference(t *testing.T) {
	skipNoGPU(t)
	vb, _ := backend.Get(backend.Vulkan)
	ref, _ := backend.Get(backend.Ref)
	cases := []struct {
		l, d  int
		gamma float64
	}{
		{1, 4, 0.9}, {5, 8, 0.9}, {8, 16, 0.99}, {12, 6, 0.5}, {16, 32, 1.0}, {10, 64, 0.968},
	}
	for _, c := range cases {
		q := bench.RandF32(tensor.Shape{c.l, c.d}, 1)
		k := bench.RandF32(tensor.Shape{c.l, c.d}, 2)
		v := bench.RandF32(tensor.Shape{c.l, c.d}, 3)
		ins := []*tensor.Tensor{q, k, v}
		attrs := backend.RetentionAttrs{Gamma: c.gamma}
		gv, err := backend.Execute(backend.NewContext().WithBackend(vb), backend.OpRetention, ins, attrs)
		if err != nil {
			t.Fatalf("vulkan retention %+v: %v", c, err)
		}
		gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpRetention, ins, attrs)
		if err != nil {
			t.Fatal(err)
		}
		if !gv[0].Shape().Equal(tensor.Shape{c.l, c.d}) {
			t.Fatalf("%+v: shape %v", c, gv[0].Shape())
		}
		rtol, atol := crossTol(c.d+c.l), 1e-5
		for i := range gv[0].Numel() {
			idx := tensor.Unravel(i, gv[0].Shape())
			g, r := gv[0].AtF64(idx...), gr[0].AtF64(idx...)
			if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r))+atol {
				t.Errorf("%+v %v: vulkan %g vs ref %g (Δ %g)", c, idx, g, r, math.Abs(g-r))
			}
		}
	}
}

// TestVulkanRetentionBackwardCrossReference V-CROSSes the Vulkan retention backward against the f64
// CPU reference (§T175, the twin of Metal §T174): vulkan dQ/dK/dV == ref within crossTol, with the
// shared dK/dV atomically accumulated on MoltenVK (VK_EXT_shader_atomic_float).
func TestVulkanRetentionBackwardCrossReference(t *testing.T) {
	skipNoGPU(t)
	vb, _ := backend.Get(backend.Vulkan)
	ref, _ := backend.Get(backend.Ref)
	cases := []struct {
		l, d  int
		gamma float64
	}{
		{1, 4, 0.9}, {5, 8, 0.9}, {8, 16, 0.99}, {12, 6, 0.5}, {16, 32, 1.0}, {10, 64, 0.968},
	}
	for _, c := range cases {
		q := bench.RandF32(tensor.Shape{c.l, c.d}, 1)
		k := bench.RandF32(tensor.Shape{c.l, c.d}, 2)
		v := bench.RandF32(tensor.Shape{c.l, c.d}, 3)
		dO := bench.RandF32(tensor.Shape{c.l, c.d}, 4)
		ins := []*tensor.Tensor{q, k, v, dO}
		attrs := backend.RetentionAttrs{Gamma: c.gamma}
		gv, err := backend.Execute(backend.NewContext().WithBackend(vb), backend.OpRetentionBackward, ins, attrs)
		if err != nil {
			t.Fatalf("vulkan retention-backward %+v: %v", c, err)
		}
		gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpRetentionBackward, ins, attrs)
		if err != nil {
			t.Fatal(err)
		}
		rtol, atol := crossTol(c.d+c.l), 1e-4
		for oi, name := range []string{"dQ", "dK", "dV"} {
			for i := range gv[oi].Numel() {
				idx := tensor.Unravel(i, gv[oi].Shape())
				g, r := gv[oi].AtF64(idx...), gr[oi].AtF64(idx...)
				if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r))+atol {
					t.Errorf("%+v %s%v: vulkan %g vs ref %g (Δ %g)", c, name, idx, g, r, math.Abs(g-r))
				}
			}
		}
	}
}

// TestVulkanRetentionTrainsE2E proves retention TRAINS on Vulkan: a Vulkan-tape retention fwd+bwd
// produces the same Q/K/V gradients as a reference tape (§T175).
func TestVulkanRetentionTrainsE2E(t *testing.T) {
	skipNoGPU(t)
	vb, _ := backend.Get(backend.Vulkan)
	ref, _ := backend.Get(backend.Ref)
	const l, d, gamma = 8, 16, 0.95
	q := bench.RandF32(tensor.Shape{l, d}, 1)
	k := bench.RandF32(tensor.Shape{l, d}, 2)
	v := bench.RandF32(tensor.Shape{l, d}, 3)
	grads := func(be backend.Backend) (gq, gk, gv *tensor.Tensor) {
		tape := autograd.NewTapeOn(be)
		ctx := tape.Context()
		o, err := backend.Execute(ctx, backend.OpRetention, []*tensor.Tensor{q, k, v}, backend.RetentionAttrs{Gamma: gamma})
		if err != nil {
			t.Fatal(err)
		}
		loss, err := backend.Execute(ctx, backend.OpSum, []*tensor.Tensor{o[0]}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if err := tape.Backward(loss[0]); err != nil {
			t.Fatal(err)
		}
		return tape.Grad(q), tape.Grad(k), tape.Grad(v)
	}
	vq, vk, vv := grads(vb)
	rq, rk, rv := grads(ref)
	rtol, atol := crossTol(d+l), 1e-4
	for _, pair := range []struct {
		name string
		g, r *tensor.Tensor
	}{{"dQ", vq, rq}, {"dK", vk, rk}, {"dV", vv, rv}} {
		if pair.g == nil || pair.r == nil {
			t.Fatalf("%s: nil grad", pair.name)
		}
		for i := range pair.r.Numel() {
			idx := tensor.Unravel(i, pair.r.Shape())
			if math.Abs(pair.g.AtF64(idx...)-pair.r.AtF64(idx...)) > rtol*math.Max(1, math.Abs(pair.r.AtF64(idx...)))+atol {
				t.Errorf("%s%v: vulkan-tape %g vs ref-tape %g", pair.name, idx, pair.g.AtF64(idx...), pair.r.AtF64(idx...))
			}
		}
	}
}

// TestVulkanRetentionValueExpansion: retention with d_v≠d_k (RetNet value expansion) runs natively on
// the Vulkan kernel — FORWARD (§T179) and BACKWARD (§T192, the twin of Metal T191) — matching the
// reference within the §V11 f32 cross-tolerance, so the faithful (d_v=2·d_k) MSR block both infers and
// TRAINS on the GPU via MoltenVK.
func TestVulkanRetentionValueExpansion(t *testing.T) {
	skipNoGPU(t)
	vb, _ := backend.Get(backend.Vulkan)
	ref, _ := backend.Get(backend.Ref)
	cases := []struct{ l, dk, dv int }{{6, 3, 6}, {8, 4, 8}, {5, 8, 16}, {12, 16, 32}, {4, 5, 3}}
	for _, c := range cases {
		q := bench.RandF32(tensor.Shape{c.l, c.dk}, 1)
		k := bench.RandF32(tensor.Shape{c.l, c.dk}, 2)
		v := bench.RandF32(tensor.Shape{c.l, c.dv}, 3)
		attrs := backend.RetentionAttrs{Gamma: 0.9}
		gm, err := backend.Execute(backend.NewContext().WithBackend(vb), backend.OpRetention, []*tensor.Tensor{q, k, v}, attrs)
		if err != nil {
			t.Fatalf("%+v: vulkan retention d_v≠d_k: %v", c, err)
		}
		gr, _ := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpRetention, []*tensor.Tensor{q, k, v}, attrs)
		if !gm[0].Shape().Equal(tensor.Shape{c.l, c.dv}) {
			t.Fatalf("%+v: shape %v, want [%d %d]", c, gm[0].Shape(), c.l, c.dv)
		}
		rtol, atol := crossTol(c.dk+c.l), 1e-5
		for i := range gr[0].Numel() {
			idx := tensor.Unravel(i, gr[0].Shape())
			g, r := gm[0].AtF64(idx...), gr[0].AtF64(idx...)
			if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r))+atol {
				t.Errorf("%+v %v: vulkan %g vs ref %g (Δ %g)", c, idx, g, r, math.Abs(g-r))
			}
		}
	}
	// BACKWARD d_v≠d_k now runs on the GPU too (§T192) — dQ,dK [L,dk], dV [L,dv] match ref @crossTol.
	for _, c := range cases {
		q := bench.RandF32(tensor.Shape{c.l, c.dk}, 1)
		k := bench.RandF32(tensor.Shape{c.l, c.dk}, 2)
		v := bench.RandF32(tensor.Shape{c.l, c.dv}, 3)
		dO := bench.RandF32(tensor.Shape{c.l, c.dv}, 4)
		attrs := backend.RetentionAttrs{Gamma: 0.9}
		bm, err := backend.Execute(backend.NewContext().WithBackend(vb), backend.OpRetentionBackward, []*tensor.Tensor{q, k, v, dO}, attrs)
		if err != nil {
			t.Fatalf("%+v: vulkan retention-backward d_v≠d_k: %v", c, err)
		}
		br, _ := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpRetentionBackward, []*tensor.Tensor{q, k, v, dO}, attrs)
		wantShapes := []tensor.Shape{{c.l, c.dk}, {c.l, c.dk}, {c.l, c.dv}}
		rtol, atol := crossTol(c.dk+c.dv+c.l), 1e-5
		for oi := range 3 {
			if !bm[oi].Shape().Equal(wantShapes[oi]) {
				t.Fatalf("%+v backward[%d] shape %v, want %v", c, oi, bm[oi].Shape(), wantShapes[oi])
			}
			for i := range bm[oi].Numel() {
				idx := tensor.Unravel(i, bm[oi].Shape())
				g, r := bm[oi].AtF64(idx...), br[oi].AtF64(idx...)
				if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r))+atol {
					t.Errorf("%+v backward[%d] %v: vulkan %g vs ref %g (Δ %g)", c, oi, idx, g, r, math.Abs(g-r))
				}
			}
		}
	}
}

// §V-CROSS: the GPU RMSNorm matches the Pure-Go reference across shapes (including a
// row count not aligned to the 64-per-workgroup dispatch) and both an explicit and a
// defaulted (0→1e-5) epsilon. The GPU sums in f32 vs the reference's f64, so a
// dim-scaled tolerance applies (§V11).
func TestVulkanRMSNormCrossReference(t *testing.T) {
	skipNoGPU(t)
	vb, ok := backend.Get(backend.Vulkan)
	if !ok {
		t.Fatal("vulkan available but not registered")
	}
	ref, _ := backend.Get(backend.Ref)
	shapes := []tensor.Shape{{1, 1}, {2, 8}, {7, 33}, {65, 16}, {130, 128}, {4, 5, 64}}
	for _, sh := range shapes {
		d := sh[len(sh)-1]
		x := bench.RandF32(sh, 1)
		gamma := bench.RandF32(tensor.Shape{d}, 2)
		for _, eps := range []float64{0, 1e-5, 1e-2} {
			attrs := backend.NormAttrs{Eps: eps}
			gv, err := backend.Execute(backend.NewContext().WithBackend(vb), backend.OpRMSNorm, []*tensor.Tensor{x, gamma}, attrs)
			if err != nil {
				t.Fatalf("vulkan %v eps=%g: %v", sh, eps, err)
			}
			gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpRMSNorm, []*tensor.Tensor{x, gamma}, attrs)
			if err != nil {
				t.Fatal(err)
			}
			if !gv[0].Shape().Equal(gr[0].Shape()) {
				t.Fatalf("shape %v: vulkan %v vs ref %v", sh, gv[0].Shape(), gr[0].Shape())
			}
			rtol := crossTol(d)
			for i := range gv[0].Numel() {
				idx := tensor.Unravel(i, gv[0].Shape())
				g, r := gv[0].AtF64(idx...), gr[0].AtF64(idx...)
				if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r)) {
					t.Fatalf("shape %v eps=%g [%d]: vulkan %v vs ref %v (rtol %g)", sh, eps, i, g, r, rtol)
				}
			}
		}
	}
}

// §V-CROSS: the GPU RoPE matches the Pure-Go reference across shapes, head counts,
// position offsets, and the PI/YaRN frequency variants (which flow through the
// host-precomputed inv[]/posDiv). f32 cos/sin vs the reference's f64 → a small absolute
// tolerance (§V11). The xPos variant is exercised as a fallback (still == ref).
func TestVulkanRoPECrossReference(t *testing.T) {
	skipNoGPU(t)
	vb, ok := backend.Get(backend.Vulkan)
	if !ok {
		t.Fatal("vulkan available but not registered")
	}
	ref, _ := backend.Get(backend.Ref)
	cases := []struct {
		seq, width int
		attrs      backend.RoPEAttrs
	}{
		{1, 8, backend.RoPEAttrs{}},
		{4, 16, backend.RoPEAttrs{}},
		{7, 64, backend.RoPEAttrs{Heads: 2}},                                   // multi-head, hd=32
		{130, 128, backend.RoPEAttrs{Heads: 4}},                                // rows past one workgroup, hd=32
		{16, 32, backend.RoPEAttrs{PosOffset: 100}},                            // KV-cache decode offset
		{16, 32, backend.RoPEAttrs{PosScale: 4}},                               // linear PI
		{16, 64, backend.RoPEAttrs{YaRNScale: 2, YaRNOrigCtx: 2048, Heads: 2}}, // YaRN
		{8, 16, backend.RoPEAttrs{XPos: true}},                                 // xPos → ref fallback path
	}
	for _, c := range cases {
		q := bench.RandF32(tensor.Shape{c.seq, c.width}, 1)
		gv, err := backend.Execute(backend.NewContext().WithBackend(vb), backend.OpRoPE, []*tensor.Tensor{q}, c.attrs)
		if err != nil {
			t.Fatalf("vulkan %+v: %v", c, err)
		}
		gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpRoPE, []*tensor.Tensor{q}, c.attrs)
		if err != nil {
			t.Fatal(err)
		}
		for i := range gv[0].Numel() {
			idx := tensor.Unravel(i, gv[0].Shape())
			g, r := gv[0].AtF64(idx...), gr[0].AtF64(idx...)
			if math.Abs(g-r) > 1e-4*math.Max(1, math.Abs(r)) {
				t.Fatalf("case %+v [%d]: vulkan %v vs ref %v", c, i, g, r)
			}
		}
	}
}

// §V-CROSS: the GPU softmax matches the Pure-Go reference across shapes (rows past one
// workgroup, a 3-D input) — the row max-shift and normalization are exact up to the f32
// exp/division tolerance (§V11); rows include large-magnitude logits (§V12 stability).
func TestVulkanSoftmaxCrossReference(t *testing.T) {
	skipNoGPU(t)
	vb, ok := backend.Get(backend.Vulkan)
	if !ok {
		t.Fatal("vulkan available but not registered")
	}
	ref, _ := backend.Get(backend.Ref)
	shapes := []tensor.Shape{{1, 1}, {2, 8}, {7, 33}, {130, 64}, {4, 5, 16}}
	for _, sh := range shapes {
		x := bench.RandF32(sh, 3)
		gv, err := backend.Execute(backend.NewContext().WithBackend(vb), backend.OpSoftmax, []*tensor.Tensor{x}, nil)
		if err != nil {
			t.Fatalf("vulkan %v: %v", sh, err)
		}
		gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpSoftmax, []*tensor.Tensor{x}, nil)
		if err != nil {
			t.Fatal(err)
		}
		for i := range gv[0].Numel() {
			idx := tensor.Unravel(i, gv[0].Shape())
			g, r := gv[0].AtF64(idx...), gr[0].AtF64(idx...)
			if math.Abs(g-r) > 1e-5*math.Max(1, math.Abs(r)) {
				t.Fatalf("shape %v [%d]: vulkan %v vs ref %v", sh, i, g, r)
			}
		}
	}
}

// §V-CROSS: the GPU LayerNorm matches the Pure-Go reference across shapes (rows past one
// workgroup, a 3-D input) and explicit/defaulted epsilon. f32 mean/variance vs the
// reference's f64 → a dim-scaled tolerance (§V11).
func TestVulkanLayerNormCrossReference(t *testing.T) {
	skipNoGPU(t)
	vb, ok := backend.Get(backend.Vulkan)
	if !ok {
		t.Fatal("vulkan available but not registered")
	}
	ref, _ := backend.Get(backend.Ref)
	shapes := []tensor.Shape{{1, 1}, {2, 8}, {7, 33}, {130, 64}, {4, 5, 16}}
	for _, sh := range shapes {
		d := sh[len(sh)-1]
		x := bench.RandF32(sh, 1)
		gamma := bench.RandF32(tensor.Shape{d}, 2)
		beta := bench.RandF32(tensor.Shape{d}, 3)
		for _, eps := range []float64{0, 1e-5, 1e-2} {
			attrs := backend.NormAttrs{Eps: eps}
			gv, err := backend.Execute(backend.NewContext().WithBackend(vb), backend.OpLayerNorm, []*tensor.Tensor{x, gamma, beta}, attrs)
			if err != nil {
				t.Fatalf("vulkan %v eps=%g: %v", sh, eps, err)
			}
			gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpLayerNorm, []*tensor.Tensor{x, gamma, beta}, attrs)
			if err != nil {
				t.Fatal(err)
			}
			rtol := crossTol(d)
			for i := range gv[0].Numel() {
				idx := tensor.Unravel(i, gv[0].Shape())
				g, r := gv[0].AtF64(idx...), gr[0].AtF64(idx...)
				if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r)) {
					t.Fatalf("shape %v eps=%g [%d]: vulkan %v vs ref %v (rtol %g)", sh, eps, i, g, r, rtol)
				}
			}
		}
	}
}

// §V-CROSS: the generic GPU unary elementwise ops match the Pure-Go reference. Log/Sqrt
// use positive inputs (their domain); the rest use general inputs. f32 vs f64 → §V11 tol.
func TestVulkanUnaryCrossReference(t *testing.T) {
	skipNoGPU(t)
	vb, ok := backend.Get(backend.Vulkan)
	if !ok {
		t.Fatal("vulkan available but not registered")
	}
	ref, _ := backend.Get(backend.Ref)
	ops := []struct {
		op  backend.Op
		pos bool // restrict to positive inputs (Log/Sqrt domain)
	}{
		{backend.OpNeg, false}, {backend.OpExp, false}, {backend.OpLog, true},
		{backend.OpTanh, false}, {backend.OpReLU, false}, {backend.OpSigmoid, false},
		{backend.OpSiLU, false}, {backend.OpSqrt, true}, {backend.OpAbs, false},
		{backend.OpGELU, false}, // §T352: exact-erf GELU (A&S approx) now on-GPU
	}
	for _, sh := range []tensor.Shape{{1}, {300}, {4, 5}, {2, 3, 7}} {
		for _, o := range ops {
			x := bench.RandF32(sh, 5)
			if o.pos { // map to (0, ~3] so Log/Sqrt stay in-domain
				for i := range x.Numel() {
					idx := tensor.Unravel(i, sh)
					x.SetF64(math.Abs(x.AtF64(idx...))+0.1, idx...)
				}
			}
			gv, err := backend.Execute(backend.NewContext().WithBackend(vb), o.op, []*tensor.Tensor{x}, nil)
			if err != nil {
				t.Fatalf("vulkan %v %v: %v", o.op, sh, err)
			}
			gr, _ := backend.Execute(backend.NewContext().WithBackend(ref), o.op, []*tensor.Tensor{x}, nil)
			for i := range gv[0].Numel() {
				idx := tensor.Unravel(i, sh)
				g, r := gv[0].AtF64(idx...), gr[0].AtF64(idx...)
				if math.Abs(g-r) > 1e-5*math.Max(1, math.Abs(r)) {
					t.Fatalf("%v %v [%d]: vulkan %v vs ref %v", o.op, sh, i, g, r)
				}
			}
		}
	}
}

// §V-CROSS: the measured host-resident bias-gradient route is bit-identical to
// the reference's row-order F64 accumulation, including a non-contiguous input.
func TestVulkanAddBiasBackwardCrossReference(t *testing.T) {
	skipNoGPU(t)
	vb, ok := backend.Get(backend.Vulkan)
	if !ok {
		t.Fatal("vulkan available but not registered")
	}
	ref, _ := backend.Get(backend.Ref)
	inputs := []*tensor.Tensor{
		bench.RandF32(tensor.Shape{1, 1}, 1),
		bench.RandF32(tensor.Shape{3, 5}, 1),
		bench.RandF32(tensor.Shape{256, 2048}, 1),
		bench.RandF32(tensor.Shape{7, 512}, 1),
	}
	noncontiguous, err := bench.RandF32(tensor.Shape{9, 6}, 2).Transpose(0, 1)
	if err != nil {
		t.Fatal(err)
	}
	inputs = append(inputs, noncontiguous)
	for _, g := range inputs {
		sh := g.Shape()
		ins := []*tensor.Tensor{g}
		gv, err := backend.Execute(backend.NewContext().WithBackend(vb), backend.OpAddBiasBackward, ins, nil)
		if err != nil {
			t.Fatalf("vulkan addbias-backward %v: %v", sh, err)
		}
		gr, _ := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpAddBiasBackward, ins, nil)
		for j := range gv[0].Numel() {
			a, r := gv[0].AtF64(j), gr[0].AtF64(j)
			if a != r {
				t.Fatalf("addbias-backward %v [%d]: vulkan %v vs ref %v", sh, j, a, r)
			}
		}
	}
}

// §V-CROSS (§T356): matmul with a TRANSPOSED-view operand (the matmul backward's dO·Wᵀ /
// Xᵀ·dO) matches the reference — the shader reads it transposed via a flag instead of a CPU
// strided-gather copy, so the transposed and contiguous paths must agree.
func TestVulkanMatMulTransposedCrossReference(t *testing.T) {
	skipNoGPU(t)
	vb, ok := backend.Get(backend.Vulkan)
	if !ok {
		t.Fatal("vulkan available but not registered")
	}
	ref, _ := backend.Get(backend.Ref)
	check := func(name string, ins []*tensor.Tensor) {
		gv, err := backend.Execute(backend.NewContext().WithBackend(vb), backend.OpMatMul, ins, nil)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		gr, _ := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpMatMul, ins, nil)
		for i := range gv[0].Numel() {
			idx := tensor.Unravel(i, gv[0].Shape())
			g, r := gv[0].AtF64(idx...), gr[0].AtF64(idx...)
			if math.Abs(g-r) > 1e-4*math.Max(1, math.Abs(r)) {
				t.Fatalf("%s [%d]: vulkan %v vs ref %v", name, i, g, r)
			}
		}
	}
	a := bench.RandF32(tensor.Shape{5, 7}, 1)
	w := bench.RandF32(tensor.Shape{6, 7}, 2)
	wt, _ := w.Transpose(0, 1)
	check("a·Wᵀ", []*tensor.Tensor{a, wt}) // transB, edge dims (not TS-multiples)
	b := bench.RandF32(tensor.Shape{5, 6}, 3)
	at, _ := a.Transpose(0, 1)
	check("Xᵀ·b", []*tensor.Tensor{at, b}) // transA
	a2 := bench.RandF32(tensor.Shape{4, 5}, 6)
	b2 := bench.RandF32(tensor.Shape{6, 4}, 7)
	a2t, _ := a2.Transpose(0, 1)
	b2t, _ := b2.Transpose(0, 1)
	check("A2ᵀ·B2ᵀ", []*tensor.Tensor{a2t, b2t}) // both transposed
	big := bench.RandF32(tensor.Shape{256, 512}, 4)
	bw := bench.RandF32(tensor.Shape{2048, 512}, 5)
	bwt, _ := bw.Transpose(0, 1)
	check("256×512·(2048×512)ᵀ", []*tensor.Tensor{big, bwt}) // FFN-sized
}

// §V-CROSS (§T362): the GPU SiLU backward dx=g·silu'(x) matches the Pure-Go reference.
func TestVulkanSiLUBackwardCrossReference(t *testing.T) {
	skipNoGPU(t)
	vb, ok := backend.Get(backend.Vulkan)
	if !ok {
		t.Fatal("vulkan available but not registered")
	}
	ref, _ := backend.Get(backend.Ref)
	for _, sh := range []tensor.Shape{{1}, {300}, {256, 2048}, {2, 3, 7}} {
		x := bench.RandF32(sh, 1)
		g := bench.RandF32(sh, 2)
		ins := []*tensor.Tensor{x, g}
		gv, err := backend.Execute(backend.NewContext().WithBackend(vb), backend.OpSiLUBackward, ins, nil)
		if err != nil {
			t.Fatalf("vulkan silu-backward %v: %v", sh, err)
		}
		gr, _ := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpSiLUBackward, ins, nil)
		for i := range gv[0].Numel() {
			idx := tensor.Unravel(i, sh)
			gg, r := gv[0].AtF64(idx...), gr[0].AtF64(idx...)
			if math.Abs(gg-r) > 1e-5*math.Max(1, math.Abs(r)) {
				t.Fatalf("silu-backward %v [%d]: vulkan %v vs ref %v", sh, i, gg, r)
			}
		}
	}
}

// §V-CROSS (§T353): the GPU GELU backward dx=g·gelu'(x) matches the Pure-Go reference.
func TestVulkanGELUBackwardCrossReference(t *testing.T) {
	skipNoGPU(t)
	vb, ok := backend.Get(backend.Vulkan)
	if !ok {
		t.Fatal("vulkan available but not registered")
	}
	ref, _ := backend.Get(backend.Ref)
	for _, sh := range []tensor.Shape{{1}, {300}, {256, 2048}, {2, 3, 7}} {
		x := bench.RandF32(sh, 1)
		g := bench.RandF32(sh, 2)
		ins := []*tensor.Tensor{x, g}
		gv, err := backend.Execute(backend.NewContext().WithBackend(vb), backend.OpGELUBackward, ins, nil)
		if err != nil {
			t.Fatalf("vulkan gelu-backward %v: %v", sh, err)
		}
		gr, _ := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpGELUBackward, ins, nil)
		for i := range gv[0].Numel() {
			idx := tensor.Unravel(i, sh)
			gg, r := gv[0].AtF64(idx...), gr[0].AtF64(idx...)
			if math.Abs(gg-r) > 1e-5*math.Max(1, math.Abs(r)) {
				t.Fatalf("gelu-backward %v [%d]: vulkan %v vs ref %v", sh, i, gg, r)
			}
		}
	}
}

// §V-CROSS (§T352): the GPU bias add O[i,j]=X[i,j]+B[j] matches the Pure-Go reference.
func TestVulkanAddBiasCrossReference(t *testing.T) {
	skipNoGPU(t)
	vb, ok := backend.Get(backend.Vulkan)
	if !ok {
		t.Fatal("vulkan available but not registered")
	}
	ref, _ := backend.Get(backend.Ref)
	for _, sh := range []tensor.Shape{{1, 1}, {3, 5}, {256, 512}, {7, 2048}} {
		x := bench.RandF32(sh, 1)
		b := bench.RandF32(tensor.Shape{sh[1]}, 2)
		ins := []*tensor.Tensor{x, b}
		gv, err := backend.Execute(backend.NewContext().WithBackend(vb), backend.OpAddBias, ins, nil)
		if err != nil {
			t.Fatalf("vulkan addbias %v: %v", sh, err)
		}
		gr, _ := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpAddBias, ins, nil)
		for i := range gv[0].Numel() {
			idx := tensor.Unravel(i, sh)
			g, r := gv[0].AtF64(idx...), gr[0].AtF64(idx...)
			if math.Abs(g-r) > 1e-5*math.Max(1, math.Abs(r)) {
				t.Fatalf("addbias %v [%d]: vulkan %v vs ref %v", sh, i, g, r)
			}
		}
	}
}

// §V-CROSS: the generic GPU same-shape binary elementwise ops match the Pure-Go reference;
// a broadcasting case confirms the reference fallback. f32 vs f64 → §V11 tol.
func TestVulkanBinaryCrossReference(t *testing.T) {
	skipNoGPU(t)
	vb, ok := backend.Get(backend.Vulkan)
	if !ok {
		t.Fatal("vulkan available but not registered")
	}
	ref, _ := backend.Get(backend.Ref)
	ops := []backend.Op{backend.OpAdd, backend.OpSub, backend.OpMul, backend.OpDiv, backend.OpMaximum, backend.OpMinimum}
	for _, sh := range []tensor.Shape{{1}, {300}, {4, 5}, {2, 3, 7}} {
		for _, op := range ops {
			a := bench.RandF32(sh, 5)
			b := bench.RandF32(sh, 6)
			for i := range b.Numel() { // keep |b| away from 0 so Div stays well-conditioned
				idx := tensor.Unravel(i, sh)
				b.SetF64(math.Copysign(math.Abs(b.AtF64(idx...))+0.5, b.AtF64(idx...)), idx...)
			}
			gv, err := backend.Execute(backend.NewContext().WithBackend(vb), op, []*tensor.Tensor{a, b}, nil)
			if err != nil {
				t.Fatalf("vulkan %v %v: %v", op, sh, err)
			}
			gr, _ := backend.Execute(backend.NewContext().WithBackend(ref), op, []*tensor.Tensor{a, b}, nil)
			for i := range gv[0].Numel() {
				idx := tensor.Unravel(i, sh)
				g, r := gv[0].AtF64(idx...), gr[0].AtF64(idx...)
				if math.Abs(g-r) > 1e-5*math.Max(1, math.Abs(r)) {
					t.Fatalf("%v %v [%d]: vulkan %v vs ref %v", op, sh, i, g, r)
				}
			}
		}
	}
	// broadcasting → reference fallback still correct
	a := bench.RandF32(tensor.Shape{4, 3}, 1)
	b := bench.RandF32(tensor.Shape{3}, 2)
	gv, err := backend.Execute(backend.NewContext().WithBackend(vb), backend.OpMul, []*tensor.Tensor{a, b}, nil)
	if err != nil {
		t.Fatalf("broadcast mul: %v", err)
	}
	gr, _ := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpMul, []*tensor.Tensor{a, b}, nil)
	for i := range gv[0].Numel() {
		idx := tensor.Unravel(i, gv[0].Shape())
		if math.Abs(gv[0].AtF64(idx...)-gr[0].AtF64(idx...)) > 1e-5 {
			t.Fatalf("broadcast mul [%d]: %v vs %v", i, gv[0].AtF64(idx...), gr[0].AtF64(idx...))
		}
	}
}

// §V-CROSS: the GPU RoPE BACKWARD (inverse rotation) matches the Pure-Go reference across
// shapes/heads/PosOffset/PI/YaRN, and falls back correctly for xPos. Enables GPU training
// of the RoPE path. f32 cos/sin vs f64 → abs/rel tol 1e-4 (§V11).
func TestVulkanRoPEBackwardCrossReference(t *testing.T) {
	skipNoGPU(t)
	vb, ok := backend.Get(backend.Vulkan)
	if !ok {
		t.Fatal("vulkan available but not registered")
	}
	ref, _ := backend.Get(backend.Ref)
	cases := []struct {
		seq, width int
		attrs      backend.RoPEAttrs
	}{
		{4, 16, backend.RoPEAttrs{}},
		{7, 64, backend.RoPEAttrs{Heads: 2}},
		{130, 128, backend.RoPEAttrs{Heads: 4}},
		{16, 32, backend.RoPEAttrs{PosOffset: 100}},
		{16, 32, backend.RoPEAttrs{PosScale: 4}},
		{16, 64, backend.RoPEAttrs{YaRNScale: 2, YaRNOrigCtx: 2048, Heads: 2}},
		{8, 16, backend.RoPEAttrs{XPos: true}}, // → ref fallback
	}
	for _, c := range cases {
		g := bench.RandF32(tensor.Shape{c.seq, c.width}, 7)
		gv, err := backend.Execute(backend.NewContext().WithBackend(vb), backend.OpRoPEBackward, []*tensor.Tensor{g}, c.attrs)
		if err != nil {
			t.Fatalf("vulkan %+v: %v", c, err)
		}
		gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpRoPEBackward, []*tensor.Tensor{g}, c.attrs)
		if err != nil {
			t.Fatal(err)
		}
		for i := range gv[0].Numel() {
			idx := tensor.Unravel(i, gv[0].Shape())
			gg, r := gv[0].AtF64(idx...), gr[0].AtF64(idx...)
			if math.Abs(gg-r) > 1e-4*math.Max(1, math.Abs(r)) {
				t.Fatalf("case %+v [%d]: vulkan %v vs ref %v", c, i, gg, r)
			}
		}
	}
}

// §V-CROSS: the GPU RMSNorm BACKWARD matches the Pure-Go reference for both dx (per-row)
// and dgamma (atomic cross-row reduction). Enables GPU RMSNorm training. f32 vs f64 →
// dim-scaled tol (§V11); dgamma sums over rows so its tolerance scales with rows too.
func TestVulkanRMSNormBackwardCrossReference(t *testing.T) {
	skipNoGPU(t)
	vb, ok := backend.Get(backend.Vulkan)
	if !ok {
		t.Fatal("vulkan available but not registered")
	}
	ref, _ := backend.Get(backend.Ref)
	shapes := []tensor.Shape{{1, 1}, {2, 8}, {7, 33}, {130, 64}, {4, 5, 16}}
	for _, sh := range shapes {
		d := sh[len(sh)-1]
		rows := sh.Numel() / d
		x := bench.RandF32(sh, 1)
		gamma := bench.RandF32(tensor.Shape{d}, 2)
		g := bench.RandF32(sh, 3)
		for _, eps := range []float64{0, 1e-5} {
			attrs := backend.NormAttrs{Eps: eps}
			gv, err := backend.Execute(backend.NewContext().WithBackend(vb), backend.OpRMSNormBackward, []*tensor.Tensor{x, gamma, g}, attrs)
			if err != nil {
				t.Fatalf("vulkan %v eps=%g: %v", sh, eps, err)
			}
			gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpRMSNormBackward, []*tensor.Tensor{x, gamma, g}, attrs)
			if err != nil {
				t.Fatal(err)
			}
			// dx
			for i := range gv[0].Numel() {
				idx := tensor.Unravel(i, gv[0].Shape())
				a, b := gv[0].AtF64(idx...), gr[0].AtF64(idx...)
				if math.Abs(a-b) > crossTol(d)*math.Max(1, math.Abs(b)) {
					t.Fatalf("dx %v eps=%g [%d]: vulkan %v vs ref %v", sh, eps, i, a, b)
				}
			}
			// dgamma (sum over rows → looser dim+rows-scaled tolerance)
			dgTol := 1e-6 * math.Sqrt(float64(d*rows+1))
			for i := range gv[1].Numel() {
				a, b := gv[1].AtF64(i), gr[1].AtF64(i)
				if math.Abs(a-b) > dgTol*math.Max(1, math.Abs(b)) {
					t.Fatalf("dgamma %v eps=%g [%d]: vulkan %v vs ref %v (tol %g)", sh, eps, i, a, b, dgTol)
				}
			}
		}
	}
}

// §V-CROSS: the GPU LayerNorm BACKWARD matches the Pure-Go reference for dx (per-row) and
// dgamma/dbeta (atomic cross-row reductions). Enables GPU LayerNorm training.
func TestVulkanLayerNormBackwardCrossReference(t *testing.T) {
	skipNoGPU(t)
	vb, ok := backend.Get(backend.Vulkan)
	if !ok {
		t.Fatal("vulkan available but not registered")
	}
	ref, _ := backend.Get(backend.Ref)
	shapes := []tensor.Shape{{1, 1}, {2, 8}, {7, 33}, {130, 64}, {4, 5, 16}}
	for _, sh := range shapes {
		d := sh[len(sh)-1]
		rows := sh.Numel() / d
		x := bench.RandF32(sh, 1)
		gamma := bench.RandF32(tensor.Shape{d}, 2)
		g := bench.RandF32(sh, 3)
		for _, eps := range []float64{0, 1e-5} {
			attrs := backend.NormAttrs{Eps: eps}
			gv, err := backend.Execute(backend.NewContext().WithBackend(vb), backend.OpLayerNormBackward, []*tensor.Tensor{x, gamma, g}, attrs)
			if err != nil {
				t.Fatalf("vulkan %v eps=%g: %v", sh, eps, err)
			}
			gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpLayerNormBackward, []*tensor.Tensor{x, gamma, g}, attrs)
			if err != nil {
				t.Fatal(err)
			}
			for i := range gv[0].Numel() { // dx
				idx := tensor.Unravel(i, gv[0].Shape())
				a, b := gv[0].AtF64(idx...), gr[0].AtF64(idx...)
				if math.Abs(a-b) > crossTol(d)*math.Max(1, math.Abs(b)) {
					t.Fatalf("dx %v eps=%g [%d]: vulkan %v vs ref %v", sh, eps, i, a, b)
				}
			}
			dgTol := 1e-6 * math.Sqrt(float64(d*rows+1))
			for _, out := range []int{1, 2} { // dgamma, dbeta
				for i := range gv[out].Numel() {
					a, b := gv[out].AtF64(i), gr[out].AtF64(i)
					if math.Abs(a-b) > dgTol*math.Max(1, math.Abs(b)) {
						t.Fatalf("out%d %v eps=%g [%d]: vulkan %v vs ref %v (tol %g)", out, sh, eps, i, a, b, dgTol)
					}
				}
			}
		}
	}
}

// §V-CROSS: the GPU cross-entropy backward matches the Pure-Go reference across batch/class
// shapes and the label-smoothing / z-loss variants. Seeds the backward pass on the GPU.
func TestVulkanCrossEntropyBackwardCrossReference(t *testing.T) {
	skipNoGPU(t)
	vb, ok := backend.Get(backend.Vulkan)
	if !ok {
		t.Fatal("vulkan available but not registered")
	}
	ref, _ := backend.Get(backend.Ref)
	shapes := []struct{ b, c int }{{1, 2}, {3, 5}, {130, 64}, {8, 200}}
	variants := []backend.CrossEntropyAttrs{{}, {LabelSmoothing: 0.1}, {ZLoss: 1e-3}, {LabelSmoothing: 0.1, ZLoss: 1e-4}}
	for _, sh := range shapes {
		z := bench.RandF32(tensor.Shape{sh.b, sh.c}, 1)
		tg := tensor.New(tensor.F64, tensor.Shape{sh.b})
		for i := range sh.b {
			tg.SetF64(float64(i%sh.c), i) // valid class indices
		}
		gup := tensor.FromFloat64(tensor.Shape{}, []float64{1.7}) // scalar upstream
		for _, at := range variants {
			gv, err := backend.Execute(backend.NewContext().WithBackend(vb), backend.OpCrossEntropyBackward, []*tensor.Tensor{z, tg, gup}, at)
			if err != nil {
				t.Fatalf("vulkan %+v %+v: %v", sh, at, err)
			}
			gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpCrossEntropyBackward, []*tensor.Tensor{z, tg, gup}, at)
			if err != nil {
				t.Fatal(err)
			}
			for i := range gv[0].Numel() {
				idx := tensor.Unravel(i, gv[0].Shape())
				a, b := gv[0].AtF64(idx...), gr[0].AtF64(idx...)
				if math.Abs(a-b) > 1e-5*math.Max(1, math.Abs(b)) {
					t.Fatalf("%+v %+v [%d]: vulkan %v vs ref %v", sh, at, i, a, b)
				}
			}
		}
	}
}

// §V-CROSS: the Vulkan backend's deterministic host-resident embedding backward matches the
// Pure-Go reference, including repeated indices that accumulate into one table row.
func TestVulkanEmbedBackwardCrossReference(t *testing.T) {
	skipNoGPU(t)
	vb, ok := backend.Get(backend.Vulkan)
	if !ok {
		t.Fatal("vulkan available but not registered")
	}
	ref, _ := backend.Get(backend.Ref)
	cases := []struct{ n, d, m int }{{4, 3, 6}, {10, 8, 1}, {2, 5, 100}, {256, 64, 130}}
	for _, cs := range cases {
		table := tensor.New(tensor.F32, tensor.Shape{cs.n, cs.d}) // values unused
		idx := tensor.New(tensor.F64, tensor.Shape{cs.m})
		for i := range cs.m {
			idx.SetF64(float64((i*7)%cs.n), i) // repeats when m>n → tests collisions
		}
		g := bench.RandF32(tensor.Shape{cs.m, cs.d}, 4)
		gv, err := backend.Execute(backend.NewContext().WithBackend(vb), backend.OpEmbedBackward, []*tensor.Tensor{table, idx, g}, nil)
		if err != nil {
			t.Fatalf("vulkan %+v: %v", cs, err)
		}
		gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpEmbedBackward, []*tensor.Tensor{table, idx, g}, nil)
		if err != nil {
			t.Fatal(err)
		}
		// dtable sums up to m gradient rows into one → tolerance scales with m
		tol := 1e-6 * math.Sqrt(float64(cs.m+1))
		for i := range gv[0].Numel() {
			idx2 := tensor.Unravel(i, gv[0].Shape())
			a, b := gv[0].AtF64(idx2...), gr[0].AtF64(idx2...)
			if math.Abs(a-b) > tol*math.Max(1, math.Abs(b)) {
				t.Fatalf("%+v [%d]: vulkan %v vs ref %v (tol %g)", cs, i, a, b, tol)
			}
		}
	}
}
