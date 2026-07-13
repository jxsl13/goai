//go:build darwin && cgo

package metal_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// skipNoGPU implements §V4: missing accel → skip WITH log, never silent pass.
func skipNoGPU(t *testing.T) {
	t.Helper()
	if !metal.Available() {
		t.Skip("metal: no MPS-capable GPU on this machine — skipping (§V4)")
	}
}

// crossTol is the §V11 tolerance for GPU f32 GEMM: MPS accumulates in f32 and
// reorders sums, so rtol scales with reduction length: rtol(K) = 1e-6·√K.
func crossTol(k int) float64 { return 1e-6 * math.Sqrt(float64(k)) }

// §V3/§V11: metal matmul matches the Pure-Go reference within the K-scaled
// tolerance across shapes.
func TestMetalCrossReference(t *testing.T) {
	skipNoGPU(t)
	mb, ok := backend.Get(backend.Metal)
	if !ok {
		t.Fatal("metal available but not registered")
	}
	ref, _ := backend.Get(backend.Ref)

	shapes := []struct{ m, k, n int }{
		{1, 1, 1}, {2, 3, 4}, {33, 65, 17}, {128, 128, 128}, {256, 512, 128},
	}
	for _, s := range shapes {
		a := bench.RandF32(tensor.Shape{s.m, s.k}, 1)
		b := bench.RandF32(tensor.Shape{s.k, s.n}, 2)
		gm, err := backend.Execute(backend.NewContext().WithBackend(mb), backend.OpMatMul, []*tensor.Tensor{a, b}, nil)
		if err != nil {
			t.Fatalf("metal %v: %v", s, err)
		}
		gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpMatMul, []*tensor.Tensor{a, b}, nil)
		if err != nil {
			t.Fatal(err)
		}
		rtol := crossTol(s.k)
		for i := range gm[0].Numel() {
			idx := tensor.Unravel(i, gm[0].Shape())
			g, r := gm[0].AtF64(idx...), gr[0].AtF64(idx...)
			if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r)) {
				t.Fatalf("shape %v [%d]: metal %v vs ref %v (rtol %g)", s, i, g, r, rtol)
			}
		}
	}
}

// §V3/§V11: metal fused attention (OpMHA) matches the Pure-Go reference within an
// f32 tolerance, across heads, GQA, causal, KV-cache (sq<sk) and attn-scale.
func TestMetalMHACrossReference(t *testing.T) {
	skipNoGPU(t)
	mb, _ := backend.Get(backend.Metal)
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
		{"gqa", 5, 5, 4, 2, 8, true, 1},         // grouped-query
		{"mqa", 8, 8, 8, 1, 4, false, 1},        // multi-query
		{"kvcache", 3, 7, 2, 2, 8, true, 1},     // sq<sk incremental decode
		{"attnscale", 6, 6, 4, 4, 8, true, 1.2}, // YaRN attn temperature
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
			gm, err := backend.Execute(backend.NewContext().WithBackend(mb), backend.OpMHA, ins, attrs)
			if err != nil {
				t.Fatalf("metal mha: %v", err)
			}
			gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpMHA, ins, attrs)
			if err != nil {
				t.Fatal(err)
			}
			// f32 softmax attention: dot over dk plus a weighted sum over ≤sk keys,
			// both f32 on the GPU vs f64 on ref → an rtol scaling with the reduction
			// length, plus a small atol for near-zero outputs.
			rtol := crossTol(c.dk + c.sk)
			atol := 1e-5
			for i := range gm[0].Numel() {
				idx := tensor.Unravel(i, gm[0].Shape())
				g, r := gm[0].AtF64(idx...), gr[0].AtF64(idx...)
				if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r))+atol {
					t.Fatalf("%s [%d]: metal %v vs ref %v (rtol %g)", c.name, i, g, r, rtol)
				}
			}
		})
	}
}

// §V3/§V11: metal fused attention with SLIDING-WINDOW (§T114, §R62) matches the
// reference — each query attends only its `window` most recent keys — across window
// sizes, heads, GQA and causal/non-causal, including a window wider than the sequence
// (≡ full attention) and the KV-cache case (sq<sk).
func TestMetalMHAWindowCrossReference(t *testing.T) {
	skipNoGPU(t)
	mb, _ := backend.Get(backend.Metal)
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
		{"window>=seq", 6, 6, 2, 2, 4, 100, true}, // ≡ full attention
		{"swa-kvcache", 3, 9, 2, 2, 8, 4, true},   // sq<sk incremental decode
		{"window1", 7, 7, 2, 2, 4, 1, true},       // only the query's own key
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
			gm, err := backend.Execute(backend.NewContext().WithBackend(mb), backend.OpMHA, ins, attrs)
			if err != nil {
				t.Fatalf("metal mha window: %v", err)
			}
			gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpMHA, ins, attrs)
			if err != nil {
				t.Fatal(err)
			}
			rtol := crossTol(c.dk + c.window)
			for i := range gm[0].Numel() {
				idx := tensor.Unravel(i, gm[0].Shape())
				g, r := gm[0].AtF64(idx...), gr[0].AtF64(idx...)
				if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r))+1e-5 {
					t.Fatalf("%s [%d]: metal %v vs ref %v (rtol %g)", c.name, i, g, r, rtol)
				}
			}
		})
	}
}

// §V3/§V11: metal FlashAttention-2 forward (OpFlashAttn) matches BOTH the reference
// flashattn AND the reference naive attention (OpMHA) within an f32 tolerance — the
// online-softmax tiling is exact, differing only by float reassociation — across
// heads, causal masking and per-head dims.
func TestMetalFlashAttnCrossReference(t *testing.T) {
	skipNoGPU(t)
	mb, _ := backend.Get(backend.Metal)
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
			gm, err := backend.Execute(backend.NewContext().WithBackend(mb), backend.OpFlashAttn, ins, attrs)
			if err != nil {
				t.Fatalf("metal flashattn: %v", err)
			}
			// reference flashattn (same op) and reference naive attention (OpMHA)
			gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpFlashAttn, ins, attrs)
			if err != nil {
				t.Fatal(err)
			}
			gmha, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpMHA, ins, attrs)
			if err != nil {
				t.Fatal(err)
			}
			rtol := crossTol(c.dk + c.seq)
			atol := 1e-5
			for i := range gm[0].Numel() {
				idx := tensor.Unravel(i, gm[0].Shape())
				g, r, mh := gm[0].AtF64(idx...), gr[0].AtF64(idx...), gmha[0].AtF64(idx...)
				if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r))+atol {
					t.Fatalf("%s [%d]: metal-flash %v vs ref-flash %v (rtol %g)", c.name, i, g, r, rtol)
				}
				if math.Abs(g-mh) > rtol*math.Max(1, math.Abs(mh))+atol {
					t.Fatalf("%s [%d]: metal-flash %v vs ref-mha %v (not exact attention)", c.name, i, g, mh)
				}
			}
		})
	}
}

// §V3/§V11: the metal SDPA backward (OpMHABackward) matches the reference dQ/dK/dV
// within an f32 tolerance. Atomic accumulation into shared dK/dV reorders sums, so
// the tolerance scales with seq.
func TestMetalMHABackwardCrossReference(t *testing.T) {
	skipNoGPU(t)
	mb, _ := backend.Get(backend.Metal)
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
		{"swa", 8, 2, 2, 4, true, 1, 3},     // sliding-window backward (§T128)
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
			g := bench.RandF32(tensor.Shape{c.seq, dm}, 4) // upstream dO
			attrs := backend.AttnAttrs{Heads: c.heads, KVHeads: c.kv, Causal: c.causal, Scale: c.scale, Window: c.window}
			ins := []*tensor.Tensor{q, k, v, g}
			gm, err := backend.Execute(backend.NewContext().WithBackend(mb), backend.OpMHABackward, ins, attrs)
			if err != nil {
				t.Fatalf("metal mha-backward: %v", err)
			}
			gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpMHABackward, ins, attrs)
			if err != nil {
				t.Fatal(err)
			}
			rtol := crossTol(c.seq*c.dk) + 1e-5
			atol := 2e-4
			names := []string{"dQ", "dK", "dV"}
			for o := range gm {
				for i := range gm[o].Numel() {
					idx := tensor.Unravel(i, gm[o].Shape())
					gg, rr := gm[o].AtF64(idx...), gr[o].AtF64(idx...)
					if math.Abs(gg-rr) > rtol*math.Max(1, math.Abs(rr))+atol {
						t.Fatalf("%s %s[%d]: metal %v vs ref %v (rtol %g)", c.name, names[o], i, gg, rr, rtol)
					}
				}
			}
		})
	}
}

// End-to-end GPU training (§T86): a full attention forward+backward on a metal-
// backed tape produces the same dQ/dK/dV as a reference-backed tape — the mha VJP
// dispatches OpMHABackward, so on a metal tape the backward runs on the GPU.
func TestMetalMHATrainingMatchesRef(t *testing.T) {
	skipNoGPU(t)
	mb, _ := backend.Get(backend.Metal)
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
	gm, gr := grads(mb), grads(ref)
	names := []string{"dQ", "dK", "dV"}
	for o := range gm {
		if gm[o] == nil || gr[o] == nil {
			t.Fatalf("%s: nil gradient", names[o])
		}
		for i := range gm[o].Numel() {
			idx := tensor.Unravel(i, gm[o].Shape())
			if math.Abs(gm[o].AtF64(idx...)-gr[o].AtF64(idx...)) > 1e-3*math.Max(1, math.Abs(gr[o].AtF64(idx...)))+3e-4 {
				t.Fatalf("%s[%d]: metal-tape %v vs ref-tape %v", names[o], i, gm[o].AtF64(idx...), gr[o].AtF64(idx...))
			}
		}
	}
}

// §I4: attention features the metal kernel does not implement (ALiBi) fall back to
// the reference and still match it bit-for-bit. Sliding window now runs on the GPU
// (§T114), so it is covered by TestMetalMHAWindowCrossReference instead.
func TestMetalMHAFallbackFeatures(t *testing.T) {
	skipNoGPU(t)
	mb, _ := backend.Get(backend.Metal)
	ref, _ := backend.Get(backend.Ref)
	q := bench.RandF32(tensor.Shape{6, 8}, 1)
	k := bench.RandF32(tensor.Shape{6, 8}, 2)
	v := bench.RandF32(tensor.Shape{6, 8}, 3)
	ins := []*tensor.Tensor{q, k, v}
	for _, attrs := range []backend.AttnAttrs{
		{Heads: 2, Causal: true, ALiBi: true},
	} {
		gm, err := backend.Execute(backend.NewContext().WithBackend(mb), backend.OpMHA, ins, attrs)
		if err != nil {
			t.Fatalf("metal fallback %+v: %v", attrs, err)
		}
		gr, _ := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpMHA, ins, attrs)
		for i := range gm[0].Numel() {
			idx := tensor.Unravel(i, gm[0].Shape())
			if math.Abs(gm[0].AtF64(idx...)-gr[0].AtF64(idx...)) > 1e-12 {
				t.Fatalf("fallback %+v [%d]: metal %v != ref %v (must be identical)", attrs, i, gm[0].AtF64(idx...), gr[0].AtF64(idx...))
			}
		}
	}
}

// §V3/§V11: metal 2-D convolution matches the Pure-Go reference within an f32
// tolerance, across stride, padding, multi-channel/filter, and with/without bias.
func TestMetalConv2DCrossReference(t *testing.T) {
	skipNoGPU(t)
	mb, _ := backend.Get(backend.Metal)
	ref, _ := backend.Get(backend.Ref)

	cases := []struct {
		name                   string
		n, c, h, wd, f, kh, kw int
		stride, pad            int
		bias                   bool
	}{
		{"basic", 1, 1, 5, 5, 1, 3, 3, 1, 0, false},
		{"pad", 2, 3, 6, 6, 4, 3, 3, 1, 1, true},
		{"stride2", 1, 3, 8, 8, 5, 3, 3, 2, 1, true},
		{"1x1", 2, 4, 5, 5, 6, 1, 1, 1, 0, true},
		{"nonsquare", 1, 2, 7, 5, 3, 2, 3, 2, 1, false},
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
			gm, err := backend.Execute(backend.NewContext().WithBackend(mb), backend.OpConv2D, ins, attrs)
			if err != nil {
				t.Fatalf("metal conv2d: %v", err)
			}
			gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpConv2D, ins, attrs)
			if err != nil {
				t.Fatal(err)
			}
			if !gm[0].Shape().Equal(gr[0].Shape()) {
				t.Fatalf("shape %v vs ref %v", gm[0].Shape(), gr[0].Shape())
			}
			rtol := crossTol(c.c*c.kh*c.kw) + 1e-6
			for i := range gm[0].Numel() {
				idx := tensor.Unravel(i, gm[0].Shape())
				g, r := gm[0].AtF64(idx...), gr[0].AtF64(idx...)
				if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r))+1e-5 {
					t.Fatalf("%s [%d]: metal %v vs ref %v (rtol %g)", c.name, i, g, r, rtol)
				}
			}
		})
	}
}

// §V3/§V11: the metal conv2d backward (OpConv2DBackward) matches the reference
// dX/dW/dBias within an f32 tolerance. Shared dX/dW/dBias accumulate via float
// atomics, so the tolerance scales with the number of contributions (§T101).
func TestMetalConv2DBackwardCrossReference(t *testing.T) {
	skipNoGPU(t)
	mb, _ := backend.Get(backend.Metal)
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
			gm, err := backend.Execute(backend.NewContext().WithBackend(mb), backend.OpConv2DBackward, ins, attrs)
			if err != nil {
				t.Fatalf("metal conv2d-backward: %v", err)
			}
			gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpConv2DBackward, ins, attrs)
			if err != nil {
				t.Fatal(err)
			}
			rtol := crossTol(c.n*ho*wo) + 1e-5
			names := []string{"dX", "dW", "dBias"}
			for o := range gm {
				for i := range gm[o].Numel() {
					idx := tensor.Unravel(i, gm[o].Shape())
					g, r := gm[o].AtF64(idx...), gr[o].AtF64(idx...)
					if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r))+2e-4 {
						t.Fatalf("%s %s[%d]: metal %v vs ref %v (rtol %g)", c.name, names[o], i, g, r, rtol)
					}
				}
			}
		})
	}
}

// Fallback: ops the metal backend does not serve route to the reference (§I4).
func TestMetalFallback(t *testing.T) {
	skipNoGPU(t)
	mb, _ := backend.Get(backend.Metal)
	x := bench.RandF32(tensor.Shape{8}, 3)
	out, err := backend.Execute(backend.NewContext().WithBackend(mb), backend.OpExp, []*tensor.Tensor{x}, nil)
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

// §C3 gate benchmarks: metal vs the ceiling-optimized Pure-Go cpu backend.
func BenchmarkMatMulF32_512_metal(b *testing.B)  { benchMatMulOn(b, backend.Metal, 512) }
func BenchmarkMatMulF32_512_cpu(b *testing.B)    { benchMatMulOn(b, backend.CPU, 512) }
func BenchmarkMatMulF32_1024_metal(b *testing.B) { benchMatMulOn(b, backend.Metal, 1024) }
func BenchmarkMatMulF32_1024_cpu(b *testing.B)   { benchMatMulOn(b, backend.CPU, 1024) }

func benchMHAOn(b *testing.B, name backend.Name, seq, heads, dk int) {
	be, ok := backend.Get(name)
	if !ok {
		b.Skipf("%s not available", name)
	}
	dm := heads * dk
	q := bench.RandF32(tensor.Shape{seq, dm}, 1)
	k := bench.RandF32(tensor.Shape{seq, dm}, 2)
	v := bench.RandF32(tensor.Shape{seq, dm}, 3)
	attrs := backend.AttnAttrs{Heads: heads, Causal: true}
	ctx := backend.NewContext().WithBackend(be)
	ins := []*tensor.Tensor{q, k, v}
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpMHA, ins, attrs); err != nil {
			b.Fatal(err)
		}
	}
}

// §C3: attention is compute-bound (O(seq²·dk)), so per-op GPU offload is justified
// (unlike memory-bound elementwise, ADR-0008). metal vs the cpu backend (→ref).
func BenchmarkMHA_512x8x64_metal(b *testing.B) { benchMHAOn(b, backend.Metal, 512, 8, 64) }
func BenchmarkMHA_512x8x64_cpu(b *testing.B)   { benchMHAOn(b, backend.CPU, 512, 8, 64) }

func benchMHABwdOn(b *testing.B, name backend.Name, seq, heads, dk int) {
	be, ok := backend.Get(name)
	if !ok {
		b.Skipf("%s not available", name)
	}
	dm := heads * dk
	q := bench.RandF32(tensor.Shape{seq, dm}, 1)
	k := bench.RandF32(tensor.Shape{seq, dm}, 2)
	v := bench.RandF32(tensor.Shape{seq, dm}, 3)
	g := bench.RandF32(tensor.Shape{seq, dm}, 4)
	attrs := backend.AttnAttrs{Heads: heads, Causal: true}
	ctx := backend.NewContext().WithBackend(be)
	ins := []*tensor.Tensor{q, k, v, g}
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpMHABackward, ins, attrs); err != nil {
			b.Fatal(err)
		}
	}
}

// §T86: the attention backward (training) is compute-bound too — GPU-offloaded.
func BenchmarkMHABwd_512x8x64_metal(b *testing.B) { benchMHABwdOn(b, backend.Metal, 512, 8, 64) }
func BenchmarkMHABwd_512x8x64_cpu(b *testing.B)   { benchMHABwdOn(b, backend.CPU, 512, 8, 64) }

func benchConv2DOn(b *testing.B, name backend.Name, n, c, hw, f, k int) {
	be, ok := backend.Get(name)
	if !ok {
		b.Skipf("%s not available", name)
	}
	x := bench.RandF32(tensor.Shape{n, c, hw, hw}, 1)
	w := bench.RandF32(tensor.Shape{f, c, k, k}, 2)
	attrs := backend.ConvAttrs{Stride: 1, Pad: 1}
	ctx := backend.NewContext().WithBackend(be)
	ins := []*tensor.Tensor{x, w}
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpConv2D, ins, attrs); err != nil {
			b.Fatal(err)
		}
	}
}

// §C3: conv2d is compute-bound (O(N·F·ho·wo·C·KH·KW)) → GPU offload justified.
func BenchmarkConv2D_8x16x32x32_metal(b *testing.B) {
	benchConv2DOn(b, backend.Metal, 8, 16, 32, 32, 3)
}
func BenchmarkConv2D_8x16x32x32_cpu(b *testing.B) { benchConv2DOn(b, backend.CPU, 8, 16, 32, 32, 3) }

// TestMetalRetentionCrossReference V-CROSSes the GPU RetNet retention forward against the f64 CPU
// reference OpRetention over several sequence lengths, head dims and decay rates (§T172). Both
// compute the parallel form (QKᵀ⊙D)V; only the accumulation precision/order differs, so they agree
// within crossTol (§V3/§V11).
func TestMetalRetentionCrossReference(t *testing.T) {
	skipNoGPU(t)
	mb, _ := backend.Get(backend.Metal)
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
		gm, err := backend.Execute(backend.NewContext().WithBackend(mb), backend.OpRetention, ins, attrs)
		if err != nil {
			t.Fatalf("metal retention %+v: %v", c, err)
		}
		gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpRetention, ins, attrs)
		if err != nil {
			t.Fatal(err)
		}
		if !gm[0].Shape().Equal(tensor.Shape{c.l, c.d}) {
			t.Fatalf("%+v: shape %v", c, gm[0].Shape())
		}
		rtol, atol := crossTol(c.d+c.l), 1e-5
		for i := range gm[0].Numel() {
			idx := tensor.Unravel(i, gm[0].Shape())
			g, r := gm[0].AtF64(idx...), gr[0].AtF64(idx...)
			if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r))+atol {
				t.Errorf("%+v %v: metal %g vs ref %g (Δ %g)", c, idx, g, r, math.Abs(g-r))
			}
		}
	}
}

// TestMetalRetentionBackwardCrossReference V-CROSSes the GPU retention backward against the f64 CPU
// reference OpRetentionBackward (§T174, mirroring T86): metal dQ/dK/dV == ref within crossTol, over
// several shapes and decay rates, with the shared dK/dV atomically accumulated on-device.
func TestMetalRetentionBackwardCrossReference(t *testing.T) {
	skipNoGPU(t)
	mb, _ := backend.Get(backend.Metal)
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
		gm, err := backend.Execute(backend.NewContext().WithBackend(mb), backend.OpRetentionBackward, ins, attrs)
		if err != nil {
			t.Fatalf("metal retention-backward %+v: %v", c, err)
		}
		gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpRetentionBackward, ins, attrs)
		if err != nil {
			t.Fatal(err)
		}
		rtol, atol := crossTol(c.d+c.l), 1e-4
		for oi, name := range []string{"dQ", "dK", "dV"} {
			for i := range gm[oi].Numel() {
				idx := tensor.Unravel(i, gm[oi].Shape())
				g, r := gm[oi].AtF64(idx...), gr[oi].AtF64(idx...)
				if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r))+atol {
					t.Errorf("%+v %s%v: metal %g vs ref %g (Δ %g)", c, name, idx, g, r, math.Abs(g-r))
				}
			}
		}
	}
}

// TestMetalRetentionTrainsE2E proves retention TRAINS on the GPU: a Metal-tape retention forward +
// backward produces the same Q/K/V gradients as a reference tape (the VJP dispatches OpRetentionBackward
// to the active backend, §T174).
func TestMetalRetentionTrainsE2E(t *testing.T) {
	skipNoGPU(t)
	mb, _ := backend.Get(backend.Metal)
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
	mq, mk, mv := grads(mb)
	rq, rk, rv := grads(ref)
	rtol, atol := crossTol(d+l), 1e-4
	for _, pair := range []struct {
		name string
		m, r *tensor.Tensor
	}{{"dQ", mq, rq}, {"dK", mk, rk}, {"dV", mv, rv}} {
		if pair.m == nil || pair.r == nil {
			t.Fatalf("%s: nil grad (metal=%v ref=%v)", pair.name, pair.m, pair.r)
		}
		for i := range pair.r.Numel() {
			idx := tensor.Unravel(i, pair.r.Shape())
			if math.Abs(pair.m.AtF64(idx...)-pair.r.AtF64(idx...)) > rtol*math.Max(1, math.Abs(pair.r.AtF64(idx...)))+atol {
				t.Errorf("%s%v: metal-tape %g vs ref-tape %g", pair.name, idx, pair.m.AtF64(idx...), pair.r.AtF64(idx...))
			}
		}
	}
}

// TestMetalRetentionValueExpansion: retention with d_v≠d_k (RetNet value expansion) runs natively on
// the Metal kernel — FORWARD (§T178) and BACKWARD (§T191) — matching the reference within the §V11
// f32 cross-tolerance, so the faithful (d_v=2·d_k) MSR block both infers and TRAINS on the GPU.
func TestMetalRetentionValueExpansion(t *testing.T) {
	skipNoGPU(t)
	mb, _ := backend.Get(backend.Metal)
	ref, _ := backend.Get(backend.Ref)
	cases := []struct{ l, dk, dv int }{{6, 3, 6}, {8, 4, 8}, {5, 8, 16}, {12, 16, 32}, {4, 5, 3}}
	for _, c := range cases {
		q := bench.RandF32(tensor.Shape{c.l, c.dk}, 1)
		k := bench.RandF32(tensor.Shape{c.l, c.dk}, 2)
		v := bench.RandF32(tensor.Shape{c.l, c.dv}, 3)
		attrs := backend.RetentionAttrs{Gamma: 0.9}
		gm, err := backend.Execute(backend.NewContext().WithBackend(mb), backend.OpRetention, []*tensor.Tensor{q, k, v}, attrs)
		if err != nil {
			t.Fatalf("%+v: metal retention d_v≠d_k: %v", c, err)
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
				t.Errorf("%+v %v: metal %g vs ref %g (Δ %g)", c, idx, g, r, math.Abs(g-r))
			}
		}
	}
	// BACKWARD d_v≠d_k now runs on the GPU too (§T191) — dQ,dK [L,dk], dV [L,dv] match ref @crossTol.
	for _, c := range cases {
		q := bench.RandF32(tensor.Shape{c.l, c.dk}, 1)
		k := bench.RandF32(tensor.Shape{c.l, c.dk}, 2)
		v := bench.RandF32(tensor.Shape{c.l, c.dv}, 3)
		dO := bench.RandF32(tensor.Shape{c.l, c.dv}, 4)
		attrs := backend.RetentionAttrs{Gamma: 0.9}
		bm, err := backend.Execute(backend.NewContext().WithBackend(mb), backend.OpRetentionBackward, []*tensor.Tensor{q, k, v, dO}, attrs)
		if err != nil {
			t.Fatalf("%+v: metal retention-backward d_v≠d_k: %v", c, err)
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
					t.Errorf("%+v backward[%d] %v: metal %g vs ref %g (Δ %g)", c, oi, idx, g, r, math.Abs(g-r))
				}
			}
		}
	}
}

// §V-CROSS: the GPU RMSNorm matches the Pure-Go reference across shapes (including
// non-power-of-two rows/dims and a 3-D input) and an explicit, defaulted (0→1e-5) and
// large epsilon. The GPU sums in f32 vs the reference's f64, so a dim-scaled tolerance
// applies (§V11).
func TestMetalRMSNormCrossReference(t *testing.T) {
	skipNoGPU(t)
	mb, ok := backend.Get(backend.Metal)
	if !ok {
		t.Fatal("metal available but not registered")
	}
	ref, _ := backend.Get(backend.Ref)
	shapes := []tensor.Shape{{1, 1}, {2, 8}, {7, 33}, {65, 16}, {130, 128}, {4, 5, 64}}
	for _, sh := range shapes {
		d := sh[len(sh)-1]
		x := bench.RandF32(sh, 1)
		gamma := bench.RandF32(tensor.Shape{d}, 2)
		for _, eps := range []float64{0, 1e-5, 1e-2} {
			attrs := backend.NormAttrs{Eps: eps}
			gv, err := backend.Execute(backend.NewContext().WithBackend(mb), backend.OpRMSNorm, []*tensor.Tensor{x, gamma}, attrs)
			if err != nil {
				t.Fatalf("metal %v eps=%g: %v", sh, eps, err)
			}
			gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpRMSNorm, []*tensor.Tensor{x, gamma}, attrs)
			if err != nil {
				t.Fatal(err)
			}
			if !gv[0].Shape().Equal(gr[0].Shape()) {
				t.Fatalf("shape %v: metal %v vs ref %v", sh, gv[0].Shape(), gr[0].Shape())
			}
			rtol := crossTol(d)
			for i := range gv[0].Numel() {
				idx := tensor.Unravel(i, gv[0].Shape())
				g, r := gv[0].AtF64(idx...), gr[0].AtF64(idx...)
				if math.Abs(g-r) > rtol*math.Max(1, math.Abs(r)) {
					t.Fatalf("shape %v eps=%g [%d]: metal %v vs ref %v (rtol %g)", sh, eps, i, g, r, rtol)
				}
			}
		}
	}
}

// §V-CROSS: the GPU RoPE matches the Pure-Go reference across shapes, head counts,
// position offsets, and the PI/YaRN frequency variants (which flow through the
// host-precomputed inv[]/posDiv). f32 cos/sin vs f64 → a small absolute tolerance (§V11).
// The xPos variant is exercised as a fallback (still == ref).
func TestMetalRoPECrossReference(t *testing.T) {
	skipNoGPU(t)
	mb, ok := backend.Get(backend.Metal)
	if !ok {
		t.Fatal("metal available but not registered")
	}
	ref, _ := backend.Get(backend.Ref)
	cases := []struct {
		seq, width int
		attrs      backend.RoPEAttrs
	}{
		{1, 8, backend.RoPEAttrs{}},
		{4, 16, backend.RoPEAttrs{}},
		{7, 64, backend.RoPEAttrs{Heads: 2}},
		{130, 128, backend.RoPEAttrs{Heads: 4}},
		{16, 32, backend.RoPEAttrs{PosOffset: 100}},
		{16, 32, backend.RoPEAttrs{PosScale: 4}},
		{16, 64, backend.RoPEAttrs{YaRNScale: 2, YaRNOrigCtx: 2048, Heads: 2}},
		{8, 16, backend.RoPEAttrs{XPos: true}},
	}
	for _, c := range cases {
		q := bench.RandF32(tensor.Shape{c.seq, c.width}, 1)
		gv, err := backend.Execute(backend.NewContext().WithBackend(mb), backend.OpRoPE, []*tensor.Tensor{q}, c.attrs)
		if err != nil {
			t.Fatalf("metal %+v: %v", c, err)
		}
		gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpRoPE, []*tensor.Tensor{q}, c.attrs)
		if err != nil {
			t.Fatal(err)
		}
		for i := range gv[0].Numel() {
			idx := tensor.Unravel(i, gv[0].Shape())
			g, r := gv[0].AtF64(idx...), gr[0].AtF64(idx...)
			if math.Abs(g-r) > 1e-4*math.Max(1, math.Abs(r)) {
				t.Fatalf("case %+v [%d]: metal %v vs ref %v", c, i, g, r)
			}
		}
	}
}

// §V-CROSS: the GPU softmax matches the Pure-Go reference across shapes (rows past one
// workgroup, a 3-D input), within the f32 exp/division tolerance (§V11).
func TestMetalSoftmaxCrossReference(t *testing.T) {
	skipNoGPU(t)
	mb, ok := backend.Get(backend.Metal)
	if !ok {
		t.Fatal("metal available but not registered")
	}
	ref, _ := backend.Get(backend.Ref)
	shapes := []tensor.Shape{{1, 1}, {2, 8}, {7, 33}, {130, 64}, {4, 5, 16}}
	for _, sh := range shapes {
		x := bench.RandF32(sh, 3)
		gv, err := backend.Execute(backend.NewContext().WithBackend(mb), backend.OpSoftmax, []*tensor.Tensor{x}, nil)
		if err != nil {
			t.Fatalf("metal %v: %v", sh, err)
		}
		gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpSoftmax, []*tensor.Tensor{x}, nil)
		if err != nil {
			t.Fatal(err)
		}
		for i := range gv[0].Numel() {
			idx := tensor.Unravel(i, gv[0].Shape())
			g, r := gv[0].AtF64(idx...), gr[0].AtF64(idx...)
			if math.Abs(g-r) > 1e-5*math.Max(1, math.Abs(r)) {
				t.Fatalf("shape %v [%d]: metal %v vs ref %v", sh, i, g, r)
			}
		}
	}
}

// §V-CROSS: the GPU LayerNorm matches the Pure-Go reference across shapes (rows past one
// workgroup, a 3-D input) and explicit/defaulted epsilon, within the f32 tolerance (§V11).
func TestMetalLayerNormCrossReference(t *testing.T) {
	skipNoGPU(t)
	mb, ok := backend.Get(backend.Metal)
	if !ok {
		t.Fatal("metal available but not registered")
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
			gv, err := backend.Execute(backend.NewContext().WithBackend(mb), backend.OpLayerNorm, []*tensor.Tensor{x, gamma, beta}, attrs)
			if err != nil {
				t.Fatalf("metal %v eps=%g: %v", sh, eps, err)
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
					t.Fatalf("shape %v eps=%g [%d]: metal %v vs ref %v (rtol %g)", sh, eps, i, g, r, rtol)
				}
			}
		}
	}
}

// §V-CROSS: the generic GPU unary elementwise ops match the Pure-Go reference. Log/Sqrt
// use positive inputs (their domain); the rest use general inputs. f32 vs f64 → §V11 tol.
func TestMetalUnaryCrossReference(t *testing.T) {
	skipNoGPU(t)
	mb, ok := backend.Get(backend.Metal)
	if !ok {
		t.Fatal("metal available but not registered")
	}
	ref, _ := backend.Get(backend.Ref)
	ops := []struct {
		op  backend.Op
		pos bool
	}{
		{backend.OpNeg, false}, {backend.OpExp, false}, {backend.OpLog, true},
		{backend.OpTanh, false}, {backend.OpReLU, false}, {backend.OpSigmoid, false},
		{backend.OpSiLU, false}, {backend.OpSqrt, true}, {backend.OpAbs, false},
		{backend.OpGELU, false}, // §T352: exact-erf GELU now on-GPU (was a ref fallback)
	}
	for _, sh := range []tensor.Shape{{1}, {300}, {4, 5}, {2, 3, 7}} {
		for _, o := range ops {
			x := bench.RandF32(sh, 5)
			if o.pos {
				for i := range x.Numel() {
					idx := tensor.Unravel(i, sh)
					x.SetF64(math.Abs(x.AtF64(idx...))+0.1, idx...)
				}
			}
			gv, err := backend.Execute(backend.NewContext().WithBackend(mb), o.op, []*tensor.Tensor{x}, nil)
			if err != nil {
				t.Fatalf("metal %v %v: %v", o.op, sh, err)
			}
			gr, _ := backend.Execute(backend.NewContext().WithBackend(ref), o.op, []*tensor.Tensor{x}, nil)
			for i := range gv[0].Numel() {
				idx := tensor.Unravel(i, sh)
				g, r := gv[0].AtF64(idx...), gr[0].AtF64(idx...)
				if math.Abs(g-r) > 1e-5*math.Max(1, math.Abs(r)) {
					t.Fatalf("%v %v [%d]: metal %v vs ref %v", o.op, sh, i, g, r)
				}
			}
		}
	}
}

// §V-CROSS: the generic GPU same-shape binary elementwise ops match the Pure-Go reference;
// a broadcasting case confirms the reference fallback. f32 vs f64 → §V11 tol.
func TestMetalBinaryCrossReference(t *testing.T) {
	skipNoGPU(t)
	mb, ok := backend.Get(backend.Metal)
	if !ok {
		t.Fatal("metal available but not registered")
	}
	ref, _ := backend.Get(backend.Ref)
	ops := []backend.Op{backend.OpAdd, backend.OpSub, backend.OpMul, backend.OpDiv, backend.OpMaximum, backend.OpMinimum}
	for _, sh := range []tensor.Shape{{1}, {300}, {4, 5}, {2, 3, 7}} {
		for _, op := range ops {
			a := bench.RandF32(sh, 5)
			b := bench.RandF32(sh, 6)
			for i := range b.Numel() {
				idx := tensor.Unravel(i, sh)
				b.SetF64(math.Copysign(math.Abs(b.AtF64(idx...))+0.5, b.AtF64(idx...)), idx...)
			}
			gv, err := backend.Execute(backend.NewContext().WithBackend(mb), op, []*tensor.Tensor{a, b}, nil)
			if err != nil {
				t.Fatalf("metal %v %v: %v", op, sh, err)
			}
			gr, _ := backend.Execute(backend.NewContext().WithBackend(ref), op, []*tensor.Tensor{a, b}, nil)
			for i := range gv[0].Numel() {
				idx := tensor.Unravel(i, sh)
				g, r := gv[0].AtF64(idx...), gr[0].AtF64(idx...)
				if math.Abs(g-r) > 1e-5*math.Max(1, math.Abs(r)) {
					t.Fatalf("%v %v [%d]: metal %v vs ref %v", op, sh, i, g, r)
				}
			}
		}
	}
	// broadcasting → reference fallback
	a := bench.RandF32(tensor.Shape{4, 3}, 1)
	b := bench.RandF32(tensor.Shape{3}, 2)
	gv, err := backend.Execute(backend.NewContext().WithBackend(mb), backend.OpMul, []*tensor.Tensor{a, b}, nil)
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

// §V-CROSS (§T356): matmul with a TRANSPOSED-view operand (the matmul backward's dO·Wᵀ /
// Xᵀ·dO) matches the reference — MPS transposes via a flag instead of a CPU strided-gather
// copy, so the transposed and contiguous paths must give identical results.
func TestMetalMatMulTransposedCrossReference(t *testing.T) {
	skipNoGPU(t)
	mb, _ := backend.Get(backend.Metal)
	ref, _ := backend.Get(backend.Ref)
	check := func(name string, ins []*tensor.Tensor) {
		gv, err := backend.Execute(backend.NewContext().WithBackend(mb), backend.OpMatMul, ins, nil)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		gr, _ := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpMatMul, ins, nil)
		for i := range gv[0].Numel() {
			idx := tensor.Unravel(i, gv[0].Shape())
			g, r := gv[0].AtF64(idx...), gr[0].AtF64(idx...)
			if math.Abs(g-r) > 1e-4*math.Max(1, math.Abs(r)) {
				t.Fatalf("%s [%d]: metal %v vs ref %v", name, i, g, r)
			}
		}
	}
	a := bench.RandF32(tensor.Shape{5, 7}, 1)
	// transB: W[6,7] contiguous → Wᵀ[7,6] view; a[5,7]·Wᵀ[7,6]
	w := bench.RandF32(tensor.Shape{6, 7}, 2)
	wt, _ := w.Transpose(0, 1)
	check("a·Wᵀ", []*tensor.Tensor{a, wt})
	// transA: X[5,7] → Xᵀ[7,5] view; Xᵀ[7,5]·b[5,6]
	b := bench.RandF32(tensor.Shape{5, 6}, 3)
	at, _ := a.Transpose(0, 1)
	check("Xᵀ·b", []*tensor.Tensor{at, b})
	// both transposed: A2[4,5]→A2ᵀ[5,4], B2[6,4]→B2ᵀ[4,6] → [5,6]
	a2 := bench.RandF32(tensor.Shape{4, 5}, 6)
	b2 := bench.RandF32(tensor.Shape{6, 4}, 7)
	a2t, _ := a2.Transpose(0, 1)
	b2t, _ := b2.Transpose(0, 1)
	check("A2ᵀ·B2ᵀ", []*tensor.Tensor{a2t, b2t})
	// FFN-sized (exercises the ~8ms copy path the fix removes)
	big := bench.RandF32(tensor.Shape{256, 512}, 4)
	bw := bench.RandF32(tensor.Shape{2048, 512}, 5)
	bwt, _ := bw.Transpose(0, 1)
	check("256×512·(2048×512)ᵀ", []*tensor.Tensor{big, bwt})
}

// §V-CROSS (§T354): the GPU bias-add backward db[j]=Σ_i g[i,j] matches the Pure-Go reference.
func TestMetalAddBiasBackwardCrossReference(t *testing.T) {
	skipNoGPU(t)
	mb, _ := backend.Get(backend.Metal)
	ref, _ := backend.Get(backend.Ref)
	for _, sh := range []tensor.Shape{{1, 1}, {3, 5}, {256, 2048}, {7, 512}} {
		g := bench.RandF32(sh, 1)
		ins := []*tensor.Tensor{g}
		gv, err := backend.Execute(backend.NewContext().WithBackend(mb), backend.OpAddBiasBackward, ins, nil)
		if err != nil {
			t.Fatalf("metal addbias-backward %v: %v", sh, err)
		}
		gr, _ := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpAddBiasBackward, ins, nil)
		for j := range gv[0].Numel() {
			a, r := gv[0].AtF64(j), gr[0].AtF64(j)
			if math.Abs(a-r) > 1e-5*math.Max(1, math.Abs(r)) {
				t.Fatalf("addbias-backward %v [%d]: metal %v vs ref %v", sh, j, a, r)
			}
		}
	}
}

// §V-CROSS (§T362): the GPU SiLU backward dx=g·silu'(x) matches the Pure-Go reference.
// SiLU is SwiGLU's activation; its VJP was a ~29ms CPU scalar loop for Llama-style training.
func TestMetalSiLUBackwardCrossReference(t *testing.T) {
	skipNoGPU(t)
	mb, _ := backend.Get(backend.Metal)
	ref, _ := backend.Get(backend.Ref)
	for _, sh := range []tensor.Shape{{1}, {300}, {256, 2048}, {2, 3, 7}} {
		x := bench.RandF32(sh, 1)
		g := bench.RandF32(sh, 2)
		ins := []*tensor.Tensor{x, g}
		gv, err := backend.Execute(backend.NewContext().WithBackend(mb), backend.OpSiLUBackward, ins, nil)
		if err != nil {
			t.Fatalf("metal silu-backward %v: %v", sh, err)
		}
		gr, _ := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpSiLUBackward, ins, nil)
		for i := range gv[0].Numel() {
			idx := tensor.Unravel(i, sh)
			gg, r := gv[0].AtF64(idx...), gr[0].AtF64(idx...)
			if math.Abs(gg-r) > 1e-5*math.Max(1, math.Abs(r)) {
				t.Fatalf("silu-backward %v [%d]: metal %v vs ref %v", sh, i, gg, r)
			}
		}
	}
}

// §V-CROSS (§T353): the GPU GELU backward dx=g·gelu'(x) matches the Pure-Go reference.
// The elementwise VJP was a ~30ms CPU scalar loop dominating the FFN's training step.
func TestMetalGELUBackwardCrossReference(t *testing.T) {
	skipNoGPU(t)
	mb, _ := backend.Get(backend.Metal)
	ref, _ := backend.Get(backend.Ref)
	for _, sh := range []tensor.Shape{{1}, {300}, {256, 2048}, {2, 3, 7}} {
		x := bench.RandF32(sh, 1)
		g := bench.RandF32(sh, 2)
		ins := []*tensor.Tensor{x, g}
		gv, err := backend.Execute(backend.NewContext().WithBackend(mb), backend.OpGELUBackward, ins, nil)
		if err != nil {
			t.Fatalf("metal gelu-backward %v: %v", sh, err)
		}
		gr, _ := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpGELUBackward, ins, nil)
		for i := range gv[0].Numel() {
			idx := tensor.Unravel(i, sh)
			gg, r := gv[0].AtF64(idx...), gr[0].AtF64(idx...)
			if math.Abs(gg-r) > 1e-5*math.Max(1, math.Abs(r)) {
				t.Fatalf("gelu-backward %v [%d]: metal %v vs ref %v", sh, i, gg, r)
			}
		}
	}
}

// §V-CROSS (§T352): the GPU bias add O[i,j]=X[i,j]+B[j] matches the Pure-Go reference.
// This op dominated the FFN as a CPU fallback before it moved to the GPU.
func TestMetalAddBiasCrossReference(t *testing.T) {
	skipNoGPU(t)
	mb, _ := backend.Get(backend.Metal)
	ref, _ := backend.Get(backend.Ref)
	for _, sh := range []tensor.Shape{{1, 1}, {3, 5}, {256, 512}, {7, 2048}} {
		x := bench.RandF32(sh, 1)
		b := bench.RandF32(tensor.Shape{sh[1]}, 2)
		ins := []*tensor.Tensor{x, b}
		gv, err := backend.Execute(backend.NewContext().WithBackend(mb), backend.OpAddBias, ins, nil)
		if err != nil {
			t.Fatalf("metal addbias %v: %v", sh, err)
		}
		gr, _ := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpAddBias, ins, nil)
		for i := range gv[0].Numel() {
			idx := tensor.Unravel(i, sh)
			g, r := gv[0].AtF64(idx...), gr[0].AtF64(idx...)
			if math.Abs(g-r) > 1e-5*math.Max(1, math.Abs(r)) {
				t.Fatalf("addbias %v [%d]: metal %v vs ref %v", sh, i, g, r)
			}
		}
	}
}

// §V-CROSS: the GPU RoPE BACKWARD (inverse rotation) matches the Pure-Go reference across
// shapes/heads/PosOffset/PI/YaRN, xPos→ref fallback. Enables GPU training of the RoPE path.
func TestMetalRoPEBackwardCrossReference(t *testing.T) {
	skipNoGPU(t)
	mb, ok := backend.Get(backend.Metal)
	if !ok {
		t.Fatal("metal available but not registered")
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
		{8, 16, backend.RoPEAttrs{XPos: true}},
	}
	for _, c := range cases {
		g := bench.RandF32(tensor.Shape{c.seq, c.width}, 7)
		gv, err := backend.Execute(backend.NewContext().WithBackend(mb), backend.OpRoPEBackward, []*tensor.Tensor{g}, c.attrs)
		if err != nil {
			t.Fatalf("metal %+v: %v", c, err)
		}
		gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpRoPEBackward, []*tensor.Tensor{g}, c.attrs)
		if err != nil {
			t.Fatal(err)
		}
		for i := range gv[0].Numel() {
			idx := tensor.Unravel(i, gv[0].Shape())
			gg, r := gv[0].AtF64(idx...), gr[0].AtF64(idx...)
			if math.Abs(gg-r) > 1e-4*math.Max(1, math.Abs(r)) {
				t.Fatalf("case %+v [%d]: metal %v vs ref %v", c, i, gg, r)
			}
		}
	}
}

// §V-CROSS: the GPU RMSNorm BACKWARD matches the Pure-Go reference for both dx (per-row)
// and dgamma (atomic cross-row reduction). Enables GPU RMSNorm training.
func TestMetalRMSNormBackwardCrossReference(t *testing.T) {
	skipNoGPU(t)
	mb, ok := backend.Get(backend.Metal)
	if !ok {
		t.Fatal("metal available but not registered")
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
			gv, err := backend.Execute(backend.NewContext().WithBackend(mb), backend.OpRMSNormBackward, []*tensor.Tensor{x, gamma, g}, attrs)
			if err != nil {
				t.Fatalf("metal %v eps=%g: %v", sh, eps, err)
			}
			gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpRMSNormBackward, []*tensor.Tensor{x, gamma, g}, attrs)
			if err != nil {
				t.Fatal(err)
			}
			for i := range gv[0].Numel() {
				idx := tensor.Unravel(i, gv[0].Shape())
				a, b := gv[0].AtF64(idx...), gr[0].AtF64(idx...)
				if math.Abs(a-b) > crossTol(d)*math.Max(1, math.Abs(b)) {
					t.Fatalf("dx %v eps=%g [%d]: metal %v vs ref %v", sh, eps, i, a, b)
				}
			}
			dgTol := 1e-6 * math.Sqrt(float64(d*rows+1))
			for i := range gv[1].Numel() {
				a, b := gv[1].AtF64(i), gr[1].AtF64(i)
				if math.Abs(a-b) > dgTol*math.Max(1, math.Abs(b)) {
					t.Fatalf("dgamma %v eps=%g [%d]: metal %v vs ref %v (tol %g)", sh, eps, i, a, b, dgTol)
				}
			}
		}
	}
}

// §V-CROSS: the GPU LayerNorm BACKWARD matches the Pure-Go reference for dx (per-row) and
// dgamma/dbeta (atomic cross-row reductions). Enables GPU LayerNorm training.
func TestMetalLayerNormBackwardCrossReference(t *testing.T) {
	skipNoGPU(t)
	mb, ok := backend.Get(backend.Metal)
	if !ok {
		t.Fatal("metal available but not registered")
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
			gv, err := backend.Execute(backend.NewContext().WithBackend(mb), backend.OpLayerNormBackward, []*tensor.Tensor{x, gamma, g}, attrs)
			if err != nil {
				t.Fatalf("metal %v eps=%g: %v", sh, eps, err)
			}
			gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpLayerNormBackward, []*tensor.Tensor{x, gamma, g}, attrs)
			if err != nil {
				t.Fatal(err)
			}
			for i := range gv[0].Numel() {
				idx := tensor.Unravel(i, gv[0].Shape())
				a, b := gv[0].AtF64(idx...), gr[0].AtF64(idx...)
				if math.Abs(a-b) > crossTol(d)*math.Max(1, math.Abs(b)) {
					t.Fatalf("dx %v eps=%g [%d]: metal %v vs ref %v", sh, eps, i, a, b)
				}
			}
			dgTol := 1e-6 * math.Sqrt(float64(d*rows+1))
			for _, out := range []int{1, 2} {
				for i := range gv[out].Numel() {
					a, b := gv[out].AtF64(i), gr[out].AtF64(i)
					if math.Abs(a-b) > dgTol*math.Max(1, math.Abs(b)) {
						t.Fatalf("out%d %v eps=%g [%d]: metal %v vs ref %v (tol %g)", out, sh, eps, i, a, b, dgTol)
					}
				}
			}
		}
	}
}

// §V-CROSS: the GPU cross-entropy backward matches the Pure-Go reference across shapes and
// the label-smoothing / z-loss variants. Seeds the backward pass on the GPU.
func TestMetalCrossEntropyBackwardCrossReference(t *testing.T) {
	skipNoGPU(t)
	mb, ok := backend.Get(backend.Metal)
	if !ok {
		t.Fatal("metal available but not registered")
	}
	ref, _ := backend.Get(backend.Ref)
	shapes := []struct{ b, c int }{{1, 2}, {3, 5}, {130, 64}, {8, 200}}
	variants := []backend.CrossEntropyAttrs{{}, {LabelSmoothing: 0.1}, {ZLoss: 1e-3}, {LabelSmoothing: 0.1, ZLoss: 1e-4}}
	for _, sh := range shapes {
		z := bench.RandF32(tensor.Shape{sh.b, sh.c}, 1)
		tg := tensor.New(tensor.F64, tensor.Shape{sh.b})
		for i := range sh.b {
			tg.SetF64(float64(i%sh.c), i)
		}
		gup := tensor.FromFloat64(tensor.Shape{}, []float64{1.7})
		for _, at := range variants {
			gv, err := backend.Execute(backend.NewContext().WithBackend(mb), backend.OpCrossEntropyBackward, []*tensor.Tensor{z, tg, gup}, at)
			if err != nil {
				t.Fatalf("metal %+v %+v: %v", sh, at, err)
			}
			gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpCrossEntropyBackward, []*tensor.Tensor{z, tg, gup}, at)
			if err != nil {
				t.Fatal(err)
			}
			for i := range gv[0].Numel() {
				idx := tensor.Unravel(i, gv[0].Shape())
				a, b := gv[0].AtF64(idx...), gr[0].AtF64(idx...)
				if math.Abs(a-b) > 1e-5*math.Max(1, math.Abs(b)) {
					t.Fatalf("%+v %+v [%d]: metal %v vs ref %v", sh, at, i, a, b)
				}
			}
		}
	}
}

// §V-CROSS: the GPU embedding backward (atomic scatter-add) matches the Pure-Go reference,
// including REPEATED indices (collisions on a table row). Enables GPU embedding training.
func TestMetalEmbedBackwardCrossReference(t *testing.T) {
	skipNoGPU(t)
	mb, ok := backend.Get(backend.Metal)
	if !ok {
		t.Fatal("metal available but not registered")
	}
	ref, _ := backend.Get(backend.Ref)
	cases := []struct{ n, d, m int }{{4, 3, 6}, {10, 8, 1}, {2, 5, 100}, {256, 64, 130}}
	for _, cs := range cases {
		table := tensor.New(tensor.F32, tensor.Shape{cs.n, cs.d})
		idx := tensor.New(tensor.F64, tensor.Shape{cs.m})
		for i := range cs.m {
			idx.SetF64(float64((i*7)%cs.n), i)
		}
		g := bench.RandF32(tensor.Shape{cs.m, cs.d}, 4)
		gv, err := backend.Execute(backend.NewContext().WithBackend(mb), backend.OpEmbedBackward, []*tensor.Tensor{table, idx, g}, nil)
		if err != nil {
			t.Fatalf("metal %+v: %v", cs, err)
		}
		gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpEmbedBackward, []*tensor.Tensor{table, idx, g}, nil)
		if err != nil {
			t.Fatal(err)
		}
		tol := 1e-6 * math.Sqrt(float64(cs.m+1))
		for i := range gv[0].Numel() {
			idx2 := tensor.Unravel(i, gv[0].Shape())
			a, b := gv[0].AtF64(idx2...), gr[0].AtF64(idx2...)
			if math.Abs(a-b) > tol*math.Max(1, math.Abs(b)) {
				t.Fatalf("%+v [%d]: metal %v vs ref %v (tol %g)", cs, i, a, b, tol)
			}
		}
	}
}

// §T367 (ADR-0019 Phase 1): the device-resident buffer primitive round-trips f32 data
// (upload → download identical) and Release is idempotent. Nothing dispatches over it yet.
func TestMetalDeviceBufferRoundTrip(t *testing.T) {
	skipNoGPU(t)
	for _, n := range []int{0, 1, 7, 4096, 2048 * 512} {
		src := make([]float32, n)
		for i := range src {
			src[i] = float32(i)*0.5 - 3
		}
		db, err := metal.NewDeviceBufferF32(src)
		if err != nil {
			t.Fatalf("n=%d upload: %v", n, err)
		}
		if db.Len() != n {
			t.Fatalf("n=%d Len=%d", n, db.Len())
		}
		dst := make([]float32, n)
		if err := db.DownloadF32(dst); err != nil {
			t.Fatalf("n=%d download: %v", n, err)
		}
		for i := range src {
			if dst[i] != src[i] {
				t.Fatalf("n=%d [%d]: got %v want %v", n, i, dst[i], src[i])
			}
		}
		db.Release()
		db.Release() // idempotent
	}
}
