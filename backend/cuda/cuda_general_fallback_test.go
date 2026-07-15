//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/ref"
)

// The general CUDA kernels must NEVER change a result relative to Pure-Go: any case the
// device path doesn't serve (F64 dtype, non-contiguous input, broadcast binary, XPos RoPE,
// ALiBi/sliding-window MHA, mismatched norm weights) must fall back to the reference and
// produce the same value (or the same error). This locks the fallback contract so a future
// kernel change can't silently diverge.
func TestCUDAGeneralFallbackContract(t *testing.T) {
	skipNoGPU(t)
	cu, ok := backend.Get(backend.CUDA)
	if !ok {
		t.Skip("cuda backend not registered")
	}
	ref, _ := backend.Get(backend.Ref)
	cctx := backend.NewContext().WithBackend(cu)
	rctx := backend.NewContext().WithBackend(ref)

	ck := func(tag string, op backend.Op, a backend.Attrs, ins ...*tensor.Tensor) {
		g, ge := backend.Execute(cctx, op, ins, a)
		r, re := backend.Execute(rctx, op, ins, a)
		if (ge == nil) != (re == nil) {
			t.Errorf("%s: error mismatch — cuda=%v ref=%v", tag, ge, re)
			return
		}
		if ge != nil {
			t.Logf("%s: both error (ok)", tag)
			return
		}
		gf := g[0].Cast(tensor.F32).Contiguous().Storage().F32()
		rf := r[0].Cast(tensor.F32).Contiguous().Storage().F32()
		if len(gf) != len(rf) {
			t.Errorf("%s: len %d != %d", tag, len(gf), len(rf))
			return
		}
		for i := range gf {
			if d := math.Abs(float64(gf[i] - rf[i])); d > 1e-3+1e-3*math.Abs(float64(rf[i])) {
				t.Errorf("%s[%d]: cuda %v != ref %v", tag, i, gf[i], rf[i])
				return
			}
		}
		t.Logf("%s: falls back == reference", tag)
	}

	// --- F64 dtype → Pure-Go (Kernel returns false for non-F32) ---
	x64 := bench.RandF64(tensor.Shape{3, 8}, 1)
	g64 := bench.RandF64(tensor.Shape{8}, 2)
	ck("silu/f64", backend.OpSiLU, nil, x64)
	ck("add/f64", backend.OpAdd, nil, x64, x64)
	ck("matmul/f64", backend.OpMatMul, nil, bench.RandF64(tensor.Shape{3, 4}, 3), bench.RandF64(tensor.Shape{4, 5}, 4))
	ck("rmsnorm/f64", backend.OpRMSNorm, backend.NormAttrs{Eps: 1e-5}, x64, g64)

	// --- non-contiguous F32 (transposed view) → handled via Contiguous() ---
	tv, _ := bench.RandF32(tensor.Shape{8, 3}, 5).Transpose(0, 1)
	ck("gelu/noncontig", backend.OpGELU, nil, tv)
	ck("softmax/noncontig", backend.OpSoftmax, nil, tv)

	// --- single element ---
	ck("silu/1elem", backend.OpSiLU, nil, bench.RandF32(tensor.Shape{1, 1}, 6))

	// --- broadcast binary (mismatched numel) → reference broadcast ---
	ck("add/broadcast", backend.OpAdd, nil, bench.RandF32(tensor.Shape{4, 4}, 7), bench.RandF32(tensor.Shape{1, 4}, 8))

	// --- XPos RoPE (device path doesn't apply the xPos magnitude) → reference ---
	ck("rope/xpos", backend.OpRoPE, backend.RoPEAttrs{Base: 10000, Heads: 4, XPos: true},
		bench.RandF32(tensor.Shape{4, 32}, 9))

	// --- MHA ALiBi and sliding-window → reference ---
	q := bench.RandF32(tensor.Shape{6, 32}, 10)
	kk := bench.RandF32(tensor.Shape{6, 32}, 11)
	vv := bench.RandF32(tensor.Shape{6, 32}, 12)
	ck("mha/alibi", backend.OpMHA, backend.AttnAttrs{Heads: 4, Causal: true, ALiBi: true}, q, kk, vv)
	ck("mha/window", backend.OpMHA, backend.AttnAttrs{Heads: 4, Causal: true, Window: 2}, q, kk, vv)

	// --- mismatched gamma length → both error ---
	ck("rmsnorm/badgamma", backend.OpRMSNorm, backend.NormAttrs{Eps: 1e-5},
		bench.RandF32(tensor.Shape{3, 8}, 13), bench.RandF32(tensor.Shape{7}, 14))
}
