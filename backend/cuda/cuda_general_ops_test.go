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

// The general CUDA backend (backend.Get(CUDA)) must now run the elementwise + softmax ops
// on the GPU with results matching the CPU reference within f32 tolerance — proving the
// backend is a fuller general backend, not just OpMatMul (previously everything but matmul
// fell back to Pure-Go).
func TestCUDAGeneralOps(t *testing.T) {
	skipNoGPU(t)
	cu, ok := backend.Get(backend.CUDA)
	if !ok {
		t.Skip("cuda backend not registered")
	}
	ref, _ := backend.Get(backend.Ref)
	cctx := backend.NewContext().WithBackend(cu)
	rctx := backend.NewContext().WithBackend(ref)

	exec := func(ctx *backend.Context, op backend.Op, ins ...*tensor.Tensor) *tensor.Tensor {
		o, e := backend.Execute(ctx, op, ins, nil)
		must(t, e)
		return o[0]
	}
	closeTo := func(tag string, g, r *tensor.Tensor) {
		gf := g.Contiguous().Storage().F32()
		rf := r.Cast(tensor.F32).Contiguous().Storage().F32()
		if len(gf) != len(rf) {
			t.Fatalf("%s: len %d != %d", tag, len(gf), len(rf))
		}
		const tol = 2e-4
		for i := range gf {
			if d := math.Abs(float64(gf[i] - rf[i])); d > tol+tol*math.Abs(float64(rf[i])) {
				t.Fatalf("%s[%d]: cuda %v vs ref %v (Δ%g)", tag, i, gf[i], rf[i], d)
			}
		}
		t.Logf("%s: CUDA general backend == reference", tag)
	}

	shape := tensor.Shape{4, 50}
	x := bench.RandF32(shape, 1)
	y := bench.RandF32(shape, 2)

	// Unary: SiLU, GELU.
	closeTo("OpSiLU", exec(cctx, backend.OpSiLU, x), exec(rctx, backend.OpSiLU, x))
	closeTo("OpGELU", exec(cctx, backend.OpGELU, x), exec(rctx, backend.OpGELU, x))
	// Binary: Add, Mul.
	closeTo("OpAdd", exec(cctx, backend.OpAdd, x, y), exec(rctx, backend.OpAdd, x, y))
	closeTo("OpMul", exec(cctx, backend.OpMul, x, y), exec(rctx, backend.OpMul, x, y))
	// Softmax over the last axis.
	closeTo("OpSoftmax", exec(cctx, backend.OpSoftmax, x), exec(rctx, backend.OpSoftmax, x))

	// RMSNorm (x, gamma) and LayerNorm (x, gamma, beta) over the last axis.
	gamma := bench.RandF32(tensor.Shape{50}, 3)
	beta := bench.RandF32(tensor.Shape{50}, 4)
	na := backend.NormAttrs{Eps: 1e-5}
	rmsG, _ := backend.Execute(cctx, backend.OpRMSNorm, []*tensor.Tensor{x, gamma}, na)
	rmsR, _ := backend.Execute(rctx, backend.OpRMSNorm, []*tensor.Tensor{x, gamma}, na)
	closeTo("OpRMSNorm", rmsG[0], rmsR[0])
	lnG, _ := backend.Execute(cctx, backend.OpLayerNorm, []*tensor.Tensor{x, gamma, beta}, na)
	lnR, _ := backend.Execute(rctx, backend.OpLayerNorm, []*tensor.Tensor{x, gamma, beta}, na)
	closeTo("OpLayerNorm", lnG[0], lnR[0])

	// RoPE on a 2D [seq, heads·hd] tensor.
	q := bench.RandF32(tensor.Shape{6, 64}, 5)
	ra := backend.RoPEAttrs{Base: 10000, Heads: 8, PosOffset: 3}
	ropeG, _ := backend.Execute(cctx, backend.OpRoPE, []*tensor.Tensor{q}, ra)
	ropeR, _ := backend.Execute(rctx, backend.OpRoPE, []*tensor.Tensor{q}, ra)
	closeTo("OpRoPE", ropeG[0], ropeR[0])

	// MHA (softmax → looser f32 tolerance): standard causal, GQA, and KV-cache decode (sq<sk).
	closeMHA := func(tag string, g, r *tensor.Tensor) {
		gf := g.Contiguous().Storage().F32()
		rf := r.Cast(tensor.F32).Contiguous().Storage().F32()
		if len(gf) != len(rf) {
			t.Fatalf("%s: len %d != %d", tag, len(gf), len(rf))
		}
		for i := range gf {
			if d := math.Abs(float64(gf[i] - rf[i])); d > 1e-3+1e-3*math.Abs(float64(rf[i])) {
				t.Fatalf("%s[%d]: cuda %v vs ref %v (Δ%g)", tag, i, gf[i], rf[i], d)
			}
		}
		t.Logf("%s: CUDA general backend == reference", tag)
	}
	mha := func(tag string, sq, sk, heads, kvHeads, dk int, causal bool) {
		dm := heads * dk
		kvw := kvHeads * dk
		qq := bench.RandF32(tensor.Shape{sq, dm}, 10)
		kk := bench.RandF32(tensor.Shape{sk, kvw}, 11)
		vv := bench.RandF32(tensor.Shape{sk, kvw}, 12)
		aa := backend.AttnAttrs{Heads: heads, KVHeads: kvHeads, Causal: causal}
		gg, _ := backend.Execute(cctx, backend.OpMHA, []*tensor.Tensor{qq, kk, vv}, aa)
		rr, _ := backend.Execute(rctx, backend.OpMHA, []*tensor.Tensor{qq, kk, vv}, aa)
		closeMHA(tag, gg[0], rr[0])
	}
	mha("OpMHA/standard", 6, 6, 8, 8, 8, true)
	mha("OpMHA/gqa", 6, 6, 8, 2, 8, true)
	mha("OpMHA/decode", 1, 6, 8, 2, 8, true)
}
