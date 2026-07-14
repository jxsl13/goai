//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// End-to-end: a FULL single-head Llama decoder layer — pre-norm attention with
// Q/K/V/O projections + RoPE + causal attention + residual, then pre-norm
// SwiGLU FFN + residual — composed ENTIRELY from the on-device primitives, and
// verified against the same layer computed op-by-op on the Pure-Go reference.
// This is the capstone: every resident kernel (RMSNorm, matmul, RoPE, QKᵀ,
// scale+mask, softmax, ·V, SiLU, Mul, Add, Clone) chained into a whole layer.
func TestCUDADecoderLayerMatchesRef(t *testing.T) {
	skipNoGPU(t)
	ref, _ := backend.Get(backend.Ref)
	const seq, d, ffn = 6, 32, 88
	rp := backend.RoPEAttrs{}
	scale := float32(1 / math.Sqrt(float64(d)))

	x := bench.RandF32(tensor.Shape{seq, d}, 1)
	g1 := bench.RandF32(tensor.Shape{d}, 2)
	wq := bench.RandF32(tensor.Shape{d, d}, 3)
	wk := bench.RandF32(tensor.Shape{d, d}, 4)
	wv := bench.RandF32(tensor.Shape{d, d}, 5)
	wo := bench.RandF32(tensor.Shape{d, d}, 6)
	g2 := bench.RandF32(tensor.Shape{d}, 7)
	wg := bench.RandF32(tensor.Shape{d, ffn}, 8)
	wu := bench.RandF32(tensor.Shape{d, ffn}, 9)
	wd := bench.RandF32(tensor.Shape{ffn, d}, 10)

	// ---------------- device: fully resident ----------------
	rg1, _ := cuda.NewResidentVec(g1)
	rg2, _ := cuda.NewResidentVec(g2)
	rWq, _ := cuda.NewResidentB(wq)
	rWk, _ := cuda.NewResidentB(wk)
	rWv, _ := cuda.NewResidentB(wv)
	rWo, _ := cuda.NewResidentB(wo)
	rWg, _ := cuda.NewResidentB(wg)
	rWu, _ := cuda.NewResidentB(wu)
	rWd, _ := cuda.NewResidentB(wd)
	defer func() {
		for _, r := range []interface{ Free() }{rg1, rg2, rWq, rWk, rWv, rWo, rWg, rWu, rWd} {
			r.Free()
		}
	}()

	dx, _ := cuda.UploadF32(x) // residual 1
	dh, _ := dx.Clone()
	must(t, dh.RMSNorm(rg1, 1e-5))
	dq, err := rWq.MatMulDevice(dh)
	must(t, err)
	dk, err := rWk.MatMulDevice(dh)
	must(t, err)
	dv, err := rWv.MatMulDevice(dh)
	must(t, err)
	must(t, dq.RoPE(rp))
	must(t, dk.RoPE(rp))
	scores, err := dq.MatMulBT(dk)
	must(t, err)
	must(t, scores.CausalScaleMask(scale, 0))
	must(t, scores.Softmax())
	dattn, err := scores.MatMul(dv)
	must(t, err)
	dao, err := rWo.MatMulDevice(dattn)
	must(t, err)
	must(t, dao.Add(dx)) // x1 = x + attn·Wo
	// FFN
	dh2, _ := dao.Clone()
	must(t, dh2.RMSNorm(rg2, 1e-5))
	dgate, err := rWg.MatMulDevice(dh2)
	must(t, err)
	dup, err := rWu.MatMulDevice(dh2)
	must(t, err)
	must(t, dgate.SiLU())
	must(t, dgate.Mul(dup))
	ddown, err := rWd.MatMulDevice(dgate)
	must(t, err)
	must(t, ddown.Add(dao)) // y = x1 + ffn
	got, err := ddown.ToHost()
	must(t, err)
	for _, f := range []*cuda.DeviceF32{dx, dh, dq, dk, dv, scores, dattn, dao, dh2, dgate, dup, ddown} {
		f.Free()
	}

	// ---------------- reference: op by op ----------------
	rc := backend.NewContext().WithBackend(ref)
	ex := func(op backend.Op, a backend.Attrs, ins ...*tensor.Tensor) *tensor.Tensor {
		o, e := backend.Execute(rc, op, ins, a)
		must(t, e)
		return o[0]
	}
	h := ex(backend.OpRMSNorm, backend.NormAttrs{Eps: 1e-5}, x, g1)
	q := ex(backend.OpRoPE, rp, ex(backend.OpMatMul, nil, h, wq))
	k := ex(backend.OpRoPE, rp, ex(backend.OpMatMul, nil, h, wk))
	v := ex(backend.OpMatMul, nil, h, wv)
	kt, _ := k.Transpose(0, 1)
	sc := ex(backend.OpMatMul, nil, q, kt)
	masked := tensor.New(tensor.F32, tensor.Shape{seq, seq})
	for i := 0; i < seq; i++ {
		for j := 0; j < seq; j++ {
			if j > i {
				masked.SetF64(math.Inf(-1), i, j)
			} else {
				masked.SetF64(sc.AtF64(i, j)*float64(scale), i, j)
			}
		}
	}
	attn := ex(backend.OpMatMul, nil, ex(backend.OpSoftmax, nil, masked), v)
	x1 := ex(backend.OpAdd, nil, x, ex(backend.OpMatMul, nil, attn, wo))
	h2 := ex(backend.OpRMSNorm, backend.NormAttrs{Eps: 1e-5}, x1, g2)
	act := ex(backend.OpMul, nil, ex(backend.OpSiLU, nil, ex(backend.OpMatMul, nil, h2, wg)), ex(backend.OpMatMul, nil, h2, wu))
	want := ex(backend.OpAdd, nil, x1, ex(backend.OpMatMul, nil, act, wd))

	var maxRel float64
	for i := range got.Numel() {
		idx := tensor.Unravel(i, got.Shape())
		g, w := got.AtF64(idx...), want.AtF64(idx...)
		if r := math.Abs(g-w) / (math.Abs(w) + 1e-3); r > maxRel {
			maxRel = r
		}
		if math.Abs(g-w) > 1e-2*math.Max(1, math.Abs(w)) {
			t.Fatalf("[%d]: device-layer %v vs ref %v", i, g, w)
		}
	}
	t.Logf("full decoder layer device-vs-ref max rel err = %.2e", maxRel)
}
