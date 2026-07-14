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

// End-to-end: a full MULTI-HEAD Llama decoder layer — pre-norm multi-head causal
// attention (Q/K/V/O projections + RoPE + batched MHA + residual) then pre-norm
// SwiGLU FFN + residual — composed from the on-device primitives (with the
// batched MultiHeadAttention), verified against a per-head reference on the
// Pure-Go backend. Confirms multi-head attention works inside a whole layer.
func TestCUDAMultiHeadDecoderLayerMatchesRef(t *testing.T) {
	skipNoGPU(t)
	ref, _ := backend.Get(backend.Ref)
	const seq, heads, hd, ffn = 6, 4, 16, 176
	const d = heads * hd // 64
	rp := backend.RoPEAttrs{Heads: heads}
	scale := 1 / math.Sqrt(float64(hd))

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

	// ---------------- device ----------------
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

	dx, _ := cuda.UploadF32(x)
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
	dattn, err := cuda.MultiHeadAttention(dq, dk, dv, heads, true)
	must(t, err)
	dao, err := rWo.MatMulDevice(dattn)
	must(t, err)
	must(t, dao.Add(dx))
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
	must(t, ddown.Add(dao))
	got, err := ddown.ToHost()
	must(t, err)
	for _, f := range []*cuda.DeviceF32{dx, dh, dq, dk, dv, dattn, dao, dh2, dgate, dup, ddown} {
		f.Free()
	}

	// ---------------- reference ----------------
	rc := backend.NewContext().WithBackend(ref)
	ex := func(op backend.Op, a backend.Attrs, ins ...*tensor.Tensor) *tensor.Tensor {
		o, e := backend.Execute(rc, op, ins, a)
		must(t, e)
		return o[0]
	}
	h := ex(backend.OpRMSNorm, backend.NormAttrs{Eps: 1e-5}, x, g1)
	qp := ex(backend.OpRoPE, rp, ex(backend.OpMatMul, nil, h, wq))
	kp := ex(backend.OpRoPE, rp, ex(backend.OpMatMul, nil, h, wk))
	vp := ex(backend.OpMatMul, nil, h, wv)
	attn := tensor.New(tensor.F32, tensor.Shape{seq, d})
	for hh := 0; hh < heads; hh++ {
		qh, kh, vh := headSlice(qp, hh, hd), headSlice(kp, hh, hd), headSlice(vp, hh, hd)
		kt, _ := kh.Transpose(0, 1)
		sc := ex(backend.OpMatMul, nil, qh, kt)
		masked := tensor.New(tensor.F32, tensor.Shape{seq, seq})
		for i := 0; i < seq; i++ {
			for j := 0; j < seq; j++ {
				if j > i {
					masked.SetF64(math.Inf(-1), i, j)
				} else {
					masked.SetF64(sc.AtF64(i, j)*scale, i, j)
				}
			}
		}
		oh := ex(backend.OpMatMul, nil, ex(backend.OpSoftmax, nil, masked), vh)
		for i := 0; i < seq; i++ {
			for dd := 0; dd < hd; dd++ {
				attn.SetF64(oh.AtF64(i, dd), i, hh*hd+dd)
			}
		}
	}
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
			t.Fatalf("[%d]: mh-layer %v vs ref %v", i, g, w)
		}
	}
	t.Logf("multi-head decoder layer device-vs-ref max rel err = %.2e", maxRel)
}
