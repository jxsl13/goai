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

// The causal scale+mask kernel: scores[i][j] *= scale, and → −inf for j>i.
func TestCUDADeviceCausalScaleMask(t *testing.T) {
	skipNoGPU(t)
	const seq = 6
	x := bench.RandF32(tensor.Shape{seq, seq}, 3)
	dx, _ := cuda.UploadF32(x)
	const scale = 0.25
	if err := dx.CausalScaleMask(scale, 0); err != nil {
		t.Fatal(err)
	}
	got, _ := dx.ToHost()
	for i := 0; i < seq; i++ {
		for j := 0; j < seq; j++ {
			g := got.AtF64(i, j)
			if j > i {
				if !math.IsInf(g, -1) {
					t.Fatalf("[%d,%d] should be -inf, got %v", i, j, g)
				}
			} else if want := x.AtF64(i, j) * scale; math.Abs(g-want) > 1e-6*math.Max(1, math.Abs(want)) {
				t.Fatalf("[%d,%d] scaled %v want %v", i, j, g, want)
			}
		}
	}
	dx.Free()
}

// End-to-end single-head causal attention composed entirely from on-device
// primitives — RoPE(Q), RoPE(K), Q·Kᵀ, scale+causal-mask, softmax, ·V — must
// match the same computation on the Pure-Go reference (RoPE + matmuls + softmax,
// with the mask applied on the host), within an attention f32 tolerance.
func TestCUDAAttentionBlockMatchesRef(t *testing.T) {
	skipNoGPU(t)
	ref, _ := backend.Get(backend.Ref)
	const seq, hd = 8, 32
	rp := backend.RoPEAttrs{}
	scale := float32(1 / math.Sqrt(float64(hd)))
	q := bench.RandF32(tensor.Shape{seq, hd}, 1)
	k := bench.RandF32(tensor.Shape{seq, hd}, 2)
	v := bench.RandF32(tensor.Shape{seq, hd}, 3)

	// --- device, fully resident ---
	dq, _ := cuda.UploadF32(q)
	dk, _ := cuda.UploadF32(k)
	dv, _ := cuda.UploadF32(v)
	must(t, dq.RoPE(rp))
	must(t, dk.RoPE(rp))
	scores, err := dq.MatMulBT(dk) // [seq,seq]
	must(t, err)
	must(t, scores.CausalScaleMask(scale, 0))
	must(t, scores.Softmax())
	out, err := scores.MatMul(dv) // [seq,hd]
	must(t, err)
	got, err := out.ToHost()
	must(t, err)
	for _, f := range []*cuda.DeviceF32{dq, dk, dv, scores, out} {
		f.Free()
	}

	// --- reference, op by op ---
	rc := backend.NewContext().WithBackend(ref)
	ex := func(op backend.Op, a backend.Attrs, ins ...*tensor.Tensor) *tensor.Tensor {
		o, e := backend.Execute(rc, op, ins, a)
		must(t, e)
		return o[0]
	}
	qr := ex(backend.OpRoPE, rp, q)
	kr := ex(backend.OpRoPE, rp, k)
	krT, _ := kr.Transpose(0, 1)
	scoresR := ex(backend.OpMatMul, nil, qr, krT) // [seq,seq]
	masked := tensor.New(tensor.F32, tensor.Shape{seq, seq})
	for i := 0; i < seq; i++ {
		for j := 0; j < seq; j++ {
			if j > i {
				masked.SetF64(math.Inf(-1), i, j)
			} else {
				masked.SetF64(scoresR.AtF64(i, j)*float64(scale), i, j)
			}
		}
	}
	sm := ex(backend.OpSoftmax, nil, masked)
	want := ex(backend.OpMatMul, nil, sm, v) // [seq,hd]

	for i := range got.Numel() {
		idx := tensor.Unravel(i, got.Shape())
		g, w := got.AtF64(idx...), want.AtF64(idx...)
		if math.Abs(g-w) > 5e-3*math.Max(1, math.Abs(w)) {
			t.Fatalf("[%d]: device-attn %v vs ref %v", i, g, w)
		}
	}
}
