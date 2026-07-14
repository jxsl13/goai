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

// The on-device elementwise Mul (SwiGLU gate·up) must match the reference.
func TestCUDADeviceMulMatchesRef(t *testing.T) {
	skipNoGPU(t)
	ref, _ := backend.Get(backend.Ref)
	for _, s := range []struct{ r, c int }{{1, 1}, {4, 8}, {33, 17}, {8, 512}} {
		a := bench.RandF32(tensor.Shape{s.r, s.c}, 1)
		b := bench.RandF32(tensor.Shape{s.r, s.c}, 2)
		da, _ := cuda.UploadF32(a)
		db, _ := cuda.UploadF32(b)
		if err := da.Mul(db); err != nil {
			t.Fatal(err)
		}
		got, _ := da.ToHost()
		gr, _ := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpMul, []*tensor.Tensor{a, b}, nil)
		for i := range got.Numel() {
			idx := tensor.Unravel(i, got.Shape())
			g, r := got.AtF64(idx...), gr[0].AtF64(idx...)
			if math.Abs(g-r) > 1e-6*math.Max(1, math.Abs(r)) {
				t.Fatalf("%v [%d]: device-mul %v vs ref %v", s, i, g, r)
			}
		}
		da.Free()
		db.Free()
	}
}

// End-to-end: a full Llama SwiGLU FFN block composed ENTIRELY from the on-device
// primitives — y = x + (SiLU(RMSNorm(x)·Wgate) ⊙ (RMSNorm(x)·Wup)) · Wdown —
// must match the same block computed op-by-op on the Pure-Go reference, within a
// whole-block f32 tolerance. This is the composition test: the kernels are
// individually verified; here they chain resident on one stream with no host
// round-trip, and the assembled result still tracks the reference.
func TestCUDASwiGLUBlockMatchesRef(t *testing.T) {
	skipNoGPU(t)
	ref, _ := backend.Get(backend.Ref)
	const rows, d, ffn = 8, 256, 688 // Llama ~2.7× ffn ratio
	x := bench.RandF32(tensor.Shape{rows, d}, 1)
	gamma := bench.RandF32(tensor.Shape{d}, 2)
	// scale weights down so a two-matmul chain stays well-conditioned
	wg := bench.RandF32(tensor.Shape{d, ffn}, 3)
	wu := bench.RandF32(tensor.Shape{d, ffn}, 4)
	wd := bench.RandF32(tensor.Shape{ffn, d}, 5)

	// --- on device, fully resident ---
	gv, _ := cuda.NewResidentVec(gamma)
	rWg, _ := cuda.NewResidentB(wg)
	rWu, _ := cuda.NewResidentB(wu)
	rWd, _ := cuda.NewResidentB(wd)
	defer func() { gv.Free(); rWg.Free(); rWu.Free(); rWd.Free() }()

	dresid, _ := cuda.UploadF32(x)
	dh, _ := cuda.UploadF32(x)
	must(t, dh.RMSNorm(gv, 1e-5))
	dgate, err := rWg.MatMulDevice(dh)
	must(t, err)
	dup, err := rWu.MatMulDevice(dh)
	must(t, err)
	must(t, dgate.SiLU())
	must(t, dgate.Mul(dup))
	ddown, err := rWd.MatMulDevice(dgate)
	must(t, err)
	must(t, ddown.Add(dresid))
	got, err := ddown.ToHost()
	must(t, err)
	for _, f := range []*cuda.DeviceF32{dresid, dh, dgate, dup, ddown} {
		f.Free()
	}

	// --- reference, op by op ---
	rc := backend.NewContext().WithBackend(ref)
	ex := func(op backend.Op, a backend.Attrs, ins ...*tensor.Tensor) *tensor.Tensor {
		o, e := backend.Execute(rc, op, ins, a)
		must(t, e)
		return o[0]
	}
	h := ex(backend.OpRMSNorm, backend.NormAttrs{Eps: 1e-5}, x, gamma)
	gate := ex(backend.OpSiLU, nil, ex(backend.OpMatMul, nil, h, wg))
	up := ex(backend.OpMatMul, nil, h, wu)
	act := ex(backend.OpMul, nil, gate, up)
	want := ex(backend.OpAdd, nil, x, ex(backend.OpMatMul, nil, act, wd))

	var maxRel float64
	for i := range got.Numel() {
		idx := tensor.Unravel(i, got.Shape())
		g, w := got.AtF64(idx...), want.AtF64(idx...)
		if r := math.Abs(g-w) / (math.Abs(w) + 1e-4); r > maxRel {
			maxRel = r
		}
		if math.Abs(g-w) > 5e-3*math.Max(1, math.Abs(w)) {
			t.Fatalf("[%d]: device-block %v vs ref %v", i, g, w)
		}
	}
	t.Logf("SwiGLU block device-vs-ref max rel err = %.2e", maxRel)
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

// Latency of the whole SwiGLU FFN block, fully device-resident (one upload, one
// download; RMSNorm + 3 matmuls + SiLU + Mul + residual all on the stream).
func BenchmarkSwiGLUBlockResident_M8_D2048(b *testing.B) {
	if _, ok := backend.Get(backend.CUDA); !ok {
		b.Skip("cuda not available")
	}
	const rows, d, ffn = 8, 2048, 5504
	x := bench.RandF32(tensor.Shape{rows, d}, 1)
	gv, _ := cuda.NewResidentVec(bench.RandF32(tensor.Shape{d}, 2))
	rWg, _ := cuda.NewResidentB(bench.RandF32(tensor.Shape{d, ffn}, 3))
	rWu, _ := cuda.NewResidentB(bench.RandF32(tensor.Shape{d, ffn}, 4))
	rWd, _ := cuda.NewResidentB(bench.RandF32(tensor.Shape{ffn, d}, 5))
	defer func() { gv.Free(); rWg.Free(); rWu.Free(); rWd.Free() }()
	b.ResetTimer()
	for range b.N {
		dresid, _ := cuda.UploadF32(x)
		dh, _ := cuda.UploadF32(x)
		dh.RMSNorm(gv, 1e-5)
		dgate, _ := rWg.MatMulDevice(dh)
		dup, _ := rWu.MatMulDevice(dh)
		dgate.SiLU()
		dgate.Mul(dup)
		ddown, _ := rWd.MatMulDevice(dgate)
		ddown.Add(dresid)
		if _, err := ddown.ToHost(); err != nil {
			b.Fatal(err)
		}
		dresid.Free()
		dh.Free()
		dgate.Free()
		dup.Free()
		ddown.Free()
	}
}
