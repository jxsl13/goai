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
)

// cpuOp runs an op on the CPU backend and returns output 0.
func cpuOp(t *testing.T, op backend.Op, a backend.Attrs, ins ...*tensor.Tensor) *tensor.Tensor {
	cpu, _ := backend.Get(backend.CPU)
	o, err := backend.Execute(backend.NewContext().WithBackend(cpu), op, ins, a)
	if err != nil {
		t.Fatal(err)
	}
	return o[0]
}

// CUDA LayerNorm must match the CPU reference (torch semantics) within f32 tol.
func TestCUDALayerNormMatchesRef(t *testing.T) {
	skipNoGPU(t)
	const rows, cols = 6, 2048
	x := bench.RandF32(tensor.Shape{rows, cols}, 1)
	g := bench.RandF32(tensor.Shape{cols}, 2)
	b := bench.RandF32(tensor.Shape{cols}, 3)
	want := cpuOp(t, backend.OpLayerNorm, backend.NormAttrs{Eps: 1e-5}, x, g, b)

	dx, _ := cuda.UploadF32(x)
	dg, _ := cuda.NewResidentVec(g)
	db, _ := cuda.NewResidentVec(b)
	out, _ := cuda.NewDeviceF32(rows, cols)
	defer func() { dx.Free(); dg.Free(); db.Free(); out.Free() }()
	must(t, dx.LayerNormInto(dg, db, 1e-5, out))
	got, _ := out.ToHost()

	var maxRel float64
	for i := range got.Numel() {
		idx := tensor.Unravel(i, got.Shape())
		gv, wv := got.AtF64(idx...), want.AtF64(idx...)
		if r := math.Abs(gv-wv) / (math.Abs(wv) + 1e-2); r > maxRel {
			maxRel = r
		}
	}
	if maxRel > 1e-4 {
		t.Fatalf("CUDA LayerNorm vs ref maxRel %.3e", maxRel)
	}
	t.Logf("CUDA LayerNorm == ref (maxRel %.2e)", maxRel)
}

// CUDA AddBias must match the CPU reference (row-broadcast) exactly (add is exact).
func TestCUDAAddBiasMatchesRef(t *testing.T) {
	skipNoGPU(t)
	const rows, n = 5, 1536
	x := bench.RandF32(tensor.Shape{rows, n}, 4)
	b := bench.RandF32(tensor.Shape{n}, 5)
	want := cpuOp(t, backend.OpAddBias, nil, x, b)

	dx, _ := cuda.UploadF32(x)
	db, _ := cuda.NewResidentVec(b)
	defer func() { dx.Free(); db.Free() }()
	must(t, dx.AddBias(db))
	got, _ := dx.ToHost()

	var maxAbs float64
	for i := range got.Numel() {
		idx := tensor.Unravel(i, got.Shape())
		if d := math.Abs(got.AtF64(idx...) - want.AtF64(idx...)); d > maxAbs {
			maxAbs = d
		}
	}
	if maxAbs > 1e-5 {
		t.Fatalf("CUDA AddBias vs ref maxAbs %.3e", maxAbs)
	}
	t.Logf("CUDA AddBias == ref (maxAbs %.2e)", maxAbs)
}
