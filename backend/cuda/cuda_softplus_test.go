//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/tensor"
)

// TestCUDASoftplus checks cu_softplus_f32 (the softplus Unary op, the last Mamba decode primitive —
// it keeps Δ strictly positive) matches the reference OpSoftplus across a range including large
// positive/negative inputs (where the naive log(1+e^x) would overflow).
func TestCUDASoftplus(t *testing.T) {
	skipNoGPU(t)
	const softplusOp = 12 // recUnarySoftplus
	in := []float32{-40, -5, -1, -0.1, 0, 0.1, 1, 5, 40, 12.5, -12.5}

	xt := tensor.New(tensor.F32, tensor.Shape{len(in)})
	copy(xt.Storage().F32(), in)
	refT, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()), backend.OpSoftplus,
		[]*tensor.Tensor{xt}, nil)
	if err != nil {
		t.Fatalf("ref OpSoftplus: %v", err)
	}

	rec, err := cuda.NewRecorder()
	if err != nil {
		t.Fatalf("NewRecorder: %v", err)
	}
	defer rec.Free()

	dx, _ := cuda.NewDeviceBufferF32(in)
	do, _ := cuda.NewDeviceF32(1, len(in))
	defer dx.Free()
	defer do.Free()
	if err := rec.Unary(dx, do, softplusOp); err != nil {
		t.Fatalf("Unary(softplus): %v", err)
	}
	if err := rec.Wait(); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, len(in))
	if err := do.DownloadF32(got); err != nil {
		t.Fatal(err)
	}
	for i := range in {
		want := refT[0].AtF64(i)
		if math.IsNaN(float64(got[i])) || math.Abs(float64(got[i])-want) > 1e-4*math.Max(1, math.Abs(want)) {
			t.Fatalf("softplus[%d] (x=%v): gpu %v vs ref %v", i, in[i], got[i], want)
		}
	}
	t.Logf("cu_softplus_f32 matches the reference OpSoftplus over %d inputs incl. ±40 (Mamba Δ primitive)", len(in))
}
