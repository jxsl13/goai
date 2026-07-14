package ref

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestAddBiasF32Parity exercises the devirtualised F32 fast paths (§T646) of both
// OpAddBias forward (out[i,j]=x[i,j]+b[j]) and OpAddBiasBackward (dbias[j]=Σ_i g[i,j]).
// The existing suite only covers F64, so this guards F32 ≡ float32(F64) (≤1e-3): the
// F32 kernels read inputs as float64 and accumulate the column-sum in float64, so they
// must match the F64 reference rounded to float32.
func TestAddBiasF32Parity(t *testing.T) {
	ref, ok := backend.Get(backend.Ref)
	if !ok {
		t.Fatal("ref backend not registered")
	}
	ctx := backend.NewContext().WithBackend(ref)

	const m, n = 5, 4
	xd := make([]float64, m*n)
	gd := make([]float64, m*n)
	bd := make([]float64, n)
	for i := range xd {
		xd[i] = float64(i)*0.37 - 3.1
		gd[i] = float64(i)*-0.21 + 1.7
	}
	for j := range bd {
		bd[j] = float64(j)*0.9 - 1.25
	}

	xf := make([]float32, len(xd))
	gf := make([]float32, len(gd))
	bf := make([]float32, len(bd))
	for i := range xd {
		xf[i] = float32(xd[i])
		gf[i] = float32(gd[i])
	}
	for j := range bd {
		bf[j] = float32(bd[j])
	}

	// ---- Forward: y = x + bias ----
	x64 := tensor.FromFloat64(tensor.Shape{m, n}, xd)
	b64 := tensor.FromFloat64(tensor.Shape{n}, bd)
	x32 := tensor.FromFloat32(tensor.Shape{m, n}, xf)
	b32 := tensor.FromFloat32(tensor.Shape{n}, bf)

	y64, err := backend.Execute(ctx, backend.OpAddBias, []*tensor.Tensor{x64, b64}, nil)
	if err != nil {
		t.Fatalf("addbias f64 forward: %v", err)
	}
	y32, err := backend.Execute(ctx, backend.OpAddBias, []*tensor.Tensor{x32, b32}, nil)
	if err != nil {
		t.Fatalf("addbias f32 forward: %v", err)
	}
	if y32[0].Dtype() != tensor.F32 {
		t.Fatalf("forward: expected F32 output, got %v", y32[0].Dtype())
	}
	for i := range m {
		for j := range n {
			got := y32[0].AtF64(i, j)
			want := float64(float32(y64[0].AtF64(i, j)))
			if math.Abs(got-want) > 1e-3 {
				t.Errorf("forward (%d,%d): f32=%v want float32(f64)=%v", i, j, got, want)
			}
		}
	}

	// ---- Backward: dbias[j] = Σ_i g[i,j] ----
	g64 := tensor.FromFloat64(tensor.Shape{m, n}, gd)
	g32 := tensor.FromFloat32(tensor.Shape{m, n}, gf)

	db64, err := backend.Execute(ctx, backend.OpAddBiasBackward, []*tensor.Tensor{g64}, nil)
	if err != nil {
		t.Fatalf("addbias-backward f64: %v", err)
	}
	db32, err := backend.Execute(ctx, backend.OpAddBiasBackward, []*tensor.Tensor{g32}, nil)
	if err != nil {
		t.Fatalf("addbias-backward f32: %v", err)
	}
	if db32[0].Dtype() != tensor.F32 {
		t.Fatalf("backward: expected F32 output, got %v", db32[0].Dtype())
	}
	for j := range n {
		got := db32[0].AtF64(j)
		want := float64(float32(db64[0].AtF64(j)))
		if math.Abs(got-want) > 1e-3 {
			t.Errorf("backward db[%d]: f32=%v want float32(f64)=%v", j, got, want)
		}
	}
}
