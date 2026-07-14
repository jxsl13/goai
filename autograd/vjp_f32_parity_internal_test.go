package autograd

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// F32 parity for the §T632–T635 devirtualization fast paths (§T637). Like the MoE
// tests, TestGradCheckAllOps runs F64-only, so the F32 branch of each fast-path
// function is otherwise unverified. One op per shared fast-path function suffices:
// unaryVJP, broadcastVJP, scaleInto (dot/nrm2/axpy), extremumVJP, prodVJP. Each F32
// branch computes in float64 and only rounds the store, so it must equal
// float32(the F64 path). Helpers moeF64/moeCastF32/moeAssertClose are in
// vjp_moe_f32_internal_test.go (same package).

func rowReduceF64(x *tensor.Tensor, prod bool) *tensor.Tensor {
	r, c := x.Shape()[0], x.Shape()[1]
	xs := x.Storage().F64()
	out := tensor.New(tensor.F64, tensor.Shape{r})
	os := out.Storage().F64()
	for i := 0; i < r; i++ {
		acc := xs[i*c]
		if prod {
			for j := 1; j < c; j++ {
				acc *= xs[i*c+j]
			}
		} else {
			for j := 1; j < c; j++ {
				if xs[i*c+j] > acc {
					acc = xs[i*c+j]
				}
			}
		}
		os[i] = acc
	}
	return out
}

func TestUnaryVJPF32Parity(t *testing.T) {
	vjp := vjps[backend.OpTanh] // exercises unaryVJP's F32 branch
	x := moeF64(tensor.Shape{8, 8}, 1, false)
	y := moeF64(tensor.Shape{8, 8}, 2, false) // arbitrary y, identical across calls
	g := moeF64(tensor.Shape{8, 8}, 3, false)
	o64, err := vjp(nil, []*tensor.Tensor{x}, []*tensor.Tensor{y}, nil, g)
	if err != nil {
		t.Fatal(err)
	}
	o32, err := vjp(nil, []*tensor.Tensor{moeCastF32(x)}, []*tensor.Tensor{moeCastF32(y)}, nil, moeCastF32(g))
	if err != nil {
		t.Fatal(err)
	}
	moeAssertClose(t, "tanh.gin", o32[0], o64[0])
}

func TestBroadcastVJPF32Parity(t *testing.T) {
	vjp := vjps[backend.OpSum] // exercises broadcastVJP's F32 branch
	x := moeF64(tensor.Shape{16, 8}, 1, false)
	g := moeF64(tensor.Shape{16}, 2, false) // reduced over axis 1
	attrs := backend.ReduceAttrs{Axes: []int{1}}
	o64, err := vjp(nil, []*tensor.Tensor{x}, nil, attrs, g)
	if err != nil {
		t.Fatal(err)
	}
	o32, err := vjp(nil, []*tensor.Tensor{moeCastF32(x)}, nil, attrs, moeCastF32(g))
	if err != nil {
		t.Fatal(err)
	}
	moeAssertClose(t, "sum.gin", o32[0], o64[0])
}

func TestScaleIntoF32Parity(t *testing.T) {
	vjp := vjps[backend.OpNrm2] // exercises scaleInto's F32 branch
	x := moeF64(tensor.Shape{64}, 1, false)
	out := tensor.New(tensor.F64, tensor.Shape{})
	out.Storage().F64()[0] = 5.0 // nonzero norm
	g := tensor.New(tensor.F64, tensor.Shape{})
	g.Storage().F64()[0] = 0.9
	o64, err := vjp(nil, []*tensor.Tensor{x}, []*tensor.Tensor{out}, nil, g)
	if err != nil {
		t.Fatal(err)
	}
	o32, err := vjp(nil, []*tensor.Tensor{moeCastF32(x)}, []*tensor.Tensor{moeCastF32(out)}, nil, moeCastF32(g))
	if err != nil {
		t.Fatal(err)
	}
	moeAssertClose(t, "nrm2.gin", o32[0], o64[0])
}

func TestExtremumVJPF32Parity(t *testing.T) {
	vjp := vjps[backend.OpMax]                // exercises extremumVJP's F32 branch
	x := moeF64(tensor.Shape{8, 8}, 5, false) // well-separated normals → distinct argmax preserved under f32
	y := rowReduceF64(x, false)               // true per-row max
	g := moeF64(tensor.Shape{8}, 6, false)    // upstream grad
	attrs := backend.ReduceAttrs{Axes: []int{1}}
	o64, err := vjp(nil, []*tensor.Tensor{x}, []*tensor.Tensor{y}, attrs, g)
	if err != nil {
		t.Fatal(err)
	}
	o32, err := vjp(nil, []*tensor.Tensor{moeCastF32(x)}, []*tensor.Tensor{moeCastF32(y)}, attrs, moeCastF32(g))
	if err != nil {
		t.Fatal(err)
	}
	moeAssertClose(t, "max.gin", o32[0], o64[0])
}

func TestProdVJPF32Parity(t *testing.T) {
	vjp := vjps[backend.OpProd]              // exercises prodVJP's F32 branch (no-zeros case)
	x := moeF64(tensor.Shape{8, 4}, 7, true) // positive → products well-defined, case-0 branch
	y := rowReduceF64(x, true)               // true per-row product
	g := moeF64(tensor.Shape{8}, 8, false)
	attrs := backend.ReduceAttrs{Axes: []int{1}}
	o64, err := vjp(nil, []*tensor.Tensor{x}, []*tensor.Tensor{y}, attrs, g)
	if err != nil {
		t.Fatal(err)
	}
	o32, err := vjp(nil, []*tensor.Tensor{moeCastF32(x)}, []*tensor.Tensor{moeCastF32(y)}, attrs, moeCastF32(g))
	if err != nil {
		t.Fatal(err)
	}
	moeAssertClose(t, "prod.gin", o32[0], o64[0])
}
