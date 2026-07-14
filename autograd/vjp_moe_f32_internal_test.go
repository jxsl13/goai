package autograd

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// The MoE VJP F32 fast paths (§T636) are NOT exercised by TestGradCheckAllOps (it
// runs in F64) nor the nn MoE suite (also F64). These tests close that gap: each F32
// fast path computes internally in float64 and only rounds the stored result to
// float32, so it must equal float32(the F64 path) within f32 rounding.

func moeF64(shape tensor.Shape, seed uint64, positive bool) *tensor.Tensor {
	t := tensor.New(tensor.F64, shape)
	rng := rand.New(rand.NewPCG(seed, 0x9e3779b9))
	s := t.Storage().F64()
	for i := range s {
		v := rng.NormFloat64()
		if positive {
			v = math.Abs(v) + 0.2
		}
		s[i] = v
	}
	return t
}

func moeCastF32(x *tensor.Tensor) *tensor.Tensor {
	o := tensor.New(tensor.F32, x.Shape())
	s, d := x.Contiguous().Storage().F64(), o.Storage().F32()
	for i := range s {
		d[i] = float32(s[i])
	}
	return o
}

func moeAssertClose(t *testing.T, name string, f32out, f64out *tensor.Tensor) {
	t.Helper()
	if f32out == nil || f64out == nil {
		if f32out != f64out {
			t.Fatalf("%s: one output nil, other not", name)
		}
		return
	}
	as := f32out.Contiguous().Storage().F32()
	bs := f64out.Contiguous().Storage().F64()
	if len(as) != len(bs) {
		t.Fatalf("%s: len %d vs %d", name, len(as), len(bs))
	}
	for i := range as {
		if d := math.Abs(float64(as[i]) - bs[i]); d > 1e-3*(1+math.Abs(bs[i]))+1e-5 {
			t.Fatalf("%s [%d]: f32 %g vs f64 %g (Δ=%g)", name, i, as[i], bs[i], d)
		}
	}
}

func TestMoEBalanceF32Parity(t *testing.T) {
	vjp := vjps[backend.OpMoEBalance]
	tks, n := 16, 4
	logits := moeF64(tensor.Shape{tks, n}, 1, false)
	assign := tensor.New(tensor.F64, tensor.Shape{tks})
	for i, s := 0, assign.Storage().F64(); i < tks; i++ {
		s[i] = float64(i % n)
	}
	g := tensor.New(tensor.F64, tensor.Shape{})
	g.Storage().F64()[0] = 0.7
	attrs := backend.MoEBalanceAttrs{}
	o64, err := vjp(nil, []*tensor.Tensor{logits, assign}, nil, attrs, g)
	if err != nil {
		t.Fatal(err)
	}
	o32, err := vjp(nil, []*tensor.Tensor{moeCastF32(logits), moeCastF32(assign)}, nil, attrs, moeCastF32(g))
	if err != nil {
		t.Fatal(err)
	}
	moeAssertClose(t, "moebalance.gl", o32[0], o64[0])
}

func TestMoECombineF32Parity(t *testing.T) {
	vjp := vjps[backend.OpMoECombine]
	tks, e, d := 8, 3, 16
	in64 := []*tensor.Tensor{moeF64(tensor.Shape{tks, e}, 2, true)} // positive weights → denom>0
	for i := 0; i < e; i++ {
		in64 = append(in64, moeF64(tensor.Shape{tks, d}, uint64(10+i), false))
	}
	g := moeF64(tensor.Shape{tks, d}, 99, false)
	o64, err := vjp(nil, in64, nil, nil, g)
	if err != nil {
		t.Fatal(err)
	}
	in32 := make([]*tensor.Tensor, len(in64))
	for i := range in64 {
		in32[i] = moeCastF32(in64[i])
	}
	o32, err := vjp(nil, in32, nil, nil, moeCastF32(g))
	if err != nil {
		t.Fatal(err)
	}
	for k := range o64 {
		moeAssertClose(t, fmt.Sprintf("moecombine.out%d", k), o32[k], o64[k])
	}
}

func TestZLossF32Parity(t *testing.T) {
	vjp := vjps[backend.OpZLoss]
	b, c := 8, 5
	z := moeF64(tensor.Shape{b, c}, 3, false)
	g := tensor.New(tensor.F64, tensor.Shape{})
	g.Storage().F64()[0] = 0.5
	o64, err := vjp(nil, []*tensor.Tensor{z}, nil, nil, g)
	if err != nil {
		t.Fatal(err)
	}
	o32, err := vjp(nil, []*tensor.Tensor{moeCastF32(z)}, nil, nil, moeCastF32(g))
	if err != nil {
		t.Fatal(err)
	}
	moeAssertClose(t, "zloss.gz", o32[0], o64[0])
}
