package autograd

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// distillRefF64 is the serial reference for the OpDistill VJP: ds[i,j] = scale·(q−p)
// with p=softmax(zt_i/T), q=softmax(zs_i/T) — exactly the pre-parallelization algorithm.
func distillRefF64(zs, zt []float64, b, c int, temp, scale float64) []float64 {
	ds := make([]float64, b*c)
	p := make([]float64, c)
	q := make([]float64, c)
	for i := 0; i < b; i++ {
		base := i * c
		softmaxRowTInto(p, zt[base:base+c], temp)
		softmaxRowTInto(q, zs[base:base+c], temp)
		for j := 0; j < c; j++ {
			ds[base+j] = scale * (q[j] - p[j])
		}
	}
	return ds
}

// TestDistillVJPParallelBitExact proves the row-parallel OpDistill VJP is BIT-IDENTICAL
// to the serial reference (disjoint row writes + privatized softmax scratch), for both
// F64 and F32. Run under -race, it also guards against a scratch data race across the
// worker goroutines. b·c here is well above the parallel threshold so the parallel path
// actually executes.
func TestDistillVJPParallelBitExact(t *testing.T) {
	const b, c = 128, 4096 // b*c = 512k ≫ 1<<15 → parallel path taken
	const temp = 2.0
	fn := vjps[backend.OpDistill]
	g := tensor.New(tensor.F64, tensor.Shape{})
	g.SetF64(1.0)
	scale := 1.0 * temp / float64(b)
	attrs := backend.DistillAttrs{Temperature: temp}
	ctx := backend.NewContext()

	// F64
	zsF := make([]float64, b*c)
	ztF := make([]float64, b*c)
	for i := range zsF {
		zsF[i] = math.Sin(float64(i*3) * 0.0007)
		ztF[i] = math.Sin(float64((i+500)*3) * 0.0007)
	}
	zs := tensor.New(tensor.F64, tensor.Shape{b, c})
	zt := tensor.New(tensor.F64, tensor.Shape{b, c})
	copy(zs.Storage().F64(), zsF)
	copy(zt.Storage().F64(), ztF)
	out, err := fn(ctx, []*tensor.Tensor{zs, zt}, nil, attrs, g)
	if err != nil {
		t.Fatal(err)
	}
	want := distillRefF64(zsF, ztF, b, c, temp, scale)
	got := out[0].Storage().F64()
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("F64 ds[%d]=%v want %v (not bit-identical to serial reference)", i, got[i], want[i])
		}
	}
	if out[1] != nil {
		t.Errorf("teacher gradient must be nil (frozen)")
	}

	// F32: reference widens the same way the kernel does, narrows once per element.
	zs32 := tensor.New(tensor.F32, tensor.Shape{b, c})
	zt32 := tensor.New(tensor.F32, tensor.Shape{b, c})
	s32, t32 := zs32.Storage().F32(), zt32.Storage().F32()
	for i := range s32 {
		s32[i] = float32(zsF[i])
		t32[i] = float32(ztF[i])
	}
	out32, err := fn(ctx, []*tensor.Tensor{zs32, zt32}, nil, attrs, g)
	if err != nil {
		t.Fatal(err)
	}
	// serial F32 reference
	wantF32 := make([]float32, b*c)
	p := make([]float64, c)
	q := make([]float64, c)
	row := make([]float64, c)
	for i := 0; i < b; i++ {
		base := i * c
		for j := 0; j < c; j++ {
			row[j] = float64(t32[base+j])
		}
		softmaxRowTInto(p, row, temp)
		for j := 0; j < c; j++ {
			row[j] = float64(s32[base+j])
		}
		softmaxRowTInto(q, row, temp)
		for j := 0; j < c; j++ {
			wantF32[base+j] = float32(scale * (q[j] - p[j]))
		}
	}
	gotF32 := out32[0].Storage().F32()
	for i := range wantF32 {
		if gotF32[i] != wantF32[i] {
			t.Fatalf("F32 ds[%d]=%v want %v (not bit-identical)", i, gotF32[i], wantF32[i])
		}
	}
}
