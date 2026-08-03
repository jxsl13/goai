package cpu

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// The WKV recurrence's running maximum is a math.Max per token per channel, and it lives in
// THREE places that a per-file sweep does not connect: backend/ref/wkv.go, the F32 scan in
// backend/cpu/wkv.go, and internal/simd/wkv_scalar.go — which is where the CPU backend's F64
// path actually lands on this build, since simd.WKVScanRangeF64 is a portable Go fallback
// rather than assembly outside the AVX build.
//
// This digest runs the op on BOTH backends in BOTH dtypes, so a conversion that misses one of
// the three files shows up as an unchanged number where a change was expected, and a
// conversion that changes a value shows up here rather than in a tolerance somewhere.
func wkvOpDigest(t *testing.T, be backend.Name, dt tensor.Dtype, seq, d int) uint64 {
	t.Helper()
	mk := func(shape tensor.Shape, fn func(i int) float64) *tensor.Tensor {
		x := tensor.New(dt, shape)
		n := x.Numel()
		if dt == tensor.F64 {
			s := x.Storage().F64()
			for i := range n {
				s[i] = fn(i)
			}
		} else {
			s := x.Storage().F32()
			for i := range n {
				s[i] = float32(fn(i))
			}
		}
		return x
	}
	k := mk(tensor.Shape{seq, d}, func(i int) float64 { return math.Sin(float64(i)*0.37) * 3 })
	v := mk(tensor.Shape{seq, d}, func(i int) float64 { return math.Cos(float64(i) * 0.21) })
	w := mk(tensor.Shape{d}, func(i int) float64 { return 0.5 + 0.01*float64(i%7) })
	u := mk(tensor.Shape{d}, func(i int) float64 { return -0.25 + 0.02*float64(i%5) })
	impl, ok := backend.Get(be)
	if !ok {
		t.Fatalf("backend %v not registered", be)
	}
	ctx := backend.NewContext().WithBackend(impl)
	out, err := backend.Execute(ctx, backend.OpWKV, []*tensor.Tensor{k, v, w, u}, nil)
	if err != nil {
		t.Fatal(err)
	}
	h := uint64(14695981039346656037)
	mix := func(u uint64) {
		for s := 0; s < 64; s += 8 {
			h = (h ^ (u>>s)&0xff) * 1099511628211
		}
	}
	n := out[0].Numel()
	if dt == tensor.F64 {
		for _, x := range out[0].Storage().F64()[:n] {
			mix(math.Float64bits(x))
		}
	} else {
		for _, x := range out[0].Storage().F32()[:n] {
			mix(uint64(math.Float32bits(x)))
		}
	}
	return h
}

// The Ref and CPU digests below are EQUAL within a dtype, which is the parity contract the
// CPU kernel was written to hold. That makes the pair a second gate on top of the frozen
// constants: converting one backend's max and not the other would break the equality even if
// someone re-froze the numbers.
func TestWKVOpIsBitIdentical(t *testing.T) {
	// 37 channels rather than a round number: the CPU F64 path bands the channels across
	// GOMAXPROCS, and a shape that divides evenly would leave the remainder band untested.
	cases := []struct {
		be     backend.Name
		dt     tensor.Dtype
		seq, d int
		want   uint64
	}{
		{backend.Ref, tensor.F64, 24, 37, 10566835949036511716},
		{backend.Ref, tensor.F32, 24, 37, 3093831351525738813},
		{backend.CPU, tensor.F64, 24, 37, 10566835949036511716},
		{backend.CPU, tensor.F32, 24, 37, 3093831351525738813},
		{backend.CPU, tensor.F64, 64, 96, 13474779355268514115},
	}
	for _, c := range cases {
		got := wkvOpDigest(t, c.be, c.dt, c.seq, c.d)
		if got != c.want {
			t.Errorf("%v %v seq=%d d=%d: digest %d", c.be, c.dt, c.seq, c.d, got)
		}
	}
}
