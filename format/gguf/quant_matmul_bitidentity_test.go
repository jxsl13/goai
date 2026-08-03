package gguf

import (
	"encoding/base64"
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// TestQMatMulIsBitIdentical freezes every bit of the quantized matmul's output. The existing
// parity test compares against a Python golden at 1e-5 relative, which is the right gate for the
// DEQUANT but cannot witness bit identity: unrolling the activation-row loop claims to change no
// value — each output keeps its own accumulator summing ascending ki — and a reassociation would
// pass a 1e-5 check with room to spare.
//
// The row counts straddle the jam: 1 and 3 run entirely in the tail, 16 is a whole number of
// groups, and 11 and 19 leave remainders of 3 after one and two full groups.
func TestQMatMulIsBitIdentical(t *testing.T) {
	raw, err := os.ReadFile("testdata/qmatmul.json")
	if err != nil {
		t.Fatalf("read golden (run make golden): %v", err)
	}
	var g qmatGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	wb, err := base64.StdEncoding.DecodeString(g.Q8)
	if err != nil {
		t.Fatal(err)
	}
	// WHAT THIS DIGEST CAN AND CANNOT SEE. Both typed paths narrow every output to float32 on
	// store — the f64 activation path included — so a perturbation below about 6e-8 relative
	// vanishes in the narrowing. Verified rather than assumed: injecting 1e-10 into an
	// accumulator moved NO digest on either path, while 1.5x reddened both. A ulp-scale
	// reassociation is therefore NOT observable through this API, and what this test does gate
	// is the failure modes a jam actually has — a bound that reads past the last row, a dropped
	// tail row, and a row wired to the wrong accumulator, each of which moves a whole value.
	//
	// BOTH TYPED PATHS ARE STILL RUN, because they are separate code. The f32 path narrows every output to
	// float32 on store, so a perturbation smaller than about 6e-8 relative vanishes in the
	// narrowing — verified: injecting 1e-10 into an f32 accumulator moved NO digest, while 1.5x
	// did. A reassociation is a ulp-scale change, so only the f64 path, whose outputs are stored
	// as float64, can witness the claim this jam actually makes.
	for _, c := range []struct {
		m    int
		f64  bool
		want uint64
	}{
		{1, false, 13012536463104661477}, {3, false, 2112980319355452432},
		{11, false, 6908458912348333419}, {16, false, 11374628099757688998},
		{19, false, 7573144450394698791},
		// FIFTEEN IS THE ONE THAT PUTS mi+7 EXACTLY AT THE LAST ROW, which is where an
		// off-by-one in the jam's bound reads past the activation matrix. With only the other
		// five lengths that mutation stayed green.
		{15, false, 10400726524783390276},
		{1, true, 4818529750319929701}, {3, true, 1358956243421026043}, {11, true, 6211164677040096810},
		{16, true, 949920475172802122}, {19, true, 4053913506136142786}, {15, true, 8332907117224403576},
	} {
		// F32 ACTIVATIONS, BECAUSE THAT IS THE ARM THE JAM LIVES IN. Built with FromFloat64 the
		// activation is F64 and QMatMul takes its other typed path: a perturbation injected into
		// the unrolled f32 block then moved NOTHING here, which is how this test was caught
		// being vacuous for the change it was written to gate.
		dt := tensor.F32
		if c.f64 {
			dt = tensor.F64
		}
		x := tensor.New(dt, tensor.Shape{c.m, g.K})
		for i := range c.m * g.K {
			x.SetF64(math.Sin(float64(i*13+7))*0.5, i/g.K, i%g.K)
		}
		y, err := QMatMul(x, wb, Q8_0, g.N, g.K)
		if err != nil {
			t.Fatal(err)
		}
		h := uint64(14695981039346656037)
		for i := range y.Numel() {
			u := math.Float64bits(y.AtF64(tensor.Unravel(i, y.Shape())...))
			for s := 0; s < 64; s += 8 {
				h = (h ^ (u>>s)&0xff) * 1099511628211
			}
		}
		if h != c.want {
			t.Fatalf("m=%d f64=%v digest %d, want %d", c.m, c.f64, h, c.want)
		}
	}
}
