package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// Tolerance-0 digest of the chunkwise output over the benchmark's own inputs.
//
// These loops were rewritten twice independently — once by interchange (accumulate i-outer
// into a dv-length buffer, then scale, then add the V term in a third pass) and once by
// 4-way blocking over the output channel with register accumulators. Both claimed
// bit-identity with the original, which means they had to agree with EACH OTHER; this
// digest is what established that rather than trusting two comments. It held, and the
// blocked form measured 1.62x (chunkwise) and 1.68x (recurrent) faster, so that is the one
// kept. The pinned values guard any future rewrite of the same loops.
func TestRetentionChunkwiseArmDigest(t *testing.T) {
	const L, dk, dv = 512, 64, 64
	mk := func(fn func(i int) float64, c int) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape{L, c})
		s := x.Storage().F64()
		for i := range s {
			s[i] = fn(i)
		}
		return x
	}
	q := mk(func(i int) float64 { return math.Sin(float64(i) * 0.01) }, dk)
	k := mk(func(i int) float64 { return math.Cos(float64(i) * 0.013) }, dk)
	v := mk(func(i int) float64 { return math.Sin(float64(i) * 0.017) }, dv)
	out, err := nn.RetentionChunkwise(q, k, v, 0.968, 64)
	if err != nil {
		t.Fatal(err)
	}
	var xr uint64
	var sum float64
	for _, f := range out.Storage().F64() {
		sum += f
		xr ^= math.Float64bits(f) // order-independent: moves if any single bit moves
	}
	const wantSum, wantXor uint64 = 0x40b2bae2cf78b812, 0xbf5a8e6dc0e400bb
	if got := math.Float64bits(sum); got != wantSum {
		t.Errorf("sum digest = %016x, want %016x", got, wantSum)
	}
	if xr != wantXor {
		t.Errorf("xor digest = %016x, want %016x", xr, wantXor)
	}
}
