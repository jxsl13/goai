package cpu

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// benchWKV measures OpWKV (RWKV-4 time-mixing recurrence) at a realistic
// [seq, d] shape. OpWKV currently has no CPU kernel — it falls through to the
// scalar ref backend, which runs 4 math.Exp per (channel, time) in a per-channel
// scan. This benchmark is the pre-baseline for a channel-vectorized CPU kernel
// (the channels are independent → vectorizable over d with expF64x4): profile it
// with -cpuprofile to see wkvKernel + math.archExp dominate. The eventual kernel
// must stay bit-exact with RWKV's decode step (TestRWKVDecodeMatchesForward), so
// forward and decode both need the same channel-vectorized computation.
func benchWKV(b *testing.B, seq, d int) {
	k := bench.RandF64(tensor.Shape{seq, d}, 1)
	v := bench.RandF64(tensor.Shape{seq, d}, 2)
	w := bench.RandF64(tensor.Shape{d}, 3)
	u := bench.RandF64(tensor.Shape{d}, 4)
	in := []*tensor.Tensor{k, v, w, u}
	ctx := backend.NewContext()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpWKV, in, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkWKV_512x1024(b *testing.B)  { benchWKV(b, 512, 1024) }
func BenchmarkWKV_1024x2048(b *testing.B) { benchWKV(b, 1024, 2048) }
