package cpu_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	cpucpu "github.com/jxsl13/goai/backend/cpu"
	"github.com/jxsl13/goai/tensor"
)

// TestMHAMaskedF32CPUByteIdenticalToRef locks the fresh F32 CPU masked-attention fast path
// byte-for-byte to backend/ref's F32 devirtualised scan (f64Data widens F32→F64, computes
// scores/softmax/·V in F64, narrows only on store). The CPU kernel widens each F32 read per
// element in the same ascending-j / j-outer-d-inner order and fans the independent
// (head,query-row) pairs across workers, so it must be BYTE-IDENTICAL, not merely within f32
// tolerance. Includes a causal −Inf mask to exercise the masked-out branch and GQA
// (kvHeads < heads) + per-head masks.
func TestMHAMaskedF32CPUByteIdenticalToRef(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	f := func(shape tensor.Shape) *tensor.Tensor {
		tn := tensor.New(tensor.F32, shape)
		s := tn.Storage().F32()
		for i := range s {
			s[i] = float32(rng.NormFloat64())
		}
		return tn
	}
	causal := func(heads, sq, sk int, perHead bool) *tensor.Tensor {
		var sh tensor.Shape
		if perHead {
			sh = tensor.Shape{heads, sq, sk}
		} else {
			sh = tensor.Shape{sq, sk}
		}
		mk := tensor.New(tensor.F32, sh)
		flat := mk.Storage().F32()
		set := func(i, j int, base int) {
			if j > i { // future key → masked
				flat[base+i*sk+j] = float32(math.Inf(-1))
			}
		}
		if perHead {
			for h := 0; h < heads; h++ {
				for i := 0; i < sq; i++ {
					for j := 0; j < sk; j++ {
						set(i, j, h*sq*sk)
					}
				}
			}
		} else {
			for i := 0; i < sq; i++ {
				for j := 0; j < sk; j++ {
					set(i, j, 0)
				}
			}
		}
		return mk
	}
	for _, cfg := range []struct {
		sq, sk, dk, heads, kvh int
		perHead, causalMask    bool
	}{
		{4, 4, 8, 2, 2, false, false}, {5, 7, 16, 4, 2, false, false},
		{5, 7, 16, 4, 4, true, false}, {64, 96, 32, 8, 8, false, false},
		// THE JAM TAKES EIGHT KEYS AT A TIME AND THESE TWO LENGTHS ARE WHAT REACH ITS EDGES.
		// sk=15 puts j+7 exactly at the last key, which is where an off-by-one in the bound
		// reads past the row; sk=11 leaves a partial group AFTER a full one. The other lengths
		// are 4 and 7 (below the group, remainder only) and 96 (a whole number of groups), so
		// neither edge was reachable.
		{5, 15, 16, 4, 2, false, false}, {5, 11, 16, 4, 4, false, false},
		{48, 48, 32, 8, 8, false, true}, {48, 48, 32, 8, 2, true, true},
	} {
		dm := cfg.dk * cfg.heads
		q, k, v := f(tensor.Shape{cfg.sq, dm}), f(tensor.Shape{cfg.sk, cfg.kvh * cfg.dk}), f(tensor.Shape{cfg.sk, cfg.kvh * cfg.dk})
		var mask *tensor.Tensor
		if cfg.causalMask {
			mask = causal(cfg.heads, cfg.sq, cfg.sk, cfg.perHead)
		} else if cfg.perHead {
			mask = f(tensor.Shape{cfg.heads, cfg.sq, cfg.sk})
		} else {
			mask = f(tensor.Shape{cfg.sq, cfg.sk})
		}
		in := []*tensor.Tensor{q, k, v, mask}
		attr := backend.AttnAttrs{Heads: cfg.heads, KVHeads: cfg.kvh}
		gc, err := backend.Execute(backend.NewContext(), backend.OpMHAMasked, in, attr)
		if err != nil {
			t.Fatal(err)
		}
		gr, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()), backend.OpMHAMasked, in, attr)
		if err != nil {
			t.Fatal(err)
		}
		cs, rs := gc[0].Storage().F32(), gr[0].Storage().F32()
		for i := range cs {
			if cpucpu.F32NativeKernelsEnabled() {
				// Perf build routes masked attention through the f32-native SIMD gemm + vexp softmax
				// (mhaMaskedFwdGemmF32), which accumulates in f32 and uses the ~1ulp poly exp — the same
				// ADR-0021 5e-5 tolerant parity the unmasked MHA gemm path carries, not byte-exact vs
				// the f64-accumulating ref.
				if d := math.Abs(float64(cs[i] - rs[i])); d > 5e-5*math.Max(1, math.Abs(float64(rs[i]))) {
					t.Fatalf("cfg=%+v idx=%d cpu=%v ref=%v (rel > 5e-5)", cfg, i, cs[i], rs[i])
				}
				continue
			}
			if math.Float32bits(cs[i]) != math.Float32bits(rs[i]) {
				t.Fatalf("cfg=%+v idx=%d cpu=%v ref=%v", cfg, i, cs[i], rs[i])
			}
		}
	}
}
