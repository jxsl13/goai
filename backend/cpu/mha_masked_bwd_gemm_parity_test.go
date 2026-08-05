package cpu_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	cpucpu "github.com/jxsl13/goai/backend/cpu"
	"github.com/jxsl13/goai/tensor"
)

func TestMHAMaskedBwdGemmCPUvsRef(t *testing.T) {
	if !cpucpu.F32NativeKernelsEnabled() {
		t.Skip("cpu masked-bwd gemm path only active on the perf build")
	}
	rng := rand.New(rand.NewSource(3))
	rnd := func(sh tensor.Shape) *tensor.Tensor {
		x := tensor.New(tensor.F32, sh)
		s := x.Storage().F32()
		for i := range s {
			s[i] = float32(rng.NormFloat64())
		}
		return x
	}
	type cfg struct {
		sq, sk, dk, heads, kvh int
		perHead, causal        bool
	}
	for _, c := range []cfg{
		{64, 64, 16, 4, 4, false, false},
		{64, 64, 16, 4, 4, false, true},
		{64, 64, 16, 4, 4, true, true},
		{128, 128, 32, 8, 8, true, false}, // per-head, larger
	} {
		dm := c.dk * c.heads
		q, k, v, dO := rnd(tensor.Shape{c.sq, dm}), rnd(tensor.Shape{c.sk, c.kvh * c.dk}), rnd(tensor.Shape{c.sk, c.kvh * c.dk}), rnd(tensor.Shape{c.sq, dm})
		var mask *tensor.Tensor
		if c.perHead {
			mask = tensor.New(tensor.F32, tensor.Shape{c.heads, c.sq, c.sk})
		} else {
			mask = tensor.New(tensor.F32, tensor.Shape{c.sq, c.sk})
		}
		ms := mask.Storage().F32()
		// fill mask: additive bias + causal -Inf upper triangle if causal
		idx := 0
		fill := func(i, j int) {
			if c.causal && j > i {
				ms[idx] = float32(math.Inf(-1))
			} else {
				ms[idx] = float32(rng.NormFloat64()) * 0.3
			}
			idx++
		}
		if c.perHead {
			for h := 0; h < c.heads; h++ {
				for i := 0; i < c.sq; i++ {
					for j := 0; j < c.sk; j++ {
						fill(i, j)
					}
				}
			}
		} else {
			for i := 0; i < c.sq; i++ {
				for j := 0; j < c.sk; j++ {
					fill(i, j)
				}
			}
		}
		attr := backend.AttnAttrs{Heads: c.heads, KVHeads: c.kvh}
		in := []*tensor.Tensor{q, k, v, mask, dO}
		gc, err := backend.Execute(backend.NewContext(), backend.OpMHAMaskedBackward, in, attr)
		if err != nil {
			t.Fatalf("cfg %+v cpu: %v", c, err)
		}
		gr, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()), backend.OpMHAMaskedBackward, in, attr)
		if err != nil {
			t.Fatalf("cfg %+v ref: %v", c, err)
		}
		names := []string{"dQ", "dK", "dV", "dMask"}
		for o := range gc {
			cs, rs := gc[o].Storage().F32(), gr[o].Storage().F32()
			var maxRel float64
			for i := range cs {
				d := math.Abs(float64(cs[i] - rs[i]))
				rel := d / math.Max(1, math.Abs(float64(rs[i])))
				if rel > maxRel {
					maxRel = rel
				}
			}
			if maxRel > 5e-5 {
				t.Errorf("cfg %+v %s: maxRel %.2e > 5e-5", c, names[o], maxRel)
			}
		}
	}
}
