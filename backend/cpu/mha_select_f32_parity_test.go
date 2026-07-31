package cpu_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/cpu"
	"github.com/jxsl13/goai/tensor"
)

// TestMHASelectF32CPUByteIdenticalToRef locks the fresh F32 CPU selective-attention fast
// path byte-for-byte to backend/ref's F32 devirtualised scan (f64Data widens F32→F64,
// F64 score/softmax/·V, narrow only on store). The cpu path widens each F32 read per element
// in the same ascending-j order and fans the independent (head,query-row) pairs across
// workers → BYTE-IDENTICAL. Exercises both selector branches (sv==0 → source1, sv!=0 →
// source2), a −Inf sel entry (masked-out key), and GQA (kvHeads < heads).
func TestMHASelectF32CPUByteIdenticalToRef(t *testing.T) {
	rng := rand.New(rand.NewSource(21))
	f := func(shape tensor.Shape) *tensor.Tensor {
		tn := tensor.New(tensor.F32, shape)
		s := tn.Storage().F32()
		for i := range s {
			s[i] = float32(rng.NormFloat64())
		}
		return tn
	}
	for _, cfg := range []struct {
		sq, sk, dk, heads, kvh int
	}{{4, 4, 8, 2, 2}, {5, 7, 16, 4, 2}, {64, 96, 32, 8, 8}, {128, 128, 32, 8, 4}} {
		dm := cfg.dk * cfg.heads
		kd := cfg.kvh * cfg.dk
		selT := tensor.New(tensor.F32, tensor.Shape{cfg.sq, cfg.sk})
		s := selT.Storage().F32()
		for i := range s {
			r := rng.Float64()
			switch {
			case r < 0.45:
				s[i] = 1 // source2
			case r > 0.97:
				s[i] = float32(math.Inf(-1)) // masked-out
			default:
				s[i] = 0 // source1
			}
		}
		in := []*tensor.Tensor{f(tensor.Shape{cfg.sq, dm}), f(tensor.Shape{cfg.sk, kd}), f(tensor.Shape{cfg.sq, dm}), f(tensor.Shape{cfg.sk, kd}), f(tensor.Shape{cfg.sk, kd}), selT}
		attr := backend.AttnAttrs{Heads: cfg.heads, KVHeads: cfg.kvh}
		gc, err := backend.Execute(backend.NewContext(), backend.OpMHASelect, in, attr)
		if err != nil {
			t.Fatal(err)
		}
		gr, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()), backend.OpMHASelect, in, attr)
		if err != nil {
			t.Fatal(err)
		}
		cs, rs := gc[0].Storage().F32(), gr[0].Storage().F32()
		for i := range cs {
			if math.Float32bits(cs[i]) != math.Float32bits(rs[i]) {
				t.Fatalf("cfg=%+v idx=%d cpu=%v ref=%v", cfg, i, cs[i], rs[i])
			}
		}
	}
}
