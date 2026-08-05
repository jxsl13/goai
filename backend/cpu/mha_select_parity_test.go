package cpu_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	cpucpu "github.com/jxsl13/goai/backend/cpu"
	"github.com/jxsl13/goai/tensor"
)

func TestMHASelectCPUByteIdenticalToRef(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	// BOTH DTYPE ARMS. They were line-for-line duplicates and now share one body, which is
	// exactly the situation where testing one of them proves nothing about the other.
	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		f := func(shape tensor.Shape) *tensor.Tensor {
			tn := tensor.New(dt, shape)
			n := 1
			for _, d := range shape {
				n *= d
			}
			for i := range n {
				tn.SetF64(rng.NormFloat64(), tensor.Unravel(i, shape)...)
			}
			return tn
		}
		for _, cfg := range []struct{ sq, sk, dk, heads, kvh int }{{4, 4, 8, 2, 2}, {5, 7, 16, 4, 2}, {64, 96, 32, 8, 8}} {
			dm := cfg.dk * cfg.heads
			kd := cfg.kvh * cfg.dk
			selT := tensor.New(dt, tensor.Shape{cfg.sq, cfg.sk})
			// A MASKED KEY IS -inf, AND THE MASK MUST LAND UNEVENLY. Both selector values were 0 or
			// 1 here, so the whole masked path went untested — and a jam that takes four keys per
			// pass has to fall back for any group that straddles the mask, which is new code that
			// only a mixed row reaches. About a fifth masked puts masked and unmasked keys in the
			// same group of four at every one of these widths.
			for i := range cfg.sq * cfg.sk {
				v := 0.0
				switch r := rng.Float64(); {
				case r < 0.35:
					v = 1
				case r < 0.55:
					v = math.Inf(-1)
				}
				selT.SetF64(v, i/cfg.sk, i%cfg.sk)
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
			for i := 0; i < gc[0].Numel(); i++ {
				c := tensor.Unravel(i, gc[0].Shape())
				if dt == tensor.F32 && cpucpu.F32NativeKernelsEnabled() {
					// Perf build routes f32 selective attention through the gemm+vexp pipeline
					// (5e-5 tolerant parity), not byte-exact vs the f64-accumulating ref.
					if d := math.Abs(gc[0].AtF64(c...) - gr[0].AtF64(c...)); d > 5e-5*math.Max(1, math.Abs(gr[0].AtF64(c...))) {
						t.Fatalf("dt=%v cfg=%+v idx=%d cpu=%v ref=%v (rel > 5e-5)",
							dt, cfg, i, gc[0].AtF64(c...), gr[0].AtF64(c...))
					}
					continue
				}
				if math.Float64bits(gc[0].AtF64(c...)) != math.Float64bits(gr[0].AtF64(c...)) {
					t.Fatalf("dt=%v cfg=%+v idx=%d cpu=%v ref=%v",
						dt, cfg, i, gc[0].AtF64(c...), gr[0].AtF64(c...))
				}
			}
		}
	}
}

func benchMHASelect(b *testing.B, sq, sk, dk, heads int, ref bool) {
	rng := rand.New(rand.NewSource(1))
	dm := dk * heads
	f := func(shape tensor.Shape) *tensor.Tensor {
		t := tensor.New(tensor.F64, shape)
		for i, s := 0, t.Storage().F64(); i < len(s); i++ {
			s[i] = rng.NormFloat64()
		}
		return t
	}
	selT := tensor.New(tensor.F64, tensor.Shape{sq, sk})
	for i, s := 0, selT.Storage().F64(); i < len(s); i++ {
		if rng.Float64() < 0.5 {
			s[i] = 1
		}
	}
	in := []*tensor.Tensor{f(tensor.Shape{sq, dm}), f(tensor.Shape{sk, dm}), f(tensor.Shape{sq, dm}), f(tensor.Shape{sk, dm}), f(tensor.Shape{sk, dm}), selT}
	attr := backend.AttnAttrs{Heads: heads}
	ctx := backend.NewContext()
	if ref {
		ctx = ctx.WithBackend(backend.Reference())
	}
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpMHASelect, in, attr); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMHASelect_cpu_512(b *testing.B) { benchMHASelect(b, 512, 512, 64, 8, false) }
func BenchmarkMHASelect_ref_512(b *testing.B) { benchMHASelect(b, 512, 512, 64, 8, true) }
