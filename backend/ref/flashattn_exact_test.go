package ref

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

type flashAttnExactCase struct {
	name        string
	dtype       tensor.Dtype
	seq, heads  int
	kvHeads, dk int
	block       int
	causal      bool
}

// TestFlashAttnMatchesBlockedOnlineOracle locks the reference kernel to its
// defining blocked online-softmax schedule. A naive attention oracle is not
// suitable here: it computes the same real-valued function with different
// floating-point max, rescale, and normalization boundaries.
func TestFlashAttnMatchesBlockedOnlineOracle(t *testing.T) {
	cases := []flashAttnExactCase{
		{name: "f64_causal_dk1_short_block", dtype: tensor.F64, seq: 3, heads: 1, kvHeads: 1, dk: 1, block: 8, causal: true},
		{name: "f64_noncausal_nondivisible", dtype: tensor.F64, seq: 7, heads: 2, kvHeads: 2, dk: 3, block: 3},
		{name: "f64_causal_gqa", dtype: tensor.F64, seq: 9, heads: 4, kvHeads: 2, dk: 5, block: 4, causal: true},
		{name: "f32_noncausal_mqa", dtype: tensor.F32, seq: 6, heads: 4, kvHeads: 1, dk: 4, block: 2},
		{name: "f32_causal_unit_blocks", dtype: tensor.F32, seq: 5, heads: 2, kvHeads: 2, dk: 2, block: 1, causal: true},
		{name: "f64_default_block", dtype: tensor.F64, seq: 4, heads: 2, kvHeads: 1, dk: 3, block: 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			q, k, v := flashAttnExactInputs(tc)
			attrs := backend.AttnAttrs{
				Heads: tc.heads, KVHeads: tc.kvHeads,
				Block: tc.block, Causal: tc.causal,
			}
			got, err := flashAttnKernel(backend.NewContext(), []*tensor.Tensor{q, k, v}, attrs)
			if err != nil {
				t.Fatal(err)
			}
			want := flashAttnBlockedOracle(q, k, v, attrs, flashAttnNoMutation)
			flashAttnRequireExact(t, tc.dtype, got[0], want)
		})
	}
}

// TestFlashAttnOracleDetectsScheduleDrift proves the exact comparison is not
// vacuous. Each mutation represents a distinct implementation defect: a QK
// index error, a changed running-max boundary, or a missing old-block rescale.
func TestFlashAttnOracleDetectsScheduleDrift(t *testing.T) {
	tc := flashAttnExactCase{
		name: "mutation_probe", dtype: tensor.F64,
		seq: 7, heads: 2, kvHeads: 1, dk: 3, block: 2,
	}
	q, k, v := flashAttnExactInputs(tc)
	attrs := backend.AttnAttrs{Heads: tc.heads, KVHeads: tc.kvHeads, Block: tc.block}
	want := flashAttnBlockedOracle(q, k, v, attrs, flashAttnNoMutation)
	for name, mutation := range map[string]flashAttnMutation{
		"qk_index":    flashAttnMutateQKIndex,
		"running_max": flashAttnMutateRunningMax,
		"rescale":     flashAttnDropRescale,
	} {
		t.Run(name, func(t *testing.T) {
			mutated := flashAttnBlockedOracle(q, k, v, attrs, mutation)
			if flashAttnTensorsExact(tc.dtype, mutated, want) {
				t.Fatal("mutation produced a bit-identical output; oracle probe is not discriminating")
			}
		})
	}
}

func flashAttnExactInputs(tc flashAttnExactCase) (q, k, v *tensor.Tensor) {
	q = tensor.New(tc.dtype, tensor.Shape{tc.seq, tc.heads * tc.dk})
	k = tensor.New(tc.dtype, tensor.Shape{tc.seq, tc.kvHeads * tc.dk})
	v = tensor.New(tc.dtype, tensor.Shape{tc.seq, tc.kvHeads * tc.dk})
	for i := range tc.seq {
		for d := range tc.heads * tc.dk {
			x := float64((i + 1) * (d + 3))
			q.SetF64(math.Sin(x)*0.75+math.Cos(x*0.37)*0.125, i, d)
		}
		for d := range tc.kvHeads * tc.dk {
			x := float64((i + 2) * (d + 1))
			k.SetF64(math.Cos(x*0.61)*0.625+float64(i)*0.0625-float64(d)*0.03125, i, d)
			v.SetF64(math.Sin(x*0.43)*1.25+math.Cos(x*0.17)*0.375+float64(i-d)*0.015625, i, d)
		}
	}
	return q, k, v
}

type flashAttnMutation uint8

const (
	flashAttnNoMutation flashAttnMutation = iota
	flashAttnMutateQKIndex
	flashAttnMutateRunningMax
	flashAttnDropRescale
)

// flashAttnBlockedOracle deliberately mirrors the scalar kernel schedule:
// key blocks, ascending QK reductions, one running-max/rescale update per
// block, ascending P*V reductions, and one final normalization.
func flashAttnBlockedOracle(q, k, v *tensor.Tensor, attrs backend.AttnAttrs, mutation flashAttnMutation) *tensor.Tensor {
	pa := attrs.WithDefaults()
	seq, dm := q.Shape()[0], q.Shape()[1]
	dk := dm / pa.Heads
	rep := pa.Heads / pa.KVHeads
	block := pa.Block
	if block <= 0 {
		block = seq
	}
	scale := 1 / math.Sqrt(float64(dk))
	out := tensor.New(q.Dtype(), q.Shape())
	acc := make([]float64, dk)
	p := make([]float64, block)
	for h := range pa.Heads {
		qOff := h * dk
		kvOff := (h / rep) * dk
		for i := range seq {
			jmax := seq
			if pa.Causal {
				jmax = i + 1
			}
			m := math.Inf(-1)
			l := 0.0
			clear(acc)
			for j0 := 0; j0 < jmax; j0 += block {
				j1 := min(j0+block, jmax)
				mBlk := math.Inf(-1)
				for j := j0; j < j1; j++ {
					var s float64
					for d := range dk {
						kd := d
						if mutation == flashAttnMutateQKIndex && dk > 1 {
							kd = (d + 1) % dk
						}
						s += q.AtF64(i, qOff+d) * k.AtF64(j, kvOff+kd)
					}
					s *= scale
					p[j-j0] = s
					if s > mBlk {
						mBlk = s
					}
				}
				mNew := m
				if mBlk > mNew {
					mNew = mBlk
				}
				if mutation == flashAttnMutateRunningMax && j0 > 0 {
					mNew += 0.25
				}
				corr := math.Exp(m - mNew)
				if mutation == flashAttnDropRescale && j0 > 0 {
					corr = 1
				}
				var pSum float64
				for j := j0; j < j1; j++ {
					e := math.Exp(p[j-j0] - mNew)
					p[j-j0] = e
					pSum += e
				}
				l = corr*l + pSum
				for d := range dk {
					var pv float64
					for j := j0; j < j1; j++ {
						pv += p[j-j0] * v.AtF64(j, kvOff+d)
					}
					acc[d] = corr*acc[d] + pv
				}
				m = mNew
			}
			for d := range dk {
				out.SetF64(acc[d]/l, i, qOff+d)
			}
		}
	}
	return out
}

func flashAttnRequireExact(t *testing.T, dtype tensor.Dtype, got, want *tensor.Tensor) {
	t.Helper()
	for i := range got.Shape()[0] {
		for d := range got.Shape()[1] {
			g, w := got.AtF64(i, d), want.AtF64(i, d)
			if !flashAttnValueExact(dtype, g, w) {
				t.Fatalf("[%d,%d]: got %.17g, want %.17g; blocked schedules differ", i, d, g, w)
			}
		}
	}
}

func flashAttnTensorsExact(dtype tensor.Dtype, a, b *tensor.Tensor) bool {
	for i := range a.Shape()[0] {
		for d := range a.Shape()[1] {
			if !flashAttnValueExact(dtype, a.AtF64(i, d), b.AtF64(i, d)) {
				return false
			}
		}
	}
	return true
}

func flashAttnValueExact(dtype tensor.Dtype, a, b float64) bool {
	if dtype == tensor.F32 {
		return math.Float32bits(float32(a)) == math.Float32bits(float32(b))
	}
	return math.Float64bits(a) == math.Float64bits(b)
}
