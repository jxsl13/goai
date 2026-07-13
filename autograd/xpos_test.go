package autograd_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

func xpos(t *testing.T, q *tensor.Tensor, downscale bool) *tensor.Tensor {
	t.Helper()
	out, err := backend.Execute(backend.NewContext(), backend.OpRoPE,
		[]*tensor.Tensor{q}, backend.RoPEAttrs{XPos: true, XPosDownscale: downscale})
	if err != nil {
		t.Fatal(err)
	}
	return out[0]
}

func zeta(i, hd int, gamma float64) float64 {
	half := hd / 2
	return (float64(i)/float64(half) + gamma) / (1 + gamma)
}

// §V16 tier-1: XPos off is byte-for-byte plain RoPE (ξ=0 ⇒ pure RoPE, §R125).
func TestXPosOffIsRoPE(t *testing.T) {
	q := tensor.FromFloat64(tensor.Shape{4, 4}, []float64{
		1, 2, 3, 4, 0.5, -1, 2, 0.3, -2, 1, 0.7, -0.4, 1.1, 0.2, -1.5, 2,
	})
	plain, _ := backend.Execute(backend.NewContext(), backend.OpRoPE, []*tensor.Tensor{q}, nil)
	off, _ := backend.Execute(backend.NewContext(), backend.OpRoPE, []*tensor.Tensor{q}, backend.RoPEAttrs{XPos: false})
	for i := range q.Numel() {
		idx := tensor.Unravel(i, q.Shape())
		if plain[0].AtF64(idx...) != off[0].AtF64(idx...) {
			t.Errorf("XPos:false must equal plain RoPE at %v", idx)
		}
	}
}

// §V16 tier-1: xPos output = ζ_i^(±n)·(RoPE rotation). Dividing the xPos output by
// the plain-RoPE output recovers exactly ζ_i^(+n) (query) / ζ_i^(−n) (key), the
// same factor on both elements of pair i (§R125).
func TestXPosMagnitude(t *testing.T) {
	const seq, hd = 5, 6
	qd := make([]float64, seq*hd)
	for i := range qd {
		qd[i] = float64(i%7) - 3 + 0.5
	}
	q := tensor.FromFloat64(tensor.Shape{seq, hd}, qd)
	rope, _ := backend.Execute(backend.NewContext(), backend.OpRoPE, []*tensor.Tensor{q}, nil)
	half := hd / 2
	for _, down := range []bool{false, true} {
		xp := xpos(t, q, down)
		for n := range seq {
			for i := range half {
				e := float64(n)
				if down {
					e = -float64(n)
				}
				want := math.Pow(zeta(i, hd, 0.4), e)
				for _, col := range []int{i, i + half} {
					r := rope[0].AtF64(n, col)
					if math.Abs(r) < 1e-9 {
						continue // ratio ill-defined where the rotation is ~0
					}
					got := xp.AtF64(n, col) / r
					if math.Abs(got-want) > 1e-9*math.Max(1, want) {
						t.Errorf("down=%v n=%d pair=%d col=%d: scale %g, want ζ^e=%g", down, n, i, col, got, want)
					}
				}
			}
		}
	}
}

// §V16 tier-1: THE defining property — with a query scaled ζ^n and a key scaled
// ζ^(−m), the attention score picks up an exponential ζ^(n−m) decay in the
// relative distance (§R125). Single rotation pair (hd=2) isolates a single ζ so
// the whole score scales by one factor: score_xpos(n,m) = ζ^(n−m)·score_rope(n,m).
func TestXPosScoreDecay(t *testing.T) {
	const seq, hd = 6, 2
	qd := []float64{0.7, -0.4, 1.2, 0.9, -1.1, 0.5, 0.3, -0.8, 1.5, 0.2, -0.6, 1.0}
	kd := []float64{0.2, 1.1, -0.5, 0.8, 0.9, -1.2, 1.3, 0.4, -0.7, 0.6, 0.5, -0.9}
	q := tensor.FromFloat64(tensor.Shape{seq, hd}, qd)
	k := tensor.FromFloat64(tensor.Shape{seq, hd}, kd)

	ropeQ, _ := backend.Execute(backend.NewContext(), backend.OpRoPE, []*tensor.Tensor{q}, nil)
	ropeK, _ := backend.Execute(backend.NewContext(), backend.OpRoPE, []*tensor.Tensor{k}, nil)
	xq := xpos(t, q, false) // query: ζ^(+n)
	xk := xpos(t, k, true)  // key:   ζ^(−m)
	z := zeta(0, hd, 0.4)

	dot := func(a, b *tensor.Tensor, n, m int) float64 {
		return a.AtF64(n, 0)*b.AtF64(m, 0) + a.AtF64(n, 1)*b.AtF64(m, 1)
	}
	for n := range seq {
		for m := range seq {
			sr := dot(ropeQ[0], ropeK[0], n, m)
			sx := dot(xq, xk, n, m)
			want := math.Pow(z, float64(n-m)) * sr
			if math.Abs(sx-want) > 1e-9*math.Max(1, math.Abs(want)) {
				t.Errorf("n=%d m=%d: xpos score %g, want ζ^(n−m)·rope %g", n, m, sx, want)
			}
		}
	}
}

// §V2: finite differences of ½·Σ(xPos output)² w.r.t. q match the analytic VJP
// (the xPos magnitude scaling flows through the gradient).
func TestXPosGradcheck(t *testing.T) {
	const seq, hd = 4, 4
	qd := []float64{0.5, -1.5, 2.0, 1.0, -0.5, 0.25, 0.8, -0.3, 1.1, -0.7, 0.6, -1.2, 0.4, 0.9, -0.2, 1.3}
	attrs := backend.RoPEAttrs{XPos: true, XPosDownscale: true}
	lossAt := func(d []float64) float64 {
		out, _ := backend.Execute(backend.NewContext(), backend.OpRoPE,
			[]*tensor.Tensor{tensor.FromFloat64(tensor.Shape{seq, hd}, d)}, attrs)
		var s float64
		for i := range out[0].Numel() {
			v := out[0].AtF64(tensor.Unravel(i, out[0].Shape())...)
			s += v * v
		}
		return s / 2
	}
	data := append([]float64(nil), qd...)
	qT := tensor.FromFloat64(tensor.Shape{seq, hd}, data)
	tape := autograd.NewTape()
	out, err := backend.Execute(tape.Context(), backend.OpRoPE, []*tensor.Tensor{qT}, attrs)
	if err != nil {
		t.Fatal(err)
	}
	sq, _ := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{out[0], out[0]}, nil)
	loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{sq[0]}, nil)
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(qT)
	const h = 1e-6
	for i := range data {
		o := data[i]
		data[i] = o + h
		lp := lossAt(data)
		data[i] = o - h
		lm := lossAt(data)
		data[i] = o
		num := (lp - lm) / (2 * h)
		ana := 0.5 * g.AtF64(tensor.Unravel(i, qT.Shape())...) // loss uses ½Σ; grad is of Σ
		if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
			t.Errorf("grad[%d]: numeric %.8g vs analytic %.8g", i, num, ana)
		}
	}
}

// §V16 tier-1: at position 0 the xPos scale ζ_i^0 = 1, so row 0 is exactly plain
// RoPE (identity of the magnitude scaling at the origin).
func TestXPosPosZeroIsRoPE(t *testing.T) {
	q := tensor.FromFloat64(tensor.Shape{3, 4}, []float64{1, 2, 3, 4, 0.5, -1, 2, 0.3, -2, 1, 0.7, -0.4})
	rope, _ := backend.Execute(backend.NewContext(), backend.OpRoPE, []*tensor.Tensor{q}, nil)
	xp := xpos(t, q, false)
	for col := range 4 {
		if math.Abs(xp.AtF64(0, col)-rope[0].AtF64(0, col)) > 1e-12 {
			t.Errorf("row 0 col %d: xpos %g != rope %g", col, xp.AtF64(0, col), rope[0].AtF64(0, col))
		}
	}
}
