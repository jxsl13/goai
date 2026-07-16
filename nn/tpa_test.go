package nn_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// tpaReconRef reconstructs Q/K/V from TPA factors by hand — plain float64
// loops, no tensor ops: a[t]=x[t]·wa, b[t]=x[t]·wb, then the outer-product sum
// out[t, h·dh+j] = (1/r)·Σ_r a[t, r·heads+h]·b[t, r·dh+j]. The independent
// reference the layer's einsum contraction is checked against.
func tpaReconRef(x, wa, wb *tensor.Tensor, r, heads, dh int) *tensor.Tensor {
	seq, d := x.Shape()[0], x.Shape()[1]
	out := tensor.New(tensor.F64, tensor.Shape{seq, heads * dh})
	for t := range seq {
		a := make([]float64, r*heads)
		for c := range a {
			for i := range d {
				a[c] += x.AtF64(t, i) * wa.AtF64(i, c)
			}
		}
		b := make([]float64, r*dh)
		for c := range b {
			for i := range d {
				b[c] += x.AtF64(t, i) * wb.AtF64(i, c)
			}
		}
		for h := range heads {
			for j := range dh {
				var s float64
				for rr := range r {
					s += a[rr*heads+h] * b[rr*dh+j]
				}
				out.SetF64(s/float64(r), t, h*dh+j)
			}
		}
	}
	return out
}

func tpaCmp(t *testing.T, name string, got, want *tensor.Tensor, tol float64) {
	t.Helper()
	if !got.Shape().Equal(want.Shape()) {
		t.Fatalf("%s: shape %v, want %v", name, got.Shape(), want.Shape())
	}
	for i := range got.Numel() {
		idx := tensor.Unravel(i, got.Shape())
		g, w := got.AtF64(idx...), want.AtF64(idx...)
		if math.Abs(g-w) > tol*math.Max(1, math.Abs(w)) {
			t.Fatalf("%s[%v]: got %.15g want %.15g (diff %.3g)", name, idx, g, w, math.Abs(g-w))
		}
	}
}

// §V16 tier-1 RECONSTRUCTION anchor: with random weights and asymmetric ranks,
// the layer's factor→Q/K/V einsum contraction must match the hand-computed
// outer-product sum (pure float64 loops), and its full forward must equal
// feeding those hand-reconstructed Q,K,V to the standard OpMHA path + W_O.
// A subtly-wrong contraction (rank/head axis swap) cannot pass this.
func TestTPAReconstructionHandComputed(t *testing.T) {
	const seq, d, heads, dh = 3, 5, 2, 3
	const rq, rk, rv = 2, 2, 1 // asymmetric on purpose: catches rank-axis mixups
	m, err := nn.NewTPA(tensor.F64, d, heads, dh, true, 11, nn.WithTPARanks(rq, rk, rv))
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(3, 5))
	x := randMat(rng, seq, d)

	q := tpaReconRef(x, m.WAQ, m.WBQ, rq, heads, dh)
	k := tpaReconRef(x, m.WAK, m.WBK, rk, heads, dh)
	v := tpaReconRef(x, m.WAV, m.WBV, rv, heads, dh)

	ctx := backend.NewContext()
	attn, err := backend.Execute(ctx, backend.OpMHA, []*tensor.Tensor{q, k, v},
		backend.AttnAttrs{Heads: heads, Causal: true})
	if err != nil {
		t.Fatal(err)
	}
	want, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{attn[0], m.WO}, nil)
	if err != nil {
		t.Fatal(err)
	}
	got, err := m.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	tpaCmp(t, "tpa vs hand-reconstructed MHA", got, want[0], 1e-12)
}

// §V16 tier-1 COLLAPSE anchor: at full rank (R=heads) with identity-structured
// head factors, TPA IS vanilla multi-head attention. The input carries a
// constant-1 last feature so the linear head-factor projection can emit the
// constant one-hot pattern a_{t,r}=heads·e_r; the token-factor projections
// hold arbitrary standard Wq/Wk/Wv. Then (1/R)Σ_r a_r⊗b_r reproduces
// q=x·Wq exactly and the layer must equal OpMHA(x·Wq, x·Wk, x·Wv)·W_O.
// The RoPE variant additionally pins the paper's §3.3 commute property:
// rotating the b factors ≡ rotating the reconstructed per-head q/k.
func TestTPAFullRankCollapseToMHA(t *testing.T) {
	const seq, d, heads, dh = 4, 5, 2, 4
	rng := rand.New(rand.NewPCG(9, 2))
	x := randMat(rng, seq, d)
	for i := range seq {
		x.SetF64(1, i, d-1) // constant feature: lets a linear map emit constants
	}
	wq, wk, wv := randMat(rng, d, heads*dh), randMat(rng, d, heads*dh), randMat(rng, d, heads*dh)

	for _, tc := range []struct {
		name string
		rope bool
	}{{"plain", false}, {"rope", true}} {
		t.Run(tc.name, func(t *testing.T) {
			opts := []nn.TPAOption{nn.WithTPARanks(heads, heads, heads)}
			if tc.rope {
				opts = append(opts, nn.WithTPARoPE(10000))
			}
			m, err := nn.NewTPA(tensor.F64, d, heads, dh, true, 13, opts...)
			if err != nil {
				t.Fatal(err)
			}
			// Token factors = the vanilla per-head weights (rank r ↔ head r, so the
			// [d, R·dh] rank-major layout coincides with the head-concat layout).
			copyInto := func(dst, src *tensor.Tensor) {
				for i := range src.Numel() {
					idx := tensor.Unravel(i, src.Shape())
					dst.SetF64(src.AtF64(idx...), idx...)
				}
			}
			copyInto(m.WBQ, wq)
			copyInto(m.WBK, wk)
			copyInto(m.WBV, wv)
			// Head factors: zero except the constant-feature row, which emits
			// a_{t, r·heads+h} = heads·δ_{rh} — so (1/R)Σ_r a_r⊗b_r = b_h per head.
			for _, wa := range []*tensor.Tensor{m.WAQ, m.WAK, m.WAV} {
				nn.Zeros(wa)
				for r := range heads {
					wa.SetF64(float64(heads), d-1, r*heads+r)
				}
			}

			// Vanilla reference: q=x·Wq, k=x·Wk, v=x·Wv (+ per-head RoPE on q,k in
			// the rope variant), fused causal OpMHA, then the layer's own W_O.
			ctx := backend.NewContext()
			proj := func(w *tensor.Tensor) *tensor.Tensor {
				out, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{x, w}, nil)
				if err != nil {
					t.Fatal(err)
				}
				return out[0]
			}
			q, k, v := proj(wq), proj(wk), proj(wv)
			if tc.rope {
				rot := func(y *tensor.Tensor) *tensor.Tensor {
					out, err := backend.Execute(ctx, backend.OpRoPE, []*tensor.Tensor{y},
						backend.RoPEAttrs{Base: 10000, Heads: heads})
					if err != nil {
						t.Fatal(err)
					}
					return out[0]
				}
				q, k = rot(q), rot(k)
			}
			attn, err := backend.Execute(ctx, backend.OpMHA, []*tensor.Tensor{q, k, v},
				backend.AttnAttrs{Heads: heads, Causal: true})
			if err != nil {
				t.Fatal(err)
			}
			want, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{attn[0], m.WO}, nil)
			if err != nil {
				t.Fatal(err)
			}
			got, err := m.Forward(backend.NewContext(), x)
			if err != nil {
				t.Fatal(err)
			}
			tpaCmp(t, "full-rank TPA vs vanilla MHA", got, want[0], 1e-12)
		})
	}
}

// §V16 (c)/§V2 GRADIENT FLOW: central finite differences of the summed forward
// output confirm the analytic gradients on EVERY parameter — all six factor
// projections (through the einsum contraction, the 1/R scale, RoPE on the b^Q
// and b^K factors, and OpMHA) and the out-proj W_O. Asymmetric ranks + RoPE on
// so every distinct code path is on the tape.
func TestTPAGradCheck(t *testing.T) {
	const seq, d, heads, dh = 3, 4, 2, 4
	m, err := nn.NewTPA(tensor.F64, d, heads, dh, true, 17,
		nn.WithTPARanks(3, 2, 1), nn.WithTPARoPE(10000))
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(21, 4))
	x := randMat(rng, seq, d)

	tape := autograd.NewTape()
	out, err := m.Forward(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := sumAll(tape.Context(), out)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	sumLoss := func() float64 {
		o, err := m.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		var s float64
		for i := range o.Numel() {
			s += o.AtF64(tensor.Unravel(i, o.Shape())...)
		}
		return s
	}
	const h = 1e-6
	names := []string{"WAQ", "WBQ", "WAK", "WBK", "WAV", "WBV", "WO"}
	var entries int
	var maxRel float64
	for pi, p := range m.Params() {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("%s: nil grad — no gradient reached this parameter", names[pi])
		}
		var nz bool
		for idx := range p.Numel() {
			coord := tensor.Unravel(idx, p.Shape())
			o := p.AtF64(coord...)
			p.SetF64(o+h, coord...)
			lp := sumLoss()
			p.SetF64(o-h, coord...)
			lm := sumLoss()
			p.SetF64(o, coord...)
			num, ana := (lp-lm)/(2*h), g.AtF64(coord...)
			if math.Abs(ana) > 1e-9 {
				nz = true
			}
			rel := math.Abs(num-ana) / math.Max(1, math.Abs(num))
			if rel > maxRel {
				maxRel = rel
			}
			entries++
			if rel > 1e-4 {
				t.Errorf("%s grad%v: numeric %.8g vs analytic %.8g", names[pi], coord, num, ana)
			}
		}
		if !nz {
			t.Errorf("%s: all analytic grads zero — parameter not exercised", names[pi])
		}
	}
	t.Logf("gradcheck: %d entries, max rel err %.3g", entries, maxRel)
}

// TPA's defining property (§V16): the per-token decode cache of K/V FACTORS,
// (R_k+R_v)·(heads+dh), is far smaller than MHA's full 2·heads·dh — at the
// paper-style config 8 heads × dh 64 with R_k=R_v=2 that is 288 vs 1024
// floats per token (~72% smaller).
func TestTPAKVCacheCompression(t *testing.T) {
	m, err := nn.NewTPA(tensor.F64, 512, 8, 64, true, 1) // default ranks 6/2/2
	if err != nil {
		t.Fatal(err)
	}
	tpa, mha := m.CacheFloatsPerToken()
	if tpa != 288 || mha != 1024 {
		t.Errorf("cache sizes = %d / %d, want 288 / 1024", tpa, mha)
	}
	if tpa >= mha {
		t.Errorf("TPA cache %d should be smaller than MHA %d", tpa, mha)
	}
}

// §V16: the constructor rejects non-positive dims and ranks and an odd RoPE
// head dim; Forward rejects a wrong input width and a wrong rank.
func TestTPAErrors(t *testing.T) {
	if _, err := nn.NewTPA(tensor.F64, 0, 2, 4, true, 1); err == nil {
		t.Error("d=0: want error")
	}
	if _, err := nn.NewTPA(tensor.F64, 8, 0, 4, true, 1); err == nil {
		t.Error("heads=0: want error")
	}
	if _, err := nn.NewTPA(tensor.F64, 8, 2, -1, true, 1); err == nil {
		t.Error("negative dh: want error")
	}
	if _, err := nn.NewTPA(tensor.F64, 8, 2, 4, true, 1, nn.WithTPARanks(0, 2, 2)); err == nil {
		t.Error("Rq=0: want error")
	}
	if _, err := nn.NewTPA(tensor.F64, 8, 2, 4, true, 1, nn.WithTPARanks(2, -1, 2)); err == nil {
		t.Error("negative Rk: want error")
	}
	if _, err := nn.NewTPA(tensor.F64, 8, 2, 3, true, 1, nn.WithTPARoPE(0)); err == nil {
		t.Error("RoPE with odd dh: want error")
	}
	m, err := nn.NewTPA(tensor.F64, 8, 2, 4, true, 1)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{3, 5})); err == nil {
		t.Error("wrong input width: want error")
	}
	if _, err := m.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{2, 3, 8})); err == nil {
		t.Error("rank-3 input: want error")
	}
}

// §V16 (e) E2E TRAINABILITY: a few SGD steps on a fixed MSE objective reduce
// the loss — the full forward+backward+update path through the factorized
// attention actually moves all seven weights in a useful direction.
func TestTPATrainsE2E(t *testing.T) {
	const seq, d, heads, dh = 4, 6, 2, 4
	m, err := nn.NewTPA(tensor.F64, d, heads, dh, true, 23, nn.WithTPARanks(2, 2, 2))
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(31, 7))
	x := randMat(rng, seq, d)
	target := randMat(rng, seq, d)
	// Gentler steps than the linear-attention variants use: each reconstructed
	// Q/K/V is a PRODUCT of two factor projections, so the loss surface is
	// higher-order in the weights and lr 0.05 + momentum 0.9 diverges.
	opt := nn.NewSGD(m.Params(), 0.005, 0.5)

	step := func() float64 {
		tape := autograd.NewTape()
		out, err := m.Forward(tape.Context(), x)
		if err != nil {
			t.Fatal(err)
		}
		diff, err := backend.Execute(tape.Context(), backend.OpSub, []*tensor.Tensor{out, target}, nil)
		if err != nil {
			t.Fatal(err)
		}
		sq, err := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{diff[0], diff[0]}, nil)
		if err != nil {
			t.Fatal(err)
		}
		loss, err := sumAll(tape.Context(), sq[0])
		if err != nil {
			t.Fatal(err)
		}
		l := loss.AtF64(tensor.Unravel(0, loss.Shape())...)
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		if err := opt.Step(tape.Grad); err != nil {
			t.Fatal(err)
		}
		return l
	}
	first := step()
	var last float64
	for range 40 {
		last = step()
	}
	if !(last < first) {
		t.Errorf("MSE did not decrease over training: first %.6g, last %.6g", first, last)
	}
}

// tpaRow returns row t of x[T,d] as a fresh [1,d] tensor — one decode step's
// input.
func tpaRow(x *tensor.Tensor, t int) *tensor.Tensor {
	d := x.Shape()[1]
	r := tensor.New(x.Dtype(), tensor.Shape{1, d})
	for j := range d {
		r.SetF64(x.AtF64(t, j), 0, j)
	}
	return r
}

// §V16 DECODE≡FORWARD anchor: feeding tokens one at a time through DecodeStep
// (growing the factor cache, RoPE at each token's absolute position) reproduces
// a single causal TPA.Forward over the whole [T,d] prefix to 1e-10 — the exact
// property a KV-cache must have. RoPE + asymmetric ranks so a wrong position
// offset or a factor-cache/reconstruction bug would break the match. Also pins
// (a) cache size = compact FACTORS, (R_k+R_v)·(heads+dh)·T floats, NOT the full
// heads·dh·T keys/values, and (b) determinism across two identical decodes.
func TestTPADecodeStepEqualsForward(t *testing.T) {
	// A genuinely compressing config: factors (R_k+R_v)·(heads+dh)=3·10=30/token
	// sit below full K/V 2·heads·dh=48/token — TPA's defining property.
	const seq, d, heads, dh = 5, 8, 4, 6
	const rq, rk, rv = 3, 2, 1
	m, err := nn.NewTPA(tensor.F64, d, heads, dh, true, 41,
		nn.WithTPARanks(rq, rk, rv), nn.WithTPARoPE(10000))
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(43, 8))
	x := randMat(rng, seq, d)

	full, err := m.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}

	decode := func() *tensor.Tensor {
		ctx := backend.NewContext()
		cache := m.NewCache()
		out := tensor.New(tensor.F64, tensor.Shape{seq, d})
		for tpos := range seq {
			if got := cache.Len(); got != tpos {
				t.Fatalf("cache.Len()=%d before step %d, want %d", got, tpos, tpos)
			}
			ot, err := m.DecodeStep(ctx, cache, tpaRow(x, tpos), tpos)
			if err != nil {
				t.Fatal(err)
			}
			if !ot.Shape().Equal(tensor.Shape{1, d}) {
				t.Fatalf("step %d: out shape %v, want (1, %d)", tpos, ot.Shape(), d)
			}
			for j := range d {
				out.SetF64(ot.AtF64(0, j), tpos, j)
			}
			// §V16(2) CACHE SIZE: the cache stores the compact FACTORS,
			// (R_k+R_v)·(heads+dh) floats per position, NOT the full heads·dh K/V.
			tpaFloats, _ := m.CacheFloatsPerToken()
			pos := tpos + 1
			if got := cache.Floats(); got != tpaFloats*pos {
				t.Fatalf("cache.Floats()=%d at %d tokens, want %d (=%d/token)", got, pos, tpaFloats*pos, tpaFloats)
			}
			if cache.Floats() >= 2*heads*dh*pos {
				t.Fatalf("cache stores %d floats ≥ full-K/V %d — factors not compact", cache.Floats(), 2*heads*dh*pos)
			}
			// factor tensors have exactly the rank·axis shapes, never heads·dh.
			if cache.AK.Numel() != rk*heads*pos || cache.BK.Numel() != rk*dh*pos ||
				cache.AV.Numel() != rv*heads*pos || cache.BV.Numel() != rv*dh*pos {
				t.Fatalf("factor shapes wrong at %d tokens: AK=%v BK=%v AV=%v BV=%v",
					pos, cache.AK.Shape(), cache.BK.Shape(), cache.AV.Shape(), cache.BV.Shape())
			}
		}
		return out
	}

	got := decode()
	tpaCmp(t, "decode vs forward", got, full, 1e-10)

	// §V16(4) determinism: a second identical decode is bit-for-bit equal.
	if again := decode(); !again.Shape().Equal(got.Shape()) {
		t.Fatalf("determinism: shape drift %v vs %v", again.Shape(), got.Shape())
	} else {
		for i := range got.Numel() {
			idx := tensor.Unravel(i, got.Shape())
			if got.AtF64(idx...) != again.AtF64(idx...) {
				t.Fatalf("non-deterministic decode at %v: %v vs %v", idx, got.AtF64(idx...), again.AtF64(idx...))
			}
		}
	}
}

// §V16 RoPE-POSITION correctness, second angle: the RoPE-on decode differs from
// an otherwise-identical RoPE-off decode (positions actually rotate the factors),
// AND the position guard forbids decoding a token at anything but its true
// absolute position (== current cache length) — so a caller cannot silently
// rotate at the wrong position. The decode≡forward anchor (RoPE + causal, 1e-10)
// is the primary positive proof that each step rotates at its own position.
func TestTPADecodeStepRoPEPosition(t *testing.T) {
	const seq, d, heads, dh = 4, 6, 2, 4
	mk := func(rope bool) *nn.TPA {
		opts := []nn.TPAOption{nn.WithTPARanks(2, 2, 2)}
		if rope {
			opts = append(opts, nn.WithTPARoPE(10000))
		}
		m, err := nn.NewTPA(tensor.F64, d, heads, dh, true, 51, opts...)
		if err != nil {
			t.Fatal(err)
		}
		return m
	}
	rng := rand.New(rand.NewPCG(53, 9))
	x := randMat(rng, seq, d)

	decode := func(m *nn.TPA) *tensor.Tensor {
		ctx := backend.NewContext()
		cache := m.NewCache()
		out := tensor.New(tensor.F64, tensor.Shape{seq, d})
		for tpos := range seq {
			ot, err := m.DecodeStep(ctx, cache, tpaRow(x, tpos), tpos)
			if err != nil {
				t.Fatal(err)
			}
			for j := range d {
				out.SetF64(ot.AtF64(0, j), tpos, j)
			}
		}
		return out
	}
	// Identical seed → identical weights; the RoPE flag is the only difference, so
	// any divergence past position 0 (RoPE is identity at position 0) is purely the
	// per-position rotation being applied.
	withRoPE, without := decode(mk(true)), decode(mk(false))
	var differs bool
	for tpos := 1; tpos < seq; tpos++ {
		for j := range d {
			if math.Abs(withRoPE.AtF64(tpos, j)-without.AtF64(tpos, j)) > 1e-9 {
				differs = true
			}
		}
	}
	if !differs {
		t.Error("RoPE-on and RoPE-off decodes identical past position 0 — RoPE not applied")
	}

	// Position guard: a token may only be decoded at pos == current cache length.
	m := mk(true)
	ctx := backend.NewContext()
	cache := m.NewCache()
	if _, err := m.DecodeStep(ctx, cache, tpaRow(x, 0), 3); err == nil {
		t.Error("DecodeStep at pos 3 with empty cache: want error")
	}
	if _, err := m.DecodeStep(ctx, cache, tpaRow(x, 0), -1); err == nil {
		t.Error("DecodeStep at negative pos: want error")
	}
	if _, err := m.DecodeStep(ctx, cache, tensor.New(tensor.F64, tensor.Shape{2, d}), 0); err == nil {
		t.Error("DecodeStep with 2-row input: want error")
	}
	if _, err := m.DecodeStep(ctx, cache, tpaRow(x, 0), 0); err != nil {
		t.Fatalf("valid first step errored: %v", err)
	}
	if _, err := m.DecodeStep(ctx, cache, tpaRow(x, 1), 0); err == nil {
		t.Error("DecodeStep at stale pos 0 with 1 token cached: want error")
	}
}

// TPA (arXiv:2501.06425) factorizes each token's Q, K and V as small sums of
// outer products, so a decode cache keeps only the compact K/V factors: here
// an 8-head layer caches 288 floats per token versus 1024 for standard MHA —
// a ~72% reduction — while the attention itself stays exact softmax attention.
func ExampleTPA() {
	m, _ := nn.NewTPA(tensor.F64, 512 /*d*/, 8 /*heads*/, 64 /*dh*/, true, 1,
		nn.WithTPARanks(6, 2, 2)) // the paper's base ranks: R_q=6, R_k=R_v=2
	x := tensor.New(tensor.F64, tensor.Shape{3, 512})
	for i := range 3 {
		x.SetF64(1, i, i) // arbitrary non-zero input
	}
	out, _ := m.Forward(backend.NewContext(), x)
	tpa, mha := m.CacheFloatsPerToken()
	fmt.Printf("%v params=%d cache/token: TPA %d vs MHA %d\n", out.Shape(), len(m.Params()), tpa, mha)
	// Output: (3, 512) params=7 cache/token: TPA 288 vs MHA 1024
}

// ExampleTPA_decode autoregressively decodes three tokens through a TPACache:
// each DecodeStep appends only the compact K/V FACTORS (not full keys/values)
// and attends the single query over the growing cache — so after 3 tokens the
// cache holds CacheFloatsPerToken·3 floats, far below MHA's 2·heads·dh·3.
func ExampleTPA_decode() {
	m, _ := nn.NewTPA(tensor.F64, 32 /*d*/, 4 /*heads*/, 8 /*dh*/, true, 7,
		nn.WithTPARanks(2, 2, 2), nn.WithTPARoPE(10000))
	var cache *nn.TPACache = m.NewCache() // compact per-position K/V factors
	ctx := backend.NewContext()
	for pos := range 3 {
		xt := tensor.New(tensor.F64, tensor.Shape{1, 32})
		xt.SetF64(1, 0, pos) // arbitrary non-zero token embedding
		out, _ := m.DecodeStep(ctx, cache, xt, pos)
		_ = out // [1,32] next-layer output for this token
	}
	perTok, _ := m.CacheFloatsPerToken()
	fmt.Printf("cached %d tokens, %d floats (=%d/token)\n", cache.Len(), cache.Floats(), perTok)
	// Output: cached 3 tokens, 144 floats (=48/token)
}
