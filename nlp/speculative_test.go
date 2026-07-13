package nlp

import (
	"math"
	"math/rand/v2"
	"testing"
)

// §V16 losslessness theorem (Leviathan Thm 1 / Chen): the token produced by
// speculative sampling is distributed EXACTLY as the target p, for ANY draft q.
// Monte-Carlo verification: draft ~ q, accept/resample, empirical output ≈ p.
func TestSpeculativeLossless(t *testing.T) {
	p := []float64{0.5, 0.2, 0.2, 0.1}
	q := []float64{0.1, 0.4, 0.3, 0.2} // deliberately far from p
	rng := rand.New(rand.NewPCG(42, 7))
	const N = 300000
	counts := make([]int, len(p))
	for range N {
		dt := sampleDist(q, rng) // token drafted from q
		out, _ := SpeculativeSample(p, q, dt, rng)
		counts[out]++
	}
	for i := range p {
		if emp := float64(counts[i]) / N; math.Abs(emp-p[i]) > 0.01 {
			t.Errorf("token %d: empirical %.4f vs target %.4f (must equal p, any q)", i, emp, p[i])
		}
	}
}

// §R53: the acceptance probability averages to Σ min(p,q) = 1 − TV(p,q).
func TestSpeculativeAcceptRate(t *testing.T) {
	p := []float64{0.5, 0.2, 0.2, 0.1}
	q := []float64{0.1, 0.4, 0.3, 0.2}
	want := 0.0
	for i := range p {
		want += math.Min(p[i], q[i]) // = 0.6
	}
	rng := rand.New(rand.NewPCG(1, 2))
	const N = 300000
	accepted := 0
	for range N {
		dt := sampleDist(q, rng)
		if _, a := SpeculativeSample(p, q, dt, rng); a {
			accepted++
		}
	}
	if got := float64(accepted) / N; math.Abs(got-want) > 0.01 {
		t.Errorf("accept rate %.4f, want Σmin(p,q) = %.4f", got, want)
	}
}

// When the draft equals the target every draft is accepted (min(1,p/q)=1) and the
// output is exactly the drafted token — no wasted work.
func TestSpeculativeIdenticalAlwaysAccepts(t *testing.T) {
	p := []float64{0.3, 0.5, 0.2}
	rng := rand.New(rand.NewPCG(3, 4))
	for range 2000 {
		dt := sampleDist(p, rng)
		out, a := SpeculativeSample(p, p, dt, rng)
		if !a {
			t.Fatal("p==q must always accept")
		}
		if out != dt {
			t.Fatalf("accepted token %d != draft %d", out, dt)
		}
	}
}

// A rejected token can only come from the residual support where p > q.
func TestSpeculativeResidualSupport(t *testing.T) {
	p := []float64{0.8, 0.1, 0.1}
	q := []float64{0.1, 0.8, 0.1} // p>q only at token 0
	rng := rand.New(rand.NewPCG(5, 6))
	sawReject := false
	for range 5000 {
		dt := sampleDist(q, rng)
		out, a := SpeculativeSample(p, q, dt, rng)
		if !a {
			sawReject = true
			if p[out] <= q[out] {
				t.Errorf("rejected token %d has p=%.2f ≤ q=%.2f (not in residual)", out, p[out], q[out])
			}
		}
	}
	if !sawReject {
		t.Error("expected some rejections for this p,q")
	}
}

func onehot(v, at int) []float64 {
	d := make([]float64, v)
	d[at] = 1
	return d
}

// The run emits K+1 tokens when every draft is accepted, and stops at 1 token on
// an immediate rejection (Leviathan Algorithm 1 structure).
func TestSpeculativeRunLength(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	const V = 5

	// All drafts certain-accept (draft==target one-hots) → K accepted + 1 bonus.
	tks := []int{1, 2, 3}
	target := [][]float64{onehot(V, 1), onehot(V, 2), onehot(V, 3), onehot(V, 4)}
	draft := [][]float64{onehot(V, 1), onehot(V, 2), onehot(V, 3)}
	got := SpeculativeRun(target, draft, tks, rng)
	if len(got) != 4 {
		t.Fatalf("all-accept run length %d, want K+1=4", len(got))
	}
	if got[3] != 4 {
		t.Errorf("bonus token %d, want 4 (from target[K])", got[3])
	}

	// Immediate rejection: target gives zero mass to the first drafted token.
	target2 := [][]float64{onehot(V, 0), onehot(V, 2), onehot(V, 3), onehot(V, 4)}
	draft2 := [][]float64{onehot(V, 1), onehot(V, 2), onehot(V, 3)} // drafts token 1, target wants 0
	got2 := SpeculativeRun(target2, draft2, []int{1, 2, 3}, rng)
	if len(got2) != 1 {
		t.Fatalf("immediate-reject run length %d, want 1", len(got2))
	}
	if got2[0] != 0 {
		t.Errorf("residual token %d, want 0 (target's mass)", got2[0])
	}
}
