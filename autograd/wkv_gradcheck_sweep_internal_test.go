package autograd

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestWKVVJPGradCheckSweep is the safety net a linear-time rewrite of this VJP needs, and it exists
// because the gradcheck already here cannot serve that purpose (R-01KYWM71JJEGW).
//
// TestWKVGradCheck differentiates the plain SUM of the outputs, so the upstream gradient is
// identically 1 at every position. That hides a whole class of error: dv_i is a suffix sum over t
// of g_t*p_{t,i}, and with g constant any bug that permutes, shifts or double-counts the t index
// still lands on the same total. It also uses a single seq=5, d=3 shape with w in [0.3, 1.1], while
// the numerical behavior that makes this kernel hard is governed by the product seq*w — the span of
// the exponent vector — which that fixture never pushes.
//
// So this drives the VJP directly with a NON-UNIFORM upstream gradient, including exact zeros to
// exercise the gt == 0 early-continue, and sweeps shape and decay together:
//
//   - seq 1 and 2 pin the degenerate ends, where the only term is the diagonal u + k_t;
//   - seq 17 is not a multiple of any blocking factor a rewrite might introduce;
//   - seq 64 with w = 3 gives an exponent span of ~192, far enough that a fixed-reference
//     exp() underflows to zero and a stable implementation must carry a running maximum;
//   - w = 0.02 is the opposite regime, where decay is nearly absent and every past token
//     contributes, so a recurrence that drops the tail is caught.
//
// The reference is central differences of the weighted sum Σ g[t,c]·wkv[t,c], which is exactly the
// quantity the VJP claims to differentiate.
func TestWKVVJPGradCheckSweep(t *testing.T) {
	cases := []struct {
		seq, d int
		w      float64
		name   string
	}{
		{1, 2, 0.5, "seq1_degenerate"},
		{2, 3, 0.5, "seq2"},
		{5, 3, 0.05, "near_zero_decay"},
		{17, 3, 0.7, "odd_length"},
		{64, 2, 3.0, "long_and_steep"},
		{64, 2, 0.02, "long_and_flat"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			k, v, w, u, g := wkvGradCheckInputs(c.seq, c.d, c.w)
			vjp := vjps[backend.OpWKV]
			got, err := vjp(nil, []*tensor.Tensor{k, v, w, u}, nil, nil, g)
			if err != nil {
				t.Fatal(err)
			}
			loss := func() float64 {
				out, err := backend.Execute(backend.NewContext(), backend.OpWKV,
					[]*tensor.Tensor{k, v, w, u}, nil)
				if err != nil {
					t.Fatal(err)
				}
				var s float64
				for i := range out[0].Numel() {
					idx := tensor.Unravel(i, out[0].Shape())
					s += g.AtF64(idx...) * out[0].AtF64(idx...)
				}
				return s
			}
			const h = 1e-6
			names := []string{"k", "v", "w", "u"}
			for n, in := range []*tensor.Tensor{k, v, w, u} {
				grad := got[n]
				if grad == nil {
					t.Fatalf("%s received no gradient", names[n])
				}
				for i := range in.Numel() {
					idx := tensor.Unravel(i, in.Shape())
					o := in.AtF64(idx...)
					in.SetF64(o+h, idx...)
					lp := loss()
					in.SetF64(o-h, idx...)
					lm := loss()
					in.SetF64(o, idx...)
					num, ana := (lp-lm)/(2*h), grad.AtF64(idx...)
					// Relative, with an absolute floor: a central difference of a value near zero
					// is dominated by its own truncation error, not by any error in the gradient.
					if math.Abs(num-ana) > 1e-4*math.Max(1e-3, math.Abs(num)) {
						t.Errorf("%s grad%v: numeric %.10g vs analytic %.10g", names[n], idx, num, ana)
					}
				}
			}
		})
	}
}

// wkvGradCheckInputs builds one case. The upstream gradient is deliberately non-uniform and carries
// exact zeros; the decay w is positive per the kernel's convention and constant across channels so
// that seq*w — the quantity that decides numerical stability — is what the case name says it is.
func wkvGradCheckInputs(seq, d int, decay float64) (k, v, w, u, g *tensor.Tensor) {
	mk := func(seed float64, shape tensor.Shape, f func(i int) float64) *tensor.Tensor {
		x := tensor.New(tensor.F64, shape)
		s := x.Storage().F64()
		for i := range s {
			s[i] = f(i) * math.Sin(seed+1.7*float64(i))
		}
		return x
	}
	k = mk(1.1, tensor.Shape{seq, d}, func(int) float64 { return 0.8 })
	v = mk(2.3, tensor.Shape{seq, d}, func(int) float64 { return 0.9 })
	u = mk(4.7, tensor.Shape{d}, func(int) float64 { return 0.6 })
	w = tensor.New(tensor.F64, tensor.Shape{d})
	ws := w.Storage().F64()
	for c := range d {
		ws[c] = decay * (1 + 0.1*float64(c)) // positive decay, mildly per-channel
	}
	// Every third position is exactly zero so the gt == 0 branch runs, and the rest vary in sign
	// and magnitude so a t-index error cannot cancel.
	g = mk(3.5, tensor.Shape{seq, d}, func(i int) float64 {
		if i%3 == 0 {
			return 0
		}
		return 1 + 0.5*float64(i%5)
	})
	return k, v, w, u, g
}
