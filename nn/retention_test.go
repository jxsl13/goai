package nn_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// retentionRef is an independent recomputation of the parallel retention O=(QKᵀ⊙D)V, D_nm=γ^(n−m)
// (m≤n). q,k,v are flat [L·d] row-major.
func retentionRef(q, k, v []float64, l, d int, gamma float64) []float64 {
	o := make([]float64, l*d)
	for n := range l {
		for m := 0; m <= n; m++ {
			var a float64
			for i := range d {
				a += q[n*d+i] * k[m*d+i]
			}
			p := a * math.Pow(gamma, float64(n-m))
			for j := range d {
				o[n*d+j] += p * v[m*d+j]
			}
		}
	}
	return o
}

func randSlice(rng *rand.Rand, n int) []float64 {
	s := make([]float64, n)
	for i := range s {
		s[i] = rng.NormFloat64()
	}
	return s
}

// §V16 tier-1 (THE defining RetNet property, Sun et al. 2023): the parallel retention (QKᵀ⊙D)V and
// the recurrent form S_n=γS_{n−1}+K_nᵀV_n, out=Q_nS_n produce IDENTICAL output — the parallel–
// recurrent duality — across random Q,K,V and several decay rates.
func TestRetentionDuality(t *testing.T) {
	rng := rand.New(rand.NewSource(0x9e7))
	ctx := backend.NewContext()
	for _, gamma := range []float64{1.0, 0.99, 0.9, 0.5, 0.0} {
		for _, dims := range [][2]int{{1, 4}, {5, 3}, {8, 6}, {12, 2}} {
			l, d := dims[0], dims[1]
			qd, kd, vd := randSlice(rng, l*d), randSlice(rng, l*d), randSlice(rng, l*d)
			q := tensor.FromFloat64(tensor.Shape{l, d}, qd)
			k := tensor.FromFloat64(tensor.Shape{l, d}, kd)
			v := tensor.FromFloat64(tensor.Shape{l, d}, vd)
			par, err := nn.Retention(ctx, q, k, v, gamma)
			if err != nil {
				t.Fatal(err)
			}
			rec, err := nn.RetentionRecurrent(q, k, v, gamma)
			if err != nil {
				t.Fatal(err)
			}
			for n := range l {
				for j := range d {
					if math.Abs(par.AtF64(n, j)-rec.AtF64(n, j)) > 1e-10 {
						t.Fatalf("γ=%g L=%d d=%d [%d,%d]: parallel %.12g != recurrent %.12g",
							gamma, l, d, n, j, par.AtF64(n, j), rec.AtF64(n, j))
					}
				}
			}
		}
	}
}

// §V16 tier-1: THE chunkwise duality (§R114 Eq.7) — RetentionChunkwise produces EXACTLY the parallel
// Retention output for every chunk size (1 = fully recurrent, L = one parallel pass, and everything
// between, plus a chunk larger than L), across γ and shapes including the value expansion d_v≠d_k.
func TestRetentionChunkwiseDuality(t *testing.T) {
	rng := rand.New(rand.NewSource(0x0c47))
	ctx := backend.NewContext()
	for _, gamma := range []float64{1.0, 0.99, 0.9, 0.5, 0.0} {
		for _, dims := range [][3]int{{1, 4, 4}, {5, 3, 3}, {8, 6, 6}, {6, 3, 5}, {7, 4, 8}, {10, 2, 4}} {
			l, dk, dv := dims[0], dims[1], dims[2]
			q := tensor.FromFloat64(tensor.Shape{l, dk}, randSlice(rng, l*dk))
			k := tensor.FromFloat64(tensor.Shape{l, dk}, randSlice(rng, l*dk))
			v := tensor.FromFloat64(tensor.Shape{l, dv}, randSlice(rng, l*dv))
			par, err := nn.Retention(ctx, q, k, v, gamma)
			if err != nil {
				t.Fatal(err)
			}
			for _, cs := range []int{1, 2, 3, 5, l, l + 1} {
				ch, err := nn.RetentionChunkwise(q, k, v, gamma, cs)
				if err != nil {
					t.Fatalf("γ=%g L=%d cs=%d: %v", gamma, l, cs, err)
				}
				for n := range l {
					for j := range dv {
						if math.Abs(par.AtF64(n, j)-ch.AtF64(n, j)) > 1e-10 {
							t.Fatalf("γ=%g L=%d dk=%d dv=%d cs=%d [%d,%d]: parallel %.12g != chunkwise %.12g",
								gamma, l, dk, dv, cs, n, j, par.AtF64(n, j), ch.AtF64(n, j))
						}
					}
				}
			}
		}
	}
}

func TestRetentionChunkwiseErrors(t *testing.T) {
	q := tensor.New(tensor.F64, tensor.Shape{4, 3})
	v := tensor.New(tensor.F64, tensor.Shape{4, 3})
	if _, err := nn.RetentionChunkwise(q, q, v, 0.9, 0); err == nil {
		t.Error("chunkSize < 1 must error")
	}
	if _, err := nn.RetentionChunkwise(q, tensor.New(tensor.F64, tensor.Shape{5, 3}), v, 0.9, 2); err == nil {
		t.Error("mismatched L must error")
	}
}

// RetentionChunkwise is the third RetNet form: it chunks the sequence and carries a bounded state
// between chunks, giving the same result as the parallel Retention at O(chunk²+d²) memory. Here a
// length-4 sequence processed in chunks of 2 matches the one-pass parallel form.
func ExampleRetentionChunkwise() {
	q := tensor.FromFloat64(tensor.Shape{4, 2}, []float64{1, 0, 0, 1, 1, 1, 0.5, -0.5})
	k := tensor.FromFloat64(tensor.Shape{4, 2}, []float64{0, 1, 1, 0, 1, 1, -0.5, 0.5})
	v := tensor.FromFloat64(tensor.Shape{4, 2}, []float64{1, 2, 3, 4, 5, 6, 7, 8})
	par, _ := nn.Retention(backend.NewContext(), q, k, v, 0.9)
	ch, _ := nn.RetentionChunkwise(q, k, v, 0.9, 2)
	fmt.Printf("%.4f %.4f\n", ch.AtF64(3, 0), par.AtF64(3, 0))
	// Output: -2.6495 -2.6495
}

// §V16 tier-1: retention with a value dim different from the key dim (d_v ≠ d_k, RetNet's value
// expansion) still satisfies the parallel–recurrent duality and matches an independent recomputation
// of (QKᵀ⊙D)V (i over d_k, j over d_v). Also gradchecks the generalized backward.
func TestRetentionValueExpansion(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5ee))
	const l, dk, dv, gamma = 6, 3, 5, 0.9
	qd, kd := randSlice(rng, l*dk), randSlice(rng, l*dk)
	vd := randSlice(rng, l*dv)
	q := tensor.FromFloat64(tensor.Shape{l, dk}, qd)
	k := tensor.FromFloat64(tensor.Shape{l, dk}, kd)
	v := tensor.FromFloat64(tensor.Shape{l, dv}, vd)
	// independent reference: O[n,j] = Σ_{m≤n} (Σ_i Q[n,i]K[m,i])·γ^(n−m)·V[m,j]
	ref := func(qs, ks, vs []float64) []float64 {
		o := make([]float64, l*dv)
		for n := range l {
			for m := 0; m <= n; m++ {
				var a float64
				for i := range dk {
					a += qs[n*dk+i] * ks[m*dk+i]
				}
				p := a * math.Pow(gamma, float64(n-m))
				for j := range dv {
					o[n*dv+j] += p * vs[m*dv+j]
				}
			}
		}
		return o
	}
	par, err := nn.Retention(backend.NewContext(), q, k, v, gamma)
	if err != nil {
		t.Fatal(err)
	}
	rec, err := nn.RetentionRecurrent(q, k, v, gamma)
	if err != nil {
		t.Fatal(err)
	}
	want := ref(qd, kd, vd)
	for n := range l {
		for j := range dv {
			if math.Abs(par.AtF64(n, j)-want[n*dv+j]) > 1e-12 {
				t.Errorf("golden [%d,%d]: %.12g vs %.12g", n, j, par.AtF64(n, j), want[n*dv+j])
			}
			if math.Abs(par.AtF64(n, j)-rec.AtF64(n, j)) > 1e-10 {
				t.Errorf("duality [%d,%d]: parallel %.12g != recurrent %.12g", n, j, par.AtF64(n, j), rec.AtF64(n, j))
			}
		}
	}
	// gradcheck the generalized backward (sum of output) w.r.t. Q,K,V
	tape := autograd.NewTape()
	ctx := tape.Context()
	qt := tensor.FromFloat64(tensor.Shape{l, dk}, append([]float64(nil), qd...))
	kt := tensor.FromFloat64(tensor.Shape{l, dk}, append([]float64(nil), kd...))
	vt := tensor.FromFloat64(tensor.Shape{l, dv}, append([]float64(nil), vd...))
	o, _ := nn.Retention(ctx, qt, kt, vt, gamma)
	loss, _ := backend.Execute(ctx, backend.OpSum, []*tensor.Tensor{o}, nil)
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	sumRef := func(qs, ks, vs []float64) float64 {
		o := ref(qs, ks, vs)
		s := 0.0
		for _, x := range o {
			s += x
		}
		return s
	}
	const h = 1e-6
	gc := func(name string, data []float64, grad *tensor.Tensor, f func(idx int, delta float64) float64) {
		for i := range data {
			num := (f(i, h) - f(i, -h)) / (2 * h)
			ana := grad.AtF64(tensor.Unravel(i, grad.Shape())...)
			if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("%s grad[%d]: numeric %.8g vs analytic %.8g", name, i, num, ana)
			}
		}
	}
	gc("Q", qd, tape.Grad(qt), func(i int, dl float64) float64 {
		c := append([]float64(nil), qd...)
		c[i] += dl
		return sumRef(c, kd, vd)
	})
	gc("K", kd, tape.Grad(kt), func(i int, dl float64) float64 {
		c := append([]float64(nil), kd...)
		c[i] += dl
		return sumRef(qd, c, vd)
	})
	gc("V", vd, tape.Grad(vt), func(i int, dl float64) float64 {
		c := append([]float64(nil), vd...)
		c[i] += dl
		return sumRef(qd, kd, c)
	})
}

// §V16 tier-1: the parallel retention matches an independent recomputation of (QKᵀ⊙D)V @1e-12.
func TestRetentionGolden(t *testing.T) {
	const l, d, gamma = 6, 4, 0.9
	rng := rand.New(rand.NewSource(11))
	qd, kd, vd := randSlice(rng, l*d), randSlice(rng, l*d), randSlice(rng, l*d)
	ctx := backend.NewContext()
	got, err := nn.Retention(ctx,
		tensor.FromFloat64(tensor.Shape{l, d}, qd),
		tensor.FromFloat64(tensor.Shape{l, d}, kd),
		tensor.FromFloat64(tensor.Shape{l, d}, vd), gamma)
	if err != nil {
		t.Fatal(err)
	}
	want := retentionRef(qd, kd, vd, l, d, gamma)
	for n := range l {
		for j := range d {
			if math.Abs(got.AtF64(n, j)-want[n*d+j]) > 1e-12 {
				t.Errorf("[%d,%d]: %.14g vs %.14g", n, j, got.AtF64(n, j), want[n*d+j])
			}
		}
	}
}

// §V2 GRADCHECK: the analytic gradients of retention w.r.t. Q,K,V match central finite differences
// of the summed output.
func TestRetentionGradcheck(t *testing.T) {
	const l, d, gamma, h, rtol = 4, 3, 0.9, 1e-6, 1e-4
	rng := rand.New(rand.NewSource(23))
	qd, kd, vd := randSlice(rng, l*d), randSlice(rng, l*d), randSlice(rng, l*d)
	sumRet := func(q, k, v []float64) float64 {
		o := retentionRef(q, k, v, l, d, gamma)
		s := 0.0
		for _, x := range o {
			s += x
		}
		return s
	}
	tape := autograd.NewTape()
	ctx := tape.Context()
	q := tensor.FromFloat64(tensor.Shape{l, d}, qd)
	k := tensor.FromFloat64(tensor.Shape{l, d}, kd)
	v := tensor.FromFloat64(tensor.Shape{l, d}, vd)
	o, err := nn.Retention(ctx, q, k, v, gamma)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := backend.Execute(ctx, backend.OpSum, []*tensor.Tensor{o}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	// perturb returns sum(retention) with the named input's element idx shifted by delta.
	perturb := func(name string, idx int, delta float64) float64 {
		q2, k2, v2 := qd, kd, vd
		switch name {
		case "Q":
			q2 = append([]float64(nil), qd...)
			q2[idx] += delta
		case "K":
			k2 = append([]float64(nil), kd...)
			k2[idx] += delta
		case "V":
			v2 = append([]float64(nil), vd...)
			v2[idx] += delta
		}
		return sumRet(q2, k2, v2)
	}
	check := func(name string, n int, grad *tensor.Tensor) {
		if grad == nil {
			t.Fatalf("nil grad for %s", name)
		}
		for idx := range n {
			numer := (perturb(name, idx, h) - perturb(name, idx, -h)) / (2 * h)
			ana := grad.AtF64(tensor.Unravel(idx, tensor.Shape{l, d})...)
			if math.Abs(numer-ana) > rtol*math.Max(1, math.Abs(numer)) {
				t.Errorf("%s grad[%d]: numeric %.10g vs analytic %.10g", name, idx, numer, ana)
			}
		}
	}
	check("Q", l*d, tape.Grad(q))
	check("K", l*d, tape.Grad(k))
	check("V", l*d, tape.Grad(v))
}

// §V16 tier-1: the multi-scale per-head decays are γ_h = 1 − 2^(−5−h) (Sun et al. 2023, Eq.8).
func TestRetentionDecays(t *testing.T) {
	g := nn.RetentionDecays(4)
	want := []float64{1 - 1.0/32, 1 - 1.0/64, 1 - 1.0/128, 1 - 1.0/256}
	for h := range want {
		if math.Abs(g[h]-want[h]) > 1e-15 {
			t.Errorf("γ[%d] = %.10g, want %.10g", h, g[h], want[h])
		}
	}
}

func TestRetentionErrors(t *testing.T) {
	ctx := backend.NewContext()
	q := tensor.New(tensor.F64, tensor.Shape{4, 3})
	if _, err := nn.Retention(ctx, q, tensor.New(tensor.F64, tensor.Shape{3, 3}), q, 0.9); err == nil {
		t.Error("mismatched K shape must error")
	}
	if _, err := nn.Retention(ctx, q, q, q, 1.5); err == nil {
		t.Error("gamma > 1 must error")
	}
}

// RetNet retention replaces attention's softmax with a γ-decay mask, giving a Transformer-like
// parallel form and an equivalent RNN-like recurrent form. Here the two forms agree exactly on a
// small sequence.
func ExampleRetention() {
	ctx := backend.NewContext()
	q := tensor.FromFloat64(tensor.Shape{3, 2}, []float64{1, 0, 0, 1, 1, 1})
	k := tensor.FromFloat64(tensor.Shape{3, 2}, []float64{1, 1, 0, 1, 1, 0})
	v := tensor.FromFloat64(tensor.Shape{3, 2}, []float64{2, 0, 0, 3, 1, 1})
	par, _ := nn.Retention(ctx, q, k, v, 0.9)
	rec, _ := nn.RetentionRecurrent(q, k, v, 0.9)
	fmt.Printf("%.4f %v\n", par.AtF64(2, 0), math.Abs(par.AtF64(2, 0)-rec.AtF64(2, 0)) < 1e-12)
	// Output: 4.2400 true
}
