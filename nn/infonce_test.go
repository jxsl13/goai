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

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// infonceRef independently computes the symmetric in-batch InfoNCE loss.
func infonceRef(a, b [][]float64, tau float64, normalize bool) float64 {
	n, d := len(a), len(a[0])
	na := make([][]float64, n)
	nb := make([][]float64, n)
	for i := range n {
		na[i] = append([]float64(nil), a[i]...)
		nb[i] = append([]float64(nil), b[i]...)
		if normalize {
			var sa, sb float64
			for k := range d {
				sa += a[i][k] * a[i][k]
				sb += b[i][k] * b[i][k]
			}
			sa, sb = math.Sqrt(sa), math.Sqrt(sb)
			for k := range d {
				na[i][k] /= sa
				nb[i][k] /= sb
			}
		}
	}
	// S = na·nbᵀ / τ
	s := make([][]float64, n)
	for i := range n {
		s[i] = make([]float64, n)
		for j := range n {
			for k := range d {
				s[i][j] += na[i][k] * nb[j][k]
			}
			s[i][j] /= tau
		}
	}
	rowCE := func(row []float64, tgt int) float64 {
		mx := math.Inf(-1)
		for _, v := range row {
			mx = math.Max(mx, v)
		}
		var sum float64
		for _, v := range row {
			sum += math.Exp(v - mx)
		}
		return -(row[tgt] - mx - math.Log(sum))
	}
	var lossA, lossB float64
	for i := range n {
		lossA += rowCE(s[i], i) // row i
		col := make([]float64, n)
		for j := range n {
			col[j] = s[j][i] // column i = row i of Sᵀ
		}
		lossB += rowCE(col, i)
	}
	return 0.5 * (lossA/float64(n) + lossB/float64(n))
}

func toT(m [][]float64) *tensor.Tensor {
	r, c := len(m), len(m[0])
	t := tensor.New(tensor.F64, tensor.Shape{r, c})
	for i := range r {
		for j := range c {
			t.SetF64(m[i][j], i, j)
		}
	}
	return t
}

var (
	nceA = [][]float64{{1, 0.5, -1}, {0.2, 1, 0.3}, {-0.5, 0.4, 1}}
	nceB = [][]float64{{0.9, 0.4, -0.8}, {0.1, 1.2, 0.2}, {-0.6, 0.3, 1.1}}
)

// §V16 tier-1: InfoNCE matches an independent symmetric-CE recomputation, with and
// without L2 normalization, for a fixed temperature.
func TestInfoNCEFormula(t *testing.T) {
	for _, norm := range []bool{false, true} {
		for _, tau := range []float64{0.1, 1.0} {
			got, err := nn.InfoNCE(backend.NewContext(), toT(nceA), toT(nceB), tau, norm)
			if err != nil {
				t.Fatalf("norm=%v τ=%g: %v", norm, tau, err)
			}
			want := infonceRef(nceA, nceB, tau, norm)
			if math.Abs(got.AtF64()-want) > 1e-9 {
				t.Errorf("norm=%v τ=%g: InfoNCE = %.10g, want %.10g", norm, tau, got.AtF64(), want)
			}
		}
	}
}

// §V16 tier-1: for a single pair (n=1) the loss is 0 (softmax over one logit is 1); and
// aligned embeddings (a=b, distinct rows) give a lower loss than misaligned ones.
func TestInfoNCEProperties(t *testing.T) {
	one, _ := nn.InfoNCE(backend.NewContext(), toT([][]float64{{1, 2}}), toT([][]float64{{3, 4}}), 0.5, true)
	if math.Abs(one.AtF64()) > 1e-9 {
		t.Errorf("n=1 loss = %g, want 0", one.AtF64())
	}
	aligned, _ := nn.InfoNCE(backend.NewContext(), toT(nceA), toT(nceA), 0.1, true) // a=b
	mis := [][]float64{{-0.6, 0.3, 1.1}, {1, 0.5, -1}, {0.2, 1, 0.3}}               // b = shuffled a
	misaligned, _ := nn.InfoNCE(backend.NewContext(), toT(nceA), toT(mis), 0.1, true)
	if aligned.AtF64() >= misaligned.AtF64() {
		t.Errorf("aligned loss %.4g not < misaligned %.4g", aligned.AtF64(), misaligned.AtF64())
	}
}

// §V2: the loss is differentiable w.r.t. both embedding sets (with and without L2 norm).
func TestInfoNCEGradcheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 3))
	const n, d = 4, 3
	for _, norm := range []bool{false, true} {
		a := randMat(rng, n, d)
		b := randMat(rng, n, d)
		tape := autograd.NewTape()
		loss, err := nn.InfoNCE(tape.Context(), a, b, 0.5, norm)
		if err != nil {
			t.Fatal(err)
		}
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		const h = 1e-6
		check := func(name string, p *tensor.Tensor) {
			g := tape.Grad(p)
			if g == nil {
				t.Fatalf("norm=%v %s: nil grad", norm, name)
			}
			for i := range n {
				for j := range d {
					o := p.AtF64(i, j)
					p.SetF64(o+h, i, j)
					lp, _ := nn.InfoNCE(backend.NewContext(), a, b, 0.5, norm)
					p.SetF64(o-h, i, j)
					lm, _ := nn.InfoNCE(backend.NewContext(), a, b, 0.5, norm)
					p.SetF64(o, i, j)
					if num, ana := (lp.AtF64()-lm.AtF64())/(2*h), g.AtF64(i, j); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
						t.Errorf("norm=%v %s grad[%d,%d]: numeric %.8g vs analytic %.8g", norm, name, i, j, num, ana)
					}
				}
			}
		}
		check("a", a)
		check("b", b)
	}
}

func TestInfoNCEErrors(t *testing.T) {
	a := toT(nceA)
	if _, err := nn.InfoNCE(backend.NewContext(), a, toT([][]float64{{1, 2}}), 0.5, true); err == nil {
		t.Error("shape mismatch: want error")
	}
	if _, err := nn.InfoNCE(backend.NewContext(), a, toT(nceB), 0, true); err == nil {
		t.Error("τ=0: want error")
	}
}

func ExampleInfoNCE() {
	// Two identical, well-separated embeddings ⇒ near-zero contrastive loss.
	emb := toT([][]float64{{3, 0, 0}, {0, 3, 0}, {0, 0, 3}})
	loss, _ := nn.InfoNCE(backend.NewContext(), emb, emb, 0.05, true)
	fmt.Printf("%.4f\n", loss.AtF64())
	// Output:
	// 0.0000
}

// ntxentRef independently computes the SimCLR NT-Xent loss over z=[a;b] with self-exclusion.
func ntxentRef(a, b [][]float64, tau float64, normalize bool) float64 {
	n, d := len(a), len(a[0])
	z := make([][]float64, 0, 2*n)
	for _, row := range a {
		z = append(z, append([]float64(nil), row...))
	}
	for _, row := range b {
		z = append(z, append([]float64(nil), row...))
	}
	if normalize {
		for i := range z {
			var s float64
			for k := range d {
				s += z[i][k] * z[i][k]
			}
			s = math.Sqrt(s)
			for k := range d {
				z[i][k] /= s
			}
		}
	}
	twoN := 2 * n
	sim := make([][]float64, twoN)
	for i := range twoN {
		sim[i] = make([]float64, twoN)
		for j := range twoN {
			for k := range d {
				sim[i][j] += z[i][k] * z[j][k]
			}
			sim[i][j] /= tau
		}
	}
	var loss float64
	for i := range twoN {
		pos := i + n
		if i >= n {
			pos = i - n
		}
		mx := math.Inf(-1) // LSE over k != i (self-exclusion)
		for k := range twoN {
			if k != i {
				mx = math.Max(mx, sim[i][k])
			}
		}
		var sum float64
		for k := range twoN {
			if k != i {
				sum += math.Exp(sim[i][k] - mx)
			}
		}
		loss += -(sim[i][pos] - mx - math.Log(sum))
	}
	return loss / float64(twoN)
}

// §V16 tier-1 (§R165 NT-Xent variant): NTXentLoss matches an independent 2N-way self-excluded
// softmax-CE recomputation, with and without L2 normalization, across temperatures.
func TestNTXentFormula(t *testing.T) {
	for _, norm := range []bool{false, true} {
		for _, tau := range []float64{0.1, 0.5, 1.0} {
			got, err := nn.NTXentLoss(backend.NewContext(), toT(nceA), toT(nceB), tau, norm)
			if err != nil {
				t.Fatalf("norm=%v τ=%g: %v", norm, tau, err)
			}
			want := ntxentRef(nceA, nceB, tau, norm)
			if math.Abs(got.AtF64()-want) > 1e-9 {
				t.Errorf("norm=%v τ=%g: NTXentLoss = %.10g, want %.10g", norm, tau, got.AtF64(), want)
			}
		}
	}
}

// §V16 tier-1: the DEFINING self-exclusion property — for a single pair (n=1) the only "negative"
// is the anchor itself, which is excluded, leaving one logit ⇒ loss 0; and two identical views
// (a=b) give a lower loss than misaligned views.
func TestNTXentProperties(t *testing.T) {
	one, _ := nn.NTXentLoss(backend.NewContext(), toT([][]float64{{1, 2}}), toT([][]float64{{3, 4}}), 0.5, true)
	if math.Abs(one.AtF64()) > 1e-9 {
		t.Errorf("n=1 loss = %g, want 0 (self-exclusion leaves one logit)", one.AtF64())
	}
	aligned, _ := nn.NTXentLoss(backend.NewContext(), toT(nceA), toT(nceA), 0.1, true) // both views equal
	mis := [][]float64{{-0.6, 0.3, 1.1}, {1, 0.5, -1}, {0.2, 1, 0.3}}                  // second view = shuffled
	misaligned, _ := nn.NTXentLoss(backend.NewContext(), toT(nceA), toT(mis), 0.1, true)
	if aligned.AtF64() >= misaligned.AtF64() {
		t.Errorf("aligned loss %.4g not < misaligned %.4g", aligned.AtF64(), misaligned.AtF64())
	}
}

// §V2: differentiable w.r.t. both view sets (with and without L2 norm).
func TestNTXentGradcheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 5))
	const n, d = 4, 3
	for _, norm := range []bool{false, true} {
		a := randMat(rng, n, d)
		b := randMat(rng, n, d)
		tape := autograd.NewTape()
		loss, err := nn.NTXentLoss(tape.Context(), a, b, 0.5, norm)
		if err != nil {
			t.Fatal(err)
		}
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		const h = 1e-6
		check := func(name string, p *tensor.Tensor) {
			g := tape.Grad(p)
			if g == nil {
				t.Fatalf("norm=%v %s: nil grad", norm, name)
			}
			for i := range n {
				for j := range d {
					o := p.AtF64(i, j)
					p.SetF64(o+h, i, j)
					lp, _ := nn.NTXentLoss(backend.NewContext(), a, b, 0.5, norm)
					p.SetF64(o-h, i, j)
					lm, _ := nn.NTXentLoss(backend.NewContext(), a, b, 0.5, norm)
					p.SetF64(o, i, j)
					if num, ana := (lp.AtF64()-lm.AtF64())/(2*h), g.AtF64(i, j); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
						t.Errorf("norm=%v %s grad[%d,%d]: numeric %.8g vs analytic %.8g", norm, name, i, j, num, ana)
					}
				}
			}
		}
		check("a", a)
		check("b", b)
	}
}

func TestNTXentErrors(t *testing.T) {
	a := toT(nceA)
	if _, err := nn.NTXentLoss(backend.NewContext(), a, toT([][]float64{{1, 2}}), 0.5, true); err == nil {
		t.Error("shape mismatch: want error")
	}
	if _, err := nn.NTXentLoss(backend.NewContext(), a, toT(nceB), 0, true); err == nil {
		t.Error("τ=0: want error")
	}
	if _, err := nn.NTXentLoss(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{3}), tensor.New(tensor.F64, tensor.Shape{3}), 0.5, true); err == nil {
		t.Error("rank-1 input: want error")
	}
}

func ExampleNTXentLoss() {
	// Two identical, well-separated views ⇒ near-zero NT-Xent loss.
	emb := toT([][]float64{{3, 0, 0}, {0, 3, 0}, {0, 0, 3}})
	loss, _ := nn.NTXentLoss(backend.NewContext(), emb, emb, 0.05, true)
	fmt.Printf("%.4f\n", loss.AtF64())
	// Output:
	// 0.0000
}
