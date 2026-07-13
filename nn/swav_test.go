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

// swavRef independently computes the swapped-prediction loss:
// mean_b(−Σ_k codeS·logsoftmax(scoresT/τ)) + mean_b(−Σ_k codeT·logsoftmax(scoresS/τ)).
func swavRef(scoresT, scoresS, codeT, codeS [][]float64, temp float64) float64 {
	term := func(logits, target [][]float64) float64 {
		b, k := len(logits), len(logits[0])
		var total float64
		for i := 0; i < b; i++ {
			mx := math.Inf(-1)
			for j := 0; j < k; j++ {
				if v := logits[i][j] / temp; v > mx {
					mx = v
				}
			}
			var sum float64
			for j := 0; j < k; j++ {
				sum += math.Exp(logits[i][j]/temp - mx)
			}
			lse := mx + math.Log(sum)
			for j := 0; j < k; j++ {
				total -= target[i][j] * (logits[i][j]/temp - lse)
			}
		}
		return total / float64(b)
	}
	return term(scoresT, codeS) + term(scoresS, codeT)
}

// §V16 tier-1 (paper CONFIRMED, arXiv:2006.09882 Eq.1/6): matches the independent
// swapped-prediction reference.
func TestSwAVLossFormula(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2006))
	sT, sS := randMat(rng, 4, 3), randMat(rng, 4, 3)
	qT, qS := softmaxRows(randMat(rng, 4, 3)), softmaxRows(randMat(rng, 4, 3))
	got, err := nn.SwAVLoss(backend.NewContext(), sT, sS, qT, qS, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	want := swavRef(mat(sT), mat(sS), mat(qT), mat(qS), 0.1)
	if math.Abs(got.AtF64()-want) > 1e-9 {
		t.Errorf("loss=%.12g want %.12g", got.AtF64(), want)
	}
}

// §V16 tier-1 (closed form): uniform scores give a uniform prediction (log p = −ln K), and
// uniform codes (1/K) make each swapped term ln K, so L = 2·ln K.
func TestSwAVLossUniformClosedForm(t *testing.T) {
	const b, k = 3, 5
	scores := tensor.New(tensor.F64, tensor.Shape{b, k}) // all zeros ⇒ uniform softmax
	code := tensor.New(tensor.F64, tensor.Shape{b, k})
	for i := range b {
		for j := range k {
			code.SetF64(1.0/k, i, j)
		}
	}
	got, err := nn.SwAVLoss(backend.NewContext(), scores, scores, code, code, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if want := 2 * math.Log(k); math.Abs(got.AtF64()-want) > 1e-9 {
		t.Errorf("uniform loss=%g want 2·lnK=%g", got.AtF64(), want)
	}
}

// §V2 (the SWAPPED structure): each view predicts the OTHER view's code. With scoresT
// peaked at class 0 and scoresS peaked at class 1, the correctly-swapped codes
// (codeS=class0, codeT=class1) give a low loss; feeding the codes the wrong way round
// (self-prediction) gives a strictly higher loss — proving the cross-view pairing.
func TestSwAVLossSwappedStructure(t *testing.T) {
	sT := toT([][]float64{{5, 0, 0}}) // view T confident about class 0
	sS := toT([][]float64{{0, 5, 0}}) // view S confident about class 1
	c0 := toT([][]float64{{1, 0, 0}}) // one-hot class 0
	c1 := toT([][]float64{{0, 1, 0}}) // one-hot class 1
	// swapped: T predicts S's code (c1? no) — T should predict S's code; S is class1 ⇒ codeS
	// must be c1 for a good T-prediction... but T is peaked at 0. The aligned pairing is
	// codeT=c0 (matches sT), codeS=c1 (matches sS): then T predicts codeS=c1 (mismatch) —
	// that is the swap, deliberately cross. Compare against the self pairing.
	swapped, _ := nn.SwAVLoss(backend.NewContext(), sT, sS, c0, c1, 0.1) // T→codeS=c1, S→codeT=c0
	self, _ := nn.SwAVLoss(backend.NewContext(), sT, sS, c1, c0, 0.1)    // T→codeS=c0, S→codeT=c1
	// self pairing aligns each view with its OWN peak ⇒ lower loss; swapped is higher.
	if !(self.AtF64() < swapped.AtF64()) {
		t.Errorf("expected self-aligned %g < cross %g (codes enter swapped)", self.AtF64(), swapped.AtF64())
	}
}

// §V2: differentiable w.r.t. both score matrices (codes are fixed targets).
func TestSwAVLossGradcheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 2007))
	sT, sS := randMat(rng, 3, 4), randMat(rng, 3, 4)
	qT, qS := softmaxRows(randMat(rng, 3, 4)), softmaxRows(randMat(rng, 3, 4))
	tape := autograd.NewTape()
	loss, err := nn.SwAVLoss(tape.Context(), sT, sS, qT, qS, 0.2)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	const h = 1e-6
	fwd := func() float64 {
		l, _ := nn.SwAVLoss(backend.NewContext(), sT, sS, qT, qS, 0.2)
		return l.AtF64()
	}
	check := func(name string, p *tensor.Tensor) {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("%s: nil grad", name)
		}
		for idx := range p.Numel() {
			coord := tensor.Unravel(idx, p.Shape())
			o := p.AtF64(coord...)
			p.SetF64(o+h, coord...)
			lp := fwd()
			p.SetF64(o-h, coord...)
			lm := fwd()
			p.SetF64(o, coord...)
			if num, ana := (lp-lm)/(2*h), g.AtF64(coord...); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("%s grad%v: numeric %.8g vs analytic %.8g", name, coord, num, ana)
			}
		}
	}
	check("scoresT", sT)
	check("scoresS", sS)
}

// §V2 (end-to-end): the Sinkhorn codes feed the SwAV loss — the full pipeline this task
// completes. The loss is finite and positive.
func TestSwAVLossWithSinkhornCodes(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 2008))
	const b, k = 6, 4
	sT, sS := randMat(rng, b, k), randMat(rng, b, k)
	// codes = Sinkhorn equipartition assignment over prototypes (transpose to [K,B] as SwAV
	// does, uniform marginals), then back — here we just use the scores as the cost.
	qT, err := nn.Sinkhorn(sT, uniform(b), uniform(k), 0.05, 3)
	if err != nil {
		t.Fatal(err)
	}
	qS, _ := nn.Sinkhorn(sS, uniform(b), uniform(k), 0.05, 3)
	loss, err := nn.SwAVLoss(backend.NewContext(), sT, sS, qT, qS, nn.SwAVTemperature)
	if err != nil {
		t.Fatal(err)
	}
	if l := loss.AtF64(); math.IsNaN(l) || math.IsInf(l, 0) || l <= 0 {
		t.Errorf("SwAV loss over Sinkhorn codes = %g, want finite positive", l)
	}
}

func TestSwAVLossErrors(t *testing.T) {
	r := func(shape ...int) *tensor.Tensor { return tensor.New(tensor.F64, tensor.Shape(shape)) }
	if _, err := nn.SwAVLoss(backend.NewContext(), r(4, 3), r(4, 3), r(4, 3), r(4, 2), 0.1); err == nil {
		t.Error("shape mismatch: want error")
	}
	if _, err := nn.SwAVLoss(backend.NewContext(), r(4, 3, 1), r(4, 3, 1), r(4, 3, 1), r(4, 3, 1), 0.1); err == nil {
		t.Error("rank-3: want error")
	}
	if _, err := nn.SwAVLoss(backend.NewContext(), r(4, 3), r(4, 3), r(4, 3), r(4, 3), 0); err == nil {
		t.Error("temp=0: want error")
	}
}

// softmaxRows returns a row-stochastic version of m (each row a valid code distribution).
func softmaxRows(m *tensor.Tensor) *tensor.Tensor {
	b, k := m.Shape()[0], m.Shape()[1]
	out := tensor.New(tensor.F64, tensor.Shape{b, k})
	for i := range b {
		mx := math.Inf(-1)
		for j := range k {
			if v := m.AtF64(i, j); v > mx {
				mx = v
			}
		}
		var sum float64
		for j := range k {
			sum += math.Exp(m.AtF64(i, j) - mx)
		}
		for j := range k {
			out.SetF64(math.Exp(m.AtF64(i, j)-mx)/sum, i, j)
		}
	}
	return out
}

func ExampleSwAVLoss() {
	// Two augmentations of the same image both belong to cluster 0, so both codes are
	// [1,0]; each view confidently predicts the other's code and the swapped-prediction loss
	// is small.
	sT := toT([][]float64{{6, 0}})
	sS := toT([][]float64{{5, 1}})
	codeT := toT([][]float64{{1, 0}}) // both views assigned to cluster 0
	codeS := toT([][]float64{{1, 0}})
	loss, _ := nn.SwAVLoss(backend.NewContext(), sT, sS, codeT, codeS, 0.1)
	fmt.Println(loss.AtF64() < 0.01)
	// Output:
	// true
}
