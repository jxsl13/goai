package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// plRef = mean_b Σ_{k=0}^{K-2} [logsumexp(s[b,k:]) − s[b,k]].
func plRef(rewards [][]float64) float64 {
	b, k := len(rewards), len(rewards[0])
	var total float64
	for i := 0; i < b; i++ {
		for r := 0; r < k-1; r++ {
			mx := math.Inf(-1)
			for j := r; j < k; j++ {
				mx = math.Max(mx, rewards[i][j])
			}
			var sm float64
			for j := r; j < k; j++ {
				sm += math.Exp(rewards[i][j] - mx)
			}
			total += mx + math.Log(sm) - rewards[i][r]
		}
	}
	return total / float64(b)
}

// §V16 tier-1 (paper CONFIRMED, ListMLE): matches the sequential-softmax listwise NLL.
func TestPlackettLuceFormula(t *testing.T) {
	rw := [][]float64{{2.0, 0.5, -1.0, 0.2}, {1.0, 3.0, 0.0, -0.5}}
	got, err := nn.PlackettLuceLoss(backend.NewContext(), toT(rw))
	if err != nil {
		t.Fatal(err)
	}
	if want := plRef(rw); math.Abs(got.AtF64()-want) > 1e-9 {
		t.Errorf("loss=%.12g want %.12g", got.AtF64(), want)
	}
}

// §V16 tier-1 (K=2 reduction): for two items the Plackett-Luce loss equals the pairwise
// Bradley-Terry loss with the first item chosen.
func TestPlackettLuceReducesToBradleyTerry(t *testing.T) {
	rw := [][]float64{{1.5, 0.3}, {-0.7, 0.9}, {2.0, 2.0}}
	pl, _ := nn.PlackettLuceLoss(backend.NewContext(), toT(rw))
	chosen, rejected := rvec(1.5, -0.7, 2.0), rvec(0.3, 0.9, 2.0)
	bt, _ := nn.BradleyTerryLoss(backend.NewContext(), chosen, rejected, 0)
	if math.Abs(pl.AtF64()-bt.AtF64()) > 1e-9 {
		t.Errorf("K=2 Plackett-Luce %g != Bradley-Terry %g", pl.AtF64(), bt.AtF64())
	}
}

// §V16 tier-1 (closed form): uniform scores give L = log(K!).
func TestPlackettLuceUniformIsLogKFactorial(t *testing.T) {
	const K = 5
	rw := make([][]float64, 3)
	for i := range rw {
		rw[i] = make([]float64, K) // all zero ⇒ uniform
	}
	got, _ := nn.PlackettLuceLoss(backend.NewContext(), toT(rw))
	logKFact := 0.0
	for i := 2; i <= K; i++ {
		logKFact += math.Log(float64(i))
	}
	if math.Abs(got.AtF64()-logKFact) > 1e-9 {
		t.Errorf("uniform loss=%g, want log(%d!)=%g", got.AtF64(), K, logKFact)
	}
}

// §V2: a correct (descending) ranking has a lower loss than a reversed one for the same
// scores — the loss rewards ordering the reward model got right.
func TestPlackettLuceRewardsCorrectOrder(t *testing.T) {
	good, _ := nn.PlackettLuceLoss(backend.NewContext(), toT([][]float64{{3, 2, 1, 0}}))
	bad, _ := nn.PlackettLuceLoss(backend.NewContext(), toT([][]float64{{0, 1, 2, 3}}))
	if !(good.AtF64() < bad.AtF64()) {
		t.Errorf("descending loss %g not < reversed %g", good.AtF64(), bad.AtF64())
	}
}

// §V2: differentiable w.r.t. the rewards.
func TestPlackettLuceGradcheck(t *testing.T) {
	rw := toT([][]float64{{0.7, -0.3, 1.1, 0.2}, {1.0, 0.5, -0.4, 0.9}})
	tape := autograd.NewTape()
	loss, err := nn.PlackettLuceLoss(tape.Context(), rw)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(rw)
	const h = 1e-6
	fwd := func() float64 {
		l, _ := nn.PlackettLuceLoss(backend.NewContext(), rw)
		return l.AtF64()
	}
	for idx := range rw.Numel() {
		c := tensor.Unravel(idx, rw.Shape())
		o := rw.AtF64(c...)
		rw.SetF64(o+h, c...)
		lp := fwd()
		rw.SetF64(o-h, c...)
		lm := fwd()
		rw.SetF64(o, c...)
		if num, ana := (lp-lm)/(2*h), g.AtF64(c...); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
			t.Errorf("grad%v: numeric %.8g vs analytic %.8g", c, num, ana)
		}
	}
}

func TestPlackettLuceErrors(t *testing.T) {
	r := func(shape ...int) *tensor.Tensor { return tensor.New(tensor.F64, tensor.Shape(shape)) }
	if _, err := nn.PlackettLuceLoss(backend.NewContext(), r(4)); err == nil {
		t.Error("rank-1: want error")
	}
	if _, err := nn.PlackettLuceLoss(backend.NewContext(), r(3, 1)); err == nil {
		t.Error("K<2: want error")
	}
}

func ExamplePlackettLuceLoss() {
	// One ranking of 3 items whose scores are already in the right (descending) order gets a
	// small loss.
	loss, _ := nn.PlackettLuceLoss(backend.NewContext(), toT([][]float64{{6, 0, -6}}))
	fmt.Printf("%.3f\n", loss.AtF64())
	// Output:
	// 0.005
}
