package nlp_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

func attends(mask *tensor.Tensor, i, j int) bool { return mask.AtF64(i, j) == 0 }
func blocked(mask *tensor.Tensor, i, j int) bool { return math.IsInf(mask.AtF64(i, j), -1) }

// §V16 tier-1 (arXiv:2401.10774 §2.3): a linear chain (each node's parent is the previous
// node) reproduces the ordinary causal mask, and depths are 0,1,2,…
func TestMedusaTreeMaskChainIsCausal(t *testing.T) {
	mask, depth := nlp.MedusaTreeMask([]int{-1, 0, 1, 2})
	for i := range 4 {
		for j := range 4 {
			if j <= i && !attends(mask, i, j) {
				t.Errorf("chain: %d should attend %d", i, j)
			}
			if j > i && !blocked(mask, i, j) {
				t.Errorf("chain: %d should not attend future %d", i, j)
			}
		}
		if depth[i] != i {
			t.Errorf("chain depth[%d]=%d, want %d", i, depth[i], i)
		}
	}
}

// §V16 tier-1 (ancestor-only attention): in a branching tree each node attends exactly its
// ancestors and itself — never a sibling branch — and depth is the tree depth.
func TestMedusaTreeMaskBranches(t *testing.T) {
	//        0
	//       / \
	//      1   2
	//      |   |
	//      3   4
	mask, depth := nlp.MedusaTreeMask([]int{-1, 0, 0, 1, 2})
	wantAttend := map[int][]int{
		0: {0},
		1: {0, 1},
		2: {0, 2},
		3: {0, 1, 3},
		4: {0, 2, 4},
	}
	for i := 0; i < 5; i++ {
		want := map[int]bool{}
		for _, a := range wantAttend[i] {
			want[a] = true
		}
		for j := 0; j < 5; j++ {
			if want[j] && !attends(mask, i, j) {
				t.Errorf("node %d should attend ancestor %d", i, j)
			}
			if !want[j] && !blocked(mask, i, j) {
				t.Errorf("node %d must NOT attend %d (sibling branch)", i, j)
			}
		}
	}
	if got := depth; got[0] != 0 || got[1] != 1 || got[2] != 1 || got[3] != 2 || got[4] != 2 {
		t.Errorf("depths = %v, want [0 1 1 2 2]", got)
	}
}

func TestMedusaTreeMaskPanic(t *testing.T) {
	for _, bad := range [][]int{{-1, 1}, {0}, {-1, -2}} { // parent≥i / root not −1 / parent<−1
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("parent %v: want panic", bad)
				}
			}()
			nlp.MedusaTreeMask(bad)
		}()
	}
}

// §V16 tier-1 (paper CONFIRMED, arXiv:2401.10774 §2.3.1): accept iff
// p(token) > min(ε, δ·exp(−H(p))), H in nats.
func TestTypicalAcceptanceFormula(t *testing.T) {
	probs := []float64{0.5, 0.3, 0.2}
	const eps, delta = 0.3, 0.09
	var h float64
	for _, v := range probs {
		h -= v * math.Log(v)
	}
	thresh := math.Min(eps, delta*math.Exp(-h))
	for tok := range probs {
		want := probs[tok] > thresh
		if got := nlp.TypicalAcceptance(probs, tok, eps, delta); got != want {
			t.Errorf("token %d: got %v want %v (p=%g thresh=%g)", tok, got, want, probs[tok], thresh)
		}
	}
}

// §V2: a confident (peaked) token is accepted; a deep-tail token is rejected.
func TestTypicalAcceptancePeaked(t *testing.T) {
	probs := []float64{0.94, 0.05, 0.01}
	if !nlp.TypicalAcceptance(probs, 0, nlp.MedusaEpsilon, nlp.MedusaDelta) {
		t.Error("confident top token should be accepted")
	}
	if nlp.TypicalAcceptance(probs, 2, nlp.MedusaEpsilon, nlp.MedusaDelta) {
		t.Error("deep-tail token (0.01) should be rejected")
	}
}

// §V2 (THE entropy-adaptive behavior): the SAME token probability (0.05) is accepted when
// the model is uncertain (high entropy ⇒ low threshold) but rejected when the model is
// confident (low entropy ⇒ threshold near δ). This is the defining property of typical
// acceptance vs a fixed probability floor.
func TestTypicalAcceptanceEntropyEffect(t *testing.T) {
	highEntropy := make([]float64, 20) // uniform: H=ln20≈3.0 ⇒ threshold≈0.0045
	for i := range highEntropy {
		highEntropy[i] = 0.05
	}
	lowEntropy := []float64{0.05, 0.95} // H≈0.20 ⇒ threshold≈0.074
	if !nlp.TypicalAcceptance(highEntropy, 0, nlp.MedusaEpsilon, nlp.MedusaDelta) {
		t.Error("p=0.05 should be accepted under high entropy (low threshold)")
	}
	if nlp.TypicalAcceptance(lowEntropy, 0, nlp.MedusaEpsilon, nlp.MedusaDelta) {
		t.Error("p=0.05 should be rejected under low entropy (threshold≈0.074)")
	}
}

func TestTypicalAcceptancePanic(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("token out of range: want panic")
		}
	}()
	nlp.TypicalAcceptance([]float64{0.5, 0.5}, 2, 0.3, 0.09)
}

func ExampleMedusaTreeMask() {
	// root 0 with two children 1,2; token 2 attends its ancestor 0 and itself, but NOT the
	// sibling branch 1.
	mask, depth := nlp.MedusaTreeMask([]int{-1, 0, 0})
	fmt.Printf("2→0:%v 2→1:%v depth:%v\n", mask.AtF64(2, 0) == 0, mask.AtF64(2, 1) == 0, depth)
	// Output:
	// 2→0:true 2→1:false depth:[0 1 1]
}

func ExampleTypicalAcceptance() {
	// Under a confident distribution the top token clears the threshold; the 0.02 tail does
	// not.
	probs := []float64{0.9, 0.08, 0.02}
	fmt.Println(nlp.TypicalAcceptance(probs, 0, nlp.MedusaEpsilon, nlp.MedusaDelta),
		nlp.TypicalAcceptance(probs, 2, nlp.MedusaEpsilon, nlp.MedusaDelta))
	// Output:
	// true false
}
