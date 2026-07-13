package nlp_test

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// mockLM is a deterministic causal model: the token predicted after position j is the sum
// of the prefix seq[:j+1] modulo vocab — a genuine dependency on the whole prefix.
func mockLM(vocab int) nlp.JacobiStep {
	return func(seq []int) []int {
		out := make([]int, len(seq))
		sum := 0
		for j, t := range seq {
			sum += t
			out[j] = ((sum % vocab) + vocab) % vocab
		}
		return out
	}
}

// greedyRef decodes genLen tokens sequentially (the ground truth Jacobi must match).
func greedyRef(step nlp.JacobiStep, prompt []int, genLen int) []int {
	y := make([]int, 0, genLen)
	for i := 0; i < genLen; i++ {
		full := append(append([]int(nil), prompt...), y...)
		preds := step(full)
		y = append(y, preds[len(full)-1])
	}
	return y
}

// §V16 tier-1 (arXiv:2305.10427, lossless equivalence): the Jacobi fixed point equals the
// sequential greedy autoregressive decode.
func TestJacobiEqualsGreedy(t *testing.T) {
	step := mockLM(7)
	prompt := []int{3, 1, 4}
	const genLen = 8
	want := greedyRef(step, prompt, genLen)
	got, _ := nlp.JacobiDecode(step, prompt, make([]int, genLen), 100)
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("Jacobi[%d]=%d != greedy %d (got %v want %v)", i, got[i], want[i], got, want)
		}
	}
}

// §V16 tier-1 (convergence): Jacobi converges in at most m (+1 to detect no change)
// parallel iterations — the front-to-back locking bound.
func TestJacobiConvergesInGenLen(t *testing.T) {
	step := mockLM(11)
	prompt := []int{2, 5}
	const genLen = 10
	_, iters := nlp.JacobiDecode(step, prompt, make([]int, genLen), 100)
	if iters < 1 || iters > genLen+1 {
		t.Errorf("iterations %d, want in [1, %d]", iters, genLen+1)
	}
}

// §V16 tier-1 (fixed point): applying the model once more to the result reproduces it —
// y_i = argmax p(·|prompt, y_<i) for all i.
func TestJacobiFixedPoint(t *testing.T) {
	step := mockLM(9)
	prompt := []int{1, 2, 3}
	gen, _ := nlp.JacobiDecode(step, prompt, make([]int, 6), 100)
	full := append(append([]int(nil), prompt...), gen...)
	preds := step(full)
	for i := range gen {
		if preds[len(prompt)+i-1] != gen[i] {
			t.Errorf("not a fixed point at %d: pred %d != %d", i, preds[len(prompt)+i-1], gen[i])
		}
	}
}

// §V2: the initial guess changes only the iteration count, never the converged result.
func TestJacobiInitIndependent(t *testing.T) {
	step := mockLM(6)
	prompt := []int{4, 4}
	const genLen = 7
	fromZero, _ := nlp.JacobiDecode(step, prompt, make([]int, genLen), 100)
	other := make([]int, genLen)
	for i := range other {
		other[i] = 5 // a different starting guess
	}
	fromOther, _ := nlp.JacobiDecode(step, prompt, other, 100)
	for i := range fromZero {
		if fromZero[i] != fromOther[i] {
			t.Errorf("init-dependent result at %d: %d vs %d", i, fromZero[i], fromOther[i])
		}
	}
}

func TestJacobiEdges(t *testing.T) {
	step := mockLM(5)
	if gen, iters := nlp.JacobiDecode(step, []int{1}, nil, 100); len(gen) != 0 || iters != 0 {
		t.Errorf("genLen 0: want ([],0), got (%v,%d)", gen, iters)
	}
	if _, iters := nlp.JacobiDecode(step, []int{1}, make([]int, 3), 0); iters != 0 {
		t.Errorf("maxIters 0: want 0 iters, got %d", iters)
	}
}

func ExampleJacobiDecode() {
	// A toy model: the next token is the running prefix-sum mod 5. Jacobi decoding recovers
	// the same sequence as sequential greedy, in a few parallel iterations.
	step := mockLM(5)
	gen, iters := nlp.JacobiDecode(step, []int{1, 2}, make([]int, 4), 100)
	fmt.Println(gen, "in", iters, "iters ==", greedyRef(step, []int{1, 2}, 4))
	// Output:
	// [3 1 2 4] in 5 iters == [3 1 2 4]
}
