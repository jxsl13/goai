package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
)

// Jacobi (parallel) decoding (Santilli, Severino, Postolache, Maiorca, Mancusi, Marin &
// Rodolà 2023, "Accelerating Transformer Inference for Translation via Parallel Decoding",
// ACL 2023, arXiv:2305.10427; the same fixed-point iteration underlies Lookahead Decoding,
// Fu et al. 2024). Greedy autoregressive decoding solves the triangular system
// y_i = argmax p(y_i | y_<i, x) one token at a time. Jacobi decoding solves the SAME system
// by fixed-point iteration: from an initial guess it updates EVERY position in parallel each
// step — y_i ← argmax p(y_i | y_<i, x) using the previous iteration's prefix — and repeats
// until nothing changes. The fixed point is exactly the greedy sequence (lossless), but it
// is reached in far fewer than m sequential steps because several tokens can become correct
// per parallel pass (worst case m steps, one token "locking" from the front each iteration).

// JacobiStep is a model's parallel greedy pass: given the full token sequence it returns,
// for every position j, the argmax next token the model predicts from the prefix seq[:j+1]
// (len(result) == len(seq)). For a causal LM this is a single forward over the whole
// sequence — the operation Jacobi decoding batches across all positions at once.
type JacobiStep func(seq []int) []int

// JacobiDecode runs Jacobi parallel decoding for len(init) continuation tokens after prompt,
// starting from the guess init and iterating (at most maxIters times) until the generated
// block stops changing. It returns the generated tokens — equal to sequential greedy
// decoding of the same model — and the number of parallel iterations used. The initial
// guess only affects the iteration count, never the converged result.
func JacobiDecode(step JacobiStep, prompt, init []int, maxIters int) (gen []int, iters int) {
	pl, genLen := len(prompt), len(init)
	y := append([]int(nil), init...)
	if genLen == 0 || maxIters <= 0 {
		return y, 0
	}
	full := make([]int, pl+genLen)
	copy(full, prompt)
	for iters = 1; iters <= maxIters; iters++ {
		copy(full[pl:], y)
		preds := step(full) // argmax next token predicted after each position
		changed := false
		newY := make([]int, genLen)
		for i := range genLen {
			// generation token i is the prediction made AFTER position pl+i−1, i.e. from
			// the prefix prompt + y[:i].
			newY[i] = preds[pl+i-1]
			if newY[i] != y[i] {
				changed = true
			}
		}
		y = newY
		if !changed { // fixed point: y_i = argmax p(·|prompt,y_<i) for all i = greedy
			return y, iters
		}
	}
	return y, maxIters
}

// JacobiGenerate runs Jacobi parallel decoding on the model itself (§T441, the
// generation entry point around JacobiDecode): each iteration is ONE full forward
// over prompt+guess (the model's parallel greedy pass — argmax per position), so
// the result is exactly the model's sequential greedy generation, reached in
// iters ≤ genLen parallel passes (often far fewer once tokens lock from the
// front). Returns prompt+generated, the iteration count, and the model's first
// forward error if one occurred. genLen is capped by the context window.
func (g *GPT) JacobiGenerate(prompt []int, genLen, maxIters int) ([]int, int, error) {
	if len(prompt) == 0 {
		return nil, 0, fmt.Errorf("nlp: JacobiGenerate needs a non-empty prompt")
	}
	if len(prompt)+genLen > g.Config.Ctx {
		genLen = g.Config.Ctx - len(prompt)
	}
	if genLen <= 0 {
		return append([]int(nil), prompt...), 0, nil
	}
	ctx := backend.NewContext()
	var stepErr error
	step := func(seq []int) []int {
		preds := make([]int, len(seq))
		if stepErr != nil {
			return preds
		}
		logits, err := g.Forward(ctx, seq)
		if err != nil {
			stepErr = err
			return preds
		}
		v := logits.Shape()[1]
		for j := range preds {
			best, bestV := 0, logits.AtF64(j, 0)
			for t := 1; t < v; t++ {
				if x := logits.AtF64(j, t); x > bestV {
					best, bestV = t, x
				}
			}
			preds[j] = best
		}
		return preds
	}
	gen, iters := JacobiDecode(step, prompt, make([]int, genLen), maxIters)
	if stepErr != nil {
		return nil, 0, stepErr
	}
	return append(append([]int(nil), prompt...), gen...), iters, nil
}
