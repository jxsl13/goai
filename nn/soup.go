package nn

import (
	"fmt"
	"sort"

	"github.com/jxsl13/goai/tensor"
)

// Model soups (Wortsman, Ilharco, Gadre, Roelofs, Gontijo-Lopes, Morcos, Namkoong, Farhadi,
// Carmon, Kornblith & Schmidt 2022, "Model soups: averaging weights of multiple fine-tuned models
// improves accuracy without increasing inference time", ICML, arXiv:2203.05482). Averaging the
// WEIGHTS of several models fine-tuned from the same pretrained initialization (with different
// hyperparameters/seeds/augmentations) often beats the single best model — at NO extra inference
// cost, since the soup is one model. Unlike SWA/EMA (nn.SWA/nn.EMA), which average snapshots along
// ONE training run, a soup averages INDEPENDENT fine-tunes; unlike TIES/DARE it is a plain mean.
//
// UniformSoup averages all candidates; GreedySoup (Algorithm 1) adds candidates one at a time in
// decreasing validation-accuracy order, keeping each only when it does not lower the soup's
// held-out accuracy — so the greedy soup never underperforms the best individual model on that set.

// UniformSoup returns the unweighted elementwise average of every model's parameters. Each entry
// of models is a parallel list of parameter tensors with matching shapes/dtypes.
func UniformSoup(models [][]*tensor.Tensor) ([]*tensor.Tensor, error) {
	if err := checkModels(models); err != nil {
		return nil, err
	}
	idx := make([]int, len(models))
	for i := range idx {
		idx[i] = i
	}
	return averageModels(models, idx), nil
}

// GreedySoup builds a soup by greedy selection (Algorithm 1). It sorts the candidate models by
// DECREASING valAcc, then walks them in that order, adding each to the ingredient set only if the
// uniform average of the ingredients-so-far PLUS that model scores ≥ the current ingredients'
// average under eval (the held-out accuracy of a soup, supplied by the caller). The first (best)
// model is always taken. It returns the averaged soup weights and the accepted ORIGINAL model
// indices in the order they were added (always non-empty). eval is called on candidate averaged
// weights; higher is better.
func GreedySoup(models [][]*tensor.Tensor, valAcc []float64, eval func([]*tensor.Tensor) float64) (soup []*tensor.Tensor, chosen []int, err error) {
	if err := checkModels(models); err != nil {
		return nil, nil, err
	}
	if len(valAcc) != len(models) {
		return nil, nil, fmt.Errorf("nn: GreedySoup needs one valAcc per model, got %d for %d models", len(valAcc), len(models))
	}
	if eval == nil {
		return nil, nil, fmt.Errorf("nn: GreedySoup needs a non-nil eval")
	}
	// order = model indices sorted by decreasing validation accuracy (stable on ties).
	order := make([]int, len(models))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return valAcc[order[a]] > valAcc[order[b]] })

	best := eval(averageModels(models, order[:1])) // the top model is always the first ingredient
	chosen = []int{order[0]}
	for _, m := range order[1:] {
		trial := append(append([]int(nil), chosen...), m)
		if acc := eval(averageModels(models, trial)); acc >= best {
			chosen = trial
			best = acc
		}
	}
	return averageModels(models, chosen), chosen, nil
}

// checkModels validates a non-empty model list with a consistent parameter layout.
func checkModels(models [][]*tensor.Tensor) error {
	if len(models) == 0 {
		return fmt.Errorf("nn: model soup needs ≥1 model")
	}
	ref := models[0]
	if len(ref) == 0 {
		return fmt.Errorf("nn: model soup model 0 has no parameters")
	}
	for t, m := range models {
		if len(m) != len(ref) {
			return fmt.Errorf("nn: model soup model %d has %d params, model 0 has %d", t, len(m), len(ref))
		}
		for i := range ref {
			if !m[i].Shape().Equal(ref[i].Shape()) {
				return fmt.Errorf("nn: model soup model %d param %d shape %v != %v", t, i, m[i].Shape(), ref[i].Shape())
			}
			if m[i].Dtype() != ref[i].Dtype() {
				return fmt.Errorf("nn: model soup model %d param %d dtype %v != %v", t, i, m[i].Dtype(), ref[i].Dtype())
			}
		}
	}
	return nil
}

// averageModels returns the uniform elementwise mean over the models at the given indices as fresh
// tensors matching the parameter layout (float64 accumulation, §V10).
func averageModels(models [][]*tensor.Tensor, idxs []int) []*tensor.Tensor {
	ref := models[idxs[0]]
	out := make([]*tensor.Tensor, len(ref))
	inv := 1 / float64(len(idxs))
	for i, p := range ref {
		acc := make([]float64, p.Numel())
		// Typed contiguous fast path (§base-perf; model-merge sibling of SLERP/DARE/TIES):
		// accumulate each contiguous model directly from its backing []T — no per-element
		// Unravel/AtF64 dispatch. Same model-then-element order (so the float sum is
		// unchanged) and the same *inv rescale as the generic walk → bit-identical.
		for _, t := range idxs {
			mp := models[t][i]
			if mf := flatF64(mp); mf != nil {
				for e := range acc {
					acc[e] += mf[e]
				}
			} else if mf := flatF32(mp); mf != nil {
				for e := range acc {
					acc[e] += float64(mf[e])
				}
			} else {
				for e := range mp.Numel() {
					acc[e] += mp.AtF64(tensor.Unravel(e, mp.Shape())...)
				}
			}
		}
		res := tensor.New(p.Dtype(), p.Shape())
		if rf := flatF64(res); rf != nil {
			for e := range rf {
				rf[e] = acc[e] * inv
			}
		} else if rf := flatF32(res); rf != nil {
			for e := range rf {
				rf[e] = float32(acc[e] * inv)
			}
		} else {
			for e := range res.Numel() {
				res.SetF64(acc[e]*inv, tensor.Unravel(e, p.Shape())...)
			}
		}
		out[i] = res
	}
	return out
}
