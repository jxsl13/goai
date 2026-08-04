package nlp

import "math/rand/v2"

// BERT masked language modeling objective (Devlin, Chang, Lee & Toutanova 2019, "BERT:
// Pre-training of Deep Bidirectional Transformers for Language Understanding", NAACL,
// arXiv:1810.04805, §3.1). Unlike the seq2seq span-corruption objective (T5, §R213) and the
// causal FIM reorder (§R209), MLM is the ENCODER pretraining objective: a random 15% of the token
// positions are selected for prediction and corrupted IN PLACE, and the model must recover the
// originals from the surrounding bidirectional context. Crucially the selected positions are not
// always replaced by [MASK]: 80% become [MASK], 10% a random token, 10% are left unchanged — the
// 80/10/10 split narrows the pretrain/finetune mismatch, since [MASK] never appears downstream.

// MLMIgnoreLabel marks a position that is NOT predicted (excluded from the loss) — the same −100
// convention as HuggingFace / PyTorch's ignore_index. Real token ids are non-negative, so it never
// collides with an actual label.
const MLMIgnoreLabel = -100

// MLMMask applies BERT MLM corruption to tokens. Each position is selected with probability
// maskProb (BERT: 0.15); a selected position is replaced by maskID with prob 0.8, by a uniform
// random id in [0,vocabSize) with prob 0.1, and left unchanged with prob 0.1. It returns the
// corrupted input and the labels: labels[i] is the ORIGINAL token at a selected position and
// MLMIgnoreLabel elsewhere (so the loss is computed only on the 15%). Exclude special tokens
// upstream. The original tokens are not modified; the pair is losslessly reversible
// (MLMReconstruct).
func MLMMask(tokens []int, maskProb float64, maskID, vocabSize int, rng *rand.Rand) (input, labels []int) {
	return MLMMaskExcluding(tokens, maskProb, maskID, vocabSize, nil, rng)
}

// MLMMaskExcluding is MLMMask with special-token protection: any position whose token is in
// specialIDs (e.g. [CLS], [SEP], padding) is NEVER selected for masking — its label stays
// MLMIgnoreLabel and its input is left unchanged — matching BERT/HuggingFace, which build a
// special-tokens mask so those positions never contribute to the MLM loss. The ~maskProb rate then
// applies only to the ordinary tokens. Special positions consume no randomness, so with a nil/empty
// specialIDs this reduces to MLMMask exactly (same RNG sequence). The pair is still losslessly
// reversible with MLMReconstruct.
func MLMMaskExcluding(tokens []int, maskProb float64, maskID, vocabSize int, specialIDs []int, rng *rand.Rand) (input, labels []int) {
	input = make([]int, len(tokens))
	copy(input, tokens)
	labels = make([]int, len(tokens))
	for i := range labels {
		labels[i] = MLMIgnoreLabel
	}
	var special map[int]struct{}
	if len(specialIDs) > 0 {
		special = make(map[int]struct{}, len(specialIDs))
		for _, id := range specialIDs {
			special[id] = struct{}{}
		}
	}
	for i, tok := range tokens {
		//perfscan:ignore PS3007 tiny specialIDs set, RNG-dominated dataprep loop
		if _, isSpecial := special[tok]; isSpecial {
			continue // protected: no selection, no RNG draw
		}
		if rng.Float64() >= maskProb {
			continue // not selected for prediction
		}
		labels[i] = tokens[i] // the target is the original token
		switch r := rng.Float64(); {
		case r < 0.8:
			input[i] = maskID // 80% → [MASK]
		case r < 0.9:
			input[i] = rng.IntN(vocabSize) // 10% → random token
		default:
			// 10% → unchanged (input[i] already equals tokens[i])
		}
	}
	return input, labels
}

// MLMReconstruct recovers the original document from an (input, labels) MLM pair: a predicted
// position (labels[i] ≠ MLMIgnoreLabel) takes its label — the original token — and every other
// position keeps the (unmodified) input token. It is the exact inverse of MLMMask. ok is false if
// the lengths differ.
func MLMReconstruct(input, labels []int) (doc []int, ok bool) {
	if len(input) != len(labels) {
		return nil, false
	}
	doc = make([]int, len(input))
	for i := range input {
		if labels[i] != MLMIgnoreLabel {
			doc[i] = labels[i]
		} else {
			doc[i] = input[i]
		}
	}
	return doc, true
}
