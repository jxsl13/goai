package nlp

import "math/rand/v2"

// Fill-in-the-Middle (FIM) training-data transformation (Bavarian, Jun, Tezak, Schulman,
// McLeavey, Tworek & Chen 2022, "Efficient Training of Language Models to Fill in the
// Middle", arXiv:2207.14255; the recipe behind Codex/StarCoder/Code Llama/DeepSeek-Coder
// infilling). A left-to-right causal LM cannot ordinarily condition on text AFTER the cursor.
// FIM teaches it to by REORDERING a fraction of training documents: split a document into
// prefix, middle and suffix, then move the middle to the end behind sentinel tokens so the
// model predicts it from the surrounding context. Applied to some documents it adds infilling
// "for free" without hurting normal generation, and the reordering is losslessly reversible.

// FIMSentinels are the special (reserved, non-document) token ids that mark the FIM sections.
type FIMSentinels struct {
	Pre int // <PRE> sentinel, before the prefix
	Suf int // <SUF> sentinel, before the suffix
	Mid int // <MID> sentinel, before the middle
}

// FIMTransform reorders tokens for FIM training. It splits the document at two random points
// into prefix/middle/suffix (document = prefix+middle+suffix) and emits either the PSM layout
//
//	<PRE> prefix <SUF> suffix <MID> middle
//
// or, when spm is true, the SPM layout
//
//	<PRE> <SUF> suffix <MID> prefix middle
//
// so in both cases the middle is produced last. rng draws the two split points; the sentinels
// must not appear among the document tokens. The original tokens are not modified.
func FIMTransform(tokens []int, s FIMSentinels, spm bool, rng *rand.Rand) []int {
	n := len(tokens)
	a, b := rng.IntN(n+1), rng.IntN(n+1)
	if a > b {
		a, b = b, a
	}
	prefix, middle, suffix := tokens[:a], tokens[a:b], tokens[b:]
	out := make([]int, 0, n+3)
	if spm {
		out = append(out, s.Pre, s.Suf)
		out = append(out, suffix...)
		out = append(out, s.Mid)
		out = append(out, prefix...)
		out = append(out, middle...)
	} else {
		out = append(out, s.Pre)
		out = append(out, prefix...)
		out = append(out, s.Suf)
		out = append(out, suffix...)
		out = append(out, s.Mid)
		out = append(out, middle...)
	}
	return out
}

// FIMReconstruct recovers the original document (prefix+middle+suffix) from a FIM-transformed
// sequence, returning ok=false if it is not a well-formed FIM sequence for the given mode.
// It is the exact inverse of FIMTransform, so FIMReconstruct(FIMTransform(doc)) == doc.
func FIMReconstruct(fim []int, s FIMSentinels, spm bool) (doc []int, ok bool) {
	find := func(tok, from int) int {
		for i := from; i < len(fim); i++ {
			if fim[i] == tok {
				return i
			}
		}
		return -1
	}
	if spm {
		// <PRE> <SUF> suffix <MID> prefix middle
		if len(fim) < 3 || fim[0] != s.Pre || fim[1] != s.Suf {
			return nil, false
		}
		mid := find(s.Mid, 2)
		if mid < 0 {
			return nil, false
		}
		suffix, prefixMiddle := fim[2:mid], fim[mid+1:]
		doc = append(append([]int(nil), prefixMiddle...), suffix...)
		return doc, true
	}
	// <PRE> prefix <SUF> suffix <MID> middle
	if len(fim) < 3 || fim[0] != s.Pre {
		return nil, false
	}
	suf := find(s.Suf, 1)
	if suf < 0 {
		return nil, false
	}
	mid := find(s.Mid, suf+1)
	if mid < 0 {
		return nil, false
	}
	prefix, suffix, middle := fim[1:suf], fim[suf+1:mid], fim[mid+1:]
	doc = append(append(append([]int(nil), prefix...), middle...), suffix...)
	return doc, true
}
