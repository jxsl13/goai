package nlp

import (
	"encoding/json"
	"fmt"
	"sort"
)

// HuggingFace tokenizer.json interop for byte-level BPE (the "model":{"type":"BPE"} form
// used by GPT-2, Llama-3, Qwen, Mistral). Definitional source: the tokenizers library
// schema — verified by round-trip + fuzz (§V15). Only the BPE model core (vocab + merges,
// plus any added_tokens) is read; normalizer/pre_tokenizer/decoder sections are ignored
// (the tokenizer already applies GPT-2 pre-tokenization and the byte↔unicode map).

// hfTokJSON is the minimal tokenizer.json shape needed to rebuild a byte-level BPE.
type hfTokJSON struct {
	Model struct {
		Type   string          `json:"type"`
		Vocab  map[string]int  `json:"vocab"`  // byte-mapped token → id
		Merges json.RawMessage `json:"merges"` // ["l r", …] or [["l","r"], …]
	} `json:"model"`
	AddedTokens []addedToken `json:"added_tokens"`
}

// parseMerges accepts both tokenizer.json merge encodings: a list of "left right" strings
// (classic) or a list of ["left","right"] pairs (newer tokenizers), returning "l r" form.
func parseMerges(raw json.RawMessage) ([]string, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	var asStrings []string
	if err := json.Unmarshal(raw, &asStrings); err == nil {
		return asStrings, nil
	}
	var asPairs [][]string
	if err := json.Unmarshal(raw, &asPairs); err != nil {
		return nil, fmt.Errorf("nlp: BPE merges must be []string or [][2]string: %w", err)
	}
	out := make([]string, len(asPairs))
	for i, p := range asPairs {
		if len(p) != 2 {
			return nil, fmt.Errorf("nlp: BPE merge %d has %d elements, want 2", i, len(p))
		}
		out[i] = p[0] + " " + p[1]
	}
	return out, nil
}

// BPEFromJSON builds a byte-level BPETokenizer from HuggingFace tokenizer.json bytes. It
// errors on invalid JSON, a non-BPE model type, or an empty vocabulary.
func BPEFromJSON(data []byte, opts ...BPEOption) (*BPETokenizer, error) {
	var tj hfTokJSON
	if err := json.Unmarshal(data, &tj); err != nil {
		return nil, fmt.Errorf("nlp: parse tokenizer.json: %w", err)
	}
	if tj.Model.Type != "" && tj.Model.Type != "BPE" {
		return nil, fmt.Errorf("nlp: tokenizer.json model.type=%q, want BPE", tj.Model.Type)
	}
	// place each token at its id; added_tokens (special tokens) share the id space.
	maxID := -1
	for _, id := range tj.Model.Vocab {
		if id > maxID {
			maxID = id
		}
	}
	for _, a := range tj.AddedTokens {
		if a.ID > maxID {
			maxID = a.ID
		}
	}
	if maxID < 0 {
		return nil, fmt.Errorf("nlp: tokenizer.json has an empty vocabulary")
	}
	// hostile-input guard (§B47): the vocab slice is indexed by id, so an
	// implausibly large id (fuzz found 1e13) would allocate terabytes. Real
	// tokenizers have near-dense ids; allow generous slack, reject the rest.
	if nTok := len(tj.Model.Vocab) + len(tj.AddedTokens); maxID+1 > 4*nTok+1024 {
		return nil, fmt.Errorf("nlp: tokenizer.json max token id %d implausible for %d tokens", maxID, nTok)
	}
	vocab := make([]string, maxID+1)
	for tok, id := range tj.Model.Vocab {
		if id >= 0 {
			vocab[id] = tok
		}
	}
	for _, a := range tj.AddedTokens {
		if a.ID >= 0 {
			vocab[a.ID] = a.Content
		}
	}
	merges, err := parseMerges(tj.Model.Merges)
	if err != nil {
		return nil, err
	}
	t, err := NewBPE(vocab, merges, opts...)
	if err != nil {
		return nil, err
	}
	t.AddSpecialTokens(addedTokenSpecials(tj.AddedTokens))
	return t, nil
}

// addedTokenSpecials turns a tokenizer.json added_tokens list into the EncodeSpecial
// marker set (§B60). ALL added tokens are taken, not only those flagged "special":
// HuggingFace's AddedVocabulary pre-splits on every added token, and its
// encode_special_tokens flag filters only the special-flagged subset. Plain Encode is
// unaffected either way — it treats every marker as literal text.
func addedTokenSpecials(added []addedToken) map[string]int {
	out := make(map[string]int, len(added))
	for _, a := range added {
		if a.Content != "" && a.ID >= 0 {
			out[a.Content] = a.ID
		}
	}
	return out
}

// addedToken is one tokenizer.json added_tokens entry. Special records HF's "special"
// flag; it is retained for provenance (BOS/EOS/PAD vs. an ordinary added word) even
// though the pre-split takes both kinds.
type addedToken struct {
	ID      int    `json:"id"`
	Content string `json:"content"`
	Special bool   `json:"special"`
}

// ToJSON serializes the tokenizer to the minimal HuggingFace tokenizer.json BPE form
// (model.type "BPE" with vocab and merges), so BPEFromJSON(t.ToJSON()) round-trips it.
func (t *BPETokenizer) ToJSON() ([]byte, error) {
	vocab := make(map[string]int, len(t.decoder))
	for id, tok := range t.decoder {
		vocab[tok] = id
	}
	// merges ordered by their rank (lowest first = highest priority).
	type mr struct {
		m [2]string
		r int
	}
	ms := make([]mr, 0, len(t.mergeRank))
	for m, r := range t.mergeRank {
		ms = append(ms, mr{m, r})
	}
	//perfscan:ignore PS3002,PS6009 BPE-to-HF JSON export sort; one-time serialization | same JSON-export path; cold
	sort.Slice(ms, func(a, b int) bool { return ms[a].r < ms[b].r })
	merges := make([]string, len(ms))
	for i, e := range ms {
		merges[i] = e.m[0] + " " + e.m[1] // rejoin {left,right} into the HF "left right" form
	}
	var out hfTokJSON
	out.Model.Type = "BPE"
	out.Model.Vocab = vocab
	mj, err := json.Marshal(merges)
	if err != nil {
		return nil, err
	}
	out.Model.Merges = mj
	return json.Marshal(out)
}
