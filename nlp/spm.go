package nlp

import (
	"fmt"
	"strings"
)

// SPM is the SentencePiece BPE-mode encoder of the Llama model family (Llama 1/2,
// Mistral, TinyLlama, …), matching llama.cpp's `llm_tokenizer_spm`: the text is split
// into characters (after ▁ whitespace escaping) and the neighboring pair whose
// concatenation is the highest-scoring vocabulary piece is merged, repeatedly, until
// no adjacent pair forms a known piece; anything not in the vocabulary falls back to
// the <0xNN> byte pieces.
//
// This is NOT the Unigram-LM Viterbi of nlp.Unigram. The llama-family GGUF vocabularies
// carry scores that are NEGATIVE MERGE RANKS (piece id 259+i has score −i), not
// log-probabilities: under Viterbi's score-sum maximization, two short frequent pieces
// ("▁T"=−61 + "he"=−6 = −67) beat the correct single piece ("▁The"=−156) and the
// encoding shatters into sub-word fragments the model never saw in training (§B59 —
// observed as incoherent Mistral-7B output; TinyLlama GGUFs mask the bug because their
// scores are all zero). Greedy best-score-pair merging reproduces the BPE process those
// ranks came from, token-for-token equal to llama.cpp.
type SPM struct {
	id     map[string]int // piece → id
	pieces []UnigramPiece // id → piece+score
	unkID  int            // id emitted when even byte fallback fails
	dummy  bool           // prepend a ▁ (add_dummy_prefix)
	byteID [256]int       // id of the <0xNN> byte piece, −1 when absent

	specials specialSet // markers parsed only by EncodeSpecial (§B60)
}

// SPMOption configures an SPM tokenizer (functional-options idiom, §C12).
type SPMOption func(*SPM)

// WithSPMUnkID sets the id emitted for a byte not covered by the vocabulary's <0xNN>
// pieces (with the standard 256 byte pieces present it is never used).
//
// In plain terms: the "unknown" token id of last resort. Boundary behavior — any valid
// vocab id. SPECIAL VALUE / default: the id of a piece literally named "<unk>" if
// present, else 0 (the SentencePiece convention).
func WithSPMUnkID(id int) SPMOption { return func(s *SPM) { s.unkID = id } }

// WithSPMDummyPrefix toggles prepending a ▁ (the SentencePiece space marker) before
// encoding, so a word at the START of the text tokenizes the same way it would
// mid-sentence.
//
// In plain terms: SentencePiece treats a space as part of the following word; the dummy
// prefix makes "Hello" at the start match " Hello" in the middle. Boundary behavior — a
// boolean; off only for vocabularies trained without add_dummy_prefix.
//
// Default true (research-grounded: SentencePiece's add_dummy_prefix default, which the
// Llama family uses).
func WithSPMDummyPrefix(on bool) SPMOption { return func(s *SPM) { s.dummy = on } }

// NewSPM builds an SPM tokenizer from a scored vocabulary (piece id = index). Scores
// order the merges (higher merges first); for llama-family GGUF vocabularies they are
// negative merge ranks. It errors on an empty vocabulary or duplicate pieces.
func NewSPM(vocab []UnigramPiece, opts ...SPMOption) (*SPM, error) {
	if len(vocab) == 0 {
		return nil, fmt.Errorf("nlp: SPM needs a non-empty vocabulary")
	}
	s := &SPM{
		id:     make(map[string]int, len(vocab)),
		pieces: append([]UnigramPiece(nil), vocab...),
		dummy:  true,
	}
	for i := range s.byteID {
		s.byteID[i] = -1
	}
	for i, p := range vocab {
		if _, dup := s.id[p.Piece]; dup {
			return nil, fmt.Errorf("nlp: SPM duplicate piece %q", p.Piece)
		}
		s.id[p.Piece] = i
		var b int
		if n, _ := fmt.Sscanf(p.Piece, "<0x%02X>", &b); n == 1 && len(p.Piece) == 6 {
			s.byteID[b] = i
		}
	}
	if id, ok := s.id["<unk>"]; ok {
		s.unkID = id
	}
	for _, o := range opts {
		o(s)
	}
	return s, nil
}

// Encode segments text into token ids by greedy best-score adjacent-pair merging
// (llama.cpp llm_tokenizer_spm semantics; ties break to the leftmost pair). The
// rescan-per-merge loop is O(n²) in the text length — encoding is prompt-sized, not
// corpus-sized, so clarity wins over a mergeable-pair heap.
//
// It treats text as LITERAL and never emits a registered special marker's id, so it is
// the right entry point for untrusted input (§B60); [SPM.EncodeSpecial] is the one that
// parses markers. Spaces are escaped to ▁ first — which means a literal ▁ already in
// text is indistinguishable from a space and will decode back as one, silently.
// [ContainsSpaceMeta] detects that input; SentencePiece specifies no escape for it and
// neither does GoAI (see [ContainsSpaceMeta] for why inventing one would be worse).
func (s *SPM) Encode(text string) []int {
	str := strings.ReplaceAll(text, " ", spaceMeta)
	if s.dummy {
		str = spaceMeta + str
	}
	if str == "" {
		return nil
	}
	syms := make([]string, 0, len(str))
	for _, r := range str {
		syms = append(syms, string(r))
	}
	for {
		best, bestScore := -1, 0.0
		for i := 0; i+1 < len(syms); i++ {
			merged := syms[i] + syms[i+1]
			// Registered special markers are never produced by ordinary merging (§B60):
			// a vocabulary holding the intermediates could otherwise build a control
			// token out of untrusted literal text. EncodeSpecial is the only path that
			// emits them.
			id, ok := s.id[merged]
			if ok && s.specials.blocked(merged) {
				continue
			}
			if ok && (best < 0 || s.pieces[id].Score > bestScore) {
				best, bestScore = i, s.pieces[id].Score
			}
		}
		if best < 0 {
			break
		}
		syms[best] += syms[best+1]
		syms = append(syms[:best+1], syms[best+2:]...)
	}
	var ids []int
	for _, sym := range syms {
		// blocked() again for the degenerate case of a single-character marker, which
		// merging never had to build (§B60); it falls through to byte fallback.
		if id, ok := s.id[sym]; ok && !s.specials.blocked(sym) {
			ids = append(ids, id)
			continue
		}
		for _, b := range []byte(sym) { // byte fallback for uncovered characters
			if id := s.byteID[b]; id >= 0 {
				ids = append(ids, id)
			} else {
				ids = append(ids, s.unkID)
			}
		}
	}
	return ids
}

// Decode reconstructs text from token ids and drops the leading dummy space. Each
// piece is expanded by its KIND, matching llama.cpp's token_to_piece, which switches
// on the token ATTRIBUTE (BYTE / UNKNOWN+CONTROL+USER_DEFINED / NORMAL) rather than
// post-processing one concatenated string:
//
//   - <0xNN> byte pieces → the raw byte, never whitespace-unescaped (a byte-fallback
//     run reconstructs arbitrary UTF-8, and its bytes mean themselves);
//   - registered special markers (see [SPM.AddSpecialTokens]) → VERBATIM. DeepSeek's
//     control tokens are named "<｜begin▁of▁sentence｜>", where U+2581 is part of the
//     NAME; unescaping it would return a corrupted "<｜begin of sentence｜>" for a
//     perfectly correct token id (§B74);
//   - everything else (NORMAL pieces) → ▁ mapped back to spaces.
//
// Losslessness — Decode(Encode(text)) == text — requires BOTH that the vocabulary can
// represent every character (the <0xNN> byte pieces normally guarantee this) and that
// text contains no literal ▁ (U+2581): Encode escapes " " to ▁ with no inverse
// escape, exactly as SentencePiece itself has none, so a literal ▁ arrives at
// segmentation indistinguishable from a space and returns as one. [ContainsSpaceMeta]
// tests for this up front.
func (s *SPM) Decode(ids []int) string {
	var b strings.Builder
	for _, id := range ids {
		if id < 0 || id >= len(s.pieces) {
			continue
		}
		p := s.pieces[id].Piece
		var v int
		if n, _ := fmt.Sscanf(p, "<0x%02X>", &v); n == 1 && len(p) == 6 {
			b.WriteByte(byte(v)) // BYTE: raw, no unescaping
			continue
		}
		if s.specials.blocked(p) {
			b.WriteString(p) // CONTROL / USER_DEFINED: ▁ is part of its name
			continue
		}
		b.WriteString(strings.ReplaceAll(p, spaceMeta, " "))
	}
	out := b.String()
	if s.dummy {
		out = strings.TrimPrefix(out, " ")
	}
	return out
}

// SPMFromGGUF builds the SPM (llama-family SentencePiece BPE) tokenizer from the
// metadata map of a parsed GGUF model file, reading the same keys as UnigramFromGGUF
// (tokenizer.ggml.tokens / .scores / .unknown_token_id). It requires
// tokenizer.ggml.model == "llama".
//
// Use THIS loader — not UnigramFromGGUF — for llama-family models whose GGUF carries
// real SentencePiece scores (negative merge ranks): Viterbi over merge ranks fragments
// the encoding and degrades generation (§B59). UnigramFromGGUF remains correct for true
// Unigram-LM vocabularies ("t5"/UGM), whose scores are log-probabilities.
func SPMFromGGUF(meta map[string]any) (*SPM, error) {
	model, _ := meta[ggufTokModel].(string)
	if model != "llama" {
		return nil, fmt.Errorf("nlp: GGUF tokenizer.ggml.model=%q is not the llama SPM family", model)
	}
	toks, ok := meta[ggufTokTokens].([]any)
	if !ok || len(toks) == 0 {
		return nil, fmt.Errorf("nlp: GGUF %s missing or empty", ggufTokTokens)
	}
	scores, ok := meta[ggufTokScores].([]any)
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF %s missing (SPM orders merges by score)", ggufTokScores)
	}
	if len(scores) != len(toks) {
		return nil, fmt.Errorf("nlp: GGUF tokens (%d) and scores (%d) length mismatch", len(toks), len(scores))
	}
	vocab := make([]UnigramPiece, len(toks))
	for i := range toks {
		piece, ok := toks[i].(string)
		if !ok {
			return nil, fmt.Errorf("nlp: GGUF token %d is %T, want string", i, toks[i])
		}
		score, ok := toks32(scores[i])
		if !ok {
			return nil, fmt.Errorf("nlp: GGUF score %d is %T, want float32", i, scores[i])
		}
		vocab[i] = UnigramPiece{Piece: piece, Score: score}
	}
	var opts []SPMOption
	if unk, ok := ggufTokenID(meta[ggufTokUnkID]); ok {
		opts = append(opts, WithSPMUnkID(unk))
	}
	s, err := NewSPM(vocab, opts...)
	if err != nil {
		return nil, err
	}
	// Control / user-defined / unknown tokens become the EncodeSpecial marker set (§B60).
	// Plain Encode is unaffected: it still treats every marker as literal text.
	texts := make([]string, len(vocab))
	for i, p := range vocab {
		texts[i] = p.Piece
	}
	s.AddSpecialTokens(ggufSpecials(meta, texts))
	return s, nil
}
