package nlp

import "fmt"

// The KV-cache POSITION CONTRACT, shared by every cached decode path (§T35, §R97).
//
// THE CONTRACT, in one line: the `pos` argument of a DecodeStep is the token's TRUE
// position in the stream — how many tokens the model has consumed before it — and is
// NEVER the row index the token lands on inside the cache.
//
// The two coincide for an append-only cache (row i holds the token at position i), and
// that coincidence is what made the old wording — "the token's absolute position
// (== cache length before the call)" — read as a definition when it was only an
// identity. Eviction breaks the identity: [KVCache.EvictStreaming] drops rows out of
// the middle, so a cache that has consumed 500 tokens may hold only 132 of them. The
// 501st token is still at position 500. It is not at position 132.
//
// WHY THE TRUE POSITION IS THE ONE BOTH CONSUMERS NEED. GoAI has two kinds of
// positional encoding behind DecodeStep, and it is worth being explicit that they do
// NOT want different values here:
//
//   - RoPE ([Llama.DecodeStep] and every rotary architecture) rotates each key by its
//     position as it enters the cache, and an attention score depends on the DIFFERENCE
//     between the query's rotation and the key's. The cached keys keep the rotation they
//     were given. So a new query is only comparable to them if it is rotated on the same
//     axis they were — the true stream position.
//   - A learned ABSOLUTE position embedding ([GPT.DecodeStep]) adds PosEmb[pos] to the
//     token embedding before the cached key is ever computed, so the position is likewise
//     baked into the cached row and cannot be revised afterwards. A new token must again
//     land on the same axis.
//
// Both bake position in at insert time; neither can re-derive it later. The true
// position is therefore the only self-consistent choice for both, and the contract is
// uniform rather than per-architecture.
//
// WHAT THIS MEANS AFTER EVICTION. The honest consequence is that eviction buys memory,
// not positional room: a bounded cache still walks up the same position axis, so a
// learned-absolute-PE model still stops at its Ctx limit and a RoPE model still drifts
// out toward its trained range. Renumbering the survivors down to 0..n−1 — the
// StreamingLLM paper's actual prescription — requires physically REWRITING the cached
// rows to match, which is only possible if the cache stores keys BEFORE the positional
// encoding is applied. [StreamCache] does exactly that and [Llama.StreamStep] is the
// path that streams without bound; the post-encoding caches here cannot, and pretending
// otherwise by quietly reassigning a new token's position is how the model ends up
// reading incoherent geometry while every function still returns nil.
//
// For readers new to LLM inference: think of the cache as a filing drawer of notes the
// model already wrote, each stamped with the page number it came from. Throwing away
// the middle of the drawer to save space does not renumber the pages that are left, and
// the next note still gets the next page number. Handing it the number of REMAINING
// notes instead would stamp it with a page number that is already in the drawer, and
// nothing about the resulting text would look obviously broken — it would just quietly
// be wrong. That silent-wrongness is the failure mode this contract exists to make
// impossible (§V29).
//
// HOW llama.cpp DOES IT (the design this mirrors). llama.cpp gives every KV cell its own
// llama_pos, wholly separate from the slot index it occupies, and its causal mask
// compares POSITIONS rather than indices. A caller may pass positions explicitly in
// llama_batch.pos, or leave that field NULL and let llama_decode derive each one as
// seq_pos_max(seq) + 1 — the cache's own idea of where the stream has got to.
// Positions are then validated, not trusted: a batch whose sequence does not continue
// the cache consecutively is rejected outright ("it is required that the sequence
// positions remain consecutive: Y = X + 1"), as is any batch whose positions decrease,
// and llama_decode returns −1 instead of logits. [KVCache.NextPos] is the seq_pos_max+1
// accessor; the check in each DecodeStep is the Y = X + 1 rule.
//
// One deliberate divergence, and the reason for it. When llama.cpp evicts, it closes the
// gap: llama_kv_cache_seq_rm removes a window and llama_kv_cache_seq_add slides
// everything after it down, so positions stay consecutive. It can only do that because
// it also re-rotates the cached keys to match, via the deferred K-shift pass
// (build_rope_shift) that RoPE's additive composition R(p)·R(d) = R(p+d) makes cheap.
// GoAI has no K-shift, so it must NOT renumber — the stored rows would keep their old
// rotation while the bookkeeping claimed a new one. Notably llama.cpp itself does not
// have a correct answer for the learned-absolute-PE case either: for a GPT-2-style model
// it rewrites the cell positions but skips the re-rotation graph entirely (there is
// nothing to rotate), leaving the already-summed absolute embedding stale and silent,
// which is why upstream ships context-shift OFF by default. Keeping the true axis and
// refusing to renumber is the version of that decision that cannot go quiet.

// kvPos places a KV-cache on its stream's position axis, so the cache can say where the
// next token belongs no matter how many rows have been evicted out from under it.
//
// It is deliberately two counters rather than a position per row. Positions here are
// only ever consumed in order, so the whole mapping is recoverable from where the stream
// started and how much of it has been thrown away — and, unlike a per-row list, two
// counters cost nothing to keep in step. Nothing has to be updated on APPEND: rows
// arriving push [KVCache.Len] up on their own, and the next position moves with it. That
// matters more than it looks. Every writer of these caches — DecodeStep, Prefill, the
// quantized decoders — would otherwise have to remember to bump a counter, and the one
// that forgot would desynchronize the cache silently, which is the exact failure class
// this file exists to close.
type kvPos struct {
	// anchor is the stream position of the first row the cache ever held. Zero for a
	// cache that starts at the beginning of its stream, which is every ordinary decode;
	// it is nonzero only when a caller deliberately primes an empty cache further along,
	// e.g. to reconstruct the state a bounded cache would have reached.
	anchor int
	// dropped counts rows eviction has removed. Their positions stay CONSUMED — the
	// stream moved past them — so the next position is still measured as if they were
	// present. This is the field that keeps `pos` on the true axis across eviction, and
	// the reason NextPos and Len diverge afterwards.
	dropped int
}

// next returns the stream position the next token appended to this cache must occupy,
// given how many rows the cache currently holds.
func (p kvPos) next(cached int) int { return p.anchor + p.dropped + cached }

// drop records that eviction removed n rows, whose positions remain consumed.
func (p *kvPos) drop(n int) { p.dropped += n }

// reset returns the tracker to a pristine stream, for a cache being refilled from
// scratch.
func (p *kvPos) reset() { p.anchor, p.dropped = 0, 0 }

// admit checks pos against the contract at the top of this file and records the anchor
// when this token is the first the cache has ever seen. cached is the cache's current
// row count and kind names the cache type for the error message.
//
// An EMPTY, never-evicted cache accepts any pos ≥ 0 and is anchored by it: there is
// nothing yet for the value to be inconsistent WITH, and allowing it is what lets a
// caller prime a cache at an offset (llama.cpp permits the same via an explicit
// llama_batch.pos). Every later token must then continue the stream exactly.
func (p *kvPos) admit(cached, pos int, kind string) error {
	if pos < 0 {
		return fmt.Errorf("nlp: DecodeStep position %d must be ≥ 0", pos)
	}
	if cached == 0 && p.dropped == 0 {
		p.anchor = pos // first token on this cache: it defines the axis
		return nil
	}
	want := p.next(cached)
	if pos == want {
		return nil
	}
	if p.dropped == 0 {
		return fmt.Errorf("nlp: DecodeStep position %d does not continue the cache: the next token is at position %d "+
			"(cached rows: %d) — pos is the token's position in the STREAM, not a free parameter; use %s.NextPos()",
			pos, want, cached, kind)
	}
	return fmt.Errorf("nlp: DecodeStep position %d does not continue the cache: the next token is at position %d "+
		"(stream consumed: %d, cached rows: %d, evicted: %d) — after eviction pos is NOT the cache length, it stays the "+
		"token's TRUE position in the stream; use %s.NextPos()",
		pos, want, want, cached, p.dropped, kind)
}

// NextPos returns the position the next token decoded into this cache must be given —
// the value to pass as DecodeStep's pos argument.
//
// Use it instead of counting tokens yourself. For a cache that has simply been appended
// to, it equals [KVCache.Len]; after [KVCache.EvictStreaming] it does NOT, because
// eviction frees rows without rewinding the stream, and the position of the next token
// is a fact about the stream. A cache that has consumed 500 tokens and retains 132
// reports 500 here and 132 from Len, and 500 is the number DecodeStep wants.
//
// This is the same service llama.cpp performs when llama_batch.pos is left NULL and it
// fills in seq_pos_max(seq)+1 for you.
func (c *KVCache) NextPos() int { return c.pos.next(c.Len()) }

// NextPos returns the position the next token decoded into this cache must be given —
// the value to pass as DecodeStep's pos argument.
//
// For an ordinary decode it counts up with the cache: [Llama.Prefill] over an n-token
// prompt leaves it at n, and every [Llama.DecodeStep] adds one. Reading it beats
// tracking the position in caller code, which is the mistake this exists to remove: a
// position that disagrees with the cache is rejected rather than silently rotated to the
// wrong angle.
func (c *LlamaCache) NextPos() int { return c.pos.next(c.Len()) }
