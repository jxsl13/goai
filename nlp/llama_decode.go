package nlp

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// LlamaCache holds the per-layer key/value tensors accumulated during autoregressive
// Llama decoding, so each new token attends to the cached past instead of recomputing
// attention over the whole prefix. The cached keys already carry their RoPE rotation
// (applied at the position each token entered the cache). K[l] and V[l] are
// contiguous [t, width] views over internal amortized-growth row buffers
// (rowBuf), refreshed each DecodeStep — appends cost O(1) per token instead of
// the old concatRows O(t) reallocate-and-copy.
//
// It also tracks where the stream has got to, which [LlamaCache.NextPos] reports and
// [Llama.DecodeStep] checks its pos argument against. Because the cached keys are stored
// POST-RoPE — already rotated to the position they entered at — that position axis is
// fixed once a row is written, and the contract at the top of kvcache.go explains why
// every later token has to stay on it.
type LlamaCache struct {
	K, V []*tensor.Tensor // per block; nil until the first token
	bufs kvBufs           // backing row buffers behind the K, V views
	pos  kvPos            // where the cached rows sit on the stream's position axis
}

// NewCache returns an empty KV-cache sized for this model's blocks.
func (m *Llama) NewCache() *LlamaCache {
	return &LlamaCache{K: make([]*tensor.Tensor, len(m.Blocks)), V: make([]*tensor.Tensor, len(m.Blocks))}
}

// Len returns the number of tokens currently cached, which for this cache — appended to
// but never evicted from — is also the number of tokens the stream has consumed. Use
// [LlamaCache.NextPos] when what you want is the next token's position.
func (c *LlamaCache) Len() int {
	if len(c.K) == 0 || c.K[0] == nil {
		return 0
	}
	return c.K[0].Shape()[0]
}

// embedOne returns the embedding of a single token, x[1,dim] = TokEmb[token].
func (m *Llama) embedOne(token int) (*tensor.Tensor, error) {
	if token < 0 || token >= m.Config.Vocab {
		return nil, fmt.Errorf("nlp: token %d outside vocab %d", token, m.Config.Vocab)
	}
	d := m.Config.Dim
	return embedRow(m.TokEmb, token, d), nil
}

// attnScale returns the value every Llama decode path must put in AttnAttrs.Scale.
// OpMHA (and OpMHASelect) build in a 1/√headDim and treat Scale as an EXTRA factor on
// top of it, so Granite's attention_multiplier — which REPLACES 1/√headDim rather than
// compounding with it — is passed as AttentionMult·√headDim, leaving a net pre-softmax
// scale of exactly AttentionMult. An unset AttentionMult (0, the Llama/Qwen/Mistral
// case) returns 0, which AttnAttrs.WithDefaults reads as 1 and leaves the kernel's own
// 1/√headDim untouched.
func (c LlamaConfig) attnScale() float64 {
	if c.AttentionMult == 0 {
		return 0
	}
	return c.AttentionMult * math.Sqrt(float64(c.Dim/c.Heads))
}

// attendFunc is the attention core of ONE block: given the block index and that block's
// post-bias, post-QK-norm, PRE-RoPE q, k and v, it applies whatever positional rotation,
// caching and masking its decode strategy uses and returns the attention output
// [rows, Heads·headDim] that feeds Wo.
//
// This is the ONLY thing that differs between GoAI's Llama decode strategies. The
// KV-cached [Llama.DecodeStep] rotates one token at its absolute position and appends
// it to a growing cache; StreamingLLM's [Llama.StreamStep] keeps a bounded pre-RoPE
// cache and re-rotates it at cache-relative positions; Self-Extend's
// [Llama.SelfExtendForward] builds two differently-rotated score sources and merges
// them under one softmax. Everything AROUND the attention — the norms, the projections,
// the per-architecture hooks, the SwiGLU — is identical in all three, and lives once in
// [Llama.blockStack].
//
// An attendFunc that runs OpMHA/OpMHASelect MUST pass [LlamaConfig.attnScale] as
// AttnAttrs.Scale. That is the one per-architecture feature this seam cannot apply on
// the caller's behalf, because it is a property of the attention kernel invocation
// rather than of the surrounding block.
type attendFunc func(ctx *backend.Context, block int, q, k, v *tensor.Tensor) (*tensor.Tensor, error)

// blockStack runs the Llama block stack over an embedded input x [rows, Dim] with a
// pluggable attention core, returning the PRE-final-norm residual stream. Callers
// finish with [Llama.Unembed] (final RMSNorm + output projection + Granite logits
// scaling) to turn it into logits.
//
// It is the single home of the per-architecture decode features, so a new decode
// strategy inherits all of them by supplying only an attendFunc:
//
//   - Granite embedding_multiplier, applied to the embedded input;
//   - Qwen2-family q/k/v projection biases, added before RoPE;
//   - Qwen3 per-head QK-norm, applied before RoPE;
//   - Granite residual_multiplier, applied to BOTH residual adds.
//
// For readers new to the codebase: GoAI models four architectures with one struct.
// A plain Llama leaves every optional field above nil or zero, which makes each hook a
// byte-identical no-op, and Qwen2/Qwen3/Granite switch them on. That is convenient
// until a decode path forgets one — and because a dropped hook produces *plausible*
// logits rather than an error, nothing fails loudly. Both alternative decode paths had
// in fact drifted this way: [Llama.StreamStep] and [Llama.SelfExtendForward] were each
// copied from the decode body before these hooks existed and never picked any of them
// up, so streaming a Qwen2, Qwen3 or Granite model silently produced different text
// than [Llama.Generate] on the same prompt. Route new decode paths through here rather
// than copying the block body; TestStreamGenerateMatchesGenerate and
// TestSelfExtendForwardArchParity fail if a path drifts again.
//
// Known limit: the full-sequence forward (hiddenFromEmbedTaps in llama.go) still has
// its own copy of this block body, because it additionally carries the KV-capture and
// residual-capture taps. The two are pinned against each other by the parity tests
// above, not by construction.
func (m *Llama) blockStack(ctx *backend.Context, x *tensor.Tensor, attend attendFunc) (*tensor.Tensor, error) {
	if err := validateLayerBackends(m.LayerBackends, len(m.Blocks)); err != nil {
		return nil, err
	}
	var err error
	// Granite: inputs_embeds *= embedding_multiplier (no-op for Llama, EmbeddingMult 0).
	if x, err = scaleScalar(ctx, x, m.Config.EmbeddingMult); err != nil {
		return nil, err
	}
	return m.blockRange(ctx, x, 0, len(m.Blocks), attend)
}

// blockRange runs blocks [start,end) without embedding scaling or final unembedding.
// LayerSkip verification enters here with the early pass' cached residual rows.
func (m *Llama) blockRange(ctx *backend.Context, x *tensor.Tensor, start, end int, attend attendFunc) (*tensor.Tensor, error) {
	cfg := m.Config
	kv := cfg.kvHeads()
	for l := start; l < end; l++ {
		b := m.Blocks[l]
		layerCtx := contextForLayer(ctx, m.LayerBackends, l)
		xb, err := b.AttnNorm.Forward(layerCtx, x)
		if err != nil {
			return nil, err
		}
		q, err := project(layerCtx, xb, b.Wq)
		if err != nil {
			return nil, err
		}
		k, err := project(layerCtx, xb, b.Wk)
		if err != nil {
			return nil, err
		}
		v, err := project(layerCtx, xb, b.Wv)
		if err != nil {
			return nil, err
		}
		// Qwen2-family q/k/v projection biases (added before RoPE); nil for Llama/Mistral.
		if q, err = addBiasIf(layerCtx, q, b.Bq); err != nil {
			return nil, err
		}
		if k, err = addBiasIf(layerCtx, k, b.Bk); err != nil {
			return nil, err
		}
		if v, err = addBiasIf(layerCtx, v, b.Bv); err != nil {
			return nil, err
		}
		// Qwen3 per-head QK-norm (before RoPE); nil for Llama/Qwen2.
		if q, err = applyQKNorm(layerCtx, q, b.QNorm, cfg.Heads); err != nil {
			return nil, err
		}
		if k, err = applyQKNorm(layerCtx, k, b.KNorm, kv); err != nil {
			return nil, err
		}
		a, err := attend(layerCtx, l, q, k, v)
		if err != nil {
			return nil, err
		}
		o, err := project(layerCtx, a, b.Wo)
		if err != nil {
			return nil, err
		}
		// Granite: x = x + o·residual_multiplier (no-op for Llama, ResidualMult 0).
		if o, err = scaleScalar(layerCtx, o, cfg.ResidualMult); err != nil {
			return nil, err
		}
		if x, err = exec2(layerCtx, backend.OpAdd, nil, x, o); err != nil {
			return nil, err
		}
		xf, err := b.FFNNorm.Forward(layerCtx, x)
		if err != nil {
			return nil, err
		}
		ff, err := b.FFN.Forward(layerCtx, xf)
		if err != nil {
			return nil, err
		}
		// Granite: x = x + ff·residual_multiplier (same scalar as the attention residual).
		if ff, err = scaleScalar(layerCtx, ff, cfg.ResidualMult); err != nil {
			return nil, err
		}
		if x, err = exec2(layerCtx, backend.OpAdd, nil, x, ff); err != nil {
			return nil, err
		}
	}
	return x, nil
}

// DecodeStep advances the Llama by one token using the KV-cache and returns the
// next-token logits [1,vocab]. The token's rotated k,v are appended to the cache and the
// single query attends to all cached keys. Inference-only (no tape); it produces the
// same logits as a full Forward over the prefix.
//
// pos is the token's TRUE position in the stream and is used as the RoPE offset. It must
// equal [LlamaCache.NextPos] — pass that rather than a separately maintained counter. An
// ordinary loop still counts 0, 1, 2, …, and [Llama.Prefill] over an n-token prompt
// leaves the next position at n. A pos that does not continue the cache is REJECTED:
// RoPE encodes distance as the difference between the query's rotation and each cached
// key's, so a wrong pos does not fail, it silently rotates the query to the wrong angle
// and returns perfectly ordinary-looking logits (§V29). The reasoning behind the
// convention is in the position contract at the top of kvcache.go.
//
// The block body it runs is [Llama.blockStack] — shared with StreamingLLM and
// Self-Extend decoding — so every per-architecture hook (Qwen2 biases, Qwen3 QK-norm,
// the Granite scalars) is applied here by construction rather than by remembering to
// copy it.
func (m *Llama) DecodeStep(ctx *backend.Context, cache *LlamaCache, token, pos int) (*tensor.Tensor, error) {
	if pos < 0 || pos >= m.Config.Ctx {
		return nil, fmt.Errorf("nlp: position %d outside context %d", pos, m.Config.Ctx)
	}
	if err := cache.pos.admit(cache.Len(), pos, "LlamaCache"); err != nil {
		return nil, err
	}
	cfg := m.Config
	kv := cfg.kvHeads()
	x, err := m.embedOne(token)
	if err != nil {
		return nil, err
	}
	// These attrs are identical for every layer of this step (Base/Heads/KV/pos/scale are
	// all layer-independent), so box each into the Attrs interface ONCE per token here
	// rather than re-boxing it inside the per-layer closure — as concrete structs passed to
	// exec1's Attrs parameter they were heap-boxed ~N_layers times per decoded token (T955).
	qRoPE := backend.Attrs(backend.RoPEAttrs{Base: cfg.RopeBase, Heads: cfg.Heads, PosOffset: pos})
	kRoPE := backend.Attrs(backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv, PosOffset: pos})
	attn := backend.Attrs(backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: false, Scale: cfg.attnScale()})
	h, err := m.blockStack(ctx, x, func(layerCtx *backend.Context, l int, q, k, v *tensor.Tensor) (*tensor.Tensor, error) {
		// RoPE the single token at its absolute position, then append k,v to the cache.
		q, err := exec1(layerCtx, backend.OpRoPE, qRoPE, q)
		if err != nil {
			return nil, err
		}
		if k, err = exec1(layerCtx, backend.OpRoPE, kRoPE, k); err != nil {
			return nil, err
		}
		kNew, vNew := cache.bufs.appendKV(cache.K, cache.V, l, k, v)
		cache.K[l], cache.V[l] = kNew, vNew
		// single query at the last position attends to all cached keys → no causal mask
		return exec1(layerCtx, backend.OpMHA, attn, q, kNew, vNew)
	})
	if err != nil {
		return nil, err
	}
	// Final RMSNorm + output projection + the Granite logits scaling.
	return m.Unembed(ctx, h)
}

// Prefill runs the transformer ONCE over the whole prompt, seeds the KV-cache with
// every prompt token's key/value rows, and returns the full logits [seq, vocab]
// (row seq−1 is the next-token distribution Generate samples from). It is the
// batched replacement for feeding the prompt token-by-token through [Llama.DecodeStep]:
// one full-sequence pass through the block stack instead of seq single-token
// dispatch rounds, with a bit-identical cache — the captured k/v are the same
// post-RoPE rows DecodeStep would have appended (full-sequence RoPE rotates row p
// for position p, exactly DecodeStep's PosOffset=p), so decoding may continue from
// the seeded cache as if the prompt had been stepped through one token at a time.
//
// For readers new to LLM inference: generation has two phases. "Prefill" processes
// the whole prompt at once — every token is known up front, so the model can attend
// over the full sequence in a handful of large batched kernels and store each
// layer's keys/values in the cache. "Decode" then produces one token at a time,
// where each new token only needs its own small computation plus the cached past.
// Batching the prefill is the standard first optimization of every serving stack
// (llama.cpp's prompt batch, vLLM's prefill phase): the math is unchanged, but T
// prompt tokens cost one batched pass instead of T sequential steps.
//
// cache must be empty (a fresh [Llama.NewCache]); Prefill errors on a non-empty
// cache because the full-sequence RoPE and causal mask assume positions 0..seq−1.
// Inference-only, like DecodeStep: run it on a plain backend.NewContext.
//
// Because those positions are fixed at 0..seq−1, Prefill also ANCHORS the cache's
// position axis at 0, leaving [LlamaCache.NextPos] at len(tokens) — the position the
// first generated token must be given.
func (m *Llama) Prefill(ctx *backend.Context, cache *LlamaCache, tokens []int) (*tensor.Tensor, error) {
	if n := cache.Len(); n != 0 {
		return nil, fmt.Errorf("nlp: Prefill needs an empty cache, got %d cached tokens", n)
	}
	// The full-sequence RoPE below rotates row p for position p, so the stream starts at
	// 0 regardless of anything a previous, now-emptied use of this cache anchored.
	cache.pos.reset()
	if len(cache.K) < len(m.Blocks) {
		cache.K = growSlice(cache.K, len(m.Blocks))
	}
	if len(cache.V) < len(m.Blocks) {
		cache.V = growSlice(cache.V, len(m.Blocks))
	}
	x, err := m.Embed(ctx, tokens)
	if err != nil {
		return nil, err
	}
	// The Granite embedding_multiplier (and every other Granite scalar) is applied
	// inside the shared block stack, exactly as in Forward and DecodeStep.
	h, err := m.hiddenFromEmbedCapture(ctx, x, func(l int, k, v *tensor.Tensor) {
		// One multi-row Append per layer: the whole prompt's [seq, kvWidth] k/v
		// land in the cache's row buffers in a single typed block copy.
		cache.K[l], cache.V[l] = cache.bufs.appendKV(cache.K, cache.V, l, k, v)
	})
	if err != nil {
		return nil, err
	}
	logits, err := exec1(ctx, backend.OpMatMul, nil, h, m.Out)
	if err != nil {
		return nil, err
	}
	// Granite: logits /= logits_scaling, matching ForwardFromEmbed (no-op for Llama).
	return divLogits(ctx, logits, m.Config.LogitsScale)
}

// PrefillAppend processes a non-empty token suffix in one batch against an existing
// contiguous Llama KV prefix and returns logits for the suffix [len(tokens), vocab].
// It is the compute primitive behind exact prompt-prefix reuse: callers may retain a
// cache for a stable system prompt, document, or conversation prefix, then append only
// each new turn instead of running [Llama.Prefill] over the shared prefix again.
//
// The operation is exact, not an approximation. Decoder keys and values at position t
// depend only on tokens through t, so an identical cached prefix supplies the same
// attention state; rectangular causal attention lets suffix row i see every prefix row
// plus suffix rows through i. This is the single-request contiguous-cache seam used by
// automatic prefix-caching systems such as SGLang RadixAttention. It does not itself
// provide a multi-request radix tree, LRU eviction, paging, or cache isolation.
//
// An empty fresh cache is supported and produces logits and K/V bit-identical to
// [Llama.Prefill]. The cache must contain every position from zero through Len()-1 in
// every model layer; offset-anchored, evicted, asymmetric, wrong-width, or over-context
// caches are rejected before any rows are appended. Inference-only (no tape), like
// Prefill and [Llama.DecodeStep].
func (m *Llama) PrefillAppend(ctx *backend.Context, cache *LlamaCache, tokens []int) (*tensor.Tensor, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("nlp: PrefillAppend needs a non-empty suffix")
	}
	prefix, err := validateLlamaPrefixState(m, cache)
	if err != nil {
		return nil, err
	}
	if prefix+len(tokens) > m.Config.Ctx {
		return nil, fmt.Errorf("nlp: PrefillAppend total length %d exceeds context %d", prefix+len(tokens), m.Config.Ctx)
	}
	for _, token := range tokens {
		if token < 0 || token >= m.Config.Vocab {
			return nil, fmt.Errorf("nlp: token %d outside vocab %d", token, m.Config.Vocab)
		}
	}
	if err := cache.pos.admit(prefix, prefix, "LlamaCache"); err != nil {
		return nil, err
	}
	x, err := m.Embed(ctx, tokens)
	if err != nil {
		return nil, err
	}
	if x, err = scaleScalar(ctx, x, m.Config.EmbeddingMult); err != nil {
		return nil, err
	}
	h, err := m.cachedBlockRange(ctx, cache, x, 0, len(m.Blocks), prefix, nil)
	if err != nil {
		return nil, err
	}
	return m.Unembed(ctx, h)
}

// Generate autoregressively decodes up to maxNew tokens after prompt with the sampler
// s, using the KV-cache (one batched [Llama.Prefill] over the prompt, then one
// [Llama.DecodeStep] per new token). Returns prompt+generated. With a
// greedy sampler the output is identical to argmax-ing a full Forward at each step.
// The decode runs on backend.Default() unless WithBackend overrides it (§T361).
//
// Pass [WithEOS] to stop early on a stop token, which is INCLUDED in the returned
// slice. Without it the loop runs the full maxNew steps whatever it draws — the
// long-standing behaviour, deliberately preserved.
func (m *Llama) Generate(prompt []int, maxNew int, s TokenSampler, opts ...GenerateOption) ([]int, error) {
	var gc genConfig
	for _, o := range opts {
		o(&gc)
	}
	ctx := backend.NewContext()
	if gc.be != nil {
		ctx = ctx.WithBackend(gc.be)
	}
	cache := m.NewCache()
	out := append([]int(nil), prompt...)
	if len(prompt) == 0 {
		return nil, fmt.Errorf("nlp: Generate needs a non-empty prompt")
	}

	// Batched prefill: one full-sequence pass seeds the cache and yields the
	// prompt's logits, replacing len(prompt) single-token DecodeStep rounds.
	full, err := m.Prefill(ctx, cache, prompt)
	if err != nil {
		return nil, err
	}
	logits, err := full.Slice(0, full.Shape()[0]-1, full.Shape()[0])
	if err != nil {
		return nil, err
	}
	pos := len(prompt)
	for range maxNew {
		if pos >= m.Config.Ctx {
			break
		}
		next := s.SampleWithHistory(rowLogits(logits), out)
		out = append(out, next)
		if gc.stopEOS(next, s) {
			break
		}
		l, err := m.DecodeStep(ctx, cache, next, pos)
		if err != nil {
			return nil, err
		}
		logits = l
		pos++
	}
	return out, nil
}
