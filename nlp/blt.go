package nlp

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// bltVocab is the byte-level vocabulary: every possible byte value. BLT and its
// entropy patcher are tokenizer-free — the "vocabulary" is fixed by the byte
// alphabet, never learned from data.
const bltVocab = 256

// bltNegInf is the additive-mask constant that zeroes an attention pair after
// softmax (exp(−1e30 − max) underflows to exactly 0 in f64) — the same constant
// nn/qknorm.go uses for its composed causal mask.
const bltNegInf = -1e30

// EntropyPatcherConfig fixes the geometry of the small next-byte language model
// that drives entropy patching (mirrors GPTConfig, minus Vocab — the byte
// alphabet fixes it at 256).
type EntropyPatcherConfig struct {
	Ctx    int     // max byte-sequence length
	Dim    int     // embedding width
	Heads  int     // attention heads (must divide Dim)
	Layers int     // transformer blocks
	Eps    float64 // LayerNorm epsilon (≤0 → 1e-5)
	// Threshold is the global entropy boundary rule: a new patch starts before
	// byte i when the model's next-byte entropy at i exceeds this value (nats).
	Threshold float64
	// MaxPatch caps the patch length: a boundary is forced once a patch reaches
	// this many bytes regardless of entropy. 0 → no cap.
	MaxPatch int
}

// EntropyPatcher is the byte-stream segmenter of the Byte Latent Transformer
// (Pagnoni et al. 2024, "Byte Latent Transformer: Patches Scale Better Than
// Tokens", arXiv:2412.09871, Meta): a SMALL causal byte-LM whose next-byte
// entropy H(xᵢ | x₍<ᵢ₎) = −Σ p·log p decides where patches begin. Wherever the
// model is uncertain about the next byte (entropy above Threshold), a new patch
// starts — so hard-to-predict regions get many short patches (more latent
// compute) and predictable regions get long ones (less). Per the paper the
// patcher is trained SEPARATELY on its own next-byte cross-entropy (Loss), and
// its boundaries enter the main model as FROZEN, non-differentiable inputs.
//
// In plain terms: before the big model reads a sentence, this small model skims
// it and draws pencil marks wherever the text becomes surprising. The chunks
// between marks — the patches — replace the fixed tokens a tokenizer would
// produce: boring stretches become one big chunk, tricky spots get chopped
// finely.
type EntropyPatcher struct {
	GPT *GPT // the small causal byte-LM backbone (Vocab=256, GPT block stack)
	// Threshold is the entropy (nats) above which a patch boundary is placed
	// (the paper's global-threshold rule). Calibrate it on training data, e.g.
	// to a target average patch length.
	Threshold float64
	// MaxPatch caps patch length (a boundary is forced at this many bytes);
	// 0 → no cap.
	MaxPatch int
}

// NewEntropyPatcher builds a randomly initialized entropy patcher: a causal GPT
// over the 256-byte alphabet (the diffusion-LM constructor pattern retargeted to
// Vocab=256). Deterministic via seed; weights are Xavier-uniform, LayerNorms
// 1/0, biases 0, parameters F64 (the repo's training/gradcheck precision).
func NewEntropyPatcher(cfg EntropyPatcherConfig, seed uint64) (*EntropyPatcher, error) {
	if cfg.Ctx <= 0 || cfg.Dim <= 0 || cfg.Layers <= 0 {
		return nil, fmt.Errorf("nlp: NewEntropyPatcher needs Ctx, Dim, Layers > 0 (got %d, %d, %d)",
			cfg.Ctx, cfg.Dim, cfg.Layers)
	}
	if cfg.Heads <= 0 || cfg.Dim%cfg.Heads != 0 {
		return nil, fmt.Errorf("nlp: NewEntropyPatcher dim %d not divisible by heads %d", cfg.Dim, cfg.Heads)
	}
	if cfg.MaxPatch < 0 {
		return nil, fmt.Errorf("nlp: NewEntropyPatcher MaxPatch must be ≥ 0, got %d", cfg.MaxPatch)
	}
	eps := cfg.Eps
	if eps <= 0 {
		eps = 1e-5
	}
	g := &GPT{
		Config: GPTConfig{Vocab: bltVocab, Ctx: cfg.Ctx, Dim: cfg.Dim, Heads: cfg.Heads, Layers: cfg.Layers, Eps: eps},
		TokEmb: bltRand(seed, bltVocab, cfg.Dim),
		PosEmb: bltRand(seed+1, cfg.Ctx, cfg.Dim),
		Head:   bltRand(seed+2, cfg.Dim, bltVocab),
		LNf:    bltLN(cfg.Dim, eps),
	}
	blocks, err := newByteBlocks(cfg.Layers, cfg.Dim, cfg.Heads, eps, seed+10)
	if err != nil {
		return nil, err
	}
	g.Blocks = blocks
	return &EntropyPatcher{GPT: g, Threshold: cfg.Threshold, MaxPatch: cfg.MaxPatch}, nil
}

// Params returns the patcher's trainable tensors (the backbone GPT's). Train
// the patcher on Loss BEFORE the main BLT — its boundaries are frozen inputs to
// the main model, never trained jointly.
func (p *EntropyPatcher) Params() []*tensor.Tensor { return p.GPT.Params() }

// Loss is the patcher's own training objective: mean next-byte cross-entropy of
// data[1:] under the causal byte-LM (logit rows 0..L−2 gathered differentiably,
// then nn.CrossEntropy). Needs at least 2 bytes. Fully differentiable w.r.t.
// Params through a recording context.
func (p *EntropyPatcher) Loss(ctx *backend.Context, data []byte) (*tensor.Tensor, error) {
	logits, err := p.GPT.Forward(ctx, bytesToInts(data))
	if err != nil {
		return nil, err
	}
	return nextByteCE(ctx, logits, data)
}

// Entropies returns the next-byte entropy signal H over data: H[i] for i ≥ 1 is
// the Shannon entropy (nats) of the model's softmax distribution for byte i
// given bytes 0..i−1 (the logit row at position i−1); H[0] is +Inf — the
// sequence start is always a patch boundary. The softmax/entropy loop runs
// host-side over the logit rows (inference only, no gradient).
func (p *EntropyPatcher) Entropies(ctx *backend.Context, data []byte) ([]float64, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("nlp: Entropies needs a non-empty byte sequence")
	}
	h := make([]float64, len(data))
	h[0] = math.Inf(1)
	if len(data) == 1 {
		return h, nil
	}
	logits, err := p.GPT.Forward(ctx, bytesToInts(data))
	if err != nil {
		return nil, err
	}
	lc := logits.Contiguous()
	for i := 1; i < len(data); i++ {
		h[i] = ByteEntropy(logitsRow(lc, i-1, bltVocab, bltVocab))
	}
	return h, nil
}

// Patch segments data into patches with the global-threshold rule: a new patch
// starts before byte i when Entropies(data)[i] > Threshold, or when the current
// patch has reached MaxPatch bytes. It returns the patch LENGTHS (each ≥ 1,
// summing to len(data)) — the frozen, non-differentiable boundary input the BLT
// forward consumes.
func (p *EntropyPatcher) Patch(ctx *backend.Context, data []byte) ([]int, error) {
	h, err := p.Entropies(ctx, data)
	if err != nil {
		return nil, err
	}
	return PatchLengths(h, p.Threshold, p.MaxPatch), nil
}

// ByteEntropy returns the Shannon entropy H = −Σ p·log p (nats) of the softmax
// distribution over one logit row — the per-position uncertainty signal entropy
// patching thresholds (arXiv:2412.09871 §2.3). Numerically stable (max-shifted
// softmax; zero-probability entries contribute exactly 0).
func ByteEntropy(logits []float64) float64 {
	probs := softmaxProb(logits)
	var h float64
	for _, p := range probs {
		if p > 0 {
			h -= p * math.Log(p)
		}
	}
	return h
}

// PatchLengths applies the global-threshold boundary rule to an entropy
// sequence: position 0 always starts the first patch; position i ≥ 1 starts a
// new patch when entropies[i] > threshold, or when the current patch already
// holds maxPatch bytes (maxPatch ≤ 0 → no cap). Returns the patch lengths
// (each ≥ 1, summing to len(entropies)); nil for an empty input.
func PatchLengths(entropies []float64, threshold float64, maxPatch int) []int {
	if len(entropies) == 0 {
		return nil
	}
	var lens []int
	cur := 1
	for i := 1; i < len(entropies); i++ {
		if entropies[i] > threshold || (maxPatch > 0 && cur >= maxPatch) {
			lens = append(lens, cur)
			cur = 1
			continue
		}
		cur++
	}
	return append(lens, cur)
}

// BLTConfig fixes the Byte Latent Transformer geometry. The byte alphabet fixes
// the vocabulary at 256; Dim is shared by the byte-level (encoder/decoder) and
// patch-level (latent) streams.
type BLTConfig struct {
	Ctx   int // max byte-sequence length
	Dim   int // embedding / model width (byte and patch streams)
	Heads int // self-attention heads (must divide Dim)
	// EncLayers is the number of local-encoder byte-level transformer blocks
	// (windowed causal self-attention) run before pooling. May be 0.
	EncLayers int
	// LatentLayers is the number of latent-transformer blocks over the patch
	// sequence (full causal attention). Must be ≥ 1 — this is the model.
	LatentLayers int
	// DecLayers is the number of local-decoder byte-level blocks run after the
	// byte→patch cross-attention. May be 0.
	DecLayers int
	// Window is the sliding-window width of the LOCAL byte-level self-attention
	// (encoder and decoder blocks): each byte attends to the Window most recent
	// bytes. 0 → full causal attention. The latent transformer always attends
	// over all previous patches.
	Window int
	// HashVocab enables hash n-gram byte embeddings when > 0: for each n in
	// HashNGrams a rolling FNV-1a hash of the trailing n bytes indexes an extra
	// [HashVocab, Dim] embedding table added to the byte embedding
	// (arXiv:2412.09871 §3.2). 0 → plain byte+position embeddings only.
	HashVocab int
	// HashNGrams lists the n-gram sizes (each ≥ 1) hashed into their own table.
	// Empty with HashVocab > 0 → {3}.
	HashNGrams []int
	Eps        float64 // LayerNorm epsilon (≤0 → 1e-5)
	// PatcherDim/PatcherHeads/PatcherLayers size the built-in entropy patcher.
	// PatcherLayers 0 → no patcher is built (Patcher stays nil): supply frozen
	// patch lengths yourself, or attach an EntropyPatcher before Generate.
	PatcherDim    int // entropy-patcher embedding width
	PatcherHeads  int // entropy-patcher attention heads
	PatcherLayers int // entropy-patcher blocks; 0 → no built-in patcher
	// EntropyThreshold seeds the built-in patcher's boundary threshold (nats).
	EntropyThreshold float64
	// MaxPatchLen seeds the built-in patcher's patch-length cap; 0 → no cap.
	MaxPatchLen int
}

// BLT is the Byte Latent Transformer (Pagnoni et al. 2024, "Byte Latent
// Transformer: Patches Scale Better Than Tokens", arXiv:2412.09871, Meta) — a
// tokenizer-free byte-level language model. Where GPT (gpt.go) spends the same
// compute on every fixed token from a learned vocabulary, BLT reads RAW BYTES
// and groups them into variable-length PATCHES wherever a small entropy model
// (EntropyPatcher) finds the stream predictable, so the expensive latent
// transformer runs once per patch, not once per byte. Four stages:
//
//  1. EntropyPatcher: a small frozen byte-LM turns next-byte entropy into patch
//     boundaries (host-side, non-differentiable — faithful to the paper).
//  2. LocalEncoder: byte + position (+ optional hash n-gram) embeddings through
//     EncLayers windowed causal blocks, then bytes→patches pooling: mean-pool
//     each patch span, refined by one residual masked cross-attention where
//     patch-query i attends ONLY the bytes of its span (composed
//     matmul→mask→softmax→matmul primitives, all differentiable).
//  3. LatentTransformer: LatentLayers causal GPT blocks over the patch sequence
//     [numPatches, Dim] — numPatches is a per-forward host-side dimension, no
//     ragged tensors needed.
//  4. LocalDecoder: each byte cross-attends (masked, gated) to the latent
//     outputs of STRICTLY EARLIER patches, then DecLayers windowed causal byte
//     blocks and a 256-way byte head produce next-byte logits.
//
// Contrast with the siblings: GPT is token-based and fixed-granularity;
// DiffusionLM (diffusion_lm.go) generates by iterative unmasking over discrete
// tokens; nn/ddpm.go diffuses continuous vectors. BLT stays autoregressive like
// GPT but replaces the tokenizer with learned, dynamically sized byte patches —
// compute follows the data's information density.
//
// In plain terms: instead of reading text through a fixed dictionary of word
// pieces, this model reads raw letters and lets a small helper decide where the
// text gets interesting. Boring stretches are skimmed as one big chunk; tricky
// spots are read letter by letter. The heavyweight "thinking" model only works
// chunk-by-chunk, and a light "spelling" model turns its thoughts back into
// letters one at a time.
type BLT struct {
	Config BLTConfig // the model hyperparameters
	// Patcher is the entropy model that segments bytes into patches. It is
	// trained SEPARATELY (Patcher.Loss) and FROZEN for the main model: Params
	// excludes it and its boundaries carry no gradient. nil when
	// Config.PatcherLayers is 0 — attach one before calling Generate.
	Patcher *EntropyPatcher
	ByteEmb *tensor.Tensor // byte embedding table [256, Dim]
	PosEmb  *tensor.Tensor // byte position embedding table [Ctx, Dim]
	// HashEmb holds one [HashVocab, Dim] table per Config.HashNGrams entry
	// (hash n-gram embeddings, added to the byte embedding); empty when off.
	HashEmb   []*tensor.Tensor
	EncBlocks []*Block // local-encoder byte-level blocks (windowed causal)
	// PoolWq/PoolWk/PoolWv/PoolWo are the bytes→patches pooling cross-attention
	// projections [Dim, Dim] (patch queries over byte keys/values, span-masked).
	PoolWq, PoolWk, PoolWv, PoolWo *tensor.Tensor
	LatentBlocks                   []*Block      // latent-transformer blocks over patches (full causal)
	LatentLN                       *nn.LayerNorm // final LayerNorm of the latent stream (the GPT LNf analogue)
	// CrossWq/CrossWk/CrossWv/CrossWo are the decoder byte→patch cross-attention
	// projections [Dim, Dim] (byte queries over latent patch keys/values).
	CrossWq, CrossWk, CrossWv, CrossWo *tensor.Tensor
	DecBlocks                          []*Block       // local-decoder byte-level blocks (windowed causal)
	DecLN                              *nn.LayerNorm  // final LayerNorm before the byte head
	Head                               *tensor.Tensor // byte LM head [Dim, 256]
}

// NewBLT builds a randomly initialized Byte Latent Transformer (and, when
// cfg.PatcherLayers > 0, its entropy patcher from a derived seed). Deterministic
// via seed; weight matrices and embeddings are Xavier-uniform (nn.XavierUniform,
// the repo's trained-from-scratch default), LayerNorms 1/0, FFN biases 0;
// parameters are F64 (the repo's training/gradcheck precision).
func NewBLT(cfg BLTConfig, seed uint64) (*BLT, error) {
	if cfg.Ctx <= 0 || cfg.Dim <= 0 {
		return nil, fmt.Errorf("nlp: NewBLT needs Ctx, Dim > 0 (got %d, %d)", cfg.Ctx, cfg.Dim)
	}
	if cfg.Heads <= 0 || cfg.Dim%cfg.Heads != 0 {
		return nil, fmt.Errorf("nlp: NewBLT dim %d not divisible by heads %d", cfg.Dim, cfg.Heads)
	}
	if cfg.LatentLayers <= 0 {
		return nil, fmt.Errorf("nlp: NewBLT needs LatentLayers ≥ 1, got %d", cfg.LatentLayers)
	}
	if cfg.EncLayers < 0 || cfg.DecLayers < 0 || cfg.Window < 0 || cfg.HashVocab < 0 {
		return nil, fmt.Errorf("nlp: NewBLT EncLayers/DecLayers/Window/HashVocab must be ≥ 0")
	}
	if cfg.HashVocab > 0 && len(cfg.HashNGrams) == 0 {
		cfg.HashNGrams = []int{3}
	}
	for _, n := range cfg.HashNGrams {
		if n < 1 {
			return nil, fmt.Errorf("nlp: NewBLT hash n-gram size must be ≥ 1, got %d", n)
		}
	}
	if cfg.Eps <= 0 {
		cfg.Eps = 1e-5
	}
	m := &BLT{
		Config:   cfg,
		ByteEmb:  bltRand(seed, bltVocab, cfg.Dim),
		PosEmb:   bltRand(seed+1, cfg.Ctx, cfg.Dim),
		PoolWq:   bltRand(seed+2, cfg.Dim, cfg.Dim),
		PoolWk:   bltRand(seed+3, cfg.Dim, cfg.Dim),
		PoolWv:   bltRand(seed+4, cfg.Dim, cfg.Dim),
		PoolWo:   bltRand(seed+5, cfg.Dim, cfg.Dim),
		CrossWq:  bltRand(seed+6, cfg.Dim, cfg.Dim),
		CrossWk:  bltRand(seed+7, cfg.Dim, cfg.Dim),
		CrossWv:  bltRand(seed+8, cfg.Dim, cfg.Dim),
		CrossWo:  bltRand(seed+9, cfg.Dim, cfg.Dim),
		Head:     bltRand(seed+10, cfg.Dim, bltVocab),
		LatentLN: bltLN(cfg.Dim, cfg.Eps),
		DecLN:    bltLN(cfg.Dim, cfg.Eps),
	}
	if cfg.HashVocab > 0 {
		for i := range cfg.HashNGrams {
			m.HashEmb = append(m.HashEmb, bltRand(seed+20+uint64(i), cfg.HashVocab, cfg.Dim))
		}
	}
	var err error
	if m.EncBlocks, err = newByteBlocks(cfg.EncLayers, cfg.Dim, cfg.Heads, cfg.Eps, seed+100); err != nil {
		return nil, err
	}
	if m.LatentBlocks, err = newByteBlocks(cfg.LatentLayers, cfg.Dim, cfg.Heads, cfg.Eps, seed+1000); err != nil {
		return nil, err
	}
	if m.DecBlocks, err = newByteBlocks(cfg.DecLayers, cfg.Dim, cfg.Heads, cfg.Eps, seed+2000); err != nil {
		return nil, err
	}
	if cfg.PatcherLayers > 0 {
		m.Patcher, err = NewEntropyPatcher(EntropyPatcherConfig{
			Ctx: cfg.Ctx, Dim: cfg.PatcherDim, Heads: cfg.PatcherHeads, Layers: cfg.PatcherLayers,
			Eps: cfg.Eps, Threshold: cfg.EntropyThreshold, MaxPatch: cfg.MaxPatchLen,
		}, seed+0x9e3779b9)
		if err != nil {
			return nil, err
		}
	}
	return m, nil
}

// Params returns every trainable tensor of the MAIN model — byte/position/hash
// embeddings, encoder blocks, pooling cross-attention, latent blocks and LN,
// decoder cross-attention, decoder blocks and LN, and the byte head — in a
// stable order for optimizers. The entropy patcher is EXCLUDED (frozen
// boundaries, per the paper); train it separately via Patcher.Params.
func (m *BLT) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.ByteEmb, m.PosEmb}
	ps = append(ps, m.HashEmb...)
	for _, b := range m.EncBlocks {
		ps = append(ps, blockParams(b)...)
	}
	ps = append(ps, m.PoolWq, m.PoolWk, m.PoolWv, m.PoolWo)
	for _, b := range m.LatentBlocks {
		ps = append(ps, blockParams(b)...)
	}
	ps = append(ps, m.LatentLN.Gamma, m.LatentLN.Beta)
	ps = append(ps, m.CrossWq, m.CrossWk, m.CrossWv, m.CrossWo)
	for _, b := range m.DecBlocks {
		ps = append(ps, blockParams(b)...)
	}
	ps = append(ps, m.DecLN.Gamma, m.DecLN.Beta, m.Head)
	return ps
}

// Forward computes next-byte logits [len(data), 256] for a byte sequence under
// frozen patch lengths (from EntropyPatcher.Patch, or any segmentation whose
// lengths are ≥ 1 and sum to len(data)). The full byte→patch→byte pass:
// embed (+hash n-grams) → encoder blocks → span-masked pooling → latent blocks
// → gated byte→patch cross-attention → decoder blocks → byte head. numPatches
// is a per-forward dimension baked into this call's dense mask/pooling tensors.
// Fully differentiable w.r.t. Params through a recording context; the patch
// lengths themselves carry no gradient (frozen, per the paper).
func (m *BLT) Forward(ctx *backend.Context, data []byte, patchLens []int) (*tensor.Tensor, error) {
	h, o, err := m.latentStates(ctx, data, patchLens)
	if err != nil {
		return nil, err
	}
	// Local decoder: byte queries over latent patch keys/values. Byte i may
	// attend only patches STRICTLY BEFORE its own (patch k's latent output
	// summarizes bytes through the end of patch k — own-patch attention would
	// leak future bytes). Bytes of patch 0 have no completed patch: their rows
	// are gated to zero instead of softmax-normalizing an all-masked row.
	mask, gate := bltDecoderMask(h.Dtype(), patchLens, len(data))
	x, err := bltCrossAttention(ctx, h, o, m.CrossWq, m.CrossWk, m.CrossWv, m.CrossWo, mask)
	if err != nil {
		return nil, err
	}
	if x, err = exec1(ctx, backend.OpMul, nil, x, gate); err != nil { // zero the patch-0 byte rows
		return nil, err
	}
	d, err := exec1(ctx, backend.OpAdd, nil, h, x)
	if err != nil {
		return nil, err
	}
	if d, err = runByteBlocks(ctx, d, m.DecBlocks, m.Config.Window); err != nil {
		return nil, err
	}
	if d, err = m.DecLN.Forward(ctx, d); err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, d, m.Head)
}

// ForwardLatent computes the LATENT-PATH logits [numPatches, 256]: the latent
// transformer's patch states through the shared byte head, skipping the local
// decoder. This is the patch-level view of the model — and the byte-GPT
// degeneracy pin: with stride-1 patches (every byte its own patch), no encoder
// blocks, and a zeroed PoolWo (identity pooling), it equals a plain byte-GPT
// built from the same latent weights exactly.
func (m *BLT) ForwardLatent(ctx *backend.Context, data []byte, patchLens []int) (*tensor.Tensor, error) {
	_, o, err := m.latentStates(ctx, data, patchLens)
	if err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, o, m.Head)
}

// Loss is the main training objective: mean next-byte cross-entropy of data[1:]
// under Forward's logits (rows 0..L−2 gathered differentiably, then
// nn.CrossEntropy). Needs at least 2 bytes. Fully differentiable w.r.t. Params
// through a recording context; the patcher and its boundaries stay frozen.
func (m *BLT) Loss(ctx *backend.Context, data []byte, patchLens []int) (*tensor.Tensor, error) {
	logits, err := m.Forward(ctx, data, patchLens)
	if err != nil {
		return nil, err
	}
	return nextByteCE(ctx, logits, data)
}

// Generate autoregressively extends prompt by up to maxNew bytes, re-running
// the entropy patcher on the growing sequence each step so patch boundaries are
// signalled ONLINE (the paper's inference mode). The sampler draws each byte
// from the last logit row (Greedy() is deterministic). Returns prompt+generated;
// stops at the context limit. Requires an attached Patcher.
func (m *BLT) Generate(prompt []byte, maxNew int, s *Sampler) ([]byte, error) {
	if m.Patcher == nil {
		return nil, fmt.Errorf("nlp: BLT.Generate needs an entropy patcher (Config.PatcherLayers > 0 or attach one)")
	}
	if len(prompt) == 0 {
		return nil, fmt.Errorf("nlp: BLT.Generate needs a non-empty prompt")
	}
	if s == nil {
		return nil, fmt.Errorf("nlp: BLT.Generate needs a non-nil sampler")
	}
	out := append([]byte(nil), prompt...)
	ctx := backend.NewContext()
	for range maxNew {
		if len(out) >= m.Config.Ctx {
			break
		}
		lens, err := m.Patcher.Patch(ctx, out)
		if err != nil {
			return nil, err
		}
		logits, err := m.Forward(ctx, out, lens)
		if err != nil {
			return nil, err
		}
		row := logitsRow(logits.Contiguous(), len(out)-1, bltVocab, bltVocab)
		out = append(out, byte(s.Sample(row)))
	}
	return out, nil
}

// latentStates runs stages 2–3: embeddings → encoder blocks → span-masked
// pooling → latent blocks → latent LN. Returns the byte states h [L, Dim] and
// the latent patch states o [numPatches, Dim].
func (m *BLT) latentStates(ctx *backend.Context, data []byte, patchLens []int) (h, o *tensor.Tensor, err error) {
	if err := validPatchLens(len(data), patchLens); err != nil {
		return nil, nil, err
	}
	if len(data) > m.Config.Ctx {
		return nil, nil, fmt.Errorf("nlp: BLT sequence length %d exceeds Ctx %d", len(data), m.Config.Ctx)
	}
	if h, err = m.embed(ctx, data); err != nil {
		return nil, nil, err
	}
	if h, err = runByteBlocks(ctx, h, m.EncBlocks, m.Config.Window); err != nil {
		return nil, nil, err
	}
	// Bytes→patches pooling: mean-pool each span (Mnorm·h — with stride-1
	// patches Mnorm is the identity), then one residual span-masked
	// cross-attention refinement (patch queries over byte keys/values).
	mnorm, span := bltPoolMask(h.Dtype(), patchLens, len(data))
	p0, err := exec1(ctx, backend.OpMatMul, nil, mnorm, h)
	if err != nil {
		return nil, nil, err
	}
	refine, err := bltCrossAttention(ctx, p0, h, m.PoolWq, m.PoolWk, m.PoolWv, m.PoolWo, span)
	if err != nil {
		return nil, nil, err
	}
	p, err := exec1(ctx, backend.OpAdd, nil, p0, refine)
	if err != nil {
		return nil, nil, err
	}
	if o, err = runByteBlocks(ctx, p, m.LatentBlocks, 0); err != nil {
		return nil, nil, err
	}
	if o, err = m.LatentLN.Forward(ctx, o); err != nil {
		return nil, nil, err
	}
	return h, o, nil
}

// embed gathers byte + position embeddings through the dispatch (gradients flow
// to the tables), then adds the optional hash n-gram embeddings: for each
// configured n, a host-side rolling FNV-1a hash of the trailing ≤n bytes indexes
// that n's table (OpEmbed, differentiable) and the rows are added in.
func (m *BLT) embed(ctx *backend.Context, data []byte) (*tensor.Tensor, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("nlp: BLT needs a non-empty byte sequence")
	}
	seq := len(data)
	tokIdx := tensor.New(m.ByteEmb.Dtype(), tensor.Shape{seq})
	posIdx := tensor.New(m.PosEmb.Dtype(), tensor.Shape{seq})
	for i, b := range data {
		tokIdx.SetF64(float64(b), i)
		posIdx.SetF64(float64(i), i)
	}
	et, err := exec1(ctx, backend.OpEmbed, nil, m.ByteEmb, tokIdx)
	if err != nil {
		return nil, err
	}
	ep, err := exec1(ctx, backend.OpEmbed, nil, m.PosEmb, posIdx)
	if err != nil {
		return nil, err
	}
	x, err := exec1(ctx, backend.OpAdd, nil, et, ep)
	if err != nil {
		return nil, err
	}
	for j, n := range m.Config.HashNGrams {
		idx := tensor.New(m.HashEmb[j].Dtype(), tensor.Shape{seq})
		for i, id := range hashNGramIDs(data, n, m.Config.HashVocab) {
			idx.SetF64(float64(id), i)
		}
		eh, err := exec1(ctx, backend.OpEmbed, nil, m.HashEmb[j], idx)
		if err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, eh); err != nil {
			return nil, err
		}
	}
	return x, nil
}

// bltCrossAttention is the composed-primitive masked single-head cross-attention
// shared by pooling (queries=patches, keys/values=bytes) and the decoder
// (queries=bytes, keys/values=patches): softmax(mask + (q·Wq)(kv·Wk)ᵀ/√Dim) ·
// (kv·Wv) · Wo. The mask is a host-built additive [len(q), len(kv)] tensor (0
// allowed, −1e30 excluded) — the nn/qknorm.go pattern, every op VJP'd, so the
// whole path is differentiable (unlike the fused inference-only OpMHAMasked).
// Single-head is a documented simplification of the paper's multi-head
// cross-attention.
func bltCrossAttention(ctx *backend.Context, q, kv, wq, wk, wv, wo, mask *tensor.Tensor) (*tensor.Tensor, error) {
	qp, err := exec1(ctx, backend.OpMatMul, nil, q, wq)
	if err != nil {
		return nil, err
	}
	kp, err := exec1(ctx, backend.OpMatMul, nil, kv, wk)
	if err != nil {
		return nil, err
	}
	vp, err := exec1(ctx, backend.OpMatMul, nil, kv, wv)
	if err != nil {
		return nil, err
	}
	kt, err := exec1(ctx, backend.OpTranspose, nil, kp)
	if err != nil {
		return nil, err
	}
	scores, err := exec1(ctx, backend.OpMatMul, nil, qp, kt)
	if err != nil {
		return nil, err
	}
	scale := tensor.New(scores.Dtype(), tensor.Shape{1, 1})
	scale.SetF64(1/math.Sqrt(float64(wq.Shape()[1])), 0, 0)
	if scores, err = exec1(ctx, backend.OpMul, nil, scores, scale); err != nil {
		return nil, err
	}
	if scores, err = exec1(ctx, backend.OpAdd, nil, scores, mask); err != nil {
		return nil, err
	}
	attn, err := exec1(ctx, backend.OpSoftmax, nil, scores)
	if err != nil {
		return nil, err
	}
	out, err := exec1(ctx, backend.OpMatMul, nil, attn, vp)
	if err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, out, wo)
}

// runByteBlocks runs pre-LN GPT blocks over x with causal self-attention
// restricted to a sliding window (0 → full causal) — the gpt.go hiddenFromEmbed
// block loop with AttnAttrs.Window exposed and no trailing LayerNorm (callers
// apply their own). With window 0 the op sequence is bit-identical to the GPT
// stack, which the byte-GPT collapse test pins.
func runByteBlocks(ctx *backend.Context, x *tensor.Tensor, blocks []*Block, window int) (*tensor.Tensor, error) {
	for _, b := range blocks {
		h, err := b.LN1.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		q, err := exec1(ctx, backend.OpMatMul, nil, h, b.Attn.Wq)
		if err != nil {
			return nil, err
		}
		k, err := exec1(ctx, backend.OpMatMul, nil, h, b.Attn.Wk)
		if err != nil {
			return nil, err
		}
		v, err := exec1(ctx, backend.OpMatMul, nil, h, b.Attn.Wv)
		if err != nil {
			return nil, err
		}
		a, err := exec1(ctx, backend.OpMHA, backend.AttnAttrs{Heads: b.Attn.Heads, Causal: true, Window: window}, q, k, v)
		if err != nil {
			return nil, err
		}
		if a, err = exec1(ctx, backend.OpMatMul, nil, a, b.Attn.Wo); err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, a); err != nil {
			return nil, err
		}
		if h, err = b.LN2.Forward(ctx, x); err != nil {
			return nil, err
		}
		if h, err = exec1(ctx, backend.OpMatMul, nil, h, b.W1); err != nil {
			return nil, err
		}
		if h, err = exec1(ctx, backend.OpAddBias, nil, h, b.B1); err != nil {
			return nil, err
		}
		if h, err = exec1(ctx, backend.OpGELU, nil, h); err != nil {
			return nil, err
		}
		if h, err = exec1(ctx, backend.OpMatMul, nil, h, b.W2); err != nil {
			return nil, err
		}
		if h, err = exec1(ctx, backend.OpAddBias, nil, h, b.B2); err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, h); err != nil {
			return nil, err
		}
	}
	return x, nil
}

// bltPoolMask builds the host-side pooling tensors for one forward: Mnorm
// [P, L] holds 1/len(span) over each patch's byte span (mean pooling; the
// identity matrix for stride-1 patches), and span [P, L] is the additive
// cross-attention mask (0 on-span, −1e30 off-span) restricting patch-query i to
// its own bytes.
func bltPoolMask(dtype tensor.Dtype, lens []int, seq int) (mnorm, span *tensor.Tensor) {
	p := len(lens)
	mnorm = tensor.New(dtype, tensor.Shape{p, seq})
	span = tensor.New(dtype, tensor.Shape{p, seq})
	start := 0
	for i, l := range lens {
		for j := range seq {
			if j < start || j >= start+l {
				span.SetF64(bltNegInf, i, j)
			}
		}
		for j := start; j < start+l; j++ {
			mnorm.SetF64(1/float64(l), i, j)
		}
		start += l
	}
	return mnorm, span
}

// bltDecoderMask builds the host-side decoder tensors for one forward: mask
// [L, P] admits byte i to patch k iff k < patchIndex(i) (strictly earlier
// patches — own-patch attention would leak future bytes through the patch
// representation), and gate [L, 1] is 0 for the bytes of patch 0 (no completed
// patch to attend) and 1 elsewhere, multiplied onto the cross-attention output
// so all-masked rows contribute exactly nothing.
func bltDecoderMask(dtype tensor.Dtype, lens []int, seq int) (mask, gate *tensor.Tensor) {
	p := len(lens)
	mask = tensor.New(dtype, tensor.Shape{seq, p})
	gate = tensor.New(dtype, tensor.Shape{seq, 1})
	pidx := make([]int, seq)
	start := 0
	for i, l := range lens {
		for j := start; j < start+l; j++ {
			pidx[j] = i
		}
		start += l
	}
	for i := range seq {
		for k := range p {
			if k >= pidx[i] {
				mask.SetF64(bltNegInf, i, k)
			}
		}
		if pidx[i] > 0 {
			gate.SetF64(1, i, 0)
		}
	}
	return mask, gate
}

// hashNGramIDs returns the per-position hash n-gram ids: id[i] is the FNV-1a
// hash of the trailing min(n, i+1) bytes ending at i, modulo vocab. Host-side
// and deterministic; the ids index the hash embedding tables.
func hashNGramIDs(data []byte, n, vocab int) []int {
	ids := make([]int, len(data))
	for i := range data {
		lo := i - n + 1
		if lo < 0 {
			lo = 0
		}
		h := uint64(14695981039346656037) // FNV-1a 64-bit offset basis
		for _, b := range data[lo : i+1] {
			h ^= uint64(b)
			h *= 1099511628211 // FNV-1a 64-bit prime
		}
		ids[i] = int(h % uint64(vocab))
	}
	return ids
}

// nextByteCE is the shared next-byte objective: gather logit rows 0..L−2
// (differentiable OpEmbed row-gather) and mean cross-entropy against data[1:].
func nextByteCE(ctx *backend.Context, logits *tensor.Tensor, data []byte) (*tensor.Tensor, error) {
	l := len(data)
	if l < 2 {
		return nil, fmt.Errorf("nlp: next-byte loss needs at least 2 bytes, got %d", l)
	}
	idx := tensor.New(logits.Dtype(), tensor.Shape{l - 1})
	targets := tensor.New(logits.Dtype(), tensor.Shape{l - 1})
	for i := range l - 1 {
		idx.SetF64(float64(i), i)
		targets.SetF64(float64(data[i+1]), i)
	}
	rows, err := exec1(ctx, backend.OpEmbed, nil, logits, idx)
	if err != nil {
		return nil, err
	}
	return nn.CrossEntropy(ctx, rows, targets)
}

// validPatchLens checks a frozen segmentation: at least one patch, every length
// ≥ 1, lengths summing to the byte count.
func validPatchLens(dataLen int, lens []int) error {
	if len(lens) == 0 {
		return fmt.Errorf("nlp: BLT needs at least one patch")
	}
	sum := 0
	for i, l := range lens {
		if l < 1 {
			return fmt.Errorf("nlp: BLT patch %d has length %d, need ≥ 1", i, l)
		}
		sum += l
	}
	if sum != dataLen {
		return fmt.Errorf("nlp: BLT patch lengths sum to %d, want %d", sum, dataLen)
	}
	return nil
}

// newByteBlocks builds n randomly initialized pre-LN GPT blocks of width dim
// (Xavier-uniform weights, causal attention) — the NewDiffusionLM block recipe
// with Causal set, shared by the BLT streams and the entropy patcher.
func newByteBlocks(n, dim, heads int, eps float64, seed uint64) ([]*Block, error) {
	var blocks []*Block
	for l := range n {
		s := seed + uint64(l)*8
		attn, err := NewMHA(heads,
			bltRand(s, dim, dim), bltRand(s+1, dim, dim),
			bltRand(s+2, dim, dim), bltRand(s+3, dim, dim))
		if err != nil {
			return nil, err
		}
		attn.Causal = true
		blocks = append(blocks, &Block{
			LN1: bltLN(dim, eps), LN2: bltLN(dim, eps), Attn: attn,
			W1: bltRand(s+4, dim, 4*dim), B1: tensor.New(tensor.F64, tensor.Shape{4 * dim}),
			W2: bltRand(s+5, 4*dim, dim), B2: tensor.New(tensor.F64, tensor.Shape{dim}),
		})
	}
	return blocks, nil
}

// blockParams lists one block's trainable tensors in the gpt.go Params order.
func blockParams(b *Block) []*tensor.Tensor {
	return []*tensor.Tensor{
		b.LN1.Gamma, b.LN1.Beta,
		b.Attn.Wq, b.Attn.Wk, b.Attn.Wv, b.Attn.Wo,
		b.LN2.Gamma, b.LN2.Beta,
		b.W1, b.B1, b.W2, b.B2,
	}
}

// bltRand returns a seeded Xavier-uniform F64 matrix [fanIn, fanOut].
func bltRand(seed uint64, fanIn, fanOut int) *tensor.Tensor {
	t := tensor.New(tensor.F64, tensor.Shape{fanIn, fanOut})
	nn.XavierUniform(t, fanIn, fanOut, seed)
	return t
}

// bltLN returns a fresh LayerNorm (γ=1, β=0) of width d with the given epsilon.
func bltLN(d int, eps float64) *nn.LayerNorm {
	l := nn.NewLayerNorm(tensor.F64, d)
	l.Eps = eps
	return l
}

// bytesToInts widens a byte sequence to the []int token form the GPT backbone
// consumes (byte value = token id; tokenizer-free).
func bytesToInts(data []byte) []int {
	out := make([]int, len(data))
	for i, b := range data {
		out[i] = int(b)
	}
	return out
}
