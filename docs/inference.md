# Inference on trained models: measured findings (§T472–§T477, §T504–§T513)

Companion to [`benchmarking.md`](benchmarking.md) (speed) and
[`alignment.md`](alignment.md) (training-time alignment): what the inference
features actually deliver on REAL trained models. Models are trained in-repo per
test run (§T434 pattern — char-grammar corpus, GPT on Metal in ~9s, Llama on the
CPU backend in ~3s), so every number re-derives with the full suite; all tests are
`-short`-skipped.

| Claim | Result | Test |
|-------|--------|------|
| Distilled drafts beat independent drafts at speculative decoding | 73% → 88% acceptance, same arch/steps/init | `llamagpu/distill_spec_test.go` |
| Mirostat controls surprise to τ | One-sided: tracks τ BELOW the model's entropy (0.5→0.87 bits from 1.80); saturates at plain sampling above it | `llamagpu/mirostat_trained_test.go` |
| Soft watermark (γ=0.25, δ=2) is detectable | z=4.08 at 50 tokens, 9.07 at 300, on ~1.8-bit/token text; plain text never flagged | `llamagpu/watermark_trained_test.go` |
| Bounded KV cache decodes at full quality | 68 rows vs 300+: +0.05 bits/token | `llamagpu/streaming_trained_test.go` |
| Streaming generates far past the training context | 4× context on a RoPE model: far-tail 0.95 bits vs 0.35 in-context — no cliff | `llamagpu/streamgen_trained_test.go` |
| Quantization costs quality | Q8_0 +0.001 bits / 99% agreement; Q4_0 −0.013 / 97% — near-lossless here | `llamagpu/quant_quality_test.go` |
| K-quants: the ladder shows in AGREEMENT | Q6_K 97%, Q4_K 97%, Q2_K 86% — Q2_K flips 14% of decisions even where its small-window CE looks harmless; agreement is the sensitive metric under aggressive quantization | `llamagpu/kquant_quality_test.go` |
| Contrastive decoding contrasts | β=0 ≡ expert exactly; β=0.5 raises the amateur's surprise (1.19→1.33 bits) while expert-surprise even drops (0.85→0.81) | `llamagpu/contrastive_trained_test.go` |
| Beam search beats greedy likelihood | beam-4 +0.31 nats over greedy (greedy ≠ sequence argmax); scores match independent scoring; diverse beam: 4/4 distinct groups at bounded cost | `llamagpu/beam_trained_test.go` |
| DoLa contrasts layers within its guarantee | α=1 ≡ greedy; α=0.1 raises early-exit surprise 1.0→3.1 bits, mature stays within the log₂(1/α) plausibility bound; α=0.5 tightens to below greedy | `llamagpu/dola_trained_test.go` |
| Tree-Medusa never decodes below chain-Medusa | trained heads, same base: tree (topK=2) 4.00 tok/round vs chain 3.92; the all-top-1 chain path is a member of every round's tree | `llamagpu/medusa_tree_trained_test.go` |
| Self-Extend extrapolates length with NO fine-tuning | trained only on 32-token windows, evaluated at 4×: plain CE 0.316→1.488 beyond training length; Self-Extend (w=8, G=8) holds 0.515 | `llamagpu/self_extend_trained_test.go` |
| Self-Extend stays nearly FLAT out to 8× | extension curve, far-half CE at 2×/4×/8× training length: plain 0.91→1.95→2.40 (marching toward uniform ≈2.9); Self-Extend (w=8, G per length) 0.57→0.68→0.70 | `llamagpu/self_extend_curve_test.go` |
| Self-Extend GENERATION stays coherent at 4× | far-half windowed surprise of generated text: Self-Extend 0.50 (≈ trained level) vs plain greedy 2.30 (degenerated) | `llamagpu/self_extend_trained_test.go` |

## Sharpened understandings

- **Masks and merged scores are different primitives.** Tree attention (Medusa's
  candidate tree) needs only an additive mask per pair (`mha_masked`, §T507);
  Self-Extend needs per-pair CHOICE between two score sources inside ONE softmax
  (`mha_select`, §T512) — an additive mask cannot express it. Getting this wrong
  cost one corrected over-claim (§T508).
- **Branching helps only where top-1 fails.** At toy scale Medusa acceptance sits
  near ceiling (143/147), so the tree's extra candidates rarely fire — the
  tree/chain gap is a headroom effect, not a constant factor (§T510).
- **Mirostat is a surprise CEILING, not a thermostat.** It only truncates
  high-surprise tokens; with τ above the model's entropy the filter vanishes and
  sampling is plain. Raising surprise needs temperature (which reshapes the
  distribution: 1.8 → 3.42 bits where the model's natural rate is 1.80).
- **The attention-sink phenomenon needs deep models.** At 2–3 layers the
  sink-free ablations matched sinks in BOTH the bounded-cache and the
  beyond-context experiments. The mechanism (pre-RoPE cache, cache-relative
  re-rotation) is implemented per the paper and the bounded-cache/long-generation
  results stand on their own; the specific sink dependence did not manifest at
  toy scale.
- **Watermarking's low-entropy caveat has a threshold.** ~1.8 bits/token was
  comfortably enough entropy for δ=2 detection from 50 tokens.

## Measurement discipline (the gotchas these experiments caught)

- **Interleave A/B timing runs** (A,B,A,B medians) — sequential blocks hand the
  first scheme any cold-start/thermal outlier (a spurious 4.57× once, §T452).
- **Compare agreement TEACHER-FORCED** — free-running comparisons diverge at the
  first mismatch and then score different contexts (a 63% "agreement" artifact
  that was really 99%, §T477).
- **Assert what the theory guarantees, log the rest.** The Mirostat τ=4 and
  reward-hacking cases both "failed" naive assertions exactly where theory says
  they must — the failures were the findings (§T473, §T469).
