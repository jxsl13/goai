# Interpretability: the Jacobian lens (J-lens / J-space)

GoAI ships a pure-Go port of Anthropic's **Jacobian lens** — the method from
"Verbalizable Representations Form a Global Workspace in Language Models"
(2026) — validated against the [reference implementation](https://github.com/anthropics/jacobian-lens).
It answers a concrete question about any GoAI decoder: *what concepts is the
model holding at each layer and position while it generates* — including
concepts it never writes into the output (the paper's "global workspace"
effect).

In plain terms: a language model's hidden state at a middle layer is not
directly readable — the usual "logit lens" trick (project it through the output
head) is distorted because the later layers still transform it. The J-lens
learns, once, how each layer's state maps to the final layer (an expected
Jacobian), and uses that to read the middle state *as if* it had finished
processing. Applied across every layer and position, it produces a grid of "what
the model is thinking here."

## The method

For each layer `l`, fit the **expected Jacobian** of the final residual stream
with respect to that layer's residual stream, averaged over a generic corpus:

```
J_l = E[ ∂h_L / ∂h_l ]
```

Reading an activation is then a corrected logit lens — transport it into the
final basis, then unembed:

```
readout(h_l) = W_U · norm(J_l · h_l)
```

The **J-space** is the set of activations expressible as a sparse, nonnegative
combination of at most `k` (default ≤ 25) of these per-token J-lens directions —
the small "verbalizable" subspace the paper identifies as the workspace.

Two details GoAI verified against the reference implementation and that the
paper's prose omits (SPEC §R250):

- The estimator masks **invalid positions**: the final position (no next-token
  target) is always excluded, and `WithJLensSkipFirst(n)` drops leading
  attention-sink positions — from both the target sum and the source-average
  divisor.
- The reference `.pt` artifact is `{"J": {layer: fp16 [d,d]}, ...}` with
  block-output layer indexing (reference layer `l` = GoAI layer `l+1`), stored
  row-major `J[out,in]`, applied verbatim.

## API

Everything lives in package `nlp`.

**Fit** — accumulate the expected Jacobians from a token corpus (one VJP per
target position per direction; bound it for large models with the options):

```go
lens, err := nlp.FitJLens(model, corpus,
	nlp.WithJLensSkipFirst(16),        // drop attention-sink positions
	nlp.WithJLensMaxSequences(1000),   // cap corpus size
	nlp.WithJLensMaxTargetsPerSeq(64)) // cap targets per sequence
```

`model` is any `ResidualModel` — every GoAI decoder (`Llama`, `GPT`, and
therefore any model loaded via `LlamaFromGGUF`) implements it. `Merge(other,
weight)` combines lenses fitted on disjoint corpus slices.

**Persist / import** — save and load GoAI-native lenses, or import a
reference/Neuronpedia-fitted `.pt`:

```go
err := lens.Save("lens.safetensors")
lens, err := nlp.LoadJLens("lens.safetensors")
lens, err := nlp.JLensFromPT("reference-lens.pt") // Anthropic / Neuronpedia artifacts
```

**Read** — apply the transported readout at a layer/position, or produce the
whole grid:

```go
readout, err := lens.ApplyAt(model, ctx, tokens, layer, position) // ranked tokens + RankOf
slice, err := lens.Slice(model, ctx, tokens)                      // [layers+1 × positions] grid
```

**Decompose** — the sparse nonnegative J-space of an activation, with the
random-direction occupancy control:

```go
decomp, err := nlp.JSpaceDecompose(lens, model, h, layer, 25,
	nlp.WithJSpaceControlDirections(64), nlp.WithJSpaceSeed(1))
```

**Visualize** — one self-contained interactive HTML page (layer × position
grid; click a cell to pin its token → rank-across-layers chart + heatmap;
optional J-space occupancy badges). Zero external requests, deterministic
output:

```go
page, err := nlp.JLensHTML(model, lens, tokens,
	nlp.WithJLensHTMLTitle("what the model holds"),
	nlp.WithJLensHTMLTokenText(func(id int) string { return vocab[id] }),
	nlp.WithJLensHTMLOccupancy(8))
```

## On a downloaded model

Because a GGUF-loaded model is a normal `*Llama`, the whole pipeline works on a
real checkpoint with no extra wiring — load, fit, render:

```go
f, _ := gguf.ReadFile("model.gguf")
model, _ := nlp.LlamaFromGGUF(f.Metadata, f.Tensors)
lens, _ := nlp.FitJLens(model, corpus)
page, _ := nlp.JLensHTML(model, lens, tokens)
```

`ExampleLlamaFromGGUF_jlens` exercises this end to end; a lens fitted on a
GGUF-round-tripped model matches the fit on the original to the F32 weight floor
(≈9e-7).

## Validation and honest scope

- **Fit and readout are tier-1 anchored** against the actual reference
  implementation (run in a venv on a tiny model): fit-parity 1.85e-07,
  readout-parity 1.79e-08, `.pt` import at the fp16 storage floor (4.7e-04).
- **The workspace effect reproduces in miniature**: on a GPT trained on lag-2
  permutation dynamics (where the model must *hold* a token two steps before
  emitting it), the J-lens ranks the held future token at mean rank 1.17 while
  the plain logit lens sits at 6.07 of vocab 12 — the "thinking about X while
  writing Y" phenomenon, on a GoAI model.
- **The J-space decomposition is tier-2** (property-anchored, not
  reference-anchored): the reference repository ships no decomposition code, so
  it follows the §R250 recipe and is validated by properties — planted sparse
  combinations recovered to 1e-12, isotropic noise at occupancy ≈ 0.2.
- **Scale caveat**: the paper's quantitative results are for Claude- and
  Gemma-scale models; the *phenomenology* is weaker on tiny models, so the
  runnable replication here is a miniature. The ported experiment/eval prompt
  sets (`nlp/testdata/jlens_prompts/`, Apache-2.0) are for replicating the
  paper's experiments on a real fitted model.

## Further reading

- The `nlp` package doc summarizes the interpretability surface; each J-lens
  function carries dual-audience godoc.
- [`SPEC.md`](../SPEC.md) §R250 records the method and the reference-verified
  corrections; §T810–T818 the build.
- The paper: Anthropic 2026, "Verbalizable Representations Form a Global
  Workspace in Language Models"; reference code at
  <https://github.com/anthropics/jacobian-lens> (Apache-2.0).
