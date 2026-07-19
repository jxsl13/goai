# GoAI

[![ci](https://github.com/jxsl13/goai/actions/workflows/ci.yml/badge.svg)](https://github.com/jxsl13/goai/actions/workflows/ci.yml)

A Go-native, full-spectrum AI library: tensors, autograd, training, transformer
inference, quantized GGUF (llama.cpp’s single-file model format) models, vision, classic ML and RL — **pure Go first,
zero dependencies, no C toolchain required**.

In plain terms: everything you need to load a language model, generate text,
or train a network — in ordinary Go, with `go get`, on any platform. GPU
acceleration (Metal, Vulkan, CUDA) is optional and switches on by itself when
compiled in; without it, everything still runs.

Every operation ships as a validated pure-Go reference first — cross-checked
against PyTorch, tiktoken and ggml goldens at fixed tolerances — and is only
then optimized against that reference. `CGO_ENABLED=0 go test ./...` is green
on Linux, macOS and Windows.

**How fast is it, really?** [`BENCHMARKS.md`](BENCHMARKS.md) puts GoAI next
to the incumbents — llama.cpp, vLLM, PyTorch, NumPy, scikit-learn — on the
same machine and the same model weights, by category, wins and losses alike:
GPU decode ahead of llama.cpp-Q8 at every measured scale (up to **1.19×** at
Mistral-7B), CPU GEMM at **torch-cpu parity** (ahead on large shapes, in pure
Go), scikit-learn **beaten or matched on every classical method**, and the
open gaps (vLLM prefill, serving throughput) stated with root causes.

## Quickstart

Load a quantized llama.cpp model and generate text — in plain terms: open a
ready-made model file, ask it a question, print its answer:

```go
f, err := gguf.ReadFile("model.gguf") // any llama.cpp GGUF: K-quants, i-quants, MXFP4
if err != nil {
	log.Fatal(err)
}

model, err := nlp.LlamaFromGGUF(f.Metadata, f.Tensors)
tok, err := nlp.BPEFromGGUF(f.Metadata)
tpl, err := nlp.ChatTemplateFromGGUF(f.Metadata) // chatml/llama3/gemma/mistral/phi3

prompt, _ := tpl.Render([]nlp.ChatMessage{
	{Role: "user", Content: "Why is the sky blue?"},
}, nlp.WithGenerationPrompt())

sampler := nlp.NewSampler(42, nlp.WithTemperature(0.8), nlp.WithTopP(0.95),
	nlp.WithMinP(0.05), nlp.WithDRY(0.8)) // DRY sequence-repetition penalty
out, err := model.Generate(tok.Encode(prompt), 256, sampler)
fmt.Println(tok.Decode(out))
```

Force the model to emit JSON that conforms to a schema (structured outputs) —
in plain terms: instead of hoping the model answers in the right format, make
any other output literally impossible:

```go
grammar, err := nlp.JSONSchemaToGrammar([]byte(`{
	"type": "object",
	"properties": {"city": {"type": "string"}, "temp_c": {"type": "number"}},
	"required": ["city", "temp_c"]
}`))
// vocab[i] is token i's text, eosID the end token — both from the GGUF metadata.
guide, err := nlp.NewGrammarGuide(grammar, vocab)          // pushdown automaton over the vocab
guided := guide.Sampler(sampler, eosID)                    // masks every off-grammar token
out, err = model.Generate(tok.Encode(prompt), 256, guided) // guaranteed parseable JSON
```

Train a model — here the Vision Transformer on a toy task — with the standard
tape/optimizer loop. In plain terms: show the model labeled examples, measure
how wrong it is, nudge every parameter downhill, repeat:

```go
model, err := vision.NewViT(1, 8, 3 /*classes*/, seed, vision.WithViTDtype(tensor.F64))
opt := nn.NewAdamW(model.Params(), 0.01, 0)
for step := 0; step < 150; step++ {
	tape := autograd.NewTape()
	logits, err := model.Forward(tape.Context(), images) // [batch, classes]
	loss, err := nn.CrossEntropy(tape.Context(), logits, targets)
	if err := tape.Backward(loss); err != nil {
		log.Fatal(err)
	}
	clipped, _ := nn.ClipGradNorm(model.Params(), tape.Grad, 1.0)
	if err := opt.Step(clipped); err != nil {
		log.Fatal(err)
	}
}
```

## What works today

- **Thirty-one loadable LLM architectures, transformers-anchored**: load real
  Hugging Face checkpoints of Llama/Mistral, Qwen2/2.5, Qwen3, Phi-1/2/3,
  Gemma/Gemma 2, Cohere Command-R, OLMo 2, GPT-NeoX/Pythia, StableLM,
  StarCoder2, Granite, Nemotron, MPT (ALiBi), Falcon (MQA), the MoE family
  (Mixtral, Qwen2/Qwen3-MoE, GraniteMoE, OLMoE), DeepSeek-V2 (Multi-head
  Latent Attention + MoE), the state-space models Mamba/Mamba-2, RWKV, the
  Jamba hybrid, plus BERT/RoBERTa/DistilBERT/T5 — every one anchored bit- or
  tolerance-exact (~1e-8) against the transformers reference, fine-tunable
  (LoRA included), and generating with architecture-native serving: batched
  prompt Prefill (bit-identical to stepwise), amortized KV caches, sparse
  top-k MoE decode, O(1) recurrent state, and DeepSeek's absorbed-latent cache
  (6.7× less KV memory).
- **Train end-to-end**: full backward validated against real torch gradients at
  f64 rtol ~1e-9, checkpoint (safetensors / .pt via a safe no-code-execution
  loader / GGUF round-trips — GGUF also loads llama.cpp community files for
  **twenty architectures** (the Llama/Qwen/Gemma/Phi families through Mixtral,
  DeepSeek-V2, Command-R, Granite, and the Mamba/Mamba-2/Jamba state-space
  tier), each converter's layout conventions verified against llama.cpp source,
  and **decodes quantized checkpoints of ALL TWENTY directly from the
  ggml Q-blocks** — packed projections, rotary row interleaves, and fused MoE
  expert tensors all unpacked losslessly on the quantized bytes, never
  dequantized and requantized; the full matrix and its verification
  methodology live in [`docs/gguf.md`](docs/gguf.md)), tokenize (BPE
  bit-exact vs tiktoken, SentencePiece-Unigram, WordPiece), and generate.
- **Structured generation**: GBNF (llama.cpp’s grammar notation) grammar-constrained decoding compiled to a
  pushdown automaton (nested JSON a regex guide structurally cannot enforce),
  a JSON-Schema→grammar compiler, regex/FSM (finite-state machine) guiding, and chat templates for
  the chatml/llama3/gemma/mistral/phi3 families byte-exact against the HF (HuggingFace)
  reference renders.
- **Sampling, complete**: temperature, top-k/p, min-p, epsilon/eta, typical,
  Mirostat, repetition/frequency/presence penalties, **DRY** sequence-repetition
  and **XTC** top-choice exclusion, classifier-free guidance, DoLa,
  contrastive search & decoding, beam/diverse beam, watermarking.
- **GPU inference (Metal + Vulkan/MoltenVK)**: the `llamagpu` batched decoders
  record a whole decode step into one command buffer over device-resident
  weights and KV (key/value attention cache) cache — measured **24× (Metal) / 21× (Vulkan)** over per-op
  decode, **41×** prompt prefill, cooperative long-context attention on every
  surface, and quantized (ggml Q-block) decode at 4–8× less weight memory.
- **Accelerated decoding, measured on real trained models**: Medusa
  (**1.81×**, 97% acceptance), lossless prompt-lookup (**1.80×**), draft-model
  speculative decoding, tree-Medusa (candidate trees verified in ONE masked
  forward), Jacobi decoding, Self-Extend (4× the training length, no
  fine-tuning). Numbers and method: [`docs/benchmarking.md`](docs/benchmarking.md),
  [`docs/inference.md`](docs/inference.md).
- **Interpretability — see what a model is "thinking"**: a pure-Go port of
  Anthropic's **Jacobian lens / J-space** (2026): fit per-layer expected
  Jacobians on any GoAI decoder (validated against Anthropic's reference
  implementation to ~1e-7), import reference-fitted `.pt` lenses, decompose
  activations into the sparse verbalizable J-space, and render a
  self-contained interactive HTML view of which concepts every layer holds at
  every position during generation — including concepts the model never writes
  down (the paper's global-workspace effect, reproduced end-to-end on a GoAI
  model in the test suite). Method, API and validation:
  [`docs/interpretability.md`](docs/interpretability.md).
- **Training toolbox** (`nn`): optimizers from SGD to Muon/SOAP/Sophia/
  Schedule-Free with composable wrappers (SAM, Lookahead, GaLore, …), the PEFT (parameter-efficient fine-tuning)
  family (LoRA (low-rank adapters: train tiny add-on matrices instead of the full model)/DoRA/PiSSA/VeRA/IA³/prefix/prompt, QLoRA end-to-end),
  quantization & pruning (AWQ/GPTQ/HQQ/NF4/SparseGPT/Wanda), post-transformer
  blocks (MoE, Mamba/Mamba-2 SSD, RWKV with O(1) recurrent inference, RetNet,
  MLA (multi-head latent attention (DeepSeek’s compressed-cache attention)), Gated DeltaNet/GLA/KDA — every family proven end-to-end as a trained
  char-LM), diffusion (DDPM/DDIM/EDM/flow matching), SSL, RLHF (reinforcement learning from human feedback)/alignment
  (GRPO/GSPO and the DPO (direct preference optimization) family on a real model, including a
  reproduced-and-mitigated reward-hacking case study —
  [`docs/alignment.md`](docs/alignment.md)), continual learning, model merging.
- **Multimodal**: Vision Transformer (torch-parity forward), SigLIP sigmoid
  contrastive loss, and a LLaVA-style projector that feeds ViT (Vision Transformer) patch tokens
  into the GPT decoder — trained image→caption end-to-end.
- **Vision, classic ML & RL**: CNN (convolutional neural network) classifier over the NCHW (the batch×channel×height×width tensor layout) conv/pool stack,
  linear/softmax regression, K-Means, PCA, RL environments and canonical
  agents.
- **Formats**: safetensors (reads every official dtype including FP8
  E4M3/E5M2 — DeepSeek-V3-class checkpoints — and all integer widths), GGUF
  (every ggml quant reads: K-quants, all 8 i-quants, MXFP4), npy/npz.

Every algorithm carries its paper citation (§R in [`SPEC.md`](SPEC.md)) and is
validated on the §V16 ladder: tier-1 parity against an official reference
where one exists, tier-2 the defining paper. Each package doc ends with
further-reading pointers (papers, surveys, textbooks) for going deeper.

## Layout

| Layer | Package | Role |
|------|---------|------|
| L0 | `tensor` | Tensor, Dtype, strides/views |
| L1 | `backend`, `backend/ref` | op dispatch + pure-Go reference (numerical truth) |
| L1b | `backend/cpu`, `backend/metal`, `backend/vulkan`, `backend/cuda` | optimized + GPU backends, auto-selected, always with fallback |
| L2 | `autograd` | reverse-mode tape + VJPs |
| L3 | `nn`, `ops`, `linalg` | layers, optimizers, losses; eager op API |
| L4 | `nlp`, `vision`, `classic`, `rl` | domain packages |
| L5 | `format` | safetensors, GGUF, npy/npz |
| — | `llamagpu` | batched GPU decoding for GPT/Llama (Metal/Vulkan) |

## Build & verify

```sh
make build        # CGO_ENABLED=0 pure-Go build
make vet test     # pure-Go gate: compiles tests too
make metal-test   # Metal/MPS backend suite (darwin, cgo)
make vulkan-test  # Vulkan backend suite (MoltenVK on macOS)
make bench-compare  # cross-backend benchmark harness
```

Requires Go 1.26+. No C toolchain needed for the default build. Architecture
and task history live in [`SPEC.md`](SPEC.md) (caveman-encoded, see
[`FORMAT.md`](FORMAT.md)); design rationale in [`docs/`](docs/); the
categorical comparison against the industry incumbents in
[`BENCHMARKS.md`](BENCHMARKS.md), with the full measurement log and policy in
[`docs/benchmarking.md`](docs/benchmarking.md); the GGUF architecture matrix
and its convention-verification methodology in [`docs/gguf.md`](docs/gguf.md).

## License

GoAI is licensed under the **Mozilla Public License 2.0** (see
[LICENSE.md](LICENSE.md)): file-level copyleft. In practice:

- **Using GoAI in your product** — statically or dynamically linked, open or
  closed source — imposes **no obligations on your own code**.
- **Modifying GoAI's files** — those modified files (and only those) must remain
  available under the MPL, with source offered to recipients of your binaries.
