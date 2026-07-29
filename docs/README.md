# GoAI documentation — reading guide

> **In plain terms:** this folder holds the project's long-form documentation.
> Each page answers one kind of question; this index tells you which page to
> open for the question you actually have. API documentation itself lives in
> the code as godoc (run `go doc` or read pkg.go.dev) — these pages are the
> measured findings, formats, and design rationale around it.

| Your question | Read |
|---|---|
| How fast is GoAI against llama.cpp, vLLM, PyTorch, NumPy, scikit-learn? | [`../BENCHMARKS.md`](../BENCHMARKS.md) — the curated, categorical comparison |
| Where do all the raw numbers, rules and optimization history live? | [`benchmarking.md`](benchmarking.md) — the running measurement log and the method (§V22) behind every speed claim |
| What do the inference features (speculative decoding, quantization, watermarking, long context, prefix caching) actually deliver? | [`inference.md`](inference.md) — measured on models the test suite trains from scratch |
| What do the training features (optimizers, wrappers, diffusion, continual learning, merging, SSL, PEFT) deliver? | [`training.md`](training.md) |
| How does alignment work here (GRPO/GSPO, the DPO family, reward models) — and what does reward hacking look like? | [`alignment.md`](alignment.md) — recipes plus a reproduced-and-mitigated failure case |
| Can I see what a model is "thinking"? | [`interpretability.md`](interpretability.md) — the Jacobian-lens port, validated against the reference implementation |
| Which GGUF (llama.cpp model-file) checkpoints load, and how were the loaders verified? | [`gguf.md`](gguf.md) — the architecture × quantization matrix and its convention-verification method |
| Why is X designed this way? | [`decisions/`](decisions/) — Architecture Decision Records (ADR-0001 …), one dated file per decision, including rejected ideas with their measurements |
| What low-level tricks did the per-backend optimization passes use? | `perf-notes-cpu.md`, `perf-notes-ref.md`, `perf-notes-cuda.md`, `perf-notes-lowlevel.md` — dated technique logs from the §V22 sweeps |
| Why is a host loop still serial, and what makes parallelizing it safe? | `perf-notes-parallel.md` — the scaling sweep, the bit-identity patterns, and the allocation axis |
| A loop is slow but the arithmetic looks minimal — what shape is it? | [`perf-notes-memory-order.md`](perf-notes-memory-order.md) — traversal order, sparse masks and sort-vs-selection, with the bit-identity precondition for each fix |
| What research grounded the project's scope and validation approach? | [`research/`](research/) — the landscape and validation snapshots the spec was built on |

Conventions all pages follow (from [`../SPEC.md`](../SPEC.md) §C10/§C13):
every page is written for two audiences at once — a practitioner who wants
the math and the paper citations, and a newcomer who wants plain language;
abbreviations are expanded at first use; every page ends with further-reading
pointers; and every number is reproducible from a committed harness — the
page tells you which command or test regenerates it.
