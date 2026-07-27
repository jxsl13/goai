---
schema: v1
---

## T-01KYJNDQJZEW0SMH58S84Z23PM Add an SGLang datapoint beside vLLM in the serving comparison
kind: task
state: approved
created: 2026-07-27
targets: docs/benchmarking.md

Extend the serving benchmark from three-way to four-way. SGLang was confirmed to install on the worker box but was never measured.

Scope: worker-machine only (RTX 3060, TinyLlama-1.1B). Use the identical protocol as the existing vLLM rows: batch=1 decode tg128, prefill pp128, n=16 and n=64 aggregates, sequential GPU-exclusive runs, versions recorded.

Output: a three-way section row in docs/benchmarking.md and a row in BENCHMARKS.md section 1.

Value: SGLang's RadixAttention prefix reuse is the industry mirror of this repo's prefix-caching arc, so a direct comparison sharpens that story. Precision must be matched and disclosed on both sides.

Migrated from cavekit SPEC.md T888.

## T-01KYJNDR38E4ZSN52KVM0PC5J9 Extend the trained-optimizer comparison with APOLLO and Q-GaLore
kind: task
state: approved
created: 2026-07-27
targets: docs/training.md

The nn package ships APOLLO and Q-GaLore, but the trained-optimizer zoo in llamagpu/optimizers_trained_test.go and the tables in docs/training.md still end at the earlier optimizer set. Godoc alone cannot give the comparative same-task cross-entropy-after-120-steps measurement, which is the value here.

Work: add APOLLO (seed-only projection default) and Q-GaLore (default QuantBits, plus the QuantBits=0 GaLore-collapse datapoint) rows to the zoo and wrapper harness using the same task, step count and protocol; refresh the zoo and wrapper tables in docs/training.md; record which defaults were used, with the research grounding for each.

Also fix the stale section-range header in training.md so it names the range actually covered.

Migrated from cavekit SPEC.md T891.

## T-01KYJNDS5MFEDAR0GVMSNG4MHD Clear the remaining documentation debt to a green apicheck gate
kind: task
state: active
created: 2026-07-27
targets: internal/apicheck

Bring internal/apicheck to exit 0 so the public-API doc gate can be enabled.

Completed so far: 118 nlp struct-field symbols documented with architecture-aware godoc referencing the upstream tensor names (Bert, Gemma, Gemma2, Jamba, Mamba, Mamba2, Mixtral, DeepSeekV2, GraniteMoE, RWKV, T5, Qwen2MoE and the 16 quantized twins), 4 further symbols (classic GradientBoostingRegressor.Predict, safetensors TensorInfo.Name/Dtype/Shape), and the 3 magic backend-name string literals replaced with backend.CPU. Undocumented count went from 140 to 18; the magic-strings test is green.

Remaining: (a) the 18 undocumented symbols are all llamagpu New*Q8CUDA and New*Q4KCUDA constructors owned by the parallel CUDA worker; (b) the runnable-Example requirement of TestPublicAPIDocumentedWithExamples is a separate pass. Justified typeExampleExempt and methodExampleExempt allowlist entries are legitimate for fixture-heavy surfaces and beat boilerplate examples.

Definition of done: go test ./internal/apicheck exits 0, checked unpiped. Unblocks enabling apicheck in the CI always-run set.

Note: godoc and Example edits are .go files, so this consumes CI and belongs to the main agent rather than the docs lane.

Migrated from cavekit SPEC.md T892.

## T-01KYJNDSP0FG4B672MFS69AQ0F Enable apicheck and mdlint in the CI always-run set
kind: task
state: active
created: 2026-07-27
targets: internal/cichange

The source-walking meta-tests (internal/apicheck, internal/mdlint) walk the whole repo's source and markdown, and no import edge connects them to what they check, so import-graph impact selection never picks them and their invariants can rot red while CI stays green.

Mechanism is already built and live: a config alwaysRun field plus a repeatable -always-run flag; Impact() appends every configured package that exists in the graph to each non-empty selection. Docs-only and empty selections stay at zero runners. Missing packages are tolerated so temporary modules and renames cannot inject a bogus target. Unit tests plus the regression pinning that an nlp-only diff selects apicheck are in place and proven non-vacuous. internal/speccheck is already wired into the default always-run, proving the path end-to-end.

Blocked on: the default alwaysRun set is deliberately empty because enabling apicheck and mdlint while they are red on the committed tree would fail CI on the first push. apicheck is red on the remaining doc debt; mdlint is red on worker markdown.

When both gates are green, this is a one-line change.

Note: the mdlint blocker changes shape once the cavekit spec files are removed, since much of the red is in the generated spec views. Re-measure before flipping.

Migrated from cavekit SPEC.md T893.

## T-01KYJNDT44EKJAXN8W0Y4QFZCE Batch the ViT encoder instead of looping over the batch dimension
kind: task
state: approved
created: 2026-07-27
targets: vision

vision.ViT.Forward processes a [batch,C,H,W] input by looping over the batch: slice image b, forwardOne, concat. A batch of N therefore runs as N independent length-(patches+1) encoder forward and backward passes. On GPU each per-image op pays the roughly 0.27ms Metal/Vulkan dispatch floor, multiplied by N, which measured about 40x behind torch-mps for ViT training. On CPU the same defect costs only 2.6x to 4.2x because there is no dispatch floor, so this is a GPU lever specifically.

Fix: carry a leading batch dimension through patch-embed, class-token prepend, positional embedding, the MHA blocks, final layer norm and the head, so attention runs one [N, patches+1, D] pass.

Feasibility, already scoped: nlp.MHA.Forward asserts x.Ndim()==2, so the shared attention is 2D-only and the ViT loops per image because attention cannot batch. Three routes: (a) add a batch dimension to OpMHA, nlp.MHA and every backend MHA kernel, the real fix but deep and cross-cutting into the worker's attention kernels; (b) a ViT-local batched attention in vision/ that bypasses nlp.MHA, moderate but duplicates attention; (c) wire the batched recorder to the autograd tape, currently parked.

Correctness bar: bit-identical to the per-image loop at batch=1 and equal to looping for batch>1, pinned by a gradient check and a per-image-versus-batched parity test.

Benchmark harness already exists at internal/benchcompare/vision_train_test.go.

Migrated from cavekit SPEC.md T908.

## T-01KYJNDTK2E8A9BK9SGAZ4VKYS Hoist per-layer Attrs boxing across the remaining decode models
kind: task
state: approved
created: 2026-07-27
targets: nlp

Class-audit sweep in nlp. A per-layer decode closure builds a backend.RoPEAttrs or backend.AttnAttrs struct literal and boxes it into the Attrs interface inside the loop, although the fields are layer-invariant.

Apply the same mechanical, provably bit-identical hoist already done for one model to the remaining decode models: cohere, falcon, gemma, gemma2, cla, blt and their siblings. Find the sites by searching nlp/*decode*.go for a dispatch call passing a backend.RoPEAttrs or backend.AttnAttrs literal.

Measure per model or as a batch, with a same-session A/B and bit-identity verification before shipping.

Coordination: re-check for a scope collision per file before editing, as a parallel worker is active in the mamba2 and quantized-mamba2 files.

Migrated from cavekit SPEC.md T956.

## ADR-01KYJNF428F8Q9RQTABB1ZSVPC What is the agent commit and push authority on this repo?
kind: adr
state: submitted
created: 2026-07-27
context: The migrated cavekit constraint C8 says the loop must never commit or push without explicit user permission and works in the working tree only. The repository history contradicts this: work lands as autonomous branch-plus-pull-request pushes, and the worker runner protocol RUN2 mandates PR-only pushes with manual merge after green CI. C8 was therefore not migrated as a rule, because encoding either reading without confirmation would put a false contract in the spec. The spec server itself runs with git.mode=offline, so it commits but never pushes.
status: proposed

kind: radio
option: Autonomous branch + PR push, manual merge after green CI (matches current practice)
option: Autonomous commit and push including direct to main
option: Working tree only - explicit user permission required before any commit or push
