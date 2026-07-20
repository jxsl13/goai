# GoAI — Autonomous Loop Prompt

> Authoritative per-iteration instruction for the autonomous build loop.
> Referenced by the cron job; changes here take effect from the next fire.
> Frame: `PLANNING_PROMPT.md` (section CONTINUOUS OPERATION), `SPEC.md` (the truth).

FULLY AUTONOMOUS, no questions back to the user. Work on the GoAI Go AI
library in this repository. Complete EXACTLY ONE task per fire, end to end.

## Procedure

0. **BOOTSTRAP** if `SPEC.md` is missing/incomplete: autonomously produce the
   planning basis, ONE phase per fire — (a) `/deep-research` →
   `docs/research/00-landscape.md`; (b) `/research` open core decisions
   → §R/§C; (c) `/spec` → `SPEC.md` (§G §C §I §V §T §B); (d) `/review` →
   harden §V + go/no-go note. Only proceed to step 1 with `SPEC.md` + a
   review "go".
1. **Selection:** pick the next dependency-free §T task; state its ID and
   definition of done.
2. **Build:** `/build` (or implement directly, plan-then-execute against
   SPEC.md).
3. **§V acceptance:** V-PARITY (goldens against the reference tolerance,
   generated via the `.venv` Python/NumPy when needed), V-GRAD, V-CROSS,
   V-PROP.
4. **Optimization tasks:** first push PURE Go to its ceiling
   (algorithm→layout→simd/avo/NEON→goroutines, every rung green + a
   benchmark delta), then the cgo gate (V-CGO): merge cgo only as an
   optional build-tag backend when a benchmark beats the §C threshold
   against the fully optimized pure-Go version — otherwise discard, park
   the idea in §B.
5. **Failure** → backprop: trace the root cause, add a §V invariant and/or
   §B row when warranted, fix. Never loosen tolerances or skip tests.
6. **Platform check:** pure Go WITHOUT a C toolchain must be green —
   MANDATORY via `CGO_ENABLED=0 go vet ./...` or `go test ./...` (§V23:
   `go build` alone compiles NO _test.go files and misses missing build
   tags, §B45); accel stays behind build tags with a fallback. Full sweeps
   (`go test ./...`) ALWAYS with `-timeout 1800s` (llamagpu exceeds the
   600s default, §B46) and the exit code checked UN-PIPED — `| grep | tail`
   masks a FAIL as exit 0 (§V24).
7. **Completion:** `go run ./internal/specgraph task set-status T<n> x`
   (⊥ hand-edit — SPEC.md is a generated view, §V40; `make spec-verify` must be
   green before push), add a short entry to
   `CHANGELOG.md`.

## Research rule (mandatory — mitigates the StructuredOutput failure)

NEVER use the built-in `/deep-research` workflow for external research: it
forces every sub-agent into a `StructuredOutput` schema call that fails 5×
under rate limits and crashes the whole workflow
(`StructuredOutput retry cap exceeded`).

ALWAYS use the repo's own `research-lite` workflow
(`.claude/workflows/research-lite.js`) instead:
- **small scope**: exactly ONE focused question per run (protects context);
- **schema-free**: no `agent({schema})` → the StructuredOutput failure is
  structurally impossible;
- **compressing sub-agents**: every angle agent returns ≤6 condensed lines,
  one synth agent condenses to ≤8 lines → never blows the context;
- **graceful**: dead agents → `null`, filtered, never a throw.

Invocation: `Workflow({ scriptPath: ".claude/workflows/research-lite.js", args: "<one precise question>" })`.
**Validation ladder (§V16, mandatory for every algorithm implementation):**
- Tier 1 = bit-/tolerance-exact parity against the official reference lib
  (torch/sklearn/gguf-py/safetensors). Necessary, NOT sufficient.
- Tier 2 (final authority) = the **scientific paper** defining the algorithm
  (arXiv/DOI/canonical textbook) — the implemented formula MUST match the
  paper's equation, cited in §R.
- File formats have no paper → the defining source is the format spec /
  reference implementation (record it explicitly as such; never invent a
  paper).
File formats: always round-trip + fuzz (§V15).

## Autonomy rule

On ambiguity or a design decision, do NOT stop, do NOT ask — make a
scientifically grounded default choice, document it in
`docs/decisions/ADR-<n>.md` plus a SPEC amendment (§C/§B), keep building.
Only genuine hard blockers (broken toolchain) → a short PushNotification,
otherwise continue. NO commits/pushes without the user's explicit
permission. The loop NEVER runs out of work: when every §T task is "done",
proceed per the "Empty backlog" rule below.

## Never idle (mandatory — user directive 2026-07-14)

NEVER be idle. Every fire must do productive work — never sit and wait, and
never spend a fire only re-scheduling the next wakeup. Waiting on an external
gate is NOT a reason to idle: when one thread is blocked, advance another.
Concretely, when blocked on the C16 push throttle (waiting for the hourly
window), a CI run, or a wakeup deadline, keep BUILDING locally — commits
accumulate in the working tree and push when the window opens. Always have
a next productive action from this list, in priority order:

1. Continue the current task's next slice/step.
2. Harden what just landed: more tests (edge cases, fuzz §V15, gradchecks
   §V2), documentation, benchmarks.
3. Verify-ahead the next task (research + primary-source extraction so the
   implementation is unblocked) — the §R234 verify-first discipline.
4. The Empty-backlog rule below (gap research, then beat-the-incumbents).

The ONLY legitimate idle is a genuine external wait with NO available local
work — which, given rules 1–4 and the empty-backlog rule, does not occur.
Idle-rescheduling a wakeup with productive work available is a process
failure. (Context-length is not a blocker either: if the context is
saturated, do the smallest safe useful increment and commit it, don't idle.)

DEFAULT WHEN IDLING (user directive 2026-07-14): whenever you would otherwise
idle — no current-task step, nothing to harden, nothing to verify-ahead —
spend that time BEATING THE PYTHON/C++ INCUMBENTS' PERFORMANCE. Pick a hot
path GoAI implements (matmul, attention, a decode step, a quantized kernel,
a tokenizer) and make it FASTER than the reference (torch/llama.cpp/ggml/
numpy), proven with a clean A/B benchmark (identical shapes + hardware,
warm-up excluded, variance reported, §V22 discipline; `make bench-compare` /
`make bench-python`). Every measured win — or an honestly documented deficit
with a root-cause analysis and the next lever — is one §T task and one
docs/benchmarking.md entry. This is the empty-backlog rule 2 promoted to the
standing default: idle time is performance-leadership time.

## Empty backlog rule (mandatory — the loop generates its own work)

When no open §T task exists, do NOT idle. In this order:

1. **Topic discovery:** autonomously research new AI topics, methods,
   architectures, formats, and techniques that this library does NOT yet
   implement — online search (current papers, llama.cpp/vLLM/SGLang release
   notes, architecture roundups, framework changelogs) plus a repo
   cross-check so only REAL gaps are booked (the §R234 method: verify every
   candidate against the code before calling it a gap). Book each confirmed
   gap as a §T row with cites (§R entry documenting source and scope) and
   work them per the normal procedure, one task per fire.
2. **Beat the incumbents:** once the library has implemented everything
   discoverable as far as possible, the standing task becomes performance
   leadership — make every implementation BETTER and FASTER than the
   industry-standard implementations (usually Python-with-C++-kernels or
   pure C++: torch/llama.cpp/ggml class), and PROVE it with clean,
   industry-standard benchmarks: identical workloads and shapes on identical
   hardware, warm-up excluded, repeated runs with variance reported,
   tokens/s / latency / GFLOPs as the branch-standard metrics, methodology
   and comparison scripts committed (`make bench-compare` /
   `docs/benchmarking.md` discipline, §V22: measure real workloads, A/B via
   file-toggle). Each measured, documented win (or honestly documented
   remaining deficit with root-cause analysis) is one §T task.

Rule 2 never completes — faster incumbents, new hardware, and new workloads
keep appearing — so the loop always has a next task.

## STATUS: loop ACTIVE — as of §T565: v0.1.0 public on github.com/jxsl13/goai, CI green on 3 OSes; source-list audit 12/12 resolved (2026-07-13)

Spec through §T565. The T364–T432 session carried ADR-0019 (batched decode)
to completion AND harvested the attention/training levers. Every win
§V22-measured, every op cross-validated against ref, both GPU backends.
T434–T444 then closed the INTEGRATION program (see its era bullet).
T472–T477 = TRAINED-MODEL MEASUREMENT SERIES (docs/inference.md):
distill-your-drafts 73→88% acceptance; Mirostat = a surprise CEILING
(one-sided); watermark δ=2 detectable from 50 tokens; bounded cache
+0.05 bits; streaming 4× beyond ctx (RoPE, pre-RoPE cache verified);
Q8/Q4 near-lossless (99%/97% teacher-forced agreement — NEVER measure
free-running). The sink phenomenon does NOT show at toy scale (2× honestly
negative). trainCharLlama (cpu, 3s) unlocks Llama measurements generally.
T480–T492 = DECODE CLOSE-OUT + FAMILY E2E SERIES (docs/training.md): every
decode strategy trained-model-verified (CD/DoLa via mechanism surprise,
beam +0.31 nats over greedy, DoLa plausibility = the log₂(1/α) guarantee);
optimizer zoo (8: SOAP leads at 1.188; Shampoo bug fixed →
WithShampooRootEvery) + 5 wrappers (GaLore 1.380 beats AdamW; SAM two-pass;
Grokfast needs LR compensation); the NEFTune claim shows (held-out ↓ while
train ↑); DDPM/DDIM + flow matching reconstruct the same ring; EWC 50→89%
retention (tasks must be jointly representable!); TIES/soup beat the
specialist worst case (soup > TIES on same-base); VQ-VAE 94% variance
through the ST bottleneck; SimSiam 94% linear probe label-free. Assert
discipline: assert theory guarantees, log the rest; judge regularizers on
held-out data; measure agreement teacher-forced.
T493–T499 = ARCHITECTURE E2Es + RL/MERGING CLOSE-OUT: Mamba, RetNet, MLA
and Mamba+MoE char-LMs (each with a structural CAUSALITY assertion +
training + generation; the MoE balance loss holds 4 experts at 20–28%);
GAE+PPOClip wired into the PPO actor-critic (return 1.0, critic
V(start)≈γ⁴); GreedySoup verified; DARE = the third SCALE-DEPENDENT claim
(a 0.9 drop clearly hurts, 0.5 = noise — the redundancy premise needs large
models). RWKV block wrapper, K-quant quality (needs a dim≥256 Llama),
Sophia GNB harness: demand-gated (all later built).
T504–T514 = BLOCKER TEARDOWN (pinned blockers opened one by one, every
claim re-verified): MHA gained THREE seams (LoRA §T504, bias §T505, mask
§T508 — nil = the exact old path, golden suites untouched); GPT2FromHF
(§T506, c_attn split, tied head); OpMHAMasked (§T507) → tree verify through
the full model (§T508, bit-identical on ref + the T461 chain verified
through a model) → MedusaGenerateTree (§T509, topK=1 ≡ chain EXACTLY) →
measurement with trained heads (§T510: tree 4.00 vs chain 3.92 tok/round;
acceptance CEILING at toy scale). IMPORTANT CORRECTION §T508: masks ≠
merged score sources — Self-Extend needs the latter → OpMHASelect (§T512,
three bit-exact collapse/symmetry tests) → Self-Extend e2e (§T513: trained
ONLY on 32-token windows, evaluated at 4×: plain CE 0.316→1.488 degrades,
Self-Extend w=8/G=8 holds 0.515 — with NO fine-tuning). Plus §T511: a
persistent worker pool in cpu.parallelWork (allocs −70–75%/op, time ±0–2%,
-race-verified; A/B via file toggle, NEVER git stash). Docs §T514.
T515–T521 = SWEEP + RWKV FAMILY + FUZZ HARDENING: full sweep green
(llamagpu now needs -timeout 1800s, §B46/§V24: check exits UN-PIPED!);
the OpWKV backward BUILT (§T516: softmax-average identity, O(T²) reverse,
gradcheck over all 4 inputs — the "own project" blocker fell in one fire)
→ nn.RWKVBlock + char-LM e2e (§T517: CE 3.01→0.12, causality bit-exact;
architecture series COMPLETE) → recurrent inference (§T518: Step ≡ Forward
≤1e−12, O(1) state). Integration audit R2 (§T519: the Self-Extend position
math was decoupled from the forward → selfExtendPos + a spec-consistency
test). Fuzz sweep over ALL 35 targets (§T520/§B47): the gguf bounds-check
uint64 OVERFLOW (2 sites, subtraction form) + tokenizer-JSON unbounded id
alloc (BPE via fuzz, WordPiece via class audit) fixed; explicit hostile
tests + a 6×60s deep fuzz green (§T521).
T523–T532 = MAINTENANCE + VULKAN PERF ARC: tree-wide race sweep 0 races
(§T523); bench regression check no drift (§T524); memory maintenance
(§T525); fuzz program complete — the Q4_K bound structurally corrected,
encoder optimal (§T526/§B48); Self-Extend extension curve: plain
0.91→2.40 monotonic, SE flat 0.57→0.70 out to 8× (§T527). VULKAN PERF:
mha_bwd matmul decomposition 71.5→4.74ms = 15× (§T528, 2 new shaders
softmax_packed/sm_jacobian, a 7-stage chain); the §T529 forward idea
retracted (T398 record) → §T531 with NEW evidence (profile: fwd 19%;
cheaper structure) forward chain +18% = default; cumulative 935→1882
tok/s = 2.01× (BenchmarkGPTTrainingStepVK new, metal-class); §T532
confirmed the GEMM ceiling via the §B39 record → arc CLOSED. Lessons:
read the record BEFORE building (T529 vs T531 = retract without / build
with new evidence); §V24 applies to single suites too.
T534–T538 = ADR-0008 ROUTING + THE B49 ARC: binary elementwise back on the
SIMD CPU (metal +7.8%, vulkan +5.8% training; the cpuPrefers gate against
ref leaks); unary/addbias routing MEASURED AS A LOSS (cpu unOp = scalar
closures) → reverted (§T535). §B49: an in-kernel re-dispatch with the
recorder preserved = the op lands on the tape twice = GRADIENTS DOUBLED —
only the full sweep's tight trained-model bars saw it (shape/parity/
gradcheck blind!). Fix → 46-site class audit (§T537) → structural: Execute
strips the recorder before invoking kernels, NEW §V25 (§T538). Final state:
metal 3219 tok/s, vulkan 935→1992 = 2.13×. Two regression tests
(RecordsOnce binary + fallback-under-tape ALiBi bit-identical vs ref).
T540–T546 = STEADY STATE + AT-SCALE ARC: demand gates closed with evidence
(§T540 cpu GELU 0.79ms > GPU 0.39ms); SelfExtendGenerate (§T541: generated
text holds 0.50 surprise at 4×, plain degenerates to 2.30); 124M SCALE with
synthetic weights (§T543–T546): GPT2FromHF mechanics + batched decode
verified at real size (batched ≡ analysis bit-equal), a 2-backend × 3-format
table — metal f32 76 / vulkan Q4_K 72 tok/s, a backend INVERSION on
quant-vs-f32, Q4_K > Q8 on both. The harness takes real weights the moment
they are permitted.
T548–T560 = SOURCE-LIST AUDIT (user reference list → 12 gaps → all
resolved): GSPO (op+VJP, exact collapse onto GRPO(β=0), trained e2e
0.04→0.96); QK-Clip (the γ² law); DeepSeekMoE (shared experts, bit
collapse); QLoRA e2e (NF4 lossless, adapters 1.19→1.08, base bit-frozen);
Mamba-2 SSD (the duality theorem ≤1e−12); the i-QUANT FAMILY complete (all
8 types f32-exact vs gguf-py — recipe: extract tables programmatically,
golden+fuzz per type; gguf-py in the .venv = a local cross-reference!);
MXFP4 (encode byte-exact); the sparse trio MoBA/NSA/DSA (collapse +
isolation/routing proofs per mechanism); KDA (channel decay, collapse onto
GatedDeltaNet — the test caught a missing L2 norm); PagedAttention →
ADR-0020 out of scope with a revisit trigger. Pattern of the era: collapse
tests against existing verified paths carry whole families.
T562–T565 = PUBLIC RELEASE: CI pipeline researched across the reference Go
repos and rebuilt (.github/workflows/ci.yml: pure-go 3-OS matrix, cgo+race,
cgo+metal on arm64 runners, vulkan-tag build, tidy, simd soft gate);
push-readiness audit (7.1 MB content, no secrets, no local info); license
MPL-2.0 (file-level copyleft — linking imposes nothing on the product);
v0.1.0 pushed and tagged. The FIRST CI run caught 3 real cross-platform
bugs arm64-darwin development could never see (amd64 FP tolerance, ±0
formatting, Windows path separators) — fixed, all 8 jobs green since. The
ubuntu-amd64 runner is the host class the parked T11b/T74 wait for.

**The big results of this project so far:**
- **ADR-0019 batched decode (T366–T412):** recorder (one command buffer per
  step) + DeviceBuffer on metal AND vulkan, every decode op in record mode
  (matmul/norm/rope/mha sq≠sk/blit KV-append/qmatmul), both architectures
  (GPT + Llama/GQA/SwiGLU). PUBLIC package `llamagpu`:
  New/NewVulkan(*nlp.Llama) → Decoder → Step/StepN/Generate. Real-model
  decode **24× metal / 21× vulkan** vs nlp per-op; logits == reference,
  greedy token-for-token identical, GGUF(F32) ok.
- **Attention MPS reformulation (T393–T400):** mha fwd as MPS(QKᵀ)→softmax→
  MPS(PV): 6.9× kernel / **1.87× GPT forward**; mha BACKWARD analogous
  (+ softmax jacobian): 27× kernel / **2.04× training**; GQA/MQA via MPS
  beta accumulation. The hand-written flash kernel was 15× slower than its
  own two matmuls (the §T393 floor measurement — prevented a wrong
  cooperative-tiling rewrite).
- **Silent-fallback audits (T401–T403):** OpCrossEntropy fwd (20ms!) +
  OpEmbed fwd were missing on metal+vulkan (the backwards existed!) →
  kernels → training cumulatively 1133→2997 tok/s (2.6×). The definitive
  audit method: GOAI_LOG_FALLBACK=1 (execute.go) on the REAL workload;
  standalone op timing misleads (the OpSum trap, §T402).
- **Op profiling (T410):** GOAI_TIME_OPS=1 (execute.go). Found the T402
  embed regression (the GPU gather uploaded an 8MB table per call → host
  f32 gather, both backends).
- **Quant decode (T413–T416):** recorder QMatMulResident (every ggml type,
  both backends) + llamagpu.NewQuant: Q8 decode 3.6× vs per-op quant; ≈16%
  slower than f32-batched → Q8's value = 4× MEMORY, not speed.
- **StepN + speculative (T418–T420):** multi-token step → **prefill 41×**
  (Generate prefills via ONE StepN); llamagpu.SpeculativeGenerate (draft
  step + target StepN + nlp.SpeculativeRun, lossless, KV rollback free via
  the position cache); costs measured: speedup 1.95×@50% / 2.65×@80%
  acceptance (needs trained, related models).
- **GPT + feature completion (T421–T426):** GPTDecoder (LayerNorm + pos-emb
  + AddBias record ops, both backends) incl. StepN; SpeculativeGenerate over
  the exported Stepper interface (both architectures); PromptLookupGenerate
  (draft-free, nlp.NgramLookup, 45% acceptance on repetitive prompts,
  lossless); 6 examples; decode/prefill as standing bench-compare benchmarks
  (205/200 tok/s batched, 27×/36× — regression guard). Final matrix:
  {Llama,GPT}×{Step,StepN,Generate,Speculative,PromptLookup}+Llama×Quant.
- **Long context / cooperative attention (T428–T432):** the two-pass MHA
  kernel was serial in sequence length → a cliff at large KV (242ms/step
  @2k). NEW cooperative kernel: ONE 32-lane simdgroup (metal,
  simd_shuffle_down) or subgroup (vulkan, SPIR-V 1.3, glslc
  --target-env=vulkan1.1) per (query row, head), lanes partition the keys,
  online-softmax partials (m,l,acc[dk≤128]) merged in registers via lane
  shuffles; a NaN guard when merging empty lanes (M==-INF→c=0, the §T428
  bug). Covers ALL attention surfaces: recorder decode (T428/T429:
  242→13.8ms, **17.6×**; standing bench T430: 72.3/71.0 tok/s @L=1920),
  recorder prefill windows sq>1 (T431: per-row jmax=sk-sq+i+1, 291→104ms),
  per-op OpMHA (T432: metal+vulkan host-slice wrappers, sq=1 @sk=1920
  ≈40→2.18ms). Short context honestly unchanged (dispatch-bound,
  §T430/T432).

- **Integration era (T434–T444), the "orphan audit" method:** systematically
  found everything exported-but-unwired and connected it to real models with
  small adapters/loops. (a) T434: speculative with GENUINELY trained models —
  "needs model files" resolved via IN-REPO TRAINING (81% acceptance, but
  only 1.12× — dispatch-bound draft/target ratio; pays only on large
  targets). (b) T435 GPT.Safetensors() checkpointing (bit round-trip).
  (c) T436 ApplyPenalties→sampler (SampleWithHistory, 7 loops). (d) T437 the
  TokenSampler interface → Mirostat can generate. (e) T438 mechanical audit:
  class empty; nlp/doc.go covered only ≈1/3 of the features — rewritten.
  (f) T439 Watermark.Sampler + RegexGuide.Sampler (composition adapters;
  sharp tests: z=4.58 detection, (ab)+ enforcement). (g) T440 CFGDecode
  (γ=1/γ=0 equivalences). (h) T441 GPT.JacobiGenerate (lossless vs greedy,
  6 tokens in 5 iterations). (i) T442 ForwardEarlyExit+DoLaDecode (bit
  identities, the α=1 equivalence). (j) T443/T444 Medusa COMPLETE:
  ForwardHidden, trainable heads (frozen-base tape), chain MedusaGenerate
  with typical acceptance — heads on base rollouts: 100% acceptance. NO
  algorithm in nlp remains without a runnable real-model path.
  TRANSFERABLE LESSONS: (1) exported-with-tests ≠ usable — call-site audits
  pay; (2) in-repo training unlocks every "needs artifacts" measurement;
  (3) sharp equivalence invariants (parameter extremes collapse new paths
  onto known ones) beat soft quality asserts.

- **Decode acceleration era (T446–T455):** batched Medusa (StepHidden/
  StepNHidden + MedusaGenerate over HiddenStepper, both architectures) —
  T446: 1.81× @97% acceptance (confirmed the T434 thesis: draft COST, not
  acceptance, is the lever); T455 halved the round cost via the
  lastTok-lead-window (proposals from the verify pass, StepNHidden) →
  **3.08×** (1152→3546 tok/s). Prompt lookup measured for real (T452):
  **1.80× LOSSLESS @15%** — a cheap round beats high acceptance.
  Measurement gotcha: sequential A/B blocks catch cold-start outliers →
  measure interleaved (A,B,A,B). All in benchmarking.md.
- **CV era (T456–T463):** conv/pool layers, the vision package (CNN, 100% on
  the spatial task, checkpoint round-trip), then the perf thread §V22
  profile-driven: cpu conv2d_backward onto im2col+GEMM (637→30.6 ms/step),
  cpu pools (bit-identical), the fallback chain active→cpu→ref (T461 —
  metal pools instantly ran faster, 39→33), a scratch sync.Pool against
  madvise churn (→24.3). CUMULATIVE 637→24.3 = **26×**; CPU beats Metal on
  small CNNs (the §T361 size dependence holds for CNNs too). Parked then:
  the parallelWork barrier floor — the persistent worker pool was built
  later on demand (§T511).
- **Alignment era (T464–T470, COMPLETE — docs/alignment.md):** a THIRD
  orphan class, "documented but never built": SequenceLogProbs existed only
  as a doc reference. Built: TokenLogProbs/SequenceLogProbs (stable composed
  log-softmax gather, gradchecked). FLAGSHIPS: GRPO trains the real GPT
  policy 0.042→0.979 reward (Generate rollouts + GroupAdvantage + GRPOLoss;
  lesson: sparse zero-reward groups ⇒ advantage 0 — longer rollouts/bigger
  groups/lower KL β); DPO: 3/3 positive margins, chosen↑/rejected↓ vs the
  reference. T468: IPO/SimPO/CPO/KTO VERIFIED (the contracts genuinely
  differ: SimPO length-normalized without a ref, KTO unpaired+labels,
  CPO+NLL) — all flip the initially NEGATIVE margin. T469: the full RM+GRPO
  pipeline — REWARD HACKING reproduced (the learned reward climbs, the true
  metric →0; at every head capacity). T470: iterated RLHF (RM refresh every
  5 updates on fresh policy samples) rescues the true metric COMPLETELY
  (0.042→1.000; an 8-update cadence only 0.104 — frequency is the lever; KL
  alone does not suffice). Also: V23 (the CGO0 gate compiles tests, §B45),
  README/doc.go audits of every package, GPT/CNN checkpointing (T435/T458).

**PARKED with numbers (do not touch again without new evidence):**
- Tape batching for training: ceiling 1.4× @S256 (§T411) — compute dominates.
- The conv gap: ≤2×, not on the LLM path, MPSCNN = a large API (§T417).
- The MPS matmul rate (≈3.5× behind torch): Apple's best; 49% of training
  time (§T410).
- Zero-copy UMA (§B42), vulkan GEMM blocking (§B39/41), mha_bwd registers
  (§B43), the vulkan attention fwd reformulation for INFERENCE
  (§T397/398: isolated 6-8×, real-workload SLOWER — attention is not
  vulkan's bottleneck there, the FFN matmuls are; the TRAINING-shape
  default was later justified with new evidence, §T531).

**MEASUREMENT DISCIPLINE (§V22, MANDATORY — so far: ≈6 surprises, ≈5
prevented wrong builds):** (1) measure/instrument the REAL workload
(GOAI_LOG_FALLBACK / GOAI_TIME_OPS / bench suites), never standalone ops.
(2) A/B-measure the floor BEFORE any rewrite (§T393/T396/T411/T417).
(3) RE-measure the real workload after every kernel (the §T410 regression).
(4) isolated gain ≠ real gain (§T397). (5) the bottleneck is
BACKEND-SPECIFIC (metal: attention was it; vulkan: matmul). (6)
same-session A/B, no git stash (shallow history), temp-swap via the
scratchpad.

**Dispatch rules (for every future kernel):** on Metal never use
maxTotalThreadsPerThreadgroup for per-row/register-heavy kernels — 64/TG or
cooperative (§T337/345); MSL/GLSL have no erf → A&S approximation (§T352);
backwards as OpXBackward on the active backend (§T353); transposed operands
via flags (§T356/357); never the GPU for tiny memory-bound ops (the gather,
§T410); recorder ops need per-op descriptor sets + explicit barriers on
vulkan (§T382), hazard tracking on metal; respect the VK_MAX_PIPELINES
headroom (the §T397 bug: 33>32 → random shader failures); apicheck forbids
magic backend strings (§C15) and wants forwards AND backwards checked per
op (the §T401 asymmetry).

**Next candidates:** CPU GEMM/archsimd (T11b/T74 — the ubuntu-amd64 CI
runner now exists; needs the user's decision on CI-iterated branch
development); MoE/MLA/further architectures on the recorder (only with
demand); real model weights (GPT-2/TinyLlama) ONLY with the user's download
permission. Almost everything remaining is externally blocked or
demand-gated — the loop is in maintenance/opportunity mode.

## Working context (last updated: 2026-07-13)

- Toolchain: Go 1.26.4 (arm64 dev host), git, a `.venv` with numpy 2.5.1 +
  torch 2.12.1 (`make golden`; torch goldens via `testdata/verify_torch.py`);
  vulkan via MoltenVK (brew; the Makefile sets `VK_ICD_FILENAMES`).
- The reference backend `ref` = numerical truth (§V9); `cpu` = the CGO0
  default; on macOS `metal` is the zero-config default (§T47/§B37);
  `vulkan` behind a build tag, host-verified (§B36).
- Benchmarks: `docs/benchmarking.md` (+ the §T338 snapshot table);
  `make bench-compare` = the cross-backend harness; ADRs:
  `docs/decisions/`.
- When A/B measuring: no `git stash` (shallow history — a stash resets to
  the base state); temporarily swap in the old variant and restore it from
  a scratchpad backup (pattern §T336/§T341).
- Public repo: github.com/jxsl13/goai (MPL-2.0, tag v0.1.0, CI green).
  Everything committed must be written in English and free of machine-local
  information (absolute paths, hostnames — §T567).
- PUSH THROTTLE (§C16, user directive 2026-07-13): push at most once per
  hour — commit completed tasks locally and push the batch when ≥1h has
  passed since the last push. EXCEPTION: changes to the CI-CLI optimizer
  (internal/cichange, internal/ciimpact, .github/workflows/*) push
  immediately — their whole point is live pipeline behavior. Every push
  still gets a CI watch.
- POST-PUSH SELECTOR VALIDATION (§V27, user directive 2026-07-13): after
  every push, validate the CI selector's choice for the pushed range
  against the toolchain oracle: `go run ./internal/cichange -validate
  <pushed-base> <pushed-head>` (§T583). Exit 1 = under-selection =
  release-blocking selector bug (§B row, full suite immediately); exit 0
  with excess = allowed, record the excess count in the fire report.
  Note the oracle is a CGO_ENABLED=0 LOWER bound — cgo-gated test edges
  legitimately appear as excess.
- CI WARNINGS = ERRORS (§C17): every CI watch also greps the full run log
  for warnings/deprecations (`gh run view ``<id>`` --log | grep -iE
  'deprecat|warning'`); new hits (beyond the accepted git
  init.defaultBranch hint inside actions/checkout, §T586) become fix
  tasks immediately.
