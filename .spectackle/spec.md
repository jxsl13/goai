---
schema: v1
prefix: ARCH
---

## ARCH-001 {applies: go:backend.Execute}
The packages above the backend layer SHALL depend only on the backend interface, never importing backend/ref, backend/cpu, backend/cuda, backend/metal or backend/vulkan.

Rationale: Layer model L0-L5: the public API stays backend-agnostic so an accelerator can be added or removed without breaking callers. Migrated from cavekit SPEC.md I1.

## ARCH-002 {applies: go:backend.Execute}
The library SHALL provide a pure-Go reference kernel in backend/ref for every registered backend.Op.

Rationale: Correctness before speed: the reference kernel is written first and every accelerated kernel is validated against it. Guarantees the CGO_ENABLED=0 path can run any operation. Migrated from cavekit SPEC.md I2 and G3.

## ARCH-003 {applies: go:tensor.Tensor}
The tensor core layer (tensor, dtype, device, allocator, views) SHALL build under CGO_ENABLED=0 and import no cgo-dependent package.

Rationale: L0 holds the data model every other layer depends on; admitting cgo there would make the whole library unbuildable without a C toolchain. Migrated from cavekit SPEC.md I3.

## ARCH-004 {applies: go:backend.Default}
WHEN a caller invokes backend.Default(), the library SHALL return the highest-preference registered backend along the descending order cuda > metal > vulkan > cpu, and otherwise the always-present backend/ref reference backend.

Rationale: Zero-config accelerator UX: building with an accelerator tag routes work to the GPU with no code change. Selection is by preference order, never by registration or import order. Order is overridable via backend.SetPreference; unknown names are skipped safely. Migrated from cavekit SPEC.md V18, C11 and I4.

## ARCH-005 {applies: go:backend.RegisterDefault}
IF an accelerated backend package is linked into the binary but its device is absent at runtime, THEN the backend registry SHALL leave that backend unregistered so backend.Default() falls back rather than claiming an absent device.

Rationale: An accelerated backend registers in init() only after detecting its device. A silent claim of an absent device would turn a missing GPU into a runtime failure instead of a fallback. Migrated from cavekit SPEC.md V18 and V4.

## ARCH-006 {applies: go:backend.Attrs}
The backend.Attrs SHALL remain a sealed interface (interface{ opAttrs() }) implemented by one concrete struct per operation, never a map[string]any or any other string-keyed bag.

Rationale: Stringly-typed keys fail silently: a typo returns the zero default and produces a wrong answer with a nil error. A typed struct makes a wrong field or type a compile error. Each struct single-sources its defaults in WithDefaults() so the forward kernel and its VJP cannot drift. Migrated from cavekit SPEC.md I6, C14, V20 and ADR-01KYCZF2W8FZBB4QMWH7M56TY7.

## ARCH-007 {applies: go:backend.Execute}
WHEN backend.Execute receives a non-nil Attrs whose concrete type is not the one its Op declares, the backend.Execute SHALL return a non-nil error naming the Op, the got Attrs type and the want Attrs type, before resolving the Kernel.

Rationale: The codebase-wide idiom pa, _ := attrs.(backend.XAttrs) discards the ok flag, which is correct for a nil Attrs but would hand a kernel a zero-valued Attrs on a type mismatch and return a wrong answer with a nil error (observed: OpSum given ConcatAttrs returned scalar 28 instead of [6 22]). The op-to-type table is kept honest by a guard that derives ground truth from the kernels themselves rather than reading the table. Migrated from cavekit SPEC.md V28.

## ARCH-008 {applies: go:backend.Execute}
The backend.Execute SHALL strip the tape recorder from the Context passed to each Kernel, so an Op is recorded exactly once after the kernel returns.

Rationale: An in-kernel re-dispatch (a reference fallback, or cpu routing per ADR-01KYCZF2W8E7B99G59FTGY0S2S) would otherwise record the same op twice and double its gradients. Enforcing this structurally at the choke point removes the need for 46 per-site WithRecorder(nil) strips, which remain only as documentation. Migrated from cavekit SPEC.md V25.

## ARCH-009 {applies: go:backend.Backend}
The Backend and Kernel interfaces SHALL define an explicit execution and synchronization model in which synchronous execution is the default and an asynchronous accelerator exposes Synchronize().

Rationale: Fixed before the first accelerator backend existed so that adding a GPU could not force a breaking change on the public API. Migrated from cavekit SPEC.md V14.

## ARCH-010 {applies: go:apicheck.TestNoMagicBackendNameStrings}
The identifier-like string literals such as backend names SHALL be typed constants of the backend.Name enum (backend.CPU, backend.Ref, backend.Metal, backend.CUDA, backend.Vulkan) rather than bare string literals.

Rationale: Exceptions are the literals that define the enum itself in backend/names.go and the tensor.DeviceKind Stringer in tensor/device.go. Enforced mechanically. Migrated from cavekit SPEC.md C15, V21 and ADR-01KYCZF2W8EZKR9TS8RX89MC7E.

## BUILD-001
The repository SHALL keep the CGO_ENABLED=0 pure-Go build and test suite green on macOS, Windows and Linux against Go 1.26.

Rationale: The pure-Go path is the product's floor: it must run everywhere, on every platform, with no C toolchain present. Migrated from cavekit SPEC.md C1, I5 and V7.

## BUILD-002
The pure-Go verification gate SHALL run CGO_ENABLED=0 go vet ./... or go test rather than go build alone.

Rationale: go build skips _test.go files, so an untagged test file referencing cgo-only symbols passes the build and then breaks the suite. Every cgo-backend test file carries its build tag, and only a gate that compiles tests can enforce that. Migrated from cavekit SPEC.md V23.

## BUILD-003
IF a proposed cgo or external-C backend has not met all three gates, THEN the library SHALL reject it and keep the optimized pure-Go implementation on the ship path, leaving CGO_ENABLED=0 fully functional.

Rationale: The three gates: (1) the pure-Go version is verification-green and optimized to its documented ceiling with a roofline, (2) a benchmark beats the speedup threshold against that optimized pure-Go version, (3) CGO_ENABLED=0 stays fully functional through a fallback. cgo is a last resort, never a first reach. Migrated from cavekit SPEC.md C2 and V7.

## BUILD-004
The significance threshold for adopting a cgo or accelerated path over pure Go SHALL be a speedup of at least 1.5x, or reaching at least 80 percent of a C++ baseline that pure Go cannot reach.

Rationale: Makes deutlich schneller a measurable gate instead of a judgment call. Revisable only through a recorded decision. Migrated from cavekit SPEC.md C3.

## BUILD-005
WHERE the Metal backend on macOS, the build SHALL gate it on //go:build darwin && cgo alone, with no opt-in build tag.

Rationale: macOS system frameworks are always present, so Metal needs no link-time dependency check; CUDA and Vulkan stay behind opt-in tags because libcublas and libvulkan are build-time link dependencies not guaranteed to exist. Runtime dlopen of libcuda.so and libvulkan.so.1 is the documented path to make those tag-free too. Migrated from cavekit SPEC.md C6 and ADR-01KYCZF2W8F84VD5DSVQ4017MV.

## BUILD-006
The pure-Go dependency set SHALL keep go.mod free of cgo-dependent modules, isolating every accelerator dependency behind a build tag.

Rationale: Migrated from cavekit SPEC.md C9.

## BUILD-007
IF a model exceeds the available device VRAM, THEN the library SHALL transparently offload the overflow to the CPU-SIMD path and keep running, never aborting with an out-of-memory failure while a fallback exists.

Rationale: No hard VRAM floor: on a low-end device the library is slower but functional. This is the legitimate use of heterogeneous per-op splitting, for functionality rather than speed. Migrated from cavekit SPEC.md C24.

## BUILD-008
WHERE heterogeneous per-op-part backend assignment, the library SHALL expose it as an explicit documented configuration surface that is off by default and enabled only when measured beneficial or forced by a low-VRAM fallback.

Rationale: On a device where the model fits in VRAM a per-op split is a transfer-dominated loss, so the default stays the measured-best single backend. The mechanism is required; silent splitting is not. Migrated from cavekit SPEC.md C23, ADR-01KYCZF2W8E7B99G59FTGY0S2S and ADR-01KYCZF2W8F3GS2X0410JSHPKZ.

## NUM-001
The operation SHALL carry a golden test against NumPy or PyTorch within the documented tolerance of rtol 1e-12 for f64 and rtol 1e-5 for f32.

Rationale: Numeric parity is the acceptance criterion: done means results match a reference within a fixed tolerance, proven rather than claimed. A missing golden file is generated reproducibly and committed. Migrated from cavekit SPEC.md V1, G5 and R6.

## NUM-002
The differentiable operation SHALL pass a central finite-difference numeric gradient check within relative tolerance 1e-4.

Rationale: Migrated from cavekit SPEC.md V2.

## NUM-003 {applies: go:backend.Execute}
The pure-Go reference backend in backend/ref SHALL be the source of numeric truth for every backend.Op, with each accelerated kernel validated against it and never the reverse.

Rationale: Migrated from cavekit SPEC.md V9.

## NUM-004
The accelerated kernel result SHALL match backend/ref within a per-op documented rtol scaling with reduction length K, such as rtol(K)=1e-6*sqrt(K) for Metal f32 GEMM.

Rationale: An exact bit match is not required because SIMD and parallel execution reorder sums. Established points: elementwise and blocked GEMM on cpu are exact at tolerance 0 because they preserve accumulation order; Metal f32 GEMM is rtol(K) = 1e-6 * sqrt(K) because MPS both accumulates in f32 and reorders. Migrated from cavekit SPEC.md V3 and V11.

## NUM-005
The reductions and GEMM accumulation over f32 inputs SHALL accumulate in f64 or use Kahan summation unless a recorded research finding justifies otherwise.

Rationale: Floating-point addition is not associative, so accumulation width and order are correctness-relevant, not performance trivia. Goroutine and parallel reduction order is documented, and non-determinism is admitted only with a justified tolerance. Migrated from cavekit SPEC.md V10.

## NUM-006
The operation SHALL document and test its policy for NaN and Inf (IEEE-754 propagation), empty tensors, zero-dimensional tensors and non-contiguous views, with those edge cases present in its golden file.

Rationale: Migrated from cavekit SPEC.md V12.

## NUM-007
The golden file SHALL record its generating environment (library version, dtype, seed, shape) in a committed sidecar, so regeneration is byte-deterministic.

Rationale: Without a recorded environment a golden file cannot be regenerated or audited, and a reference-library upgrade silently changes what parity means. Migrated from cavekit SPEC.md V13.

## NUM-008
The algorithm implementation SHALL pass tier-one parity against a reference library and tier-two agreement with the defining paper equation, cited by DOI or arXiv id.

Rationale: Tier one alone is insufficient: a reference library can encode its own approximations. File formats have no defining paper, so their reference specification or implementation is the definitional source, stated explicitly rather than invented. Paper-tier status is tracked per algorithm. Migrated from cavekit SPEC.md V16, G4 and C18.

## NUM-009
The property-based tests SHALL assert shape algebra, linearity and associativity via testing/quick generators.

Rationale: Migrated from cavekit SPEC.md V6.

## NUM-010
The complex tasks SHALL be decomposed until each leaf is pinned in isolation by a unit test, parity check, gradient check or collapse test, then assembled bottom-up.

Rationale: A leaf verified in isolation eliminates defects at that level, so a defect surfacing higher up is localized to the new composition and fault-localization stays cheap. A big-bang assembly checkable only at the top hides an unbounded search. The established repo pattern is: collapse test pinning base equivalence at 1e-10, then per-parameter gradient check, then an end-to-end test that only checks integration. Migrated from cavekit SPEC.md C26.

## CI-001
The continuous integration SHALL stay green on macOS, Windows and Linux across the pure-Go fallback and every available accelerator, skipping an unavailable accelerator with a log line rather than passing silently.

Rationale: A silent skip is indistinguishable from a pass, which is how accelerator regressions survive a green board. Migrated from cavekit SPEC.md V4.

## CI-002
The full test sweep SHALL run go test with -timeout 1800s and check the exit code unpiped, since a trailing pipe to grep or tail masks a failure with exit status 0.

Rationale: The trained-model suite legitimately exceeds the 600s default (measured 670s green, failing at the default). A sweep claimed green without both conditions is void. Migrated from cavekit SPEC.md V24.

## CI-003 {applies: go:main.TestRunAlwaysRunSelectsMetaTests}
IF internal/cichange cannot positively prove that no code changed, THEN the CI selector SHALL classify the change as code and run the full pipeline, skipping only on positive proof that no .go file changed.

Rationale: Every uncertain case (parse error, unknown file class, missing or unavailable base revision, a workflow-file change, a non-Go non-doc file, added or deleted files that are not pure docs) fails open. Skipping CI is an optimization, never a correctness decision, and each case is asserted by a unit test. Migrated from cavekit SPEC.md V26.

## CI-004
WHEN a selective CI run is pushed, the loop SHALL compare the selector output against a go list -deps -test oracle, treating any omitted package as a release-blocking under-selection.

Rationale: The oracle is the compiler's own build-tag-aware truth, so it can refute the selector rather than agree with it by construction. An under-selection means a push shipped untested code and gets an immediate full-suite run plus a selector fix before the next selective push. Migrated from cavekit SPEC.md V27.

## CI-005 {applies: go:main.TestRunAlwaysRunSelectsMetaTests}
The run path and impact path of internal/cichange SHALL apply an identical alwaysRun set to every non-empty selection, and select zero runners for an empty one.

Rationale: A package registered in alwaysRun that the impact path pads into the affected set but the run path never selects for execution never actually gates a push (observed: speccheck was impact-padded but run-skipped on nn and nlp pushes). The docs-only case must reach zero runners. Migrated from cavekit SPEC.md V40.

## CI-006
IF a runner deprecation, action deprecation or tool warning appears in a CI run, THEN the loop SHALL open a tracked fix task for it, rather than suppressing the warning without a root-cause fix.

Rationale: CI watches must report warnings, not only pass or fail conclusions. Migrated from cavekit SPEC.md C17.

## API-001 {applies: go:apicheck.TestPublicAPIDocumentedWithExamples}
The exported symbol in a public package SHALL carry a godoc comment written for two audiences: the practitioner (math, algorithm, paper citation) and the layperson (plain what, why and when).

Rationale: Docs are a first-class deliverable, not an afterthought. internal/* and the backend implementation subpackages are exempt because they are registered by blank import rather than called by users. Migrated from cavekit SPEC.md C10, C13 and V17.

## API-002 {applies: go:apicheck.TestPublicAPIDocumentedWithExamples}
The exported struct field in a public package SHALL carry its own doc or inline comment, since a grouped section comment attaches only to the first spec.

Rationale: Fields are public-facing API too. Enforced mechanically by internal/apicheck. Migrated from cavekit SPEC.md C13 and V19.

## API-003 {applies: go:apicheck.TestPublicAPIDocumentedWithExamples}
The user-facing package SHALL ship runnable Example functions in example_test.go at three levels: trivial, realistic use case, and embedded in a larger pipeline, each verified by an // Output: block.

Rationale: Examples verified by go test cannot rot, which is why they are the doc form the gate enforces. New public API is not done until its godoc and examples exist. Migrated from cavekit SPEC.md C10 and V17.

## API-004 {applies: go:apicheck.TestPublicAPIDocumentedWithExamples}
The exported user-facing type and method SHALL appear in a runnable Example, credited by its name, its New-prefixed constructor, an ExampleType function, or a call in any Example body.

Rationale: internal/apicheck parses source with go/ast rather than importing packages, so build-tagged cgo backends are checked too. Types and methods where an example is not meaningful (interfaces, functional-option types, enums, config structs, trivial accessors) sit on a justified allowlist. Migrated from cavekit SPEC.md C13 and V19.

## API-005
The a public function taking two or more optional or configuration parameters SHALL accept them as variadic functional options (opts ...Option) rather than a long positional list, keeping required tensors positional.

Rationale: The idiomatic Go convention (Rob Pike's self-referential function pattern): hyperparameters become options with defaults and invalid values are guarded. A single well-named parameter may stay positional rather than being over-abstracted. Migrated from cavekit SPEC.md C12.

## API-006
The configuration knob (functional option, config field, CLI flag, sampler or optimizer parameter) SHALL document what it does, its behavior near its boundaries, and any special values, and cite the research grounding its default.

Rationale: Defaults are grounded in literature or reference implementations (PyTorch, llama.cpp, Hugging Face) so out-of-the-box behavior is good without fiddling. Special values with special semantics (0 disables, -1 unbounded, nil auto) are stated explicitly, and why this default is part of the doc rather than a bare number. Migrated from cavekit SPEC.md C21.

## API-007
The abbreviation or acronym in human-audience documentation SHALL be expanded and explained in lay terms at first use, as in GQA (grouped-query attention: several query heads share one key/value head, shrinking the cache).

Rationale: Applies to godoc, README and docs/*. Machine-audience texts are exempt. Migrated from cavekit SPEC.md C20.

## API-008
The documentation of an implementation, novel algorithm or method SHALL reference further reading beyond the defining paper, naming surveys or canonical texts such as Goodfellow Deep Learning, Sutton and Barto, or Jurafsky and Martin where they exist.

Rationale: Depth-entry points are part of the deliverable rather than optional garnish. Migrated from cavekit SPEC.md C18.

## API-009
IF a change would alter existing public API behavior or signatures, THEN the library SHALL route it through a documented deprecation path rather than breaking callers in place.

Rationale: Migrated from cavekit SPEC.md V8.

## API-010
The internal tooling that does not need the Python scientific stack SHALL be implemented in Go rather than Python, with the .venv reserved for reference-library parity such as golden generation.

Rationale: Existing Python utilities in that class are ported when touched. Migrated from cavekit SPEC.md C19.

## PERF-001
The optimized operation SHALL record a benchmark and its baseline number, with a regression breaking CI.

Rationale: Performance is measurable or it does not exist: no faster without a benchmark and a baseline comparison. Migrated from cavekit SPEC.md V5 and G6.

## PERF-002
IF a claim names X as the bottleneck or performance floor, THEN the loop SHALL prove it by forcing X off and comparing same-session medians, treating an unchanged number as a refutation.

Rationale: A bottleneck attribution is a measurable claim, not an assumption, and must be tested before it is recorded as fact or built upon. Three notes once asserted a memcpy floor that was never measured; the zero-copy work built on it produced zero delta and was reverted. Migrated from cavekit SPEC.md V22.

## PERF-003
IF a performance claim compares this library to an external incumbent such as PyTorch, scikit-learn, tiktoken or llama.cpp, THEN the loop SHALL measure both sides on byte-identical input, cross-check output equality, record the incumbent version, and commit a reproducible script or benchmark.

Rationale: Output equality is the fairness anchor; without it the two sides may not be doing the same work. A table stating measured N versus M with no committed measurement path is a violation (observed: a benchmark asserted 23 versus 20 MB/s from a different corpus, where the real numbers were 28.2 versus 18.8). Where the incumbent runs through a foreign-language binding, its marshalling cost is part of the honest number, but the framing must say so. Migrated from cavekit SPEC.md V38.

## PERF-004
The hot paths SHALL dispatch on dtype once into typed slice kernels, hoist loop invariants, and allocate zero per element or per token.

Rationale: At library scale a one percent gain is a large global compute and energy saving, so good enough is not acceptable on a hot path and the roofline ceiling is the target. Per-element interface dispatch, per-element closures and per-element index allocation are the recurring anti-patterns. Migrated from cavekit SPEC.md C25.

## PERF-005
WHEN an optimization is measured with a best-of-N A/B at p<0.05 and verified bit-identical with package tests green, the loop SHALL ship it immediately in its own commit with a recorded benchmark row, regardless of the size of the win.

Rationale: A real one percent is a win, not churn, and the only bar is real, measured and correct rather than large. Distinct from the threshold governing whether to chase an incumbent gap: a win already in hand is never discarded for being small. Marginal is not the same as unmeasured. Migrated from cavekit SPEC.md C27.

## PERF-006
WHEN a new generic optimization pattern or anti-pattern is discovered and verified, the loop SHALL add a detector to internal/perfscan with a positive and negative fixture, or a documented heuristic in PATTERNS.md.

Rationale: Encoding the pattern as a detector makes it institutional knowledge instead of something rediscovered by re-profiling. Each finding is a candidate that still needs a hotness measurement and a bit-identity proof before shipping. Suppression is class-granular so silencing one class cannot hide another. Migrated from cavekit SPEC.md C29.

## CORE-001 {applies: go:tensor.Tensor}
The tensor storage SHALL be row-major with explicit strides, so reshape, slice and transpose are stride operations rather than copies.

Rationale: Migrated from cavekit SPEC.md C5.

## CORE-002
The dtype support SHALL cover f32 and f64 first, with f16, bf16 and int8 following later.

Rationale: Migrated from cavekit SPEC.md C4.

## CORE-003
IF a proposal targets an NPU runtime such as ANE, CoreML, DirectML or oneDNN, THEN the library SHALL record it as an explicit non-goal of the current stage rather than a silent promise, to be re-evaluated after the GPU backends.

Rationale: Migrated from cavekit SPEC.md C7 and ADR-01KYCZF2W8EZP9G0SVH1W6C85K.

## PERF-007
The a performance comparison across engines or kernels SHALL hold numeric precision equal on both sides and disclose any differing axis, whether weights, compute accumulation or activations.

Rationale: Comparing f32 against f16, or f16 against int8, is not a like-for-like win: a lower-precision kernel trades accuracy for speed. Low-precision GEMM A/Bs remain valid as speed probes provided the precision is labeled and no quality claim is attached. Migrated from the worker spec Iw8.

## PERF-008
The an A/B measurement SHALL run on the same host with a file-copy baseline toggle, exclude warmup, and report medians over at least two counts in GFLOP/s.

Rationale: A git stash baseline is unreliable against the repo's shallow history. Migrated from the worker spec Iw6.

## PROC-001
WHERE work happens on the linux-amd64-cuda worker, the loop SHALL use a dedicated branch per task and a pull request, never committing or pushing to main directly.

Rationale: Merging is manual after CI turns green, never with --auto, because the absence of branch protection would let --auto merge before checks complete. Migrated from the worker spec RUN2 and RUN3.

## PROC-002
WHEN main has moved while a worker pull request is open, the loop SHALL merge origin/main into the branch and resolve three-way, never with checkout --ours and never via stash.

Rationale: The repo's shallow history makes stash-based resolution unsafe. A conflicting pull request also starts no CI at all, because GitHub produces no merge ref, so conflicts must be resolved before checks can be expected. Migrated from the worker spec RUN3.

## PROC-003
The pre-push gate SHALL run gofmt -l clean over the whole tree, CGO_ENABLED=0 go vet ./..., affected-package tests in both cgo modes, and the markdown lint.

Rationale: CI hard-fails on formatting and go vet does not check it, so a gate omitting gofmt reddens the lane; agent-added files are frequently unformatted. Exit codes are checked unpiped. Migrated from the worker spec RUN4.

## intent
GoAI is an idiomatic, modular Go library covering the full spectrum of AI work: linear algebra, autograd, classical machine learning, deep learning, NLP and LLM inference, computer vision, reinforcement learning, and probabilistic modeling. It is pure-Go first and cgo-last: the pure-Go path is the product floor and must run on macOS, Windows and Linux, on CPU and GPU, with an accelerator only where it is measurably needed.
Core operations target parity with the established C and C++ references (Eigen, OpenBLAS, oneDNN, ggml, ONNX Runtime, PyTorch ATen) through pure-Go SIMD. Correctness comes before speed: every operation gets a valid pure-Go reference implementation first, and optimization is a separate, separately measured step. Numeric parity is the acceptance criterion, and performance is measurable or it does not exist.
A complete release means: the L0 through L3 layers are green with parity across all operations; at least one optimized CPU backend beats the scalar reference with recorded benchmark numbers; safetensors IO works; at least one end-to-end trained model converges; at least one GPU backend passes the cgo gate; and at least one LLM inference path runs.
The library is organized as a strict layer model where no upper layer knows backend internals. L0 core holds Tensor (data, dtype, shape, strides, device), Dtype, Device, Allocator and stride-based views. L1 compute holds the Backend and Kernel interfaces with the pure-Go reference backend as truth, plus the registry and feature detection. L1b accel holds swappable cpu-simd, cuda, metal and vulkan backends behind that same interface. L2 is autograd (tape, Variable, per-op VJP rules). L3 is nn (layers, init, optimizers, losses, data pipeline). L4 is the domains (transformer and LLM, vision, classical ML, RL, probabilistic). L5 is io (safetensors, GGUF, ONNX).
Mathematical and scientific grounding is required per unit of work. Numeric decisions about stability, accuracy and overflow are documented rather than implicit, and every algorithm is traced to the paper that defines it.
- ADR-01KYJ95X8QE57A9BZC4S6S4PMY ADR-0006 — Tape-based reverse-mode autograd: compact
- ADR-01KYJ95X93EBJBF5R6F4V1F2AX ADR-0004 — GELU uses the exact (erf) definition: compact
- ADR-01KYJ95X96FADRYQHMJH6T4290 ADR-0011 — NPU acceleration is a model-level, not op-level, target (§T44): compact
- ADR-01KYJ95X99EG2B679165X8K1WM ADR-0028 — one shared NEON transcendental kernel (the "vexp leaf") for every f32 activation: compact
- ADR-01KYJ95X9CFN38E4KP0KDRN00G ADR-0024 — Cross-Layer Attention (CLA): isolated KV-sharing variant: compact
- ADR-01KYJ95X9GF87RHMMYYFFSM956 ADR-0005 — Optimized `cpu` backend; SIMD-intrinsics split (T11 / T11b): compact
- ADR-01KYJ95X9MF41T82JFEQ6GH2KS ADR-0007 — CrossEntropy as a fused op (log-softmax + NLL): compact
- ADR-01KYJ95X9QF0FVTGBX7S4P1Z3V ADR-0016 — Quantized matmul as an optional backend capability: compact
- ADR-01KYJ95X9TE4ZBD6SS0G3Z7R5V ADR-0022 — second-order / create-graph autograd, and how much of it Titans/TTT actually need: compact
- ADR-01KYJ95X9XF78VZT9Z72V0V813 ADR-0015 — Typed backend names replace magic-string identifiers: compact
- ADR-01KYJ95XA0E9JAZZ9PJJNFXC4B ADR-0020: PagedAttention is out of scope (revisit trigger defined): compact
- ADR-01KYJ95XA3F818SZNW8Z62HC8N ADR-0002 — Allocator abstraction; alignment is advisory in L0: compact
- ADR-01KYJ95XA7EXVB53WEJJHP7C1D ADR-0021 — f32-native accumulation for the amd64 SIMD GEMM fast path: compact
- ADR-01KYJ95XAAE74VMTCW8EE7J3E4 ADR-0025 — Byte Latent Transformer (BLT): the "needs ragged tensors" deferral is invalid: compact
- ADR-01KYJ95XADEFESTBN7WR364V69 ADR-0018 — Zero-copy UMA for GPU ops (page-aligned storage + bytesNoCopy): compact
- ADR-01KYJ95XAGFNA9QRW70KX9X9NB ADR-0003 — Opcode dispatch, single Execute choke-point, Recorder hook: compact
- ADR-01KYJ95XAKEHQ8K9BGD2PJ8QJ0 ADR-0023 — LLaDA-style masked-diffusion language model: design + GO verdict: compact
- ADR-01KYJ95XAQFRJAR1YVPJTPYE5Y ADR-0008 — GPU strategy: offload only measured winners: compact
- ADR-01KYJ95XATF41S6T2ZH51BJYGR ADR-0029 — the build-tagged test corpora have a verification boundary, and it is where the toolchain ends: compact
- ADR-01KYJ95XAYFJR8R6AXPWAAKEZC ADR-0026 — f32-native NEON GEMM for arm64 under GOEXPERIMENT=simd: compact
- ADR-01KYJ95XB1FMZRHT6VHZVZ1W8P ADR-0001 — Type-erased tensor storage with a runtime Dtype: compact
- ADR-01KYJ95XB5F4GTJ52EA1K6J86N ADR-0021: SIMD + GPU combination — configurable split, overlap, heterogeneous: compact
- ADR-01KYJ95XB8FYSS154XEJ24WVME ADR-0013 — Tag-free Metal & zero-config accel registration (§T47): compact
- ADR-01KYJ95XBBF3Z8E87GRW1CX1WZ ADR-0017 — Pure-Go GEMM cache blocking parked (no delta on arm64): compact
- ADR-01KYJ95XBFF7MAH761ZFTAC5BT ADR-0019 — Device-resident tensors + command-buffer batching: compact
- ADR-01KYJ95XBJEM78P5K3NZ820V72 ADR-0014 — Typed per-op parameters replace the `map[string]any` attrs bag: compact
- ADR-01KYJ95XBNF60TXQS5DYW8J5M4 ADR-0027 — Apple AMX F32 GEMM: Accelerate-cgo vs raw-AMX-asm, benchmark and use the winner: compact
- ADR-01KYJ95XBSEGNB3SNAPTFRR10N ADR-0012 — Automatic backend selection by performance preference (§T46): compact
- ADR-01KYJ95XBWFP08QK2JWEB1XJ6G ADR-0009 — CUDA/cuBLAS backend (§T42): compact
- ADR-01KYJ95XC0EBAVB00V6BF4Q561 ADR-0010 — Portable Vulkan compute backend (§T43): compact
- R-01KYJAB8Q6E4KT6021QAPEBFKN Go 1.26 `simd/archsimd` ships Feb 2026 under `GOEXPERIMENT=simd`,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAEXTTE0WT4SFSQNJ97T2A GoMLX active (v0.26.0 Dec 2025) but accel via OpenXLA/`gopjrt`=cgo;…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAEY9HFPZSE33HF2GAGWZ3 no practical Pure-Go path to discrete-GPU compute;…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAEYPGFHYR7MNC6X2V6JGN parity tolerance default f64 rtol 1e-12, f32 rtol 1e-5; golden from…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAEZ3SET7A35C4CHHJDV3A safetensors low effort (JSON header + raw tensors); GGUF medium; ONNX…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAEZJVEPNT6FP5D6WSP9VK GELU exact 0.5x(1+erf(x/√2)) = PyTorch default approximate='none': Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAF00FF8D9NG0QJJ2JPZGC Adam: bias-corr m̂=m/(1−β₁ᵗ) v̂=v/(1−β₂ᵗ), ε OUTSIDE sqrt: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAF0F6EGFVVNRVS7K93ZWQ SGD momentum torch: v=μv+g, p−=lr·v: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAF0ZYF6696P7F4P85S21M LayerNorm: biased var (÷N), ε INSIDE sqrt: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAF1ECF6DBTN81X06E7Z9N conv2d in DL = cross-correlation (no flip), out=(H+2p−k)/s+1: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAF1WSFNH8BJRCJSV18FS4 GGUF magic 0x46554747 LE; alignment 32 via general.alignment(u32);…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAF2B8F4NS5SM43NPYX8NA GGUF dims innermost-first → reverse for row-major: NOT in spec text,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAF2W8EFMTT36VGDXQPS5V linalg refs (nrm2 scaled, Cholesky, Jacobi eigen, PCA): Matrix…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAF3DVFDWRT7W466BXP76D RL refs (REINFORCE baseline independence, DQN replay+target):…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAF3YXF5C94KN8KBVD2HC6 LLM/transformer refs (attention, layernorm, tokenization): Foundation…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAF4JWF0487570XSVA734D deep-research adversarial verify panel BLOCKED by session limit…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAF51VFE0T6P3HFYXB6HCG Q4_0 dequant CONFIRMED vs ggml-quants.c: low nibble qs[j]→elem j, high…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAF5JJFSSS26ZEQCR1BS38 research mechanism: built-in /deep-research forces…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAF628FMZBAKN9MSB78K95 Q8_0: 34B block = f16 d + 32×int8 qs, x=d·qs[i] NO offset (scale…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAF6N5FDGSW9QER5HMSXA2 safetensors: 8B-LE u64 header-len + JSON + data; data_offsets relative…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAF79PE8RRKF019W1NY6DS .npy: v1 uint16 / v2-3 uint32 header-len; TOTAL header padded w/ spaces…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAF7TDEPFBDMVXFBCSD7CJ attention scale = 1/√d_k with d_k = d_model/h PER-HEAD…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAF87AE8SAA7FSCJSNCMG5 nrm2 scaled recurrence (scale=0,ssq=1;…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAF8MDF4BVSMZ04VHQXJV0 AdamW decoupled wd: p←p·(1−lr·wd)−lr·m̂/(√v̂+ε) (= paper Alg.2 line12,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAF91PE719ZG0XEQZG720K LoRA h=W0x+(α/r)·BAx; A[r,k] Gaussian/Kaiming, B[d,r] zero→ΔW=0 start;…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAF9F5FN5BDR0DG2ADMWQ6 RoPE: inv_freq_i=base^(−2i/d) base=10000;…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAF9X4ESASJ3D1PTYRBF21 RMSNorm y=x/√(mean(x²)+eps)·γ, no mean-sub, no bias, eps in sqrt.…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFAC1FX98MWNXNXKCRM6W SwiGLU FFN=(SiLU(x·Wg)⊙(x·Wu))·Wd, 3 matrices, no bias;…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFAS1E1VT3D039YHZZCM4 GQA: nkv divides nh; query head h→KV head h//(nh/nkv), contiguous…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFBB2FA0TAV12J9ZVP2P2 FlashAttention: tiled softmax, O(N) mem, no materialized N×N: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFBRGEB8AH2EAMDTGW1NC BPE subword tokenization: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFC62EFYVDR9R0KP04SBN top-p nucleus = smallest desc-prob set cumsum≥p (crossing token incl),…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFCPQEYABAJRYNZJ5HRCB LayerNorm backward formula: dL/dx=(1/σ)(a−mean(a)−x̂·mean(a·x̂)), a=g⊙γ: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFD6QET3SW65GP36Z31PX macOS GPU TRAINING via MPSGraph autodiff…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFDM6E4H9ZERJ42QV44CA Apple ANE = INFERENCE-ONLY via CoreML (FP16, no gradient API); training…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFE2JF2VSJGYQMWRJVKE8 NVIDIA cuDNN full fwd+bwd (dgrad/wgrad) but cgo-ONLY (proprietary C…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFEJRE99SGF0R6WPAQF9F Vulkan/WebGPU = general GPGPU, NO built-in autodiff → Go must…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFF1BFY0AE47VZPGZRZB2 Windows DirectML trainable (DML_*_GRAD + DML_ADAM_OPTIMIZER, manual…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFFDZE4TB1GNBCPVZ6Y8J f16=IEEE754 binary16 (1/5/10 bias15); f32→f16 = round-to-nearest-even,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFFVNE02RE39C3R2XRRB3 mixed-precision training 3 techniques: (1) fp32 MASTER weights —…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFGAYFZMS1048Z6VYW2K2 cuBLAS is COLUMN-MAJOR;…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFGQHFFB9BH3FJ3T2H9DS Vulkan compute matmul orchestration: instance→enum physical…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFH4GFB8BZ03MTT4XC5T2 op-level NPU dispatch from Go = impractical (2026). ANE: CoreML…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFHJ1E7SRRMRP9GNCGEQK backend auto-select = ordered preference, first REGISTERED wins,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFHZKFA7R4FK3T0BZY299 tag-free cgo accel feasibility: (1) Metal/MPS/Foundation = macOS SYSTEM…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFJCPFVRR0NYCVXR7SHYR DPO (Rafailov et al. 2023 Eq.7): L=−E log…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFJT1EEW80BGHVP46F1V1 PPO clipped surrogate (Schulman et al. 2017 Eq.7): L^CLIP=E[min(r·Â,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFK7TFNEV5T9YHDN7T8NY GAE (Schulman et al. 2016, arXiv:1506.02438):…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFKQGEA0RQBY9S5AN33NZ knowledge distillation (Hinton et al. 2015, arXiv:1503.02531): soft…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFM5NFG1THEKXNDHR9X45 label smoothing (Szegedy et al. 2016 §7 arXiv:1512.00567; Vaswani 2017…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFMJKEEDBW4C91V8Q3DBD speculative decoding (Leviathan et al. 2023 arXiv:2211.17192…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFN0JF9HV44MW3X9F9WJ0 beam search (Sutskever et al. 2014 seq2seq; length penalty Wu et al.…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFNE4FX3BY329JS5V7XRR inverted dropout (Srivastava/Hinton et al. 2014 JMLR 15; PyTorch…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFNVMEZSSFZ0C2HXFJA8C gradient accumulation (PyTorch/HF Accelerate convention): split batch N…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFP8JE3WAWDPTW9ZMQSZ9 logit penalties: (1) REPETITION (CTRL Keskar 2019 §4.1):…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFPQGE1G95KH86ABBTVWP IPO (Azar et al. 2023 arXiv:2310.12036 Eq.17; TRL loss_type="ipo"):…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFQ45EEXTTY5260APK3R2 KTO (Ethayarajh et al. 2024 arXiv:2402.01306; TRL KTOTrainer) —…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFQKJFM3BEES9XN5HRYQ5 ALiBi (Press Smith Lewis 2021 arXiv:2108.12409 §3) — Attention with…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFR0NE12BP83WNX35CV3T MoE routing + load balancing (Switch Fedus et al. 2021 arXiv:2101.03961…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFRDSE5VBQMWN1RW16W0Y sliding-window attention (Mistral Jiang et al. 2023 arXiv:2310.06825…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFRV6EH7RTA562BE76YYB min-p sampling (Nguyen et al. 2024 arXiv:2407.01082; HF…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFS8RE96TVY9HWA4A64HX RoPE Position Interpolation (Chen et al. 2023 arXiv:2306.15595) —…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFSP4EMQSDWTYDZDPGQSP Lion (EvoLved Sign Momentum, Chen et al. 2023 arXiv:2302.06675 Alg.2;…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFT44E54VH44FCDPXNBP9 YaRN (Peng et al. 2023 arXiv:2309.00071 §3.1-3.3) — "NTK-by-parts" RoPE…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFTHREAYVDJH57592WVN6 Pure-Go perf ceiling & GEMM ladder (research-lite CONFIRMED). BLIS/Goto…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFV03EEN8XAX98SCQGK11 LLM frontier techniques 2024-25 ranked for THIS lib (value×feasibility;…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFVD0FPWS3FBVADK1BEG2 GRPO (Group Relative Policy Optimization, Shao et al. 2024 DeepSeekMath…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFVV4E85V7TAZ8XW2P5R1 DoRA (Weight-Decomposed Low-Rank Adaptation, Liu et al. 2024…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFW9AF5SVYCTEA4W9CJK8 SimPO (Meng 2024 arXiv:2405.14734 eq5-7; princeton-nlp/SimPO) + ORPO…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFWQJEZS8K0G9504G3FRJ FlashAttention-2 (Dao 2023 arXiv:2307.08691 Alg.1 fwd + Alg.2 bwd;…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFX5PFM08JNR06JVVDAPS KV-cache eviction (StreamingLLM Xiao 2023 arXiv:2309.17453 §3.1-3.2 +…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFXJEFX09X3Y8496VXC5G MLA (Multi-head Latent Attention, DeepSeek-V2 Liu 2024 arXiv:2405.04434…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFY1EFWHBTHEV94123QNJ NF4 (4-bit NormalFloat) quantization + QLoRA (Dettmers et al. 2024…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFYJ0FMXVMT6SA7WHV3FZ Mamba selective-scan (S6) (Gu & Dao 2023 arXiv:2312.00752 §3.2 Alg.2;…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFYZDFQY8XNR226GYN5QC Mamba block structure (Gu & Dao 2023 arXiv:2312.00752 §3.4/Fig.3;…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFZC6E7JRBCDWQVDA7PT4 Muon optimizer (MomentUm Orthogonalized by Newton-schulz, Jordan et al.…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAFZSAEBZTG1ZPY437ETSZ NF4 double-quantization (nested quant, QLoRA Dettmers 2023…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAG07KFA2A700B9CZR086M Adafactor (Shazeer & Stern 2018 arXiv:1804.04235, sublinear-memory…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAG0N3FKK8YNJB6N9H0DJH GaLore (Zhao et al. 2024 arXiv:2403.03507, memory-efficient full-param…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAG123E409N3ESGGRP2JXP Contrastive Decoding (Li et al. 2023 arXiv:2210.15097; reasoning…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAG1FFE4BBAS8PD8XS8QQG RLOO REINFORCE Leave-One-Out for RLHF (Ahmadian et al. 2024…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAG1WEEFTT65XSHQYS0H2G Sophia second-order optimizer (Liu et al. 2023 "Sophia: A Scalable…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAG29XFTCBH4TRHSFH7MS4 Mirostat 2.0 adaptive decoding (Basu et al. 2020 "Mirostat: A Neural…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAG2Q9F25T6WTW06VEKVEN Schedule-Free optimizer (Defazio et al. 2024 "The Road Less Scheduled"…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAG342FVXBPJ4TZHEN68ND Unigram / SentencePiece subword tokenizer (Kudo 2018 "Subword…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAG3HGE07TPSCV3MWJMFHV GGUF tokenizer metadata convention (ggml/llama.cpp; NO paper →…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAG3Y4FDWSB72Y45BFZ349 Prompt-lookup / n-gram speculative decoding (Yang et al. 2023…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAG4EYEFWRKGTQDGATN39H GPT-2 / HF byte-level BPE (Sennrich 2016 BPE §R33 + Radford 2019 GPT-2…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAG4XVE2PR645NWFCRZZV3 Epsilon + Eta truncation sampling (Hewitt, Manning & Liang 2022…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAG5ASF6GBF6BWPKSH5ZV8 Llama/Llama-2 decoder architecture (Touvron et al. 2023…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAG5S1EDMB70CX6VS2YCM6 GGUF Llama weight/metadata convention (ggml/llama.cpp; NO paper →…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAG664FX5SEZ3N1YMT1QHY ggml Q8_0/Q4_0 quantization ENCODE (ggml-quants.c…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAG6PFEMZRZ9EPNGB671QY Dr. GRPO ("GRPO Done Right", Liu et al. 2025 "Understanding…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAG74GFAPV8SSVNDSCTFS6 NEFTune noisy-embedding fine-tuning (Jain et al. 2023 "NEFTune: Noisy…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAG7HCFNKTPCJ5VXCWB094 StreamingLLM / attention sinks (Xiao et al. 2023 "Efficient Streaming…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAG80WE8A8NKXNMMD4ESCV Lookahead optimizer (Zhang, Lucas, Ba & Hinton 2019 "Lookahead…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAG8EMFN98D0JZ3VW9NWC4 Q6_K k-quant format (ggml/llama.cpp; ggml-common.h block_q6_K +…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAG8YBEQS82XK3M2BQM747 Q4_K k-quant format (ggml/llama.cpp; ggml-common.h block_q4_K +…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAG9CHEXXSBW6JN975QAFW Q4_K_M quantization mix recipe (llama.cpp src/llama-quant.cpp…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAG9TYENKTB6K9AZXWHNKV Q5_K k-quant format (ggml/llama.cpp; ggml-common.h block_q5_K +…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGA95E7PTZZKH8CEA4V9V Q3_K k-quant format (ggml/llama.cpp; ggml-common.h block_q3_K +…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGAQGE5V9MM82BDTY8ZS5 Q2_K k-quant format (ggml/llama.cpp; ggml-common.h block_q2_K +…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGB4XEE3T94XGNMW26S2E AdEMAMix optimizer (Pagliardini, Ablin & Grangier 2024 arXiv:2409.03137…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGBJKFV69PVBWS7MG9ANH Cautious optimizer / C-AdamW (Liang, Chen, Zhang, Ding, Zhai, Liu et…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGC2WETHB364G6AD42R3P LR schedules: WSD (Warmup-Stable-Decay) + inverse-sqrt/Noam…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGCG7FN5RC3XJMDR8H8YN Quantized KV cache Q8_0 (ggml/llama.cpp type_k/type_v=GGML_TYPE_Q8_0;…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGCY9EECTBCFQNZXYMC45 Gradient checkpointing / activation rematerialization (Chen, Xu, Zhang,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGDBSFEF9P2AW7A9P4ST9 Sharpness-Aware Minimization SAM (Foret, Kleiner, Mobahi, Neyshabur…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGDSEFS5TSJVYDDMKSSJ1 Regex/FSM-guided constrained decoding (Willard & Louf 2023 "Efficient…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGE8MFFSAEF3T237A1RYA Weight-space averaging: SWA + weight-EMA (research-lite CONFIRMED…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGEP5E4PSE4Y5Z2GYM3SQ z-loss / log-Z softmax regularizer (Chowdhery et al. 2022 "PaLM"…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGF3QFJ38Z6X6HQFRA15M RetNet retention (Sun/Dong/Huang/Wang et al. 2023 "Retentive Network: A…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGFHNF1VARS42Y4CEZ7X9 LU decomposition with partial pivoting + derived…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGFZQE1VS7J4PVCWK3AWF QR decomposition via Householder reflections + least-squares (Golub &…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGGD0FF8S9PMJJXVD1D8Q SVD via one-sided Jacobi (Hestenes) (Golub & Van Loan §8.6.3, Demmel…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGGTZF739NRW88G0FBHD7 SVD-derived matrix quantities + matrix norms (numpy.linalg…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGHD8EVGSVW98AFVHCJBF Symmetric eigendecomposition (eigh) via cyclic Jacobi…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGHVAE1X87J3DX39J5FX9 Cholesky decomposition + SPD solve (numpy.linalg.cholesky / LAPACK…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGJC8EK0B1BKEP97R3K95 CPO — Contrastive Preference Optimization (Xu et al. 2024…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGJTPFZ6B2PQW2Y321DHG (IA)³ — Infused Adapter by Inhibiting and Amplifying Inner Activations…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGKC1ERRBXMXQS2HZV2P8 TIES-Merging — TrIm, Elect Sign & Merge (Yadav et al. 2023…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGKS7E0WTFR5BXG4FX99Q DARE — Drop And REscale (Yu et al. 2024 arXiv:2311.03099 §3.1, ICML,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGM7ME77953S7ZZAC4597 xPos — length-extrapolatable positional encoding (Sun et al. 2022…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGMNDEZSS9P8F0KCWV96B Concatenate (numpy.concatenate) — DEFINITIONAL, no paper/algorithm…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGN2XFMJ8HK15D17ZE35Z Prompt Tuning — soft-prompt PEFT (Lester et al. 2021 arXiv:2104.08691…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGNMQEC6TCCV83NJGZFDF Slice + Split (numpy x[start:end] / numpy.split) — DEFINITIONAL, no…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGP35FSJ8GB2TZAHQ08CZ Reshape + ExpandDims + Squeeze + Stack…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGPM1EW09QMXHTGXC297P Sqrt + Abs + Clip (numpy.sqrt/abs/clip) — DEFINITIONAL math ops, no…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGQ2SF60SBQ97XZVTRC58 Maximum + Minimum + Where (numpy.maximum/minimum/where) — DEFINITIONAL…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGQJWE0X9CSZSZFGJJV1Y Bottleneck Adapter — PEFT (Houlsby et al. 2019 arXiv:1902.00751…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGR1PEEQTGDAYW5WS4C6K Prefix Tuning — PEFT (Li & Liang 2021 arXiv:2101.00190 §4, ACL,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGRG1FSXR5BBGCH2MXA54 Multi-Token Prediction (MTP) — training objective (Gloeckle et al. 2024…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGRZFE32RVEB8GZJ7K3WQ R-Drop — regularized dropout (Wu et al. 2021 arXiv:2106.14448 §2.1…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGSD0EMZRQTT2Q0AVB7J2 NumPy .npy format (numpy.lib.format v1.0) — DEFINITIONAL, no paper…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGSTRF4AB9MMWS0KP6ZR1 NumPy .npz archive (numpy.savez / numpy.load) — DEFINITIONAL, no paper…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGT88F3B9HTEGK1RX9AKV BroadcastTo (numpy.broadcast_to) — DEFINITIONAL, no paper…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGTQNEJER92BE75H78TQB Auto-broadcasting elementwise binary ops (numpy broadcasting for…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGV8BE3FVQX431BJ0VC8E Var + Std reductions (numpy.var/std, ddof=0) — DEFINITIONAL, no paper…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGVT3E11V57GX83F4GYCJ Cumsum (numpy.cumsum) — DEFINITIONAL, no paper (scan op, §V16 exempt;…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGW8GEW4SZ5MEXZ43KEMH Einsum (numpy.einsum) — DEFINITIONAL, no paper (general tensor…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGWPYEE98Q10S03QVY8VM Differentiable Einsum — the einsum-swap-rule VJP (extends §R142;…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGX88EQMRY4YFBNF2G0EP Differentiable Cholesky — reverse-mode VJP (Iain Murray 2016…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGXQAEAVSA73ZKGAMBQTB Differentiable log-determinant (SPD) — DEFINITIONAL, no paper…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGY6QEPKSX30393DBZK2W Differentiable SPD solve — DEFINITIONAL, no paper (numpy.linalg.solve +…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGYM4FWT87E6DQMKTKTDC Differentiable QR (reduced) — reverse-mode VJP (Seeger, Hetzel, Dai,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGZ2XE5RSCY5Q54SMJAZD Differentiable symmetric eigendecomposition (eigh) — reverse-mode VJP…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAGZMMFS6A34W8WE2CNZNJ Differentiable SVD (reduced/thin) — reverse-mode VJP (Townsend 2016…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAH04DE0GVPMD75H215WQ8 Spectral Normalization (Miyato, Kataoka, Koyama & Yoshida 2018…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAH0HEFTZA3KDVHSYA54ZP Weight Normalization (Salimans & Kingma 2016 "Weight Normalization: A…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAH0ZHF9FRXQPNM263PCWP Logit soft-capping (Gemma-2 technical report, Google DeepMind 2024…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAH1DNFZ1AJBTR369ZVAKF Grokfast — accelerated grokking by amplifying slow gradients (Lee, Ahn,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAH1WNEX6TCQ6YCYMG6YE0 Locally Typical Sampling (Meister, Pimentel, Wiher & Cotterell 2023…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAH2A6FSXRCNRCWFSH9GP6 Shampoo — preconditioned second-order optimizer (Gupta, Koren & Singer…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAH2QFFV4TND7VX6AVFBAX GPTQ — accurate post-training weight quantization (Frantar, Ashkboos,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAH36FFFEB4D7W5GYNVG04 AWQ — Activation-aware Weight Quantization (Lin, Tang, Tang, Yang,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAH3M6E87R6V533FAN4RN0 DeepNorm / DeepNet (Wang, Ma, Dong, Huang, Zhang & Wei 2022 "DeepNet:…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAH41VEVGBG67D3FDN49CA GKD generalized Jensen-Shannon divergence distillation (Agarwal,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAH4FEE2YSC7WBF1GX5C23 SOAP — Adam in Shampoo's eigenbasis (Vyas, Morwani, Zhao,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAH4Y8EFMVAPXWN1QB78MH Expert Choice routing for MoE (Zhou, Lei, Liu, Du, Huang, Zhao, Dai, Le…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAH5BZFR9SVRC5KD75RQF2 Soft MoE — fully-differentiable Mixture-of-Experts (Puigcerver,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAH5T4EKDRT7QQ6DS4Z9QB Tape-recorded 2-D transpose (numpy.T) — DEFINITIONAL, no paper (§V16…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAH68ZE8KRA7D4QCHY1SEJ Maximal Update Parametrization (μP / μTransfer) — Yang, Hu, Babuschkin,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAH6Q0E48RYPSHHMHGA8BZ InfoNCE / contrastive loss (symmetric in-batch negatives) — van den…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAH755EGQT0AW8HD5WCXMP Scaled-cosine Query-Key Normalization (QKNorm) — Henry, Dachapally,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAH7JAEF39Z9MD05CAM45R Differential Transformer attention (DiffAttn) — Ye, Dong, Zhang, Zhu,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAH817EBSVR10M3E9097J4 Mixture-of-Depths (MoD) — Raposo, Ritter, Richards, Lillicrap,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAH8F0FYD8F6SEYZ9ZDE90 Gated Linear Attention (GLA) — Yang, Wang, Shen, Panda & Kim 2023/2024…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAH8ZAEWWRTPK81FFCTCZA DeltaNet — delta-rule linear attention (Yang, Wang, Zhang, Shen & Kim…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAH9D4EB09YQ0KHVJDVXCA Gated DeltaNet — gated delta rule (Yang, Kang, Hofmann, Zhang, van den…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAH9WDE47VSG1M78JKNBN7 SmoothQuant activation-weight smoothing (Xiao, Lin, Seznec, Wu, Demouth…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHAE0E5DRPR02YZ3BMVST Matryoshka Representation Learning (MRL) — Kusupati, Bhatt, Rege,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHAXAF4XS7ZY1K8A7QNY6 DoLa — Decoding by Contrasting Layers (Chuang, Xie, Luo, Kim, Glass &…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHBD2EEXB93HMD8EW8ZXB Gumbel-Softmax / Concrete relaxation (Jang, Gu & Poole 2016…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHBVAFVATG45WNA2R2BCR VQ-VAE vector quantization (van den Oord, Vinyals & Kavukcuoglu 2017,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHCB9FS08H0SFN53YT7Z9 FSQ Finite Scalar Quantization (Mentzer, Minnen, Agustsson & Tschannen…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHCS9EBMVQPS9ACSYV65P Auxiliary-Loss-Free Load Balancing (Wang, Gao, Chen, Xie & Dai 2024,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHD73FX49YWBFDA0JZXZC DropPath / Stochastic Depth (Huang, Sun, Liu, Sedra & Weinberger 2016,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHDNQE3CVME2KVQ4C3YRM ColBERT late-interaction MaxSim (Khattab & Zaharia 2020, "ColBERT:…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHE47EJ9T7GBPDQ9E8EFK Classifier-Free Guidance for LLMs (Sanchez, Fan, Spangher, Levi,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHEKHEHGSZTCP9MAAX69J No-repeat n-gram blocking (HuggingFace…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHF5FF22RPPS7JK5ZTSEA Group Normalization (Wu & He 2018, "Group Normalization", ECCV,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHFN8FBKS198YZ0X8B6TW Barlow Twins self-supervised loss (Zbontar, Jing, Misra, LeCun & Deny…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHG3HETBA5D9924QFRZHX Flow Matching / Rectified Flow generative objective (Lipman, Chen,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHGJGFKEV3NT9X2D4VHZ8 DDPM Denoising Diffusion Probabilistic Models (Ho, Jain & Abbeel 2020,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHH0AFSATX85M3BC7BRKY DDIM Denoising Diffusion Implicit Models sampler (Song, Meng & Ermon…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHHH0FYBBQBWFMRP7RK29 RWKV-4 WKV operator (Peng et al. 2023, "RWKV: Reinventing RNNs for the…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHHZGFKRTCBHA0BM3XQKY VICReg Variance-Invariance-Covariance Regularization (Bardes, Ponce &…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHJEQFQ48JPCRVK7X4KBZ LLM.int8() 8-bit matmul with outlier decomposition (Dettmers, Lewis,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHJXAEDJ8XRHRBSWK3CGW Contrastive Search decoding / SimCTG (Su, Lan, Wang, Yogatama, Kong &…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHKD3ENHAFRDJ7NHF871E SimCTG contrastive TRAINING loss (Su, Lan, Wang, Yogatama, Kong &…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHKV5FH28Z7N3WA3WF40S Wanda pruning (Sun, Liu, Bhojanapalli, Vishwanathan & Kolter 2023/2024,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHMD3FJ8B7JSDAB8PM414 SparseGPT one-shot pruning (Frantar & Alistarh 2023, "SparseGPT:…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHMTDFF9BADS5YT1P289F Medusa tree-based speculative decoding primitives (Cai, Li, Geng, Peng,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHN8ME94RZ8NWZ5489J64 Sinkhorn-Knopp entropic optimal transport (Cuturi 2013, "Sinkhorn…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHNPVF12BR8S8CGWB8V9F SwAV swapped-prediction loss (Caron, Misra, Mairal, Goyal, Bojanowski &…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHP5VE6PRH7B8EEP24P50 LSQ — Learned Step Size Quantization (Esser, McKinstry, Bablani,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHPKEECQVZZJGHER916G6 DINO self-distillation loss (Caron, Touvron, Misra, Jégou, Mairal,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHQ1PF2DBWP5YWFPZA6M4 SnapKV prompt KV-cache compression (Li, Huang, Yang, Yang, Zhang, Cai,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHQKPEYBR791RSRYP7GVP Self-Extend grouped-attention length extrapolation (Jin, Han, Tang,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHR1VETA85JXFJM7Z89VS Jacobi (parallel) decoding (Santilli, Severino, Postolache, Maiorca,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHRFTE1T98FHTS596K8AG HQQ — Half-Quadratic Quantization (Badri & Shrivastava 2023,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHRYVEWRRV7W6WHSW7W5P EDM deterministic sampler / Heun 2nd-order (Karras, Aittala, Aila &…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHSEWEP4SD9CD3ARGD93Y EDM preconditioning + σ-schedule + loss weighting (Karras et al. 2022,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHSZ0F69BNCTCDN1KTXFG Sequence packing without cross-contamination (Krell, Kosec, Perez &…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHTFXFYSTBCBHJTQRQ1BG Bradley-Terry reward-model loss (Christiano, Leike, Brown, Martic, Legg…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHTXTF3KBS25QEYF9WM2B Elastic Weight Consolidation / EWC (Kirkpatrick, Pascanu, Rabinowitz,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHVCSEG48YWRV1ZCK5W4A FIM — Fill-in-the-Middle training-data transform (Bavarian, Jun, Tezak,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHVWZEYQBGWYMVM2HGMX9 PyramidKV per-layer KV budget allocation (Cai, Zhang, Yue, Ren, Zhang,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHWCZFMQBPVZ4EJ0Y4QNT Plackett-Luce ranking loss / ListMLE (Plackett 1975 "The Analysis of…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHWWNE6PTB5T3KQQCWTGR SLERP — Spherical Linear Interpolation for weight merging (Shoemake…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHXA7F178X23T1GAQXCPV T5 span-corruption denoising pretraining objective (Raffel, Shazeer,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHXV7FA8BYWVW712GQ9XF SLiC-HF Sequence Likelihood Calibration alignment loss (Zhao, Joshi,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHY9FERKSH4JWJAGZK4D8 VeRA — Vector-based Random Matrix Adaptation (Kopiczko, Blankevoort &…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHYRDE50TCC0W4SGJQ5Z4 UL2 Mixture-of-Denoisers pretraining objective (Tay, Dehghani, Tran,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHZ5WFCTV9Y52RP1VAY47 Synaptic Intelligence (SI) continual-learning importance estimator…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAHZN4EFC86QYKZPWTTY40 LAMB — Layer-wise Adaptive Moments optimizer for Batch training (You,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJ04AEM6B1ANT2SH26EV6 LLM watermarking — red-green soft watermark + z-score detector…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJ0KMEEJS9GRXB1BW6RZC Diverse Beam Search (DBS) — group-diversity decoding (Vijayakumar,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJ13XFJKRQ8XGPF42M86Z Model Soups — uniform + greedy weight-averaging of fine-tunes…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJ1M3FX58FVJHFPT18ANQ PiSSA — Principal Singular values and Singular vectors Adaptation…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJ22DF7B8V8RNSX11AC9B SimSiam — Simple Siamese self-supervised representation learning (Chen…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJ2K8FS8RHZY4BDRVK1M4 MAS — Memory Aware Synapses continual-learning importance (Aljundi,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJ350ERHV24NBZ47MKP5H RSO — Statistical Rejection Sampling Optimization for preference data…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJ3RBFZWADGASX7YYZQ99 WordPiece subword tokenizer — greedy longest-match encoding (BERT:…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJ46QEF48MM2QMY8EVTKS DyT — Dynamic Tanh, normalization-free LayerNorm replacement (Zhu,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJ4W1FD5SCQ2WD8WG37EC Focal Loss — class-imbalance training loss (Lin, Goyal, Girshick, He &…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJ5BZEEA9SW3B7WG8A8GY BERT Masked Language Modeling (MLM) masking objective (Devlin, Chang,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJ5V0FFEAEQAMN1K52Z9A Sinusoidal positional encoding — the original absolute PE (Vaswani,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJ6BAFPBB62YE7MV5PKBC Triplet margin loss (Schroff, Kalenichenko & Philbin 2015, "FaceNet: A…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJ6XDFA9TR352R5VS2D5R T5 relative position bias — log-spaced bucketing (Raffel, Shazeer,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJ7B5EJAB8T7MG1JVH9AY OneCycle / 1cycle LR policy (Leslie N. Smith 2019, "Super-Convergence:…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJ7Y4EWZR0X0991GKVDVM online gap audit 2026-07-13 (user directive): swept llama.cpp…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJ8CFEX8BXMJ163VNXBC4 govulncheck reachability (T578, verified 2026-07-13 vs pkg.go.dev…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJ8W3EM5945V7HDHPJEVQ modality discovery sweep #2 (2026-07-13, empty-backlog rule step 1 —…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJ9APEJETPQDHG3NG8504 BENCHMARK SCOREBOARD (empty-backlog rule step 2, measured 2026-07-13,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJ9S5ECK8TW6HZZY4XK8T CI caching research (T605, user directive 2026-07-14; sources: danp.net…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJA6WER9BYP7ZFST6JCMT decode-gap decomposition (T611, measured 2026-07-14, 100 steps warm,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJAQ4F7MT46Z4NAWJSXAP TurboQuant sub-4-bit KV-quant VERIFIED (research-lite, WebFetch…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJB6ZF70TTGR3JJNPTEZK HC/mHC/JPmHC math VERIFIED (research-lite, §R234 discipline, WebFetch…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJBPBEAKAV33RK269BWC8 Aux-loss-free MoE load balancing VERIFIED (research-lite, §R234,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJC79ET585T4P3NTVH3FB nGPT normalized Transformer VERIFIED (research-lite §R234, WebFetch) vs…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJCNHFDYSKQ502JWNF4QY Forgetting Transformer (FoX) VERIFIED (research-lite §R234, WebFetch +…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJD3HE0FAMVVR0T4K01PN Hymba hybrid-head VERIFIED (research-lite §R234, WebFetch + repo…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJDK8E1ER159PZS8ECJ22 PERF FLOOR (T652/T653 pprof, measured M2 Pro): after devirtualizing the…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJE36FPP9DPGB1EZ3BKKD RECURSIVE DECOMPOSITION / STEPWISE REFINEMENT (grounds §C26 decompose…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJEHXEC2BTR4SF35R0VHB MODULAR DECOMPOSITION FOR INDEPENDENT VERIFIABILITY +…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJF0KERKR63DEB6ZPTT3D BOTTOM-UP ASSEMBLY + TEST PYRAMID + FAULT-LOCALIZATION (grounds §C26…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJFG0E96V74B4E1K7BFFP J-LENS / J-SPACE (Anthropic 2026-07-06 "Verbalizable Representations…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJFYYFSPR2KJBQKJDDC6D LLaDA-style masked discrete diffusion (Nie et al. 2025,…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJGDQEKRRMKHZB64HQ8XV Cross-Layer Attention / CLA (Brandon et al. 2024, arXiv:2405.12981):…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJGX7FPV9PS7Y93QG7QV3 Coconut continuous latent reasoning (Hao et al. 2024, arXiv:2412.06769…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJHCGEN4S6HEFMHHQAWT7 BitNet b1.58 QAT (Ma et al. 2024 arXiv:2402.17764; Wang et al. 2023…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJHTXFNMRWREHF1G01W9H Process reward models (Lightman et al. 2023 arXiv:2305.20050;…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJJA0EVJTZV8GNB24S32Y EAGLE feature-level speculative decoding (Li et al. 2024…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJJT7E6K9KH1G2MRM5ZWN Byte Latent Transformer / BLT (Pagnoni et al. 2024 arXiv:2412.09871): a…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJKAKFWESMFAREJQ9P1DK APOLLO (Zhu et al. 2024): project raw full-rank gradient G through a…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJKRKEM799GXT55KHFB40 Q-GaLore has four simultaneous storage/scheduling mechanisms, not…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJM7BE22B718K29DY2PB2 Softmax-off-by-one adds a fixed zero-logit no-op candidate:…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJMS7E5JT4ESVSXDPC3GK Fira keeps GaLore's SVD low-rank optimizer state but estimates…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJN7SESQ9RAPFG7A0KXJT LayerSkip self-speculative decoding uses ONE early-exit-trained…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJNPEEJ8SB80A7T2DAHPN Official LayerSkip checkpoints are standard Llama-family causal LMs…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJP5VFX89PSW31HN2SNW4 LayerSkip TRAINING (§4.1, Eq.1-6): per-sample Bernoulli drops WHOLE…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJPM0F42SYNE1K1XZPDV3 LayerSkip KVQ self-speculation (§4.3/A.6): draft fuehrt nur erste E…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJQ39FME8D2G89SDJTPWE PREFIX-KV-REUSE: Decoder-K/V an Position t haengt nur von tokens 0..t…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJQHXEE899YQBTV05Z8GR MULTI-REQUEST PREFIX POLICY: vLLM APC identifiziert volle KV-blocks…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAJR44FZF9S0F2CX3VQDM4 PREFIX-LOOKUP POLICY: SGLang RadixAttention organisiert token-exakte…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAB951ETXAR1ZF6987FWTN `simd` pkg inlined ≈4× vs next-best, ≈16× vs plain loop; `avo` ≈3× vs…: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
- R-01KYJAB9JVFX0A514D050TKY90 `simd/archsimd` ARM64/NEON not yet covered → use Plan9-NEON on ARM64: Research was performed and consumed during the cavekit v4 era; the finding already shaped the shipped code and the migrated contracts. Imported as historical evidence, no further action.
