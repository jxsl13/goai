---
schema: v1
prefix: ARCH
---

## ARCH-011 {applies: go:backend.Execute}
The packages above the backend layer SHALL depend only on the backend interface, never importing backend/ref, backend/cpu, backend/cuda, backend/metal or backend/vulkan.

Rationale: Layer model L0-L5: the public API stays backend-agnostic so an accelerator can be added or removed without breaking callers. Migrated from cavekit SPEC.md I1.

## ARCH-012 {applies: go:backend.Execute}
The library SHALL provide a pure-Go reference kernel in backend/ref for every registered backend.Op.

Rationale: Correctness before speed: the reference kernel is written first and every accelerated kernel is validated against it. Guarantees the CGO_ENABLED=0 path can run any operation. Migrated from cavekit SPEC.md I2 and G3.

## ARCH-013 {applies: go:tensor.Tensor}
The tensor core layer (tensor, dtype, device, allocator, views) SHALL build under CGO_ENABLED=0 and import no cgo-dependent package.

Rationale: L0 holds the data model every other layer depends on; admitting cgo there would make the whole library unbuildable without a C toolchain. Migrated from cavekit SPEC.md I3.

## ARCH-014 {applies: go:backend.Default}
WHEN a caller invokes backend.Default(), the library SHALL return the highest-preference registered backend along the descending order cuda > metal > vulkan > cpu, and otherwise the always-present backend/ref reference backend.

Rationale: Zero-config accelerator UX: building with an accelerator tag routes work to the GPU with no code change. Selection is by preference order, never by registration or import order. Order is overridable via backend.SetPreference; unknown names are skipped safely. Migrated from cavekit SPEC.md V18, C11 and I4.

## ARCH-015 {applies: go:backend.RegisterDefault}
IF an accelerated backend package is linked into the binary but its device is absent at runtime, THEN the backend registry SHALL leave that backend unregistered so backend.Default() falls back rather than claiming an absent device.

Rationale: An accelerated backend registers in init() only after detecting its device. A silent claim of an absent device would turn a missing GPU into a runtime failure instead of a fallback. Migrated from cavekit SPEC.md V18 and V4.

## ARCH-016 {applies: go:backend.Attrs}
The backend.Attrs SHALL remain a sealed interface (interface{ opAttrs() }) implemented by one concrete struct per operation, never a map[string]any or any other string-keyed bag.

Rationale: Stringly-typed keys fail silently: a typo returns the zero default and produces a wrong answer with a nil error. A typed struct makes a wrong field or type a compile error. Each struct single-sources its defaults in WithDefaults() so the forward kernel and its VJP cannot drift. Migrated from cavekit SPEC.md I6, C14, V20 and ADR-01KYCZF2W8FZBB4QMWH7M56TY7.

## ARCH-017 {applies: go:backend.Execute}
WHEN backend.Execute receives a non-nil Attrs whose concrete type is not the one its Op declares, the backend.Execute SHALL return a non-nil error naming the Op, the got Attrs type and the want Attrs type, before resolving the Kernel.

Rationale: The codebase-wide idiom pa, _ := attrs.(backend.XAttrs) discards the ok flag, which is correct for a nil Attrs but would hand a kernel a zero-valued Attrs on a type mismatch and return a wrong answer with a nil error (observed: OpSum given ConcatAttrs returned scalar 28 instead of [6 22]). The op-to-type table is kept honest by a guard that derives ground truth from the kernels themselves rather than reading the table. Migrated from cavekit SPEC.md V28.

## ARCH-018 {applies: go:backend.Execute}
The backend.Execute SHALL strip the tape recorder from the Context passed to each Kernel, so an Op is recorded exactly once after the kernel returns.

Rationale: An in-kernel re-dispatch (a reference fallback, or cpu routing per ADR-01KYCZF2W8E7B99G59FTGY0S2S) would otherwise record the same op twice and double its gradients. Enforcing this structurally at the choke point removes the need for 46 per-site WithRecorder(nil) strips, which remain only as documentation. Migrated from cavekit SPEC.md V25.

## ARCH-019 {applies: go:backend.Backend}
The Backend and Kernel interfaces SHALL define an explicit execution and synchronization model in which synchronous execution is the default and an asynchronous accelerator exposes Synchronize().

Rationale: Fixed before the first accelerator backend existed so that adding a GPU could not force a breaking change on the public API. Migrated from cavekit SPEC.md V14.

## ARCH-020 {applies: go:apicheck.TestNoMagicBackendNameStrings}
The identifier-like string literals such as backend names SHALL be typed constants of the backend.Name enum (backend.CPU, backend.Ref, backend.Metal, backend.CUDA, backend.Vulkan) rather than bare string literals.

Rationale: Exceptions are the literals that define the enum itself in backend/names.go and the tensor.DeviceKind Stringer in tensor/device.go. Enforced mechanically. Migrated from cavekit SPEC.md C15, V21 and ADR-01KYCZF2W8EZKR9TS8RX89MC7E.

## BUILD-009
The repository SHALL keep the CGO_ENABLED=0 pure-Go build and test suite green on macOS, Windows and Linux against Go 1.26.

Rationale: The pure-Go path is the product's floor: it must run everywhere, on every platform, with no C toolchain present. Migrated from cavekit SPEC.md C1, I5 and V7.

## BUILD-010
The pure-Go verification gate SHALL run CGO_ENABLED=0 go vet ./... or go test rather than go build alone.

Rationale: go build skips _test.go files, so an untagged test file referencing cgo-only symbols passes the build and then breaks the suite. Every cgo-backend test file carries its build tag, and only a gate that compiles tests can enforce that. Migrated from cavekit SPEC.md V23.

## BUILD-011
IF a proposed cgo or external-C backend has not met all three gates, THEN the library SHALL reject it and keep the optimized pure-Go implementation on the ship path, leaving CGO_ENABLED=0 fully functional.

Rationale: The three gates: (1) the pure-Go version is verification-green and optimized to its documented ceiling with a roofline, (2) a benchmark beats the speedup threshold against that optimized pure-Go version, (3) CGO_ENABLED=0 stays fully functional through a fallback. cgo is a last resort, never a first reach. Migrated from cavekit SPEC.md C2 and V7.

## BUILD-012
The significance threshold for adopting a cgo or accelerated path over pure Go SHALL be a speedup of at least 1.5x, or reaching at least 80 percent of a C++ baseline that pure Go cannot reach.

Rationale: Makes deutlich schneller a measurable gate instead of a judgment call. Revisable only through a recorded decision. Migrated from cavekit SPEC.md C3.

## BUILD-013
WHERE the Metal backend on macOS, the build SHALL gate it on //go:build darwin && cgo alone, with no opt-in build tag.

Rationale: macOS system frameworks are always present, so Metal needs no link-time dependency check; CUDA and Vulkan stay behind opt-in tags because libcublas and libvulkan are build-time link dependencies not guaranteed to exist. Runtime dlopen of libcuda.so and libvulkan.so.1 is the documented path to make those tag-free too. Migrated from cavekit SPEC.md C6 and ADR-01KYCZF2W8F84VD5DSVQ4017MV.

## BUILD-014
The pure-Go dependency set SHALL keep go.mod free of cgo-dependent modules, isolating every accelerator dependency behind a build tag.

Rationale: Migrated from cavekit SPEC.md C9.

## BUILD-015
IF a model exceeds the available device VRAM, THEN the library SHALL transparently offload the overflow to the CPU-SIMD path and keep running, never aborting with an out-of-memory failure while a fallback exists.

Rationale: No hard VRAM floor: on a low-end device the library is slower but functional. This is the legitimate use of heterogeneous per-op splitting, for functionality rather than speed. Migrated from cavekit SPEC.md C24.

## BUILD-016
WHERE heterogeneous per-op-part backend assignment, the library SHALL expose it as an explicit documented configuration surface that is off by default and enabled only when measured beneficial or forced by a low-VRAM fallback.

Rationale: On a device where the model fits in VRAM a per-op split is a transfer-dominated loss, so the default stays the measured-best single backend. The mechanism is required; silent splitting is not. Migrated from cavekit SPEC.md C23, ADR-01KYCZF2W8E7B99G59FTGY0S2S and ADR-01KYCZF2W8F3GS2X0410JSHPKZ.

## NUM-011
The operation SHALL carry a golden test against NumPy or PyTorch within the documented tolerance of rtol 1e-12 for f64 and rtol 1e-5 for f32.

Rationale: Numeric parity is the acceptance criterion: done means results match a reference within a fixed tolerance, proven rather than claimed. A missing golden file is generated reproducibly and committed. Migrated from cavekit SPEC.md V1, G5 and R6.

## NUM-012
The differentiable operation SHALL pass a central finite-difference numeric gradient check within relative tolerance 1e-4.

Rationale: Migrated from cavekit SPEC.md V2.

## NUM-013 {applies: go:backend.Execute}
The pure-Go reference backend in backend/ref SHALL be the source of numeric truth for every backend.Op, with each accelerated kernel validated against it and never the reverse.

Rationale: Migrated from cavekit SPEC.md V9.

## NUM-014
The accelerated kernel result SHALL match backend/ref within a per-op documented rtol scaling with reduction length K, such as rtol(K)=1e-6*sqrt(K) for Metal f32 GEMM.

Rationale: An exact bit match is not required because SIMD and parallel execution reorder sums. Established points: elementwise and blocked GEMM on cpu are exact at tolerance 0 because they preserve accumulation order; Metal f32 GEMM is rtol(K) = 1e-6 * sqrt(K) because MPS both accumulates in f32 and reorders. Migrated from cavekit SPEC.md V3 and V11.

## NUM-015
The reductions and GEMM accumulation over f32 inputs SHALL accumulate in f64 or use Kahan summation unless a recorded research finding justifies otherwise.

Rationale: Floating-point addition is not associative, so accumulation width and order are correctness-relevant, not performance trivia. Goroutine and parallel reduction order is documented, and non-determinism is admitted only with a justified tolerance. Migrated from cavekit SPEC.md V10.

## NUM-016
The operation SHALL document and test its policy for NaN and Inf (IEEE-754 propagation), empty tensors, zero-dimensional tensors and non-contiguous views, with those edge cases present in its golden file.

Rationale: Migrated from cavekit SPEC.md V12.

## NUM-017
The golden file SHALL record its generating environment (library version, dtype, seed, shape) in a committed sidecar, so regeneration is byte-deterministic.

Rationale: Without a recorded environment a golden file cannot be regenerated or audited, and a reference-library upgrade silently changes what parity means. Migrated from cavekit SPEC.md V13.

## NUM-018
The algorithm implementation SHALL pass tier-one parity against a reference library and tier-two agreement with the defining paper equation, cited by DOI or arXiv id.

Rationale: Tier one alone is insufficient: a reference library can encode its own approximations. File formats have no defining paper, so their reference specification or implementation is the definitional source, stated explicitly rather than invented. Paper-tier status is tracked per algorithm. Migrated from cavekit SPEC.md V16, G4 and C18.

## NUM-019
The property-based tests SHALL assert shape algebra, linearity and associativity via testing/quick generators.

Rationale: Migrated from cavekit SPEC.md V6.

## NUM-020
The complex tasks SHALL be decomposed until each leaf is pinned in isolation by a unit test, parity check, gradient check or collapse test, then assembled bottom-up.

Rationale: A leaf verified in isolation eliminates defects at that level, so a defect surfacing higher up is localized to the new composition and fault-localization stays cheap. A big-bang assembly checkable only at the top hides an unbounded search. The established repo pattern is: collapse test pinning base equivalence at 1e-10, then per-parameter gradient check, then an end-to-end test that only checks integration. Migrated from cavekit SPEC.md C26.

## CI-007
The continuous integration SHALL stay green on macOS, Windows and Linux across the pure-Go fallback and every available accelerator, skipping an unavailable accelerator with a log line rather than passing silently.

Rationale: A silent skip is indistinguishable from a pass, which is how accelerator regressions survive a green board. Migrated from cavekit SPEC.md V4.

## CI-008
The full test sweep SHALL run go test with -timeout 1800s and check the exit code unpiped, since a trailing pipe to grep or tail masks a failure with exit status 0.

Rationale: The trained-model suite legitimately exceeds the 600s default (measured 670s green, failing at the default). A sweep claimed green without both conditions is void. Migrated from cavekit SPEC.md V24.

## CI-009 {applies: go:main.TestRunAlwaysRunSelectsMetaTests}
IF internal/cichange cannot positively prove that no code changed, THEN the CI selector SHALL classify the change as code and run the full pipeline, skipping only on positive proof that no .go file changed.

Rationale: Every uncertain case (parse error, unknown file class, missing or unavailable base revision, a workflow-file change, a non-Go non-doc file, added or deleted files that are not pure docs) fails open. Skipping CI is an optimization, never a correctness decision, and each case is asserted by a unit test. Migrated from cavekit SPEC.md V26.

## CI-010
WHEN a selective CI run is pushed, the loop SHALL compare the selector output against a go list -deps -test oracle, treating any omitted package as a release-blocking under-selection.

Rationale: The oracle is the compiler's own build-tag-aware truth, so it can refute the selector rather than agree with it by construction. An under-selection means a push shipped untested code and gets an immediate full-suite run plus a selector fix before the next selective push. Migrated from cavekit SPEC.md V27.

## CI-011 {applies: go:main.TestRunAlwaysRunSelectsMetaTests}
The run path and impact path of internal/cichange SHALL apply an identical alwaysRun set to every non-empty selection, and select zero runners for an empty one.

Rationale: A package registered in alwaysRun that the impact path pads into the affected set but the run path never selects for execution never actually gates a push (observed: speccheck was impact-padded but run-skipped on nn and nlp pushes). The docs-only case must reach zero runners. Migrated from cavekit SPEC.md V40.

## CI-012
IF a runner deprecation, action deprecation or tool warning appears in a CI run, THEN the loop SHALL open a tracked fix task for it, rather than suppressing the warning without a root-cause fix.

Rationale: CI watches must report warnings, not only pass or fail conclusions. Migrated from cavekit SPEC.md C17.

## API-011 {applies: go:apicheck.TestPublicAPIDocumentedWithExamples}
The exported symbol in a public package SHALL carry a godoc comment written for two audiences: the practitioner (math, algorithm, paper citation) and the layperson (plain what, why and when).

Rationale: Docs are a first-class deliverable, not an afterthought. internal/* and the backend implementation subpackages are exempt because they are registered by blank import rather than called by users. Migrated from cavekit SPEC.md C10, C13 and V17.

## API-012 {applies: go:apicheck.TestPublicAPIDocumentedWithExamples}
The exported struct field in a public package SHALL carry its own doc or inline comment, since a grouped section comment attaches only to the first spec.

Rationale: Fields are public-facing API too. Enforced mechanically by internal/apicheck. Migrated from cavekit SPEC.md C13 and V19.

## API-013 {applies: go:apicheck.TestPublicAPIDocumentedWithExamples}
The user-facing package SHALL ship runnable Example functions in example_test.go at three levels: trivial, realistic use case, and embedded in a larger pipeline, each verified by an // Output: block.

Rationale: Examples verified by go test cannot rot, which is why they are the doc form the gate enforces. New public API is not done until its godoc and examples exist. Migrated from cavekit SPEC.md C10 and V17.

## API-014 {applies: go:apicheck.TestPublicAPIDocumentedWithExamples}
The exported user-facing type and method SHALL appear in a runnable Example, credited by its name, its New-prefixed constructor, an ExampleType function, or a call in any Example body.

Rationale: internal/apicheck parses source with go/ast rather than importing packages, so build-tagged cgo backends are checked too. Types and methods where an example is not meaningful (interfaces, functional-option types, enums, config structs, trivial accessors) sit on a justified allowlist. Migrated from cavekit SPEC.md C13 and V19.

## API-015
The a public function taking two or more optional or configuration parameters SHALL accept them as variadic functional options (opts ...Option) rather than a long positional list, keeping required tensors positional.

Rationale: The idiomatic Go convention (Rob Pike's self-referential function pattern): hyperparameters become options with defaults and invalid values are guarded. A single well-named parameter may stay positional rather than being over-abstracted. Migrated from cavekit SPEC.md C12.

## API-016
The configuration knob (functional option, config field, CLI flag, sampler or optimizer parameter) SHALL document what it does, its behavior near its boundaries, and any special values, and cite the research grounding its default.

Rationale: Defaults are grounded in literature or reference implementations (PyTorch, llama.cpp, Hugging Face) so out-of-the-box behavior is good without fiddling. Special values with special semantics (0 disables, -1 unbounded, nil auto) are stated explicitly, and why this default is part of the doc rather than a bare number. Migrated from cavekit SPEC.md C21.

## API-017
The abbreviation or acronym in human-audience documentation SHALL be expanded and explained in lay terms at first use, as in GQA (grouped-query attention: several query heads share one key/value head, shrinking the cache).

Rationale: Applies to godoc, README and docs/*. Machine-audience texts are exempt. Migrated from cavekit SPEC.md C20.

## API-018
The documentation of an implementation, novel algorithm or method SHALL reference further reading beyond the defining paper, naming surveys or canonical texts such as Goodfellow Deep Learning, Sutton and Barto, or Jurafsky and Martin where they exist.

Rationale: Depth-entry points are part of the deliverable rather than optional garnish. Migrated from cavekit SPEC.md C18.

## API-019
IF a change would alter existing public API behavior or signatures, THEN the library SHALL route it through a documented deprecation path rather than breaking callers in place.

Rationale: Migrated from cavekit SPEC.md V8.

## API-020
The internal tooling that does not need the Python scientific stack SHALL be implemented in Go rather than Python, with the .venv reserved for reference-library parity such as golden generation.

Rationale: Existing Python utilities in that class are ported when touched. Migrated from cavekit SPEC.md C19.

## PERF-009
The optimized operation SHALL record a benchmark and its baseline number, with a regression breaking CI.

Rationale: Performance is measurable or it does not exist: no faster without a benchmark and a baseline comparison. Migrated from cavekit SPEC.md V5 and G6.

## PERF-010
IF a claim names X as the bottleneck or performance floor, THEN the loop SHALL prove it by forcing X off and comparing same-session medians, treating an unchanged number as a refutation.

Rationale: A bottleneck attribution is a measurable claim, not an assumption, and must be tested before it is recorded as fact or built upon. Three notes once asserted a memcpy floor that was never measured; the zero-copy work built on it produced zero delta and was reverted. Migrated from cavekit SPEC.md V22.

## PERF-011
IF a performance claim compares this library to an external incumbent such as PyTorch, scikit-learn, tiktoken or llama.cpp, THEN the loop SHALL measure both sides on byte-identical input, cross-check output equality, record the incumbent version, and commit a reproducible script or benchmark.

Rationale: Output equality is the fairness anchor; without it the two sides may not be doing the same work. A table stating measured N versus M with no committed measurement path is a violation (observed: a benchmark asserted 23 versus 20 MB/s from a different corpus, where the real numbers were 28.2 versus 18.8). Where the incumbent runs through a foreign-language binding, its marshalling cost is part of the honest number, but the framing must say so. Migrated from cavekit SPEC.md V38.

## PERF-012
The hot paths SHALL dispatch on dtype once into typed slice kernels, hoist loop invariants, and allocate zero per element or per token.

Rationale: At library scale a one percent gain is a large global compute and energy saving, so good enough is not acceptable on a hot path and the roofline ceiling is the target. Per-element interface dispatch, per-element closures and per-element index allocation are the recurring anti-patterns. Migrated from cavekit SPEC.md C25.

## PERF-013
WHEN an optimization is measured with a best-of-N A/B at p<0.05 and verified bit-identical with package tests green, the loop SHALL ship it immediately in its own commit with a recorded benchmark row, regardless of the size of the win.

Rationale: A real one percent is a win, not churn, and the only bar is real, measured and correct rather than large. Distinct from the threshold governing whether to chase an incumbent gap: a win already in hand is never discarded for being small. Marginal is not the same as unmeasured. Migrated from cavekit SPEC.md C27.

## PERF-014
WHEN a new generic optimization pattern or anti-pattern is discovered and verified, the loop SHALL add a detector to internal/perfscan with a positive and negative fixture, or a documented heuristic in PATTERNS.md.

Rationale: Encoding the pattern as a detector makes it institutional knowledge instead of something rediscovered by re-profiling. Each finding is a candidate that still needs a hotness measurement and a bit-identity proof before shipping. Suppression is class-granular so silencing one class cannot hide another. Migrated from cavekit SPEC.md C29.

## CORE-004 {applies: go:tensor.Tensor}
The tensor storage SHALL be row-major with explicit strides, so reshape, slice and transpose are stride operations rather than copies.

Rationale: Migrated from cavekit SPEC.md C5.

## CORE-005
The dtype support SHALL cover f32 and f64 first, with f16, bf16 and int8 following later.

Rationale: Migrated from cavekit SPEC.md C4.

## CORE-006
IF a proposal targets an NPU runtime such as ANE, CoreML, DirectML or oneDNN, THEN the library SHALL record it as an explicit non-goal of the current stage rather than a silent promise, to be re-evaluated after the GPU backends.

Rationale: Migrated from cavekit SPEC.md C7 and ADR-01KYCZF2W8EZP9G0SVH1W6C85K.

## PERF-015
The a performance comparison across engines or kernels SHALL hold numeric precision equal on both sides and disclose any differing axis, whether weights, compute accumulation or activations.

Rationale: Comparing f32 against f16, or f16 against int8, is not a like-for-like win: a lower-precision kernel trades accuracy for speed. Low-precision GEMM A/Bs remain valid as speed probes provided the precision is labeled and no quality claim is attached. Migrated from the worker spec Iw8.

## PERF-016
The an A/B measurement SHALL run on the same host with a file-copy baseline toggle, exclude warmup, and report medians over at least two counts in GFLOP/s.

Rationale: A git stash baseline is unreliable against the repo's shallow history. Migrated from the worker spec Iw6.

## PROC-004
WHERE work happens on the linux-amd64-cuda worker, the loop SHALL use a dedicated branch per task and a pull request, never committing or pushing to main directly.

Rationale: Merging is manual after CI turns green, never with --auto, because the absence of branch protection would let --auto merge before checks complete. Migrated from the worker spec RUN2 and RUN3.

## PROC-005
WHEN main has moved while a worker pull request is open, the loop SHALL merge origin/main into the branch and resolve three-way, never with checkout --ours and never via stash.

Rationale: The repo's shallow history makes stash-based resolution unsafe. A conflicting pull request also starts no CI at all, because GitHub produces no merge ref, so conflicts must be resolved before checks can be expected. Migrated from the worker spec RUN3.

## PROC-006
The pre-push gate SHALL run gofmt -l clean over the whole tree, CGO_ENABLED=0 go vet ./..., affected-package tests in both cgo modes, and the markdown lint.

Rationale: CI hard-fails on formatting and go vet does not check it, so a gate omitting gofmt reddens the lane; agent-added files are frequently unformatted. Exit codes are checked unpiped. Migrated from the worker spec RUN4.

## intent
GoAI is an idiomatic, modular Go library covering the full spectrum of AI work: linear algebra, autograd, classical machine learning, deep learning, NLP and LLM inference, computer vision, reinforcement learning, and probabilistic modeling. It is pure-Go first and cgo-last: the pure-Go path is the product floor and must run on macOS, Windows and Linux, on CPU and GPU, with an accelerator only where it is measurably needed.
Core operations target parity with the established C and C++ references (Eigen, OpenBLAS, oneDNN, ggml, ONNX Runtime, PyTorch ATen) through pure-Go SIMD. Correctness comes before speed: every operation gets a valid pure-Go reference implementation first, and optimization is a separate, separately measured step. Numeric parity is the acceptance criterion, and performance is measurable or it does not exist.
A complete release means: the L0 through L3 layers are green with parity across all operations; at least one optimized CPU backend beats the scalar reference with recorded benchmark numbers; safetensors IO works; at least one end-to-end trained model converges; at least one GPU backend passes the cgo gate; and at least one LLM inference path runs.
The library is organized as a strict layer model where no upper layer knows backend internals. L0 core holds Tensor (data, dtype, shape, strides, device), Dtype, Device, Allocator and stride-based views. L1 compute holds the Backend and Kernel interfaces with the pure-Go reference backend as truth, plus the registry and feature detection. L1b accel holds swappable cpu-simd, cuda, metal and vulkan backends behind that same interface. L2 is autograd (tape, Variable, per-op VJP rules). L3 is nn (layers, init, optimizers, losses, data pipeline). L4 is the domains (transformer and LLM, vision, classical ML, RL, probabilistic). L5 is io (safetensors, GGUF, ONNX).
Mathematical and scientific grounding is required per unit of work. Numeric decisions about stability, accuracy and overflow are documented rather than implicit, and every algorithm is traced to the paper that defines it.
GoAI is an idiomatic, modular Go library covering the full spectrum of AI work: linear algebra, autograd, classical machine learning, deep learning, NLP and LLM inference, computer vision, reinforcement learning, and probabilistic modeling. It is pure-Go first and cgo-last: the pure-Go path is the product floor and must run on macOS, Windows and Linux, on CPU and GPU, with an accelerator only where it is measurably needed.
Core operations target parity with the established C and C++ references (Eigen, OpenBLAS, oneDNN, ggml, ONNX Runtime, PyTorch ATen) through pure-Go SIMD. Correctness comes before speed: every operation gets a valid pure-Go reference implementation first, and optimization is a separate, separately measured step. Numeric parity is the acceptance criterion, and performance is measurable or it does not exist.
A complete release means: the L0 through L3 layers are green with parity across all operations; at least one optimized CPU backend beats the scalar reference with recorded benchmark numbers; safetensors IO works; at least one end-to-end trained model converges; at least one GPU backend passes the cgo gate; and at least one LLM inference path runs.
The library is organized as a strict layer model where no upper layer knows backend internals. L0 core holds Tensor (data, dtype, shape, strides, device), Dtype, Device, Allocator and stride-based views. L1 compute holds the Backend and Kernel interfaces with the pure-Go reference backend as truth, plus the registry and feature detection. L1b accel holds swappable cpu-simd, cuda, metal and vulkan backends behind that same interface. L2 is autograd (tape, Variable, per-op VJP rules). L3 is nn (layers, init, optimizers, losses, data pipeline). L4 is the domains (transformer and LLM, vision, classical ML, RL, probabilistic). L5 is io (safetensors, GGUF, ONNX).
Mathematical and scientific grounding is required per unit of work. Numeric decisions about stability, accuracy and overflow are documented rather than implicit, and every algorithm is traced to the paper that defines it.
- ADR-01KYJNCHP0E09BV3XD9Y66GFTD ADR-0006 — Tape-based reverse-mode autograd: compact
- ADR-01KYJNCHQ2F9BAETNPA806TJ44 ADR-0004 — GELU uses the exact (erf) definition: compact
- ADR-01KYJNCHQ5EHPVKKDF1D3HG6K4 ADR-0011 — NPU acceleration is a model-level, not op-level, target (§T44): compact
- ADR-01KYJNCHQ8ET2BBE1CH8ZRDMBV ADR-0028 — one shared NEON transcendental kernel (the "vexp leaf") for every f32 activation: compact
- ADR-01KYJNCHQCET9BDVAK7DQK3DM6 ADR-0024 — Cross-Layer Attention (CLA): isolated KV-sharing variant: compact
- ADR-01KYJNCHQFFSS96NSTTRMCQSSN ADR-0005 — Optimized `cpu` backend; SIMD-intrinsics split (T11 / T11b): compact
- ADR-01KYJNCHQJFMQVH8ZYSXTESEGJ ADR-0007 — CrossEntropy as a fused op (log-softmax + NLL): compact
- ADR-01KYJNCHQNE2HBZJTM51Y0TTH5 ADR-0016 — Quantized matmul as an optional backend capability: compact
- ADR-01KYJNCHQRE4VBXCAWR2BNTEQ5 ADR-0022 — second-order / create-graph autograd, and how much of it Titans/TTT actually need: compact
- ADR-01KYJNCHQVE5VRBSRRMQC9RA96 ADR-0015 — Typed backend names replace magic-string identifiers: compact
- ADR-01KYJNCHQYET3SW5JDKD954AFA ADR-0020: PagedAttention is out of scope (revisit trigger defined): compact
- ADR-01KYJNCHR1FEJTDV5R3S7JVK1V ADR-0002 — Allocator abstraction; alignment is advisory in L0: compact
- ADR-01KYJNCHR5FC2T80MQ51675VD8 ADR-0021 — f32-native accumulation for the amd64 SIMD GEMM fast path: compact
- ADR-01KYJNCHR8FTBBYBKM9A0SRXAB ADR-0025 — Byte Latent Transformer (BLT): the "needs ragged tensors" deferral is invalid: compact
- ADR-01KYJNCHRCEMV9E3J9PM6XJ07H ADR-0018 — Zero-copy UMA for GPU ops (page-aligned storage + bytesNoCopy): compact
- ADR-01KYJNCHRFE2JTY0KWPPTV9ZQF ADR-0003 — Opcode dispatch, single Execute choke-point, Recorder hook: compact
- ADR-01KYJNCHRJFJC8K5QNK8HGVAV1 ADR-0023 — LLaDA-style masked-diffusion language model: design + GO verdict: compact
- ADR-01KYJNCHRNFG3R3B4KH97A5HDV ADR-0008 — GPU strategy: offload only measured winners: compact
- ADR-01KYJNCHRRFAK9ZCNWZ24NA0DQ ADR-0029 — the build-tagged test corpora have a verification boundary, and it is where the toolchain ends: compact
- ADR-01KYJNCHRWFC3T5CPQQHZYM55Y ADR-0026 — f32-native NEON GEMM for arm64 under GOEXPERIMENT=simd: compact
- ADR-01KYJNCHRZENKT4FN5YAHA7ZWT ADR-0001 — Type-erased tensor storage with a runtime Dtype: compact
- ADR-01KYJNCHS2FANVXZ8SNZMKNBTV ADR-0021: SIMD + GPU combination — configurable split, overlap, heterogeneous: compact
- ADR-01KYJNCHS6E4BVSYZNTR561CJP ADR-0013 — Tag-free Metal & zero-config accel registration (§T47): compact
- ADR-01KYJNCHSAF7SAYXG9F0FDCCT4 ADR-0017 — Pure-Go GEMM cache blocking parked (no delta on arm64): compact
- ADR-01KYJNCHSDFS49SE3948KP7CTE ADR-0019 — Device-resident tensors + command-buffer batching: compact
- ADR-01KYJNCHSHEVBB7C1E864G2NDE ADR-0014 — Typed per-op parameters replace the `map[string]any` attrs bag: compact
- ADR-01KYJNCHSMFT5AM1RMHCDGHFAJ ADR-0027 — Apple AMX F32 GEMM: Accelerate-cgo vs raw-AMX-asm, benchmark and use the winner: compact
- ADR-01KYJNCHSQFVSRMZFSCVC82WZW ADR-0012 — Automatic backend selection by performance preference (§T46): compact
- ADR-01KYJNCHSVE2RB0EGKW50071HA ADR-0009 — CUDA/cuBLAS backend (§T42): compact
- ADR-01KYJNCHSYF369FXZDS7C16MBQ ADR-0010 — Portable Vulkan compute backend (§T43): compact
- T-01KYJX48GFEHZBYPPGBQA9Z4WT tensor gatherCast: hoist innermost axis out of the strided-gather odometer: ALREADY DONE — no work performed this round, and that is the finding. gatherCast in tensor/tensor.go was strip-mined by commit 29eef4b8 (perf(tensor): hoist innermost axis out of the strided-gather odometer — 3.1x). The current source carries the hoist and its bit-identity rationale verbatim: inner, sInner := shape[nd-1], strides[nd-1], a straight strided run, and the odometer ticking over axes 0..nd-2. The task was complete but never archived, so it sat in the backlog as live work. Archiving it prevents a future session from re-deriving a shipped 3.1x optimization.\n\nTHE REMAINING tensor/tensor.go PS4005 SITE IS A DIFFERENT FUNCTION: gatherGeneric (around line 219), not gatherCast. Reachability was probed as this task mandated, by panicking inside it: NONE of BenchmarkCastF16toF32, CastF32toF16, CastBF16toF32, CastF32toF64, ContiguousOffsetView or ContiguousInnerRows enters it. Reading the dispatch shows why — it is the default arm for STRIDED CROSS-DTYPE HALF CASTS (the widen-through-f64 walk), so every same-dtype and every contiguous case is shadowed by gatherCast, gatherRows or gatherBlocked2D. It is reachable in principle (a sliced f16 view cast to f32) but no benchmark in the repo enters it, so its leverage is entirely unestablished. Carried forward as its own task with that evidence attached.\n\nMETHOD NOTE: the panic-probe step this task specified is what caught both facts cheaply. Reading the timing of a benchmark that never enters the function under test is the failure it guards against, and it would have produced a confident, meaningless number here.
- T-01KYMH46N6EWQ8XACQC0KGT839 Assess tensor gatherGeneric: reachable only via strided cross-dtype half casts, leverage unestablished: ODOMETER HALF LANDED AT 1.24x; ACCESSOR HALF DELIBERATELY NOT ATTEMPTED. Measured M2 Pro darwin/arm64 go1.26.5, 4 reps of -benchtime 300ms, medians: BenchmarkStridedCastF16toF32 419,213 -> 337,936 ns (1.24x), BenchmarkStridedCastBF16toF64 433,287 -> 352,577 ns (1.23x), with BenchmarkContiguousInnerRows as an unaffected control at +2%.\n\nA FIRST RUN WAS DISCARDED RATHER THAN REPORTED: its third rep had the control drifting +27% with both candidates rising alongside, which is exactly the discard condition this task specified. The reported figures come from a clean re-run. Recording this because the contaminated rep would have shown a WORSE result for the change and could equally have been mistaken for a regression.\n\nREACHABILITY WAS PROVEN BEFORE ANY TIMING, and it was decisive: none of the six pre-existing candidate benchmarks enters gatherGeneric. Two new benchmarks build the shape that does — a sliced f16/bf16 [B,H,T,D] view materialized as f32/f64 — and were re-probed by panic to confirm entry. Timing any pre-existing benchmark would have produced a confident number about a function it never calls.\n\nCOVERAGE WAS PROBED, NOT ASSUMED: unlike autograd broadcastVJP (found completely ungated), this path IS gated — one existing test catches a broken strided walk. The added gate is stronger where it counts, byte-comparing against a frozen per-element oracle across the six S/D dtype pairs that reach here, INCLUDING the half dtypes: this path widens through float64 and f16/bf16 round-trips are lossy, so a change to WHEN the widening happens would surface only there. Both mutations turn it red.\n\nNOT DONE, and it is probably the larger win: the per-element atF64/setF64 accessor dispatch, the PS1001 half. The modest 1.24x from removing the odometer alone is itself evidence that the accessor dispatch dominates what remains. It was kept separate because a typed switch creates a dual-path function (PS6001 applies) and must reproduce the widen-through-float64 semantics exactly, including the lossy half round-trips — a tolerance-0 cross-reference across every claimed dtype pair, with each arm mutation-probed, since a case named for an arm is not proof it reaches that arm. The oracle and benchmarks added here are the harness that work would need.\n\nPS4005 tree-wide 3 -> 2; both remaining sites (autograd/vjp_reduce.go fallback, backend/ref/reduce.go) are outside tensor/.
- T-01KYJNDS5MFEDAR0GVMSNG4MHD Clear the remaining documentation debt to a green apicheck gate: Done: go test ./internal/apicheck exits 0 on fdedd4a4, checked unpiped. The last 7 gaps were NOT missing documentation. Six exported symbols (classic DBSCAN.Fit and GradientBoostingRegressor.Predict, format/gguf QMatMul, linalg LU.Solve, nn KimiDeltaAttention, tensor NewOn) had ORPHANED doc comments: a doc binds to the declaration that immediately follows it, and performance work had landed const [body truncated at tombstone retention cap]
- T-01KYJNDSP0FG4B672MFS69AQ0F Enable apicheck and mdlint in the CI always-run set: Done: internal/mdlint added to the default alwaysRun set in internal/cichange/config.go, alongside internal/perfscan and internal/apicheck, so the always-run selection is now complete. Both blockers named in the brief are gone. apicheck went green in PR #705 (six orphaned doc comments reattached plus one missing Example) and was already listed. mdlint was green without any markdown cleanup: skipDi [body truncated at tombstone retention cap]
- R-01KYZG5MGHE6M8SZN9R3DWZEGS Round T1040: WKV exp elision shipped, ref MoE exp shipped, F32 gradient bug found, three checks: Consumed: two optimizations shipped with measurements (nn.WKV -10.4 percent, ref MoE-balance -34 percent), one correctness defect fixed with a regression floor (F32 MHAMaskedBackward returned all-zero gradients), two perfscan checks added (PS3018 rebinding fix, 24 to 44 sites; PS3027 input-view-on-output-tensor, before/after validated), and two candidates measured and reverted. The unshipped candi [body truncated at tombstone retention cap]
- R-01KYZGZ670F3TSR0067WGB0N15 Round T1041: Cholesky VJP relayout -26%, pooled attention scratch -53% bytes, PS3028: Consumed: both candidates shipped with measurements (Cholesky VJP -26.2 percent at n=256 and -22.0 percent at n=512; masked-attention scratch -53.2 percent bytes at unchanged ns/op), the class became perfscan PS3028, and the validation-method defect became rule BEFORE-AFTER-MUST-NOT-REVERT-THE-TOOL-001. Four candidates remain open and are named in the body, the gguf ones blocked on missing instrum [body truncated at tombstone retention cap]
- R-01KYZJED1BEV69HHF7G1NGYVFA Round T1043: distill per-worker scratch -16%, gguf ReadFile buffering -91.7%, PS3029: Consumed: distill helper shipped at -16.6 percent with allocations down, gguf ReadFile buffering shipped at -91.7 percent on a header-heavy load, and the class became PS3029. The durable part is the benchmark: BenchmarkReadFileSynth builds its own model in a temp directory and has two shapes because the load has two independently sized halves, which unblocks the three remaining gguf candidates tha [body truncated at tombstone retention cap]
- R-01KYZKSNV3E47BC2WXTHBWCRXB Round T1045: Quantize aliasing -88% bytes, SolveSPD interchange -35.6%: Consumed: both shipped with measurements (Quantize -87.7 percent bytes and -9.1 percent time, SolveSPD -35.6 percent), each with a bit-identity or invariant gate written for it — the SolveSPD kernel had no test at all before this. The class detection already existed as PS6011; what was added is the interchange remedy and its price, so a reader can tell a two-line fix from a relayout. Four candid [body truncated at tombstone retention cap]
- R-01KZ121PQ2FEMSCG5Y3KT3RW5V Round T1055: tensor axis coalescing -8 to -12 percent, plus three fresh sweeps recorded: Consumed: axis coalescing shipped at -7.5 to -11.9 percent across four cells with an oracle-based gate, and the allocate-before-knowing lesson became a rule after it flipped one cell to +18.4 percent. The three fresh sweeps (tensor, classic, rl plus llamagpu) are summarized in the body with their top candidates and the measured impossibility of inlining AtF64, so later rounds start from them.
- T-01M088VKVJEP8AJ47MWWS5E8CG Port and validate the M2 Metal encoder profiler: Archived after successful implementation, physical M2 evidence, full local verification, and durable perfscan backpropagation.
- P-01M088T3ZBF0BTTY4MGS94NY5A Retain current-main M2 Metal encoder profiling and reject stale fusion promotions: Archived after successful implementation, physical M2 evidence, full local verification, and durable perfscan backpropagation.
- T-01M08AHZK0E7J81XBB4G7Q9M3X Expose and gate Metal recorder profiling support: Added RecorderProfilingAvailable, gated the runnable profiler example and tests on stage-boundary timestamp support, retained explicit constructor failure, and documented the capability split. Physical M2 profiler and llamagpu suites, the executable example, apicheck, and make preflight pass. This fixes the macOS CI runner where Metal exists without encoder-stage counter sampling.
- P-01M08AFT7YF5Z87D0V45TMS7X2 Make Metal profiler capability discovery explicit: Capability discovery is now explicit and verified without changing the M2 profiling fast path. The contract is API-METAL-PROFILING-CAPABILITY-001; PR #1086 carries the fix and will merge only after all CI lanes pass.
- T-01M08DK6KDEH5BV1PZXBNBWQEF Decode one safetensors tensor directly from a transient mapping: Delivered d353cf5d: LoadTensor validates one entry and reads common little-endian dtypes directly into independently owned final storage; all widening dtypes share the full-loader decoder. Ten alternating benchmark pairs improved 614,468 to 280,561 ns/op (2.19x), reduced heap 66.67 percent, and led safetensors 0.8.0 by 1.48x. The proposed transient mmap was implemented, measured 31.18 percent slow [body truncated at tombstone retention cap]
- P-01M08DGE6RE54VZ0B01V2G2XHC Eliminate safetensors single-tensor framing and duplicate copies: Proposal completed through archived child T-01M08DK6KDEH5 and implementation d353cf5d. Synthetic single-tensor framing and duplicate copies were eliminated. Direct ReadAt won and shipped; transient mmap was rejected at 31.18 percent slower and preserved in R-01M08ENS24EPV. The current M2 leadership result is 1.48x over safetensors 0.8.0, with perfscan issue 755 carrying the general detector.
- R-01M08N4Z5QEJFBZXSAH6F5591E Current M2 attribution consumes the historical 7.1x llama.cpp gap: Committed current-main attribution under internal/benchcompare/leadership/evidence/m2-llamacpp-attribution-20260817. Five alternating pairs put llama.cpp b10450 only 1.0434x ahead at matched f32-KV tg64 and 1.1193x at pp64; shipping f16-KV/FA-auto leads 1.1428x and 1.1678x. Exact GoAI and sampled llama.cpp profiles keep K-quant as the largest family, but the 4.34 percent matched residual does not [body truncated at tombstone retention cap]
- R-01M01Q0AG0EHNAC09QEG6YDF6N Diagnosed: quant kernels are per-weight-bound, not bandwidth-bound — achieved GB/s rises with block size: Consumed and corrected by R-01M08N4Z5QEJF and its pinned 2026-08-17 evidence bundle. The 7.1x throughput/16 GB/s result accurately described its historical revision but not current main after cooperative/vectorized K-quant, f16 weight-cache, attention, and recorder work. Current matched f32-KV decode trails llama.cpp b10450 by 1.0434x, so the historical diagnosis must not select another leaf rewri [body truncated at tombstone retention cap]
- T-01M08GMDSYE2D8Z938E57WAY58 Capture matched GoAI and llama.cpp M2 Metal attribution: Delivered 09c1639b. The analyzer safely launches an absolute JSON argv target without a shell, joins exact process/GPU command buffers to counters and shader intervals, supports an explicit trailing-buffer skip, records contamination, and fails closed under strict exclusivity. Five alternating strict f32-KV pairs put llama.cpp b10450 1.0434x ahead at tg64 and 1.1193x at pp64; shipping f16-KV/FA-au [body truncated at tombstone retention cap]
- P-01M08GG8GSFKTB25CEGB556A45 Attribute the current M2 K-quant decode gap against llama.cpp b10450: Completed through archived T-01M08GMDSYE2D and implementation 09c1639b. The pinned current-main matrix replaces the stale July 20x claim with a 1.0434x matched decode and 1.1193x matched pp64 deficit, preserves shipping f16-KV/FA-auto as a separate capability comparison, and records exact per-engine K-quant attribution plus trace coverage and contamination. The 4.34 percent matched residual does n [body truncated at tombstone retention cap]
- R-01M08QB38EE21TGAQ7PV7RM848 M2 split-K lane-octet leaf win reverses in the resident TinyLlama decoder: Consumed by rejected proposal P-01M08PT2KHE6M and task T-01M08PV940FFF. The measured leaf-to-resident reversal is preserved in their tombstones and https://github.com/jxsl13/perfscan/issues/757; no code change survived.
- R-01M08S3QNGFE3S1Q5T864VFTTB M2 half-resident cached-f16 FFN is exact but only 1.0348x at the full leaf seam: Consumed by the rejection of T-01M08R8SGSETD and P-01M08R4T1WEX9. Exact finite parity and a live selector were proven, but the full TinyLlama M64 FFN improved only 1.0348x against a 1.10x leaf gate. Executable changes were fully reverted; the general component-removal ceiling check is tracked in perfscan#758.
- R-01M08WMQ5MFDY8V704PG7Z81D4 Attribute the M2 pp64 command graph before another kernel rewrite: Consumed by proposal P-01M08WKNGEF49. The audit proves pp64 already uses one command buffer, encoder packing has less than one-percent ceiling, and about 131 cached-f16 quant projections each retain a cast-matmul-cast boundary. It redirects the experiment to a cached MPSGraph boundary fusion and preserves the 85584691 M2 baseline of 173.7 tg64 and 1526.0 pp64 tok/s.
- P-01M093C3MFF0E8E1THCWYA1757 Close the M2 shipping gap with an opt-in f16 KV cache: Promoted the opt-in Metal NewQuantF16KV path at 5ade743b: exact 2-byte retained K/V elements, paired f32-to-f16 append, and dk=64 f16-reading cooperative/split-K attention with f32 accumulation. Exact rounding/storage/path gates pass; trained TinyLlama preserves 76/76 greedy tokens with logit NRMSE 0.000170697. Three post-commit campaigns measure ctx512 1.0310-1.0358x, ctx1024 1.0503-1.0634x, and [body truncated at tombstone retention cap]
- T-01M09A4TMCF2E9GDEG967PVV8E Implement and gate bounded-reassociation mixed Metal QKV: Implemented heterogeneous Q4_K/Q4_K/Q6_K grouped f16 expansion, one MPS GEMM, cache-budget accounting, M<24 fallback, and exact RoPEPairSplit epilogue. Exact expansion and bounded projection parity pass. Ten-sample leaf speedups are 1.7378x at M64 and 1.2198x at M512. Three repeated trained TinyLlama invocations preserve 76/76 greedy tokens and exact checked logits; pp64 is 1.0408-1.0572x, pp512 1 [body truncated at tombstone retention cap]
- P-01M09A28JWF5ZBR8ADMM2P13MN Fuse mixed-quant M2 QKV with bounded MPS reassociation: Promoted the corrected bounded-reassociation mixed-QKV design. The frozen predecessor was rejected on an invalid bit-exact vendor-GEMM gate; ADR-01M09A3S9JFMH separates exact expansion storage from bounded numerical and trained-model semantics. The final path uses one budgeted grouped f16 expansion and one MPS GEMM for M>=24, retains raw quant kernels below M24, and fuses RoPE with q/k/v scatter. [body truncated at tombstone retention cap]
- T-01M09E0P15ERA9K4BPBM2SE39D Implement and gate M2 resident TopKN sampling: Implemented Metal DeviceBuffer.TopKN with a coherent-UMA bounded native heap, exact deterministic top-k tests through n=128000 and k=256, public Generate fast/fallback token parity, and O(k) Go result materialization. The 32k/K56 leaf is 62.669-66.691 us with 448 B and 2 allocations. Four trained TinyLlama ten-pair campaigns preserve exact 70-token sequences and deliver 1.06301x, 1.06615x, 1.06778 [body truncated at tombstone retention cap]
- P-01M09DYYWEEW9T8ERKNT02ZCNC Keep M2 Top-K generation logits device-resident: Promoted the M2 resident Top-K generation boundary in commit 47fe6387. Exact top-k and full Generate parity pass; four trained-model campaigns deliver 1.06301x-1.06778x sampled-generation speedups while forward-only readback remains only 1.00170x-1.00482x. The architecture and benchmark matrix now separate llama-bench forward-only semantics from real sampled generation. Evidence is retained under [body truncated at tombstone retention cap]
- T-01KYJNDR38E4ZSN52KVM0PC5J9 Extend the trained-optimizer comparison with APOLLO and Q-GaLore: compact
- T-01KYJNDT44EKJAXN8W0Y4QFZCE Batch the ViT encoder instead of looping over the batch dimension: compact
- ADR-01KZ3HW0ZSFE7T23XD65GPBRE6 Lower classic treeRadixCutoff from 512 to 32? It is worth 17.3 percent on the forest fit and changes which trees are grown.: compact
- ADR-01M09A3S9JFMHAYHX35BG0HRNY How should numerical equivalence be gated when a vendor GEMM changes reduction scheduling after column fusion?: compact
- ADR-01M09J0A8XFY1RH7QMCJVKVJTG What is the valid Metal residual-fusion boundary for quantized decode projections?: The decision correctly constrained the experiment to the final f32 residual addition and required bit identity. The resulting prototype preserved exactness but failed the frozen stable end-to-end leverage gates, so the decision is archived with the rejected proposal; evidence is m2-metal-quant-acc-20260818 and perfscan #769.
- P-01M0FNKC7DEJZBE5XQQ7MRF6VA Route host-resident Metal embedding backward deterministically: Retained M2 winner: deterministic host scatter replaces the synchronous upload/atomic/download route for host-resident OpEmbedBackward. All 15 frozen campaign medians cleared the 1.20x gate (3.931x to 30.762x), semantics are exact and deterministic, full local and first-pass PR #1105 CI are green, and perfscan#771 preserves the reusable detector insight.
- P-01M0FVFC9BFDCVSSYEVNKT0F6H Revalidate host-resident Metal bias-gradient reduction routing: The Metal wrapper still uploads, submits, synchronizes, and downloads each F32 bias gradient even though the CPU backend later gained an exact parallel column reduction. Freeze a same-binary direct-Metal control and production-selector candidate across representative M2 shapes. Route only a measured winner zone after three independent count-7 campaigns each clear 1.10x median speedup with candidat [body truncated at tombstone retention cap]
- T-01M0FX2EBSFC2RGQ42W0KHWCEY Benchmark and gate Metal bias-add routing: Isolate direct Metal bias add as the control and optimized CPU dispatch as the candidate without changing production routing first. Measure representative M2 Pro row and width shapes with ten untimed warmups, fixed-count count-7 samples, and three independent campaigns. Route only a contiguous winner zone whose every campaign clears 1.10x; otherwise preserve direct Metal. Add mutation-resistant tw [body truncated at tombstone retention cap]
- P-01M0FX1QZ9F088KW6S3EVH0VGR Revalidate host-resident Metal bias-add routing: The synchronous Metal OpAddBias path still uploads host-resident tensors, dispatches one memory-bound kernel, waits, and downloads the output even though ADR-0008 routes equivalent binary elementwise work through the optimized CPU backend. Establish a mutation-resistant same-binary control/candidate benchmark across representative row and width shapes on Apple M2 Pro, require three independent cou [body truncated at tombstone retention cap]
- P-01M0FYJQFBF59ARSF0MP0QXZ50 Revalidate M2 Metal activation routing after NEON kernels: Revalidated the historical T535 negative after typed arm64 NEON kernels changed the CPU candidate class. Accepted ADR-01M0FYKCJMFRE and MEASURED-METAL-SIMD-ACTIVATION-ROUTE-001: optimized CPU wins only on measured darwin/arm64 SIMD contiguous F32 shapes through 4,194,304 elements; direct Metal remains elsewhere. Production-selector campaigns and full GPT training proved positive leverage, complete [body truncated at tombstone retention cap]
- P-01M0G5H18YENC9Y7EVVM63MVNH Accelerate F32 ReLU with semantics-exact arm64 NEON: Shipped and validated exact arm64 F32 ReLU acceleration. The ordered compare/select leaf is bit-exact, improves complete CPU ReLU by 2.892x-6.197x, and supports a measured default/SIMD M2 host-route ceiling of 4,194,304 elements. Every retained selector and alternating wide-MLP campaign median passes its gate; all 15 first-pass CI checks are green. Direct Metal remains the fallback outside the mea [body truncated at tombstone retention cap]
- T-01M0G9BREWER3S6WHYQ5H229HS Implement and gate exact arm64 F32 Abs: Implemented exact arm64 F32 Abs with a 16-lane NEON kernel, payload-preserving signaling-NaN quieting, and a 1<<18 parallel threshold. Rejected vector FABS because it did not quiet sNaNs and rejected a broad quiet-bit mask because it destroyed payload bits. Three paired M2 Pro campaigns improved complete-operation throughput 1.176x-2.608x across 2K-8M elements; default/SIMD Metal routing at 4M-16M [body truncated at tombstone retention cap]
- T-01M0GABTVEF6YRS1H0VFWDC61P Fuse EAGLE Smooth-L1 after the Abs Amdahl gate: Implemented exact EAGLE Smooth-L1 forward and backward core fusion with native-backend capability gating so CUDA, Vulkan, and Metal retain the composite path unless they implement the op. Forward and VJP match the original composite graph bit-for-bit, including special values and operation-order rounding; a naive closed-form VJP was rejected after 1-ULP drift. Three M2 Pro campaigns improved exact [body truncated at tombstone retention cap]
- ADR-01M0G9E2XSE1E8K3GJ6GEM650F Which arm64 vector formulation should replace the scalar F32 widen-Abs-narrow loop without changing NaN payload semantics?: Adopted an exact arm64 Abs fast path that preserves incumbent finite, zero, infinity, quiet-NaN, signaling-NaN, and payload semantics. The implementation uses integer-domain NEON classification and quieting instead of FABS, plus a measured 1<<18 parallel threshold and an Abs-specific Metal host-routing ceiling. The decision is enforced by ARM64-EXACT-ABS-001 and MEASURED-METAL-UNARY-FALLBACK-002; [body truncated at tombstone retention cap]
- ADR-01M0GAC04JF6ARA3SE738R8MGY How should the Abs tranche clear the EAGLE end-to-end leverage gate after the exact leaf hits an Amdahl limit?: Adopted public unscaled OpSmoothL1Core and OpSmoothL1CoreBackward names to avoid claiming canonical SmoothL1 scaling semantics. EAGLE selects the fusion only when the active backend implements it natively, preventing implicit CPU fallback on accelerator backends. The exact VJP deliberately reproduces composite fan-out, rounding barriers, and accumulation order after a simpler closed form failed ra [body truncated at tombstone retention cap]
- P-01M0G9AWDGEK5TE6WASXSQS6YQ Accelerate exact F32 Abs on arm64 and remeasure M2 routing: Delivered and validated the M2-first exact arm64 Abs and EAGLE Smooth-L1 tranche. Exact F32 Abs gained 1.176x-2.608x complete-operation speedups across 2K-8M elements, while measured default/SIMD Metal routing gained at least 2.800x at 4M-16M. Because the standalone EAGLE Abs leaf remained Amdahl-limited near 1.02x-1.03x, the tranche added exact native-capability-gated Smooth-L1 core fusion, gaini [body truncated at tombstone retention cap]
- P-01M0GFFBN1F1N8ABTY2ECZ1X7S Accelerate exact F32 Neg on arm64 and remeasure M2 routing: Completed and CI-validated the exact arm64 F32 Neg acceleration and wider M2 selector winner zone. Durable rules preserve raw-bit semantics and a per-cell 1.10x campaign gate. The evidence matrix records 24 complete-CPU, 18 direct-route, and 54 production-selector passing cells, plus the focal-loss Amdahl non-claim. Generalizable findings are reported in perfscan issue 780.
- T-01M0GK57K7FF39GCDNBH43PJCJ Implement and gate fused Sigmoid Focal core and VJP: Implemented capability-gated OpSigmoidFocalCore and OpSigmoidFocalCoreBackward on CPU/reference, a detached-target VJP in autograd/vjp_sigmoid_focal.go (correcting the frozen task target typo autograd/vjp_focal.go), and an explicit mean with the established composite fallback for unsupported backends, mixed dtypes, and strided inputs. Three independent default M2 Pro count-7 campaigns passed all 1 [body truncated at tombstone retention cap]
- P-01M0GK31T4FKNRRJD8FVG6RD6E Fuse Sigmoid Focal Loss core on CPU: Completed through archived task T-01M0GK57K7FF3 and PR #1115. GoAI now capability-routes one fused sigmoid-focal elementwise core plus explicit mean and one fused logits VJP on CPU/reference, while unsupported backends, mixed dtypes, and strided inputs retain the prior composite graph. On Apple M2 Pro, every default promotion cell passed across three count-7 campaigns (1.501x to 2.553x), final rea [body truncated at tombstone retention cap]
- T-01M0GQ0JS4E2C9S3RGFJ3NRK1E Implement and gate aligned vector loads in Metal Q5_K cooperative kernel: Completed in PR #1116. Implemented aligned per-lane uint2 Q5_K qs/qh loads plus one lane-0 uint4 header load broadcast across the SIMD group. Production routing is limited to M==1 and K*N>=6291456, with historical cooperative fallback elsewhere. Three independent count=7 AB/BA campaigns produced minimum eligible-shape median speedups of mid_up 1.102x, mid_down 1.216x, gate_up 1.243x, and down 1.39 [body truncated at tombstone retention cap]
- P-01M0GPYWZ2ENZV68FJXGX2GMX5 Vector-load the M2 Metal Q5_K decode leaf: Completed in PR #1116. Implemented aligned per-lane uint2 Q5_K qs/qh loads plus one lane-0 uint4 header load broadcast across the SIMD group. Production routing is limited to M==1 and K*N>=6291456, with historical cooperative fallback elsewhere. Three independent count=7 AB/BA campaigns produced minimum eligible-shape median speedups of mid_up 1.102x, mid_down 1.216x, gate_up 1.243x, and down 1.39 [body truncated at tombstone retention cap]
- T-01M0HWBG9QEC2B5JQXEBJT2EZ9 Fuse the F64 NEON SSM recurrence on Apple arm64: validated pass by codex no attributed diff (5ad38304b535 binds the target list, not code) — Archived after PR #1128 first complete CI matrix passed 15/15 at head aab63f2dfdcfdd6aecea908d4e694d05fce06e4d. Product commit 4afa035577fbef28b64560cd5b6f0cff3b591eab and valid-domain benchmark harness 6ff0c993 produced three physical M2 Pro alternating campaigns: internal geomean -79.08%, -80.18%, -84.4 [body truncated at tombstone retention cap]
- T-01M0JBX2XYFNR95M4RC7KHTN6A Implement and benchmark closable mmap-backed raw GGUF loading: Implemented gguf.OpenRaw with explicit RawFileHandle.Close, retained read-only mappings on supported Unix systems, exact buffered fallback, capacity-clamped tensor views, single-release ownership shared across handle copies, and malformed-input cleanup. M2 TinyLlama n=10: raw open 78.824->8.860 ms (-88.76%, 8.90x); full encoded-byte consumer copy 113.81->72.72 ms (-36.11%, 1.57x); heap 652.25->15. [body truncated at tombstone retention cap]
- P-01M0JJ7DWEENN9THBDNDNYKEM1 Fuse ARM64 Q8_0 decode dot on M2: Merged by PR #1134 after all 15 CI checks passed. Main merge commit: 4aba06d12d0ba35df9cc5b8493076d618fbb2f17. Retained M2 gains: Q8_0 K4096 row dot 14.84x, QMatMul N64 4.09x, N4096 2.50x, and recurrent Mamba2 1.93x; p=0.000 n=10 with unchanged allocations. Q6_K negative control flat p=0.579. Numerical, integration, cross-build, race, spec, external perfscan, and preflight gates passed. Evidence l [body truncated at tombstone retention cap]
- T-01M0JNJ3WYE8J9Z1ND423RYHTG Implement and gate ARM64 Q5_K fused decode GEMV: validated pass by q5-post-implementation-validator diff f239106fc52d — Merged by PR #1135 at c856022516709dbdb2c09906d5784c3084833a77 after all 15 CI checks succeeded. Retained evidence: 7.24x leaf, 6.99x and 4.53x QMatMul, 2.93x recurrent decode, unchanged allocations, Q6_K flat control, and maximum relative error 9.36e-6. Generalizable selector-family finding recorded on perfscan issue #799 co [body truncated at tombstone retention cap]
- P-01M0JNF7VGE7J9PV7PQ8DN2CNN Fuse ARM64 Q5_K decode dot on M2: Q5_K ARM64 fused decode-dot shipped in PR #1135 and merge c856022516709dbdb2c09906d5784c3084833a77. The per-superblock boundary selected in ADR-01M0JNC6MSFD4 retained 2.93x end-to-end leverage with bounded numerical drift and unchanged allocations. A whole-row rewrite remains unnecessary until residual profiling justifies it; portable, M>1, Metal, and Vulkan behavior stayed unchanged.
- T-01M0JQHRA5EPEBAY8A6BPSSXV3 Implement and gate ARM64 Q3_K fused decode GEMV: validated pass by codex-root-validator diff ec0f075ffd44 :: No blocking findings after diff, numerical, benchmark, portability, and hosted-CI review. — Validated pass by codex-root-validator after the rendered current-diff pack and explicit disposition of all seven non-blocking scope/benchmark-index findings. Archived after GoAI PR #1136 completed the full 15-check CI matrix successfully at head [body truncated at tombstone retention cap]
- P-01M0JQGRZ9EANTP6DTZ44R8AX9 Fuse ARM64 Q3_K decode dot on M2: Single-task proposal completed by archived task T-01M0JQHRA5EPE and merged GoAI PR #1136 after a fully green 15-check CI matrix. The Apple ARM64 Q3_K M=1 selector now routes through a per-superblock NEON two-bit plus inverted-high-mask unpack, signed sub-block scaling, and activation dot; portable, non-ARM64, M>1, Metal, and Vulkan paths are unchanged. Physical M2 Pro retained n=10 evidence shows [body truncated at tombstone retention cap]
- T-01M0JSVYWCEYPTM9XBY8Y9M2QC Implement and gate ARM64 Q2_K fused decode GEMV: validated pass by codex-root-validator diff 95eaa96d808e :: All 12 rendered findings are accounted for. The unchanged recurrent benchmark preserves identical base/candidate harness semantics; documentation and evidence are explicitly required by the task body; all three benchmark identifiers are executable Go benchmarks; and the graph was fully reindexed with typed calls. — Merged by PR #1137 at [body truncated at tombstone retention cap]
- P-01M0JSKA87F83B9PXPGAKCBD3H Fuse ARM64 Q2_K decode dot on M2: Merged by PR #1137 at b2df5338db57b9bb30acc713546e868db32fc028. The final scalar K-quant M1 selector edge is closed on Apple ARM64 with measured end-to-end retention, strict cancellation-safe accuracy, portable fallbacks, and external perfscan learning.
- P-01M0KGTBCHF59VJBME52TDJ43A M2-first exact IQ1_S fused row dot and portable QMatMul: Proposal delivered through merged PR #1146. Direct-F32/F64 semantics were preserved; the native selector remains ARM64 contiguous F32 M1 only; every retained cell cleared the 2x gate while dequant and IQ2_S controls stayed neutral. Cross-library leadership was deliberately not claimed because the pinned llama.cpp ARM kernel consumes Q8_K activations.
- P-01M0KM3Z8YEMEA1BZ507FJAF62 M2-first exact IQ1_M fused row dot and portable QMatMul: Archived after task T-01M0KM5H42FBT completed and PR 1147 merged. IQ1_M now has exact portable F32/F64 QMatMul plus the M2 ARM64 whole-row leaf; the leadership evidence and rejected coefficient-table experiment are durable in the merged evidence bundle.
- T-01M0KPVJQVFCS9TKR58HY2TKW7 Implement and statistically gate complete Q1_0 support and M2 ARM64 fused row dot: Completed Q1_0 end-to-end support and retained the compact 2 KiB ARM64 sign-expansion leaf. Final evidence: 7.95x leaf, 7.79x N64, 5.26x N4096, all p=0.000; flat allocation counts; maximum relative error 1.138970942829309e-16. Rejected the 8 KiB uint32 mask table because twelve alternating pairs showed flat throughput while the byte table used 75% less L1 footprint. All correctness, race, cross-bu [body truncated at tombstone retention cap]
- P-01M0KPSF31FYVAEB7ZHPN5Y4AS M2-first complete Q1_0 GGUF support and fused row dot: Delivered complete ggml Q1_0 type-41 support and statistically retained the M2 ARM64 whole-row sign-XOR dot under task T-01M0KPVJQVFCS. Final candidate clears every 2x gate at 7.95x, 7.79x, and 5.26x with flat allocation counts and 1.138970942829309e-16 maximum relative error. The compact 2 KiB byte-table design replaced an equally fast 8 KiB mask table; perfscan issue 815 generalizes the result. [body truncated at tombstone retention cap]
- P-01M0KT0ECBEK5R2GZA7M6PJZ83 M2-first complete TQ1_0 GGUF support and fused row dot: Archived after task T-01M0KT35PPFRR completed and merged its five TQ1_0 contracts. Complete type-34 support and the M2 ARM64 direct-F32 leaf are validated by same-semantics benchmarks, neutral Q1_0 controls, exact pinned reference bytes, numerical gates, race and portable builds, repository and Metal preflights, external perfscan ratchet, and reproducible llama.cpp boundary evidence. The equal-sem [body truncated at tombstone retention cap]
- T-01M0KYDJXBEK088S0BM7PMNN28 Implement and gate complete TQ2_0 with M2 ARM64 fused dot: Shipped complete TQ2_0 semantics and direct-F32 ARM64 M1 acceleration with reproducible same-semantics gains of 6.54x, 6.41x, and 4.09x. Retained portable F32/F64 and M>1 fallbacks, exact raw-code-3 behavior, neutral controls, and a non-comparable llama.cpp Q8_K boundary study. Evidence lives in internal/benchcompare/leadership/evidence/m2-arm64-tq2-fused-dot-20260822; reusable analyzer guidance i [body truncated at tombstone retention cap]
- P-01M0KY9HK0FH89MPABMZ0MWX99 M2-first complete TQ2_0 GGUF support and fused row dot: Completed the M2-first TQ2_0 tranche. GoAI now supports exact type-35 encode/decode and portable F32/F64 QMatMul, with a zero-allocation direct-F32 ARM64 M1 kernel delivering 4.09x to 6.54x same-semantics speedups. All validation, control, preflight, Spectackle, and external perfscan gates passed. The pinned llama.cpp Q8_K activation path remains a boundary study and identifies multi-row reuse as [body truncated at tombstone retention cap]
- T-01M0M1625WFQGVRNDRZHCBR59E Correct and gate IQ1_S and IQ2_S GGUF type IDs: Shipped the strict pinned-enum repair: IQ1_S=19, IQ2_S=22, unsupported I8=24, with consistent public, eager, raw, and QMatMul dispatch. Preserved all decoder math and kernel performance; ten-pair controls and complete validation are committed in the evidence directory. No perfscan issue was filed because this is a correctness repair without a generalizable performance gain.
- P-01M0M5K15NFPX87P7G6KCGPSYN Complete GGUF Q4_1 and M2 fused decode: Archived after exact Q4_1 wire/API completion, 6.04x leaf and 5.69x to 4.18x QMatMul gains on M2 ARM64, honest pinned llama.cpp boundary evidence, green preflight/race/cross-build gates, and perfscan issue 819.
- T-01M0MCMHDAE2BBZV2WHG9FARE0 Implement and benchmark native Metal IQ4_NL: Implemented exact GGUF type-20 IQ4_NL scalar and two-SIMD-group Metal kernels, resident upload/recorder dispatch, full-decoder opt-in, and real GGUF Q4_1/IQ4_NL admission. Reference error max 9.961e-7; cooperative/scalar max 3.851e-6. Three fresh-process campaigns retained all samples: minimum median 4.250x GPU and 1.114x resident Metal/ARM64 wall; six-layer decode 49.93 to 212.82 tok/s (4.262x) w [body truncated at tombstone retention cap]
- P-01M0MCJGS5F459EZNA0QYG4H1K Add native M2 Metal IQ4_NL quantized decode: IQ4_NL native Metal admission is complete and benchmark-validated. The proposal established a measured routing split: ARM64 for isolated host-resident calls and compressed Metal residency for graph-amortized decode. All exactness, loader, full-decoder, three-campaign, preflight, perfscan, and zero-drift gates passed; implementation commit abf438c2 contains the reproducible evidence.
- T-01M0MFBC0WFSXT8GYGN5CC5PAZ Implement and benchmark native Metal IQ4_XS: Archived after exact IQ4_XS Metal implementation, three campaign gate, strict ARM64 host-route retention, raw-GGUF admission, full-decoder parity, zero-drift Spectackle check, external perfscan v1.80.0 differential, and versioned evidence.
- P-01M0MF818FE9XBQTDC8B3P8RN8 Add native M2 Metal IQ4_XS quantized decode: Delivered native M2 Metal IQ4_XS decode with exact GGUF type-23 semantics, compressed recorder residency, raw-GGUF model admission, conservative multi-campaign leadership evidence, verified ARM64 generic fallback, whole-decoder token parity, external perfscan reporting, and a merged living contract.
- T-01M0MJ7CE3FFPRYDZ9R1G83K72 Implement and benchmark the native M2 Metal IQ3 family: Implemented exact IQ3_XXS and IQ3_S Metal scalar and two-SIMD-group cooperative kernels, one-time GGUF-oracle codebook reconstruction with persistent Metal buffers, direct and resident APIs, recorder routing, loader admission, independent toggles, and strict tests. Three fresh-process count-seven campaigns across both formats and four M1 decoder shapes produced conservative floors of 5.592x cooper [body truncated at tombstone retention cap]
- P-01M0MJ2RYQF4QR01T9CCYCBQSS Add native M2 Metal IQ3_S and IQ3_XXS decode: Closed the M2 Metal IQ3 family tranche. IQ3_XXS and IQ3_S now retain exact compressed GGUF bytes through loader admission, one-time exact codebook reconstruction, persistent Metal residency, shared scalar/cooperative kernel semantics, and whole-token recorder execution. Three strict campaigns established conservative floors of 5.592x at the GPU boundary, 4.232x at recorder wall time, and 2.060x ve [body truncated at tombstone retention cap]
- T-01M0MN13H5FABT5D8NY04RSQ51 Implement and benchmark native Metal IQ2_XXS and IQ2_XS: Completed implementation and evidence retained in the repository; archive the closed task before PR review.
- P-01M0MMVKG2FHST9M8AWFQJ213X Add native M2 Metal IQ2 family decode: Implemented, benchmarked, and validated with durable M2 evidence; archive the completed proposal before PR review.
- T-01M0MR12RWF4182FZMHZXDZN7D Implement and gate native M2 Metal IQ1 decode: Implemented exact IQ1_S and IQ1_M scalar and cooperative M2 Metal kernels, one exact 4 KiB packed process-lifetime ternary-grid buffer reconstructed through gguf.Dequantize, independent selectors, resident recorder routing, explicit recorder-only uploads, and Llama/Phi-3 GGUF admission. Cross-reference, immutability, validation, floating-point class, resident recorder, and identical-token whole-de [body truncated at tombstone retention cap]
- P-01M0MQYAGFF3PRC2P0632PCDDB Accelerate resident IQ1 decode on M2 Metal: Delivered the M2-first IQ1_S and IQ1_M resident Metal tranche with exact packed-grid semantics, independently gated cooperative kernels, explicit recorder-only integration, loader admission, pinned correctness and leadership evidence, and zero-new-finding external perfscan ratchets. The promoted resident path clears every frozen cell with at least 4.332x cooperative/scalar GPU, 3.085x recorder-wal [body truncated at tombstone retention cap]
- ADR-01M0MR09H9ERTTFKQGTM705Z2Z How should Metal IQ1_S and IQ1_M share lookup and dispatch infrastructure while retaining independent performance gates?: The selected shared-grid architecture is implemented and validated: IQ1_S and IQ1_M share one exact 4 KiB packed process-lifetime Metal buffer while retaining separate wire parsers, cooperative selectors, correctness controls, and promotion verdicts. This avoided duplicate 64 KiB expanded grids and preserved the negative direct-host fallback decision independently from the winning resident recorde [body truncated at tombstone retention cap]
- T-01M0MTWC7GF978HFX19KW2WE6N Implement and gate native M2 Metal IQ2_S decode: Implemented exact GGUF IQ2_S wire-type 22 scalar and two-SIMD-group Metal kernels with a one-time public-decoder oracle reconstruction and an immutable 2 KiB packed device grid. Wired direct, resident, recorder, Llama, and Phi-3 paths with an independent selector. Maximum observed IQ2_S relative difference was 9.489e-6 under the 1e-4 contract. Three fresh-process campaigns established floors of 4. [body truncated at tombstone retention cap]
- P-01M0MTT417E27SQK7K6X6EM9E6 Accelerate resident IQ2_S decode on M2 Metal: Shipped resident M2 Metal IQ2_S decode with exact semantics and complete evidence. The winning design extends the IQ2 family lifecycle with a type-specific 82-byte parser, independent scalar/cooperative selector, and one immutable 2 KiB packed 1024-by-8 grid reconstructed through gguf.Dequantize. Resident performance clears every declared cell with conservative floors of 4.278x over scalar Metal G [body truncated at tombstone retention cap]
- ADR-01M0MTV0RYEXWR38M3PX8WHZ5A How should Metal IQ2_S share lookup and dispatch infrastructure with the existing IQ2_XXS and IQ2_XS family?: Decision validated in production code: the shared IQ2 lifecycle now owns one exact packed IQ2_S grid and preserves a type-specific parser and selector. The 2 KiB representation is exact against the public GGUF decoder, all declared resident cells win, and existing IQ2_XXS/IQ2_XS controls remain independent. Separate lifecycle duplication and shader-literal expansion were not used.
- T-01M0MXCMPMF3JTXQBMCSA5V48Z Implement and gate native M2 Metal TQ1_0 and TQ2_0 decode: Implemented exact native Metal TQ1_0/TQ2_0 scalar controls and two-SIMD-group resident kernels, explicit recorder-only uploads, independent selectors, GGUF loader admission, and protected CPU host fallback. Three fresh-process count-seven campaigns established 11.99x cooperative/scalar GPU, 7.771x recorder-wall, and 2.056x resident-Metal/ARM64 conservative floors with 1.005x maximum unchanged-cont [body truncated at tombstone retention cap]
- P-01M0MXA8ZRF94TCEP455E4VT5G Accelerate resident TQ1_0 and TQ2_0 decode on M2 Metal: Delivered the M2-first resident TQ1_0/TQ2_0 Metal tranche through archived task T-01M0MXCMPMF3J. Exact wire semantics and independent format pipelines are proven. Conservative three-campaign floors are 11.99x GPU, 7.771x recorder wall, and 2.056x versus fused ARM64 CPU; identical-token whole-model gains are 30.288x and 11.261x. Mixed direct-host results remain on CPU by design. Evidence is hash-pi [body truncated at tombstone retention cap]
- T-01M0N0AVG4EHQS8RNM42QXVNVY Implement and gate native M2 Metal Q1_0 and MXFP4 decode: Implemented independent native Metal Q1_0 and MXFP4 scalar/cooperative kernels across direct, resident, and recorder paths; admitted both compressed GGUF formats into Llama-family loaders and Phi-3 row slicing. Three fresh-process count-seven campaigns produced campaign-cell median floors of Q1_0 2.643x GPU, 2.017x recorder wall, 1.574x versus fused ARM64 CPU, and MXFP4 6.415x, 5.921x, 1.209x. Dir [body truncated at tombstone retention cap]
- P-01M0N04WSRFV7A6MC3TV25CE7X Complete resident Q1_0 and MXFP4 decode on M2 Metal: Completed the final resident compact-quant boundary for GGUF type 41 Q1_0 and type 39 MXFP4 on M2 Metal. Independent compiled pipelines preserved per-format specialization; resident recorder execution won every promotion cell, generic host execution did not and remains on ARM64. The task, rules, ADR, benchmarks, correctness corpus, model admission, evidence bundle, and perfscan report are complete [body truncated at tombstone retention cap]
- R-01M0N49P0YFKSVHH1K1M8EASVV Attribute M2 quant resource fragmentation with a production-geometry arena probe: Consumed into P-01M0N48W6AF54 rejection. A native full TinyLlama-geometry Q4_K command held bytes, kernels, shapes, encoder count, and dispatch order constant while comparing independent MTLBuffers with one 256-byte-aligned arena. Three fresh-process count-seven campaigns measured separate/arena GPU-duration speedups of 1.029439x, 1.002404x, and 1.077140x. The inconsistent result failed the every- [body truncated at tombstone retention cap]
- T-01M0N6AX4VFGERG5BTYVHQAFF5 Implement and gate Metal RoPE-f16-KV append fusion: Implemented the separate-QKV Metal RoPE/f16-KV append boundary with a same-binary toggle. Exact Q/K float32 and cache binary16 parity passed at positions 0, 1, and 127 with nonfinite inputs and cache sentinels; a one-ULP mutation failed every oracle case and was reverted. A 22-boundary command-buffer benchmark cleared the leaf gate. The trained Q4_K_M profile revealed only 10 of 22 layers were eli [body truncated at tombstone retention cap]
- T-01M0N76XV8FB2TDN83KH0BHED6 Implement and gate grouped-QKV RoPE-f16-KV append: Implemented the grouped-QKV Metal RoPE/f16-KV append boundary and enabled the shared fusion selector by default. Exact complete-QKV float32 and cache binary16 parity passed at positions 0, 1, and 127; a one-ULP grouped mutation failed all cases and was reverted. The combined profile replaces 20 rope, 12 rope_pair, and 22 paired-copy events with 10 separate and 12 grouped fused events, a 32-event r [body truncated at tombstone retention cap]
- P-01M0N68Z0KFF79CDKT21KC4F26 Fuse M2 single-token RoPE with f16 KV append: Promoted as one half of the combined Metal RoPE/f16-KV append design. The original 22-layer eligibility assumption was corrected by trained profiling: Q4_K_M has 10 separate-QKV and 12 grouped-QKV layers. The separate path is bit-exact and removes 20 events but alone measured only 1.0081x to 1.0087x end to end, so its frozen 1.01x gate was not weakened. The grouped sibling covered the remaining to [body truncated at tombstone retention cap]
- P-01M0N73GJ6E7S8JKQRSY1RJBH2 Fuse grouped M2 QKV RoPE with f16 KV append: Promoted the combined separate and grouped QKV RoPE/f16-KV append route on M2. The production profile falls from 54 RoPE/copy events to 22 fused events. Exact float32 and binary16 state, nonfinite behavior, sentinels, and both mutation probes passed. Final 22-boundary speedups were 1.7396x, 1.7312x, and 1.7479x; trained TinyLlama decode was 1.0574x, 1.0163x, and 1.0173x across three count-seven ca [body truncated at tombstone retention cap]
- T-01M0N9C5ATEE18E7J23JVJKTSC Rerun M2 quantized decode and compare llama.cpp CPU: Archived after commit 7d793391: raw M2 evidence, the permanent hermetic-gated production harness, refreshed claims, and the rejected pool lesson are versioned.
- P-01M0N996E4ETXTWFTDCTF17CWW Refresh M2 CPU quantized decode leadership: Archived after the M2 CPU quantized-decode matrix was refreshed: Q8_0 now beats float 1.256x in the original whole-model cell, while the production llama.cpp comparison remains explicitly unmatched and the failed pool candidate remains removed.
- T-01M0ND3KXJFPZ91B0VM9BR6DK5 Attribute and eliminate dominant M2 CPU decode allocations: Archived after green full preflight, M2 Metal, race, cross-platform compile, exact-digest production A/B, and committed evidence.
- P-01M0ND1PT6EMKSAX9S889TRKQ5 Profile and reduce M2 CPU quantized decode allocations: Archived after implementation commit 9e58e031 and complete M2-first validation. Durable learning lives in CPU-SWIGLU-INPLACE-FUSION-001, CPU-SWIGLU-INPLACE-FALLBACK-001, benchmark documentation, evidence bundle, and perfscan #828.
- T-01M0NHAMKMEDNRFHEGP8MKT2T4 Implement and gate paired Q4_K M1 CPU projections: Archived after the paired Q4_K projection implementation, permanent leaf benchmark, production TinyLlama A/B, exactness tests, full validation, evidence bundle, and perfscan issue 830 were committed.
- P-01M0NH9R8RFQT9RXN9THYAJBBK Fuse CPU QuantSwiGLU gate and up QMatMul fan-out: Archived after the child task, two contracts, exactness boundary, benchmark evidence, full validation, and perfscan issue 830 were committed. The result is an internal M2 CPU improvement, not a matched cross-library leadership claim.
- T-01M0NNSG1BFVM9ZBCY1C056216 Implement and gate mixed-shape CPU QKV fan-out: Accepted after exact implementation and physical M2 Pro validation. QMatMulTriple proportionally partitions unequal Q4_K/Q4_K/Q6_K row sets across one fan-out; eager CPU QuantLlama Forward M1 and DecodeStep route through it while recorder, accelerator, batch, layout, and unsupported-quant paths retain three projections. Twelve alternating fresh-process TinyLlama Q4_K_M pairs all won: 1.889 s to 1. [body truncated at tombstone retention cap]
- P-01M0NNS1C0FMETR9BVNZ6R7DJ2 Coalesce mixed-shape CPU QKV decode fan-out: Accepted CPU scheduler-coalescing design. It is distinct from the rejected Metal raw-QKV kernel because all row arithmetic and device paths remain unchanged. Proportional heterogeneous work distribution was necessary: contiguous flattening reduced allocations but failed the production clock, while the balanced design delivered a statistically credible -5.36% 64-step M2 CPU decode gain and -21.27% [body truncated at tombstone retention cap]
- T-01M0NWM91HF62V50EQ8X30Z6Q6 Ignore the private research-source library: Added a root-anchored /.research-sources/ ignore rule with an explicit licensed-publication warning. Verified git check-ignore maps .research-sources/MANIFEST.md to .gitignore:32, git diff --check is clean, Spectackle reports no errors, and make preflight-full passes under Go 1.26.6. Preserved all 327 MB of local research files; none enter Git.
- P-01M0NWKBGDFCQ9BQ3EF0N18N8Z Keep private research sources out of Git: Implemented PRIVATE-RESEARCH-SOURCES-ISOLATION-002 through the validated root ignore rule. The private research library is no longer a Git tracking candidate, no files were deleted, and repository preflight-full remains green.
- T-01M0NY5H7EEDRSP7BHBHQSN0QQ Implement and gate fused Q4_K pair-to-SwiGLU chunks: Implemented QMatMulPairApply with one retained gate tensor, bounded pooled up scratch, aligned concurrent consumption, and an ARM64 dual-output Q4_K row dot. The allocation-only prototype reached about 1.08x leaf and was extended rather than retained alone. Final M2 Pro medians: 1.148x leaf, 1.059x 64-step decode, 16.85% fewer decode bytes, exact digest ea3df5516f17df83. Race and preflight passed; [body truncated at tombstone retention cap]
- P-01M0NY3846EDBT6GMMC5BCM0CE Fuse paired Q4_K production into CPU SwiGLU consumption: Accepted the fused Q4_K pair-to-SwiGLU design after it cleared the declared M2 gates: 1.148x leaf, 1.059x whole decode, lower bytes, exact output. Unsupported and recorded routes retain established fallbacks. Source commit ee55a6fb; evidence commit d01a52ca; reusable analysis recorded in perfscan #833.
- T-01M0P4XWWWEQ09N23GXSN59Q46 Make the CPU production gate backend- and artifact-pinned: Implemented by daa68e8b. TestProdCPUQuantDecodeGGUF now pins backend.Preference to CPU before GGUF model construction, restores the previous preference, hashes the external fixture outside the timed decode boundary, and reports backend, SHA-256, and Go runtime. The M2 validation produced backend=cpu, digest ea3df5516f17df83, model SHA-256 9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2 [body truncated at tombstone retention cap]
- P-01M0P4X4QCF0XTB87P5M888AAP Pin CPU quant benchmark routing and model identity: Shipped the focused benchmark-attribution correction in daa68e8b. The production CPU gate now prevents linked Metal registration from silently rerouting QuantLinear, restores the caller preference, and emits the concrete backend, external GGUF SHA-256, and Go runtime outside the timed decode boundary. M2 exact output returned to digest ea3df5516f17df83 with the pinned TinyLlama artifact hash 9fecc [body truncated at tombstone retention cap]
- R-01M0P5ZH2QFJ8AE9KVGEPJNCJJ Pin llama.cpp Q8_K activation semantics for GoAI CPU decode: Consumed by proposal P-01M0P5YEXPFMV. The pinned llama.cpp bb4caa754018 Q8_K layout, quantizer, Q4_K/Q6_K ARM dot semantics, and exact-versus-approximate policy boundary are now the implementation source for the primitive and decode tasks.
- R-01M0PCJ7TSEFCTCZ45YBJ6SFVX Profile merged M2 CPU quant decode after producer-consumer fusion: Consumed by task T-01M0PCYV52EF4. Current-main 15-repetition profile preserves digest ea3df5516f17df83 and ranks paired Q4_K at 24.70 CPU seconds, independent Q4_K at 15.70, and grouped QKV at 7.48; worker wait/signal and fan-out allocations are closed targets. A 1-12 thread sweep peaks at 29.542 tok/s on 12 threads, proving compute remains parallel and selecting paired coefficient preparation as [body truncated at tombstone retention cap]
- P-01M0PCH446FH695VH9692WFSV8 Eliminate the next M2 CPU quant-decode wall-clock bottleneck: Post-fusion M2 profiling identified paired Q4_K coefficient-header extraction as the next actionable compute bottleneck after controlled thread scaling rejected a serial-scheduler hypothesis. The consumed task delivered exact 1.065x paired-row and 1.050x FFN pair-apply gains without production regression. Persistent worker pools and unchanged Q8_K activation-leaf designs remain rejected. Evidence [body truncated at tombstone retention cap]
- R-01M0PEHGT7FY285ZWC1JQ05SWB Attribute merged-main M2 CPU quant decode after Q4_K header bulk decode: Merged commit 052a4821 profiled with Go 1.26.6, 12 threads, 64 steps, and 15 retained production repetitions. Median under CPU profiling was 2.9745 s with exact digest ea3df5516f17df83. Q4_K paired block consumed 16.40% flat; independent Q4_K row consumed 10.74% cumulative, including 1.39 CPU-seconds in getScaleMinK4; Q6_K consumed 4.46%. QMatMul fan-out remained 53.25% of allocated objects but is [body truncated at tombstone retention cap]
- P-01M0PEGTNPFYBR11H1K7RDM36Q Reprofile merged M2 CPU quant decode after paired-header acceleration: Fresh profiling on merged PR 1176 selected the independent Q4_K packed-header helper as the next bounded compute target while preserving the closed status of worker-pool, caller-participation, Q8_K activation, and residual-epilogue designs. The task delivered a bit-exact 1.081x K=2048 row gain, a supporting 1.043x mixed-QKV median, and no clean-window production regression. Host contention was mea [body truncated at tombstone retention cap]
- R-01M0PGJAMGEKCT53EG9E4QMM4S Attribute post-header Apple M2 Q4_K block costs: Consumed by T-01M0PHKGVZEWN. Production profiling attributed about 18 percent of paired-row CPU time above the NEON block leaf. Post-index pointer folding was neutral; Go-side multi-block staging regressed through frame growth and zeroing; moving the full row loop and exact coefficient construction into assembly converted that attribution into a 1.069x leaf and 1.102x TinyLlama FFN boundary gain.
- T-01M0PQYE4TFZ9S3SPMTJ3K8RW6 Shuffle paired Q4_K headers with table-indexed USHL: Replaced the paired Q4_K fixed shuffle network with VTBL plus per-lane USHL. Exact arbitrary-header tests pass. Retained M2 medians improved 1.035x at the K2048 leaf (7/7), 1.078x at TinyLlama FFN pair-apply (6/7), and 1.036x for 64-step production decode (6/7) with exact digests. Evidence is in m2-cpu-q4k-header-ushl-20260823; generalized detection opportunity is perfscan issue 842.
- R-01M0PQ6CJPFMMAP21B11H6BC9B Re-profile merged M2 Q4_K vector-header kernel: Merged-main profiling identified paired Q4_K header decode as the next controllable leaf hotspot. It produced one rejected LD2R experiment and the successful table-indexed USHL task T-01M0PQYE4TFZ9; measurements and contamination handling are versioned in the associated evidence bundle.
- R-01M0PWQJZWFF9AEG1KB39ZB15Y Re-profile merged Q4_K whole-row M2 decode: Profiled exact merged PR 1181 at merge 1e5dcfef on Apple M2 Pro over 15 measured 64-step CPU decodes with exact digest ea3df5516f17df83. CPU samples attributed 16.98% flat to dotQ4KPairRowNeon, 5.43% to dotQ4KRowNeon, and 4.64% cumulative to dotQ6_KRowASM (4.21% block assembly plus 0.43% Go orchestration). Paired Q4_K now costs approximately the optimized independent per-row rate and the bounded p [body truncated at tombstone retention cap]
- P-01M0PGHM4TE7YAXSJT0Q56SSZ2 Optimize the next Apple M2 Q4_K fused block hotspot: Completed the bounded Apple M2 Q4_K hotspot program through PR 1180 and PR 1181. Table-indexed USHL header decode improved paired K2048 rows 1.035x and production 1.036x; whole-row independent assembly then improved the K2048 leaf 1.155x and exact 64-step production 1.083x. Evidence lives under internal/benchcompare/leadership/evidence/m2-cpu-q4k-header-ushl-20260823 and m2-cpu-q4k-single-row-asm- [body truncated at tombstone retention cap]
- R-01M0Q2EGE0E60T5H3P35XSCX77 Assess raw Metal M1 gate and up fusion boundary: Consumed by active proposal P-01M0Q2CZXBEXK. The boundary is not covered by prior prefill-only f16 expansion work: it targets same-type raw Q4_K or Q6_K M1 gate|up concatenation across all 22 TinyLlama FFN layers. Proceed only through exact halves parity, profiler count reduction, seven alternating trained-model decode campaigns, and pp64/pp512 controls.
- R-01M0Q39HPCECA80SSE3NVC5TSK Map non-materializing Metal Q4_K gate/up SwiGLU fusion: Consumed by proposal P-01M0Q3A75AERN. Kernel mapping proves a non-materializing pair can retain the established total Q4_K cooperative SIMD work: two SIMD groups compute matching gate/up rows, exchange four reduced scalars, and write hidden-width SwiGLU directly. M>1 remains out of scope.
- T-01M0QG4H36FC7B2P48XC15SM5E Ship and gate f16-KV fused split-K decode: Shipped the f16-KV-only fused split-K candidate and removed the failed f32 lane. Numeric parity was bit-exact in all 12 GQA/MHA cells at sk=128,129,512,1024,1536,2048. Three independent same-command count-7 M2 campaigns cleared the >=1.20 gate in every cell: medians ranged 1.251x-1.456x. Three independent 32-token interleaved TinyLlama campaigns cleared >=1.01 at contexts 512 and 1536: paired medi [body truncated at tombstone retention cap]
- P-01M0QG3MFAEQ1BVB13SKB1CN5F Fuse the M2 f16-KV split-K decode path into one threadgroup: Implemented and validated one-threadgroup-per-head fusion for eligible M2 f16-KV dk=64 split-K decode. SIMD groups preserve incumbent per-chunk arithmetic, exchange 66-float partials in threadgroup memory, and SIMD group 0 merges in ascending order. This removes the global partial round trip and second encoder. Promoted only the f16-KV lane after three leaf and three TinyLlama campaigns passed; f3 [body truncated at tombstone retention cap]
- T-01M0QSQXQNFPGBSNYW904VETG2 Validate xctrace artifacts after recorder termination: Implemented artifact-first recovery for Xcode 26.6 time-limited status-54 recordings. Recovery requires exit code 54 plus three explicit xctrace completion markers, then the existing workload, TOC, counter-schema, command-buffer, and report validations. Unit tests cover success and fail-closed cases. A real forced-timeout llama.cpp campaign passed end to end; missing-marker and failed-export captu [body truncated at tombstone retention cap]
- P-01M0QSQ4YHEY7BYWNKV1YBGBYV Preserve validated Xcode timeout Metal captures: Delivered the validated Xcode timeout capture path and pinned M2 incumbent evidence. The change accepts no arbitrary target or recorder failure: only status 54 with explicit time-limit completion markers can proceed, and all five artifact/report gates must succeed. Repository-wide short tests, vet, Spectackle check, and lint pass with only pre-existing warnings.
- R-01M0QSCP1NEQ9TZMYJGNWXCCMR Refresh matched M2 llama.cpp Metal leadership cell: Consumed by proposal P-01M0QSQ4YHEY7 and the immutable v0.2.0 evidence bundle. Five AB/BA f16-KV pairs put llama.cpp median throughput 1.0527x ahead at tg64 and 1.1211x at pp64, with material host variance retained. GoAI Q4_K plus Q6_K consume 5.251 ms of a 7.752 ms f16-KV buffer; llama.cpp shader samples are 99.985% Q4_K/Q6_K. Next work must audit the exact pinned quant dispatch structurally rath [body truncated at tombstone retention cap]
- T-01M0QY5BRFFQMAJ0XHWHNP2JK4 Validate and ship concurrent Metal decode recording: Shipped and validated dependency-tracked concurrent Metal decode recording for eligible dense pre-norm quantized f16-KV Llama graphs. Exact 24-step bitwise parity and encoder lifecycle coverage passed; go test -short -timeout 30m ./... and go vet ./... passed. The M2 promotion campaign recorded 1.0350x median GPU speedup, 1.0300x median wall speedup, 1.0053x GPU-ratio spread, exact digest d706f8fb [body truncated at tombstone retention cap]
- P-01M0QY46T8ED9TMCHR3FAQPZS8 Ship dependency-tracked concurrent Metal decode recording: Promoted a decode-only MTLDispatchTypeConcurrent recorder with explicit buffer-scope producer-consumer barriers, barrier-free independent Q/K/V and gate/up dispatches, and hard encoder termination at blit, MPS, commit, finish, and free boundaries. Standalone CGo barriers were rejected after 1.0184x GPU speedup and 0.9302x wall throughput; piggybacking the barrier bit on the next existing native ca [body truncated at tombstone retention cap]
- R-01M0QWRMAQF5GT3M25J1QMSTPP Measure dependency-tracked whole-graph Metal concurrency on M2: Consumed by proposal P-01M0QY46T8ED9 and task T-01M0QY5BRFFQM. The pinned llama.cpp v0.2.0 audit identified dependency-tracked whole-graph Metal concurrency as the leverage point; GoAI implemented the narrower eligible decode graph with explicit barriers and retained established fallbacks. The promoted M2 campaign measured 1.0350x GPU and 1.0300x wall median speedup with exact logits. The rejected [body truncated at tombstone retention cap]
- T-01M0R08R1XFV0BYT5AGW3M8NPJ Implement and gate batch-aware independent SDPA: Implemented batch-aware AttnAttrs semantics, one-op unmasked MHA.ForwardBatched, a shape-keyed M2 MPSGraph batch-axis forward, isolated portable fallbacks, and batch-safe backward. Reference, CPU, Metal, full short, Vulkan-tagged, vet, Markdown, and direct external perfscan validation passed. Three count-seven M2 campaigns improved ViT forward throughput by 49.55%, 51.43%, and 60.75%, and train-st [body truncated at tombstone retention cap]
- P-01M0R089YBE008K6PJQ009SGJV Execute independent vision attention batches in one backend operation: Completed through T-01M0R08R1XFV0. The batch cardinality contract preserves zero/one behavior, removes the 3B Slice plus B OpMHA plus Concat unmasked core, and keeps masks and gradients isolated. All frozen correctness and performance gates passed; evidence is versioned under m2-metal-batched-attention-20260823.
- R-01M0R02V7FFWKAG3EKXCMGD9TP Measure one-graph batched M2 Metal attention for vision encoders: Consumed by P-01M0R089YBE00 and T-01M0R08R1XFV0. The disposable leaf probe was bit-identical and reached 3.4453x median wall speedup with 3.1841x minimum at B=8, then was removed after production implementation. Reported GPU command timing was treated only as supporting evidence because its 49.4942x median likely omitted scheduling or transfer work.
- T-01M0R2DG90FERAHM5ZTTY67SGF Implement and gate batch-axis Metal attention backward: Implemented a cached batch-axis MPSGraph manual VJP for Metal F32 attention backward when Batch>1 and Window==0, replacing B synchronous native submissions with 1 packed graph execution while preserving batch-one, sliding-window, and safe incumbent fallback routes. Reference parity passes for causal/noncausal MHA, GQA, MQA, and ViT B=8/S=65/D=128/H=4. Three alternating count-7 M2 campaigns deliver [body truncated at tombstone retention cap]
- P-01M0R2CV3WFZCAMFD24BJSQZDD Execute batched Metal attention backward as one cached MPSGraph: Completed the proposed M2 Metal batch-axis attention backward redesign. The implementation satisfies METAL-BATCHED-ATTENTION-BACKWARD-STRUCTURE-001 and M2-VIT-BATCHED-ATTENTION-BACKWARD-PERF-001, preserves established fallback semantics, and closes the per-sequence attention-backward dispatch gap. Measured ViT train-step medians improve 1.6903x-1.7131x across three frozen campaigns with a 1.5080x [body truncated at tombstone retention cap]
- R-01M0R29CN1FD79BAX15W8RMK81 Batch Metal attention backward into one native submission: Research was consumed by proposal P-01M0R2CV3WFZC, task T-01M0R2DG90FER, and the two durable Metal batch-axis backward contracts. The decisive finding was that MPSGraph automatic differentiation does not support this broadcast-K path on the installed M2 runtime; an explicit cached softmax VJP with repetition-axis reductions provides the compatible and faster route. The result and rejected alternat [body truncated at tombstone retention cap]
- T-01M0R4T3BYESY843XEF6FA60D2 Implement and gate fused pre-norm FFN training: Implemented and gated fused pre-norm FFN training: generic forward/backward ops, explicit VJP, F32/F64 reference oracle, exact incumbent composite fallback, cached two-submission Metal MPSGraph forward/backward, and single/batched ViT routing. M2 boundary median speedup was 2.88x-3.29x across three order-alternated campaigns; full B=8 depth=4 ViT train-step median speedup was 1.34x-1.37x and all 2 [body truncated at tombstone retention cap]
- P-01M0R4K081FDDTQMBYZWCRRZ55 Fuse pre-norm FFN training boundaries on Metal: Delivered the pre-norm FFN Metal training-boundary fusion with explicit semantics and fallback contracts. The production design uses separate cached forward and backward graphs because upstream gradients arrive after forward execution; epsilon remains a runtime scalar and caches are bounded by shape. Three independent campaigns validate 2.88x-3.29x median boundary speedup and 1.34x-1.37x median fu [body truncated at tombstone retention cap]
- R-01M0R43YYWE2TTJMX2PJ09GN5S Screen a cached Metal pre-norm FFN training graph on M2: Research was consumed by proposal P-01M0R4K081FDD and task T-01M0R4T3BYESY. Profiling identified the synchronized seven-op pre-norm FFN boundary as a higher-leverage M2 target than isolated scalar kernels. A single forward-plus-backward graph was rejected because dOut is unavailable during forward; two persistent shape-keyed graphs retain exact training semantics and still yield 2.88x-3.29x bounda [body truncated at tombstone retention cap]
- T-01M0R91C3KECY8XGNX30NBTX49 Implement and gate fused pre-norm attention training: validated pass by github-actions/pr-1194 diff 5b0938fc3015 — PR 1194 delivers cached one-submission Metal pre-norm attention forward and backward graphs with exact seven-gradient semantics, portable fallbacks, mutation protection, and full ViT parity. Three M2 Pro campaigns measured 2.968x-3.059x boundary and 2.062x-2.110x depth-4 ViT training-step speedups; the weakest of 21 aligned full pairs [body truncated at tombstone retention cap]
- P-01M0R8Z9RPFAFV5FYZ0VN01KNP Fuse pre-norm attention training boundaries on Metal: Delivered by archived task T-01M0R91C3KECY and PR 1194: generic pre-norm attention ops, explicit VJP, portable reference oracle, narrow NLP routing, cached Metal forward/backward graphs, full ViT integration, correctness gates, and durable M2 benchmarks. The shape-specialized fusion passed all gates with 2.968x-3.059x boundary and 2.062x-2.110x full depth-4 ViT training-step gains; all current-hea [body truncated at tombstone retention cap]
- R-01M0R8JTW6E1AVVRZ6YJSHFVME Screen a cached Metal pre-norm attention training graph on M2: Consumed by proposal P-01M0R8Z9RPFAF and task T-01M0R91C3KECY. Screening established pre-norm attention as the next compute-heavy M2 graph boundary after FFN fusion; implementation confirmed cached native Metal graphs and one submission per direction as the leverage path. Three campaigns validated 2.968x-3.059x boundary and 2.062x-2.110x depth-4 ViT training-step gains with a 1.793x weakest aligne [body truncated at tombstone retention cap]
- T-01M0RCS0J9FTDANR9DMPW7GA54 Implement and gate fused pre-norm transformer blocks: Implemented OpPreNormTransformerBlock and its explicit 13-gradient VJP across portable F32/F64 reference, narrow NLP routing, ViT integration, and bounded shape-keyed Metal MPSGraph forward/backward caches with two runtime epsilon feeds. Control retained the separately fused attention and FFN boundaries. On M2 Pro B8 S65 D128 H4 F512, three GOMAXPROCS=1 order-alternated count-seven campaigns measu [body truncated at tombstone retention cap]
- P-01M0RCQZESFTASGRCF6MVGEAE8 Fuse complete pre-norm transformer blocks on Metal: Promoted the complete pre-norm transformer-block boundary. The design composes the already-promoted attention and exact-GELU FFN graph builders, removes the host-visible intermediate and two synchronous submissions per training step, preserves the exact two-boundary fallback, and passed all frozen M2 gates. Durable contracts cover semantics, fallback, one cached submission per direction, numerical [body truncated at tombstone retention cap]
- R-01M0RCPZNJF75SZJ766JXXPX6H Screen a one-submission Metal pre-norm transformer block on M2: Research consumed by proposal P-01M0RCQZESFTA and task T-01M0RCS0J9FTD. The frozen research body accidentally said hidden 256; the authoritative performance rule M2-PRENORM-TRANSFORMER-BLOCK-PERF-001 and actual merged ViT fixture use F512. At B8 S65 D128 H4 F512, all three boundary medians exceeded 1.15x, all depth-4 ViT medians exceeded 1.08x, and all aligned pairs exceeded 1.03x. The complete gr [body truncated at tombstone retention cap]
- T-01M0RGNBBDEF7RQWNFM2MJ2HYY Implement and gate a depth-composed Metal transformer stack: Implemented a generic depth-aware complete pre-norm transformer-stack operation, exact complete-block-loop fallback, portable reference forward and VJP, bounded Metal MPSGraph forward/backward, ViT routing, and parity/mutation/benchmark coverage. At M2 Pro F32 B8 S65 D128 H4 F512 Depth4, three GOMAXPROCS=1 order-alternated count-7 campaigns measured 2.179x, 2.203x, and 1.435x stack-boundary median [body truncated at tombstone retention cap]
- P-01M0RGMQF1ENNBTT2DFFC3P0A2 Compose pre-norm transformer stacks in one Metal graph: Promoted the depth-2-through-8 Metal stack graph after all semantic, numeric, mutation, fallback, allocation, and M2 performance gates passed. The architecture keeps intermediate block activations device-local while preserving the existing complete-block helper as the exact portable fallback.
- R-01M0RGHTRTE3CSGT21HN59B9QY Measure a depth-composed Metal pre-norm transformer stack on M2: Consumed by proposal P-01M0RGMQF1ENN and task T-01M0RGNBBDEF7. Measurement proved that composing already-promoted compute-heavy block graphs removes enough synchronous submission and host-copy overhead to improve the full ViT training step reliably; raw evidence and protocol were committed.
- T-01M0RKZ1YSFHNVTJ65SHVC4Q0X Implement and gate the fused ViT classifier boundary: Implemented a differentiable LayerNorm-sequence-classifier operation with portable F32/F64 reference semantics, an exact five-input VJP, ViT fallback when either fused direction is unavailable, cached one-submission MPSGraph forward/backward for supported unmeasured shapes, and a measured zero-Metal-submission host route for Darwin ARM64 B8/S65/D128/C10. Implementation commit def1abfe1461e6d47830a [body truncated at tombstone retention cap]
- P-01M0RKX6GTF838PM6SNEJ33XVC Fuse the ViT classifier boundary on Metal: Delivered the M2-first ViT classifier-boundary redesign through archived task T-01M0RKZ1YSFHN. The final operation preserves portable composite fallback and exact autograd semantics, keeps a bounded cached MPSGraph path for supported unmeasured shapes, and selects the measured host route only for Darwin ARM64 B8/S65/D128/C10. Accepted evidence shows 18.4516x-42.2081x classifier-boundary medians, 1 [body truncated at tombstone retention cap]
- R-01M0RKKV5RF95S1CDC657TGYH0 Attribute the remaining post-stack ViT training cost on M2: Research was consumed by proposal P-01M0RKX6GTF83 and task T-01M0RKZ1YSFHN. Attribution found the post-stack tail dominated by final normalization, batch class-row gathering, and the classifier head. A one-submission MPSGraph fusion improved the incumbent but lost materially to an exact host-fused route on the small synchronous Apple M2 unified-memory boundary. The durable result is to exploit row [body truncated at tombstone retention cap]
- R-01M0RQ3BTBEDGTR3YTW6RJJRCM Attribute the remaining ViT loss-tail cost on M2: Research consumed by rejected proposal P-01M0RQAB74FFS and task T-01M0RQASR1E37. It established that B8/C10 CrossEntropy is roughly 120x faster on host at isolated-boundary scope, but only about 1.041x median at the full ViT step when routes alternate within one binary; the promotion gate therefore correctly rejected it. Initial separate-process 1.18x-1.25x screens were confounded by thermal/order [body truncated at tombstone retention cap]
- R-01M0RRGS0YEQ48CY5QY9NC7T9F Attribute the post-stack ViT input boundary on M2: Consumed by P-01M0RRS336FYB. Two order-reversed M2 campaigns measured full-step medians 13.888/13.222 ms versus packed-token lower bounds 10.092/10.113 ms (1.376x/1.307x ceilings); the exact input boundary cost 1.631/1.695 ms and about 7.397 MB plus 720 allocations. This evidence justifies a one-forward/one-backward graph without claiming the semantics-relaxed lower bound as a candidate.
- T-01M0RRT1J6FSHREQ1W64YPRSC9 Implement and gate the fused ViT patch sequence: Archived after exact correctness, make preflight, make preflight-metal, external perfscan, and final evidence passed. Durable contracts remain PATCH-EMBED-SEQUENCE-SEMANTICS-001, VIT-PATCH-EMBED-SEQUENCE-FALLBACK-001, METAL-PATCH-EMBED-SEQUENCE-GRAPH-STRUCTURE-001, METAL-PATCH-EMBED-SEQUENCE-NUMERIC-001, and M2-PATCH-EMBED-SEQUENCE-PERF-001.

## PROC-007
WHERE a performance transform is not bit-identical, the GoAI SHALL apply it only where the value is a continuous output, and never where it feeds round, quantize, argmax, or a threshold comparison.

Rationale: Bit-identity had been the de facto shipping gate, since every perfscan optimization so far was pure traversal reordering. PS5001 reciprocal-multiply is the first class whose win requires a half-ulp change, with no bit-identical variant available. The continuous-output boundary makes the trade reviewable rather than ad hoc. Adopted by user decision.

## PROC-008
WHERE a non-bit-identical transform is shipped, the GoAI SHALL replace the bit-identical assertion with a tolerance test whose bound is justified in the commit.

Rationale: A loosened assertion is only defensible if the loosening is explicit and its magnitude argued at the point of change; silently relaxing a bit-identity test hides the numerics decision from review. Companion to PROC-007.

## PROC-009
WHERE a kernel is about to be rewritten for performance, the GoAI SHALL first probe the existing suite with a deliberate one-ulp mutation, and if it survives, write the bit-exact oracle BEFORE applying the optimization.

Rationale: QR and SolveSPD both had no correctness coverage at the level their rewrites touched: deliberate index and one-ulp mutations passed the entire backend/ref and autograd suites. Writing the oracle first is what proves it encodes the OLD behavior rather than the new. Property checks are not sufficient substitutes - Q.R == A and QtQ == I tolerate exactly the drift being guarded against, which is why the suites missed the mutations.

## PROC-010
WHERE a mutation probe returns green, the GoAI SHALL explain why before concluding the test is weak, since the mutation may be algebraically equivalent or numerically absorbed.

Rationale: Two green results in the IPO work were correct greens: regrouping (pc-rc)-(pl-rl) is exact under bench.RandF64's [-1,1) range by Sterbenz, and a 1-ulp change in an O(1) term is absorbed by subtraction from a target of 5.0. Neither indicates a weak test. The probe must also match the defect class the change can introduce - a dispatch rewrite risks index and operand errors, not reassociation.

## PROC-011
WHERE a bit-exact oracle is written for a numerical kernel, the GoAI SHALL reproduce the kernel's own algorithm, not the mathematical definition of the quantity it computes.

Rationale: Twice caught in one session. An Nrm2 oracle written as sqrt(sum x^2) disagreed with the kernel in the last digit because nrm2 uses the LAPACK dnrm2 SCALED update, which is better conditioned. A flashattn oracle via OpMHA collapse disagreed in 27 of 48 elements because flash accumulates unnormalized and divides at the end. In both cases the kernel was right and the oracle was wrong; only the bit-exact comparison revealed it, and a tolerance test would have ratified the wrong oracle.

## PROC-012
WHERE a mutation probe survives and the code is about to be called unguarded, the GoAI SHALL first confirm the mutated line executes, by a temporary panic, since unreached code always survives mutation.

Rationale: Twice mistaken this session. mha.go:596 was reported BLIND when mhaBwdGemmBand is gated by f32NativeKernels and unreached in the default build, and the AddBias benchmarks measured a broadcast path that bcastBlockApply shadows. Unreached is not unguarded; conflating them manufactures findings and, in the benchmark case, manufactured a whole investigation.

## PROC-013
WHERE an edit is applied by string substitution, the GoAI SHALL assert the anchor matched and re-read the region before reporting the change, since a missed anchor is silent.

Rationale: Three times in one session a substitution silently no-oped and the result was reported as done. A perfscan patch produced an unchanged 30-to-30 finding count briefly read as a real measurement, and two MHA scope comments were described in commit messages while the file kept stale text that contradicted the code and repeated a corrected claim. A reader trusting the comment would have concluded guarded paths were unguarded.

## PROC-014
WHERE a cross-backend parity test is about to be written, the GoAI SHALL search for an existing TestXxxCrossReferenceExact covering that op first, and mutate to confirm the existing test does not already catch the defect.

Rationale: A Conv2D parity guard was added and then removed one iteration later: TestConvCrossReferenceExact already covered four shapes, both dtypes and bias, and caught the same im2col mutation. A redundant test is not free - it runs on every CI pass and reads as independent evidence when it is not.

## PROC-015
WHERE a mutation probe reports a kernel unguarded, the GoAI SHALL run the probe over every package that could contain a cross-reference test, not only the kernel's own package.

Rationale: The ULP audit ran go test on the kernel's package only and reported 11 of 12 backend/ref kernels blind. Re-run at full scope, flashattn, conv, crossentropy and cumsum are all GUARDED from backend/cpu. A probe proves nothing about tests it did not run, which is PROC-012's unreached-versus-unguarded error at package granularity.

## PROC-016
WHERE a perfscan finding names a loop, the GoAI SHALL benchmark the enclosing operation end to end, not the loop, since a loop that is a small share of its operation cannot move the total.

Rationale: PS4006 paid 2.15x on solvespd where the substitution IS the work, and 0.93x on classic cholSolve where the O(d cubed) factorization is a small part of an O(N times d squared) Gram-matrix build. Same pattern, opposite result, decided entirely by the ratio between the flagged loop and its operation. Five prior wins did not make a sixth.

## PROC-017
WHERE a compliance or coverage check reports no gaps, the GoAI SHALL verify the check's INPUT set before believing its empty output, since an empty diff from a wrong input is indistinguishable from compliance.

Rationale: An ARCH-012 check reported zero gaps while scraping only 48 of 91 declared ops from the wrong files. The empty diff looked exactly like compliance. Re-run against backend/op.go the real counts are 91 declared and 90 ref-registered, gap OpInvalid only. Same discipline as PROC-012 for mutation probes and PROC-013 for string edits, applied to the query itself.

## PROC-MUTATION-VALID-001
WHEN a probe reports the test suite still green, the a mutation probe SHALL reject the run as INVALID unless the mutation compiled and its anchor matched exactly once — a non-compiling or unmatched mutation yields no failures and reads as proof of no coverage.

## PROC-INTERLEAVE-001
WHEN a speedup below roughly 10 percent is claimed, the a benchmark A/B SHALL toggle the change in and out within ONE session, at least three alternations, AND discard the run unless each arm spread stays near 5 percent — alternation removes drift between runs, not contention during them.

## GIT-CALLS-MUST-SCRUB-GIT-ENV-001
IF a git command runs with cmd.Dir set and inherits the process environment, THEN the developer SHALL strip every GIT_* variable from it and add 1 floor proving a decoy repo named by GIT_DIR stays untouched.

Rationale: GIT_DIR overrides repository discovery entirely: with it set, git ignores cmd.Dir and operates on the repository the variable names. Git exports GIT_DIR for every hook it runs, and the pre-push hook runs make preflight, which runs the tests, so internal/cichange committed about 50 fixture commits onto the branch being pushed and replaced the index. The production gitRun had the same defect - dir is its whole contract. The suite had been green because harness and code were BOTH redirected to the same wrong repo and therefore agreed; fixing only one side is what made the failure visible.

## NEW-DECL-GOES-ABOVE-THE-DOC-BLOCK-001
IF an optimization adds a helper, constant or type beside a documented function, THEN the developer SHALL put it above that function s doc comment; a doc binds to the next declaration, and 6 exported symbols lost theirs.

Rationale: A doc comment binds to the declaration that immediately follows it. Optimizations landing a jam constant or a parallel helper between a comment and its function rebound the comment and left DBSCAN.Fit, GradientBoostingRegressor.Predict, QMatMul, LU.Solve, KimiDeltaAttention and tensor.NewOn undocumented; apicheck caught it only at push time, after the commits had merged.

## SEPARATE-STATEMENTS-DO-NOT-STOP-FMA-001
IF a float chain claiming bit-identity is split into statements to block contraction, THEN the developer SHALL wrap each step in an explicit conversion (T(x*s) when generic) — Go fuses across statements; this diverged on 66 of 256 logits.

## BEFORE-AFTER-MUST-NOT-REVERT-THE-TOOL-001
IF a check is validated against pre-fix code recovered with a stash, THEN the developer SHALL restore only the changed FILE from git — a stash reverts the uncommitted detector too and reports 0 findings, reading as a broken check.

## DEVIRTUALIZING-REMOVES-AN-FMA-BARRIER-001
IF a typed arm replaces a closure-based one and must stay bit-identical, THEN the developer SHALL wrap any product feeding an accumulation in an explicit conversion — the closure call was blocking a fusion the typed arm allows, and 1 ulp drifted.

## SIZE-THE-CELL-PAST-L1-BEFORE-JUDGING-LAYOUT-001
IF a layout or interchange transform is judged against existing benchmark cells, THEN the developer SHALL add a cell whose working set exceeds L1 first — the same QR transform read -35.0 percent at 128x64 and nothing at 32x16.

## INTERCHANGE-BEFORE-TRANSPOSE-001
IF a column-walked operand sits in a nest whose strided index is carried by an accumulating loop, THEN the developer SHALL move that loop outermost rather than transposing — one kernel measured -38 percent at zero memory against -12.6 percent at +38 percent bytes.

## A-SELF-DEFINED-COMPARISON-IS-A-PROOF-NOT-A-GATE-001
IF a parity test defines both implementations it compares, THEN the developer SHALL extract the shipped one into a named function the test CALLS — a self-defined pair proved the identity and stayed green when the real code was mutated.

## NEVER-ASK-TAKE-THE-MEASURED-WIN-001
The agent SHALL implement the measured optimization and document any contract change, rather than raising it as a decision — this was asked once about ReadRaw aliasing and answered as a standing instruction.

## WITHHOLD-A-CHECK-THAT-MISSES-ITS-OWN-CASE-001
IF a drafted check reports 0 findings on the site that motivated it, THEN the developer SHALL withhold it and record the diagnosis in an archived R-item — 3 predicates for the invariant-nest class each failed differently, and only the written-up failures stopped a 4th.

## A-UNIVERSAL-PATH-HELPER-MUST-NOT-ALLOCATE-WHEN-IT-CANNOT-HELP-001
IF a helper added to a path every call takes allocates before knowing it can help, THEN the developer SHALL scan first and return the inputs unchanged when it cannot — 2 unconditional slice allocations turned a -12 percent win into a +18.4 percent regression.

## HOIST-A-BRANCH-BY-DUPLICATING-NOT-BY-A-FUNCTION-VALUE-001
IF a loop-invariant branch is hoisted out of a per-element loop, THEN the developer SHALL duplicate the loop body per arm — a function value measured +51 and +66.5 percent, making a direct inlinable call indirect.

## A-DTYPE-SPECIALIZED-ARM-GUARDS-EVERY-TENSOR-IT-READS-001
IF a dtype-specialized arm is entered on a check covering fewer tensors than it reads, THEN the developer SHALL guard on all of them and send mixed sets to the reference kernel — a query-only guard crashed on an F32 query with an F64 mask.

## A-PARALLEL-GATE-MUST-CLEAR-THE-SERIAL-THRESHOLD-001
IF a test compares a parallel path against its serial one, THEN the developer SHALL size the fixture past the helper own work threshold — 3 sizes under n*feat=8192 compared the serial path with itself and a row-skipping mutation passed.

## CHECK-EVERY-SITE-NOT-THE-FIRST-ONE-FOUND-001
WHEN a check validates a property against a site inside the code it inspects, the agent SHALL evaluate EVERY such site, since one correctly formed site otherwise vouches for a defective sibling and the check silently loses coverage.

## NEVER-FILTER-TESTS-WITH-RUN-X-001
WHEN a profile is taken alongside a benchmark, the agent SHALL pass -run =^$ and never -run x, which is a REGEX matching every test whose name contains an x and silently fills the profile with test allocations.

## ACCUMULATORS-SHARING-ONE-PASS-ARE-NOT-SEPARABLE-COST-001
WHEN a profile attributes cost to individual accumulation lines inside one streaming loop, the agent SHALL treat the whole pass as the cost, not the lines; caching two of three accumulators in a memory-bound SVD sweep made it 30 to 50 percent SLOWER because the second pass doubled the traffic.

## ONE-THREAD-PER-OUTPUT-IS-AN-M1-OCCUPANCY-DEFECT-001
WHEN a GPU quantized-matmul kernel dispatches one thread per output element, the loop SHALL treat M=1 as an occupancy defect and give one simdgroup or workgroup per output row, splitting K so every lane stays inside one scale group; measured 1.80x to 6.01x across 14 kernels on two backends.

Rationale: At M=1 a one-thread-per-output dispatch leaves only N threads with work, each walking all of K. Fixed for all 7 quant formats on Metal (Q3_K 6.01x, Q4_K 3.41x, Q8_0 3.02x, Q6_K 2.69-11.79x, Q5_K 2.66x, Q4_0 2.48x, Q2_K 2.21x) and all 7 on Vulkan (Q5_K 3.04x, Q3_K 2.54x, Q6_K 2.20x, Q2_K 2.19x, Q4_K 2.17x, Q8_0 1.88x, Q4_0 1.80x). The gain shrinks as dequant gets cheaper, so simple formats are worth doing last.

## REDUNDANT-GPU-WORK-IS-CHEAPER-THAN-LOST-PARALLELISM-001
WHEN a GPU optimization proposes to eliminate redundant work by giving each thread more of it, the loop SHALL compare thread counts before and after and bound the lever by probe first, because two such attempts were measured 3.9 percent and 2.2-6.3x SLOWER while every winning kernel INCREASED thread count.

Rationale: Vulkan M=1 GEMV removed 15/16 of the tiled kernel's discarded arithmetic and lost 3.9 percent: 32768 threads became 2048, and the wasted threads were supplying the occupancy that hid memory latency. Metal M-blocked mat-mat removed 8x of weight traffic and lost 2.2-6.3x for the same reason. By contrast the 14 cooperative quant kernels that won all raised thread count 32x or 64x per output row. The two classes separate on thread count, not on how much redundant work was removed.

## HARNESS-CALL-PATTERN-IS-NOT-EVIDENCE-ABOUT-THE-SYSTEM-001
WHEN a fixed per-call cost measured in a benchmark is proposed as an optimization target, the loop SHALL find the production call site and count how many real operations amortize that cost before acting, because a harness submits per op for ISOLATION while production batched a whole decode step and paid the same 149us across roughly 0.5 percent of a token.

Rationale: A ~149us per-submit floor was measured and read as a batching opportunity. llamagpu/decoder.go records one recorder per decode step with the entire per-layer loop inside it, Vulkan's Commit() is a no-op and Wait() is Finish(), so production pays that cost about twice per token — 0.3ms against a 61ms step. The floor was a property of the measuring instrument, not of the system.
## SUPPRESSION-IS-EVIDENCE-READ-IT-BEFORE-TASKING-001
WHEN a detector finding is promoted into a task or ADR, the loop SHALL read the target line for an existing perfscan:ignore and weigh that justification before writing the item; this check has pre-empted 2 of 2 such items.

Rationale: T-01KYKSAF75FQGSFSQM9Z2RAJXQ named crossentropy math.Log, suppressed as one-per-row with the c-wide exp already vexp'd, and the code confirms 256 logs against 1048576 exps at 256x4096. ADR-01KYJYY74VE27BSEH9VGZSNFMK named the FA /l norm, suppressed as O(seq.dk) against an O(seq2.dk) body, and measurement gave 4.1 percent on one dtype of one kernel. Both were written from detector output without reading the site.
## FIRST-BENCHMARK-SAMPLE-IS-NOT-COMPARABLE-001
WHEN a Go benchmark result is compared against another variant, the loop SHALL discard the first sample of each -count run and compare medians of INTERLEAVED runs of both variants, never a sweep of one variant against a sample of another.

Rationale: Two false results in one session. (1) A 1.59x small-n win was a cold first sample (2.12 ms) read against a warm sample of another run (1.33 ms); interleaved and warmup-trimmed the two variants are 0.8 percent apart. First samples ran 20-35 percent high here, 1.55-1.75 ms against a 1.30-1.36 ms warm level. (2) A clean grain sweep showed 2.6 percent that vanished to 1.4 percent inside 57 percent spreads when interleaved, and the sweep and interleaved run disagreed on absolute level for IDENTICAL code (5.2 against 6.1 ms), proving the host drifted between them.

## GUARD-TOGGLE-ARMS-MUST-DIFFER-001
WHEN a guard compares two code paths selected by toggles, the loop SHALL assert the arms DIFFER at the shape where the recorded gap is widest, since an interception upstream of both toggles makes it a path against itself; TestQ4KMatrixUnitHasNoCrossover read 1.00x where its table records 0.36x.

Rationale: The f16 short-prompt path is checked first in the resident dispatch and, with the weight cache on, its M cap becomes 1<<20, so it served BOTH arms. The test failed only on thermal noise while reporting a crossover conclusion about a comparison that never ran. A majority-of-shapes vacuity rule did NOT catch it: one shape cleared 2 percent on noise alone, so anchor to the widest shape.

## FP-GOLDENS-ARE-PER-ARCH-001
WHEN a test freezes a golden digest of floating-point output, the loop SHALL key that golden on runtime.GOARCH via internal/archgold and never build the fixture from math.Sin, math.Cos or any transcendental.

Rationale: Two independent causes make one constant unportable: math.Sin/Cos differ by 1 ulp across GOARCH (41 of 2048 swept values; math.Cos(84) ends e523 on arm64, e522 on amd64), and arm64 fuses a*b+c where amd64 v1 does not. With exact dyadic fixtures the only shape still matching was the one where the unrolled loop never ran, proving contraction is a second cause.

## AMD64-FP-GOLDENS-COME-FROM-CI-001
WHEN an amd64 floating-point golden is recorded from an Apple-silicon host, the loop SHALL harvest the value from CI logs, never from a Rosetta run.

Rationale: Rosetta reproduces amd64 FAILURES but not amd64 FP RESULTS: TestMLAVJPIsBitIdentical gives 10503053519604685430 under Rosetta at both GOAMD64=v1 and v2 against 2081554234887433254 on real x86, while ubuntu and windows agree with each other. Rosetta appears to fuse SSE multiply-add onto ARM FMA.

## TIMING-ASSERTIONS-SKIP-ON-RUNNERS-001
WHEN a test asserts a wall-clock threshold, a throughput floor or an allocation budget, the loop SHALL skip it under testing.Short so it never runs on shared CI runners, and keep any correctness assertion in the same test unconditional.

Rationale: Runner hardware inverts orderings rather than merely adding noise: the Q4_K crossover reports mmunit ahead at M=32/48/64 on GitHub macOS, the opposite of every M2 reading, and BPE floors set at one third of an M2 Pro measured 1.3 MB/s, 36x under. Allocation budgets additionally count the race detector's shadow memory (16 MiB against a 12 MiB budget).

## GOLDEN-BYTES-NEED-GITATTRIBUTES-001
WHEN a test compares golden bytes read from a file, the loop SHALL mark the path -text in .gitattributes so git performs no EOL conversion.

Rationale: An unspecified text attribute lets git convert LF to CRLF on checkout wherever core.autocrlf is on, the default on GitHub Windows runners. TestMarshalIndexMatchesTransformersGolden failed there and nowhere else; the only CRLF in the comparison was the one git introduced, and the failure is invisible on Linux and macOS.

## WORKMD-MERGE-IS-NOT-A-UNION-001
WHEN a merge conflict touches .spectackle/work.md, the loop SHALL never resolve it by keeping both sides, because work.md entries encode CURRENT STATE and a deletion IS the state change.

Rationale: Consolidation branches resolved every .spectackle conflict as a union and resurrected a tombstoned task: T-01KYKSAF75FQGSFSQM9Z2RAJXQ was closed no-action with its reject event in the journal, yet spectackle get read it as draft again, so a later session would have re-picked refused work. Union is correct for journal.ndjson and for two independent record appends only.

## ITERATIVE-SOLVER-PERTURBATION-IS-NOT-AN-ERROR-BUDGET-001
WHEN a change would alter the values an iterative solver's search consumes, the loop SHALL gate it on the MEASURED iteration count for the target data, never on the change's error bound.

Rationale: Sweeping a relative perturbation through the SVC RBF kernel: one-ulp noise (1e-16) took the fit from 79 steps to 2025 and 7.24 to 66.60 ms, while 1e-14 also stalled and 1e-15, 1e-13, 1e-9 and 1e-7 all stayed at 79 steps. Damage is not monotonic in error, because SMO's trajectory turns on which pair each step selects. Test accuracy stayed 1.0000 throughout, so the symptom is time and not correctness, and a tolerance test cannot see it.

## NARROWING-A-KERNEL-INVALIDATES-ITS-CROSSOVERS-001
WHEN a kernel stops covering input sizes it once covered, the loop SHALL re-measure every dispatch threshold naming it as fallback, and assert cost below each boundary stays within 2x the cost above it.

Rationale: The M>=24 expand-then-GEMM gate was calibrated against the cooperative kernel at M=16 0.90x. When cooperative narrowed to M==1, batches of 2 to 23 fell to the scalar kernel: 23 tokens cost 257.0 ms against 47.1 ms for 24. Correctness suites cannot see it, since the slow kernel is correct.

## PERF-M2-METAL-PROFILER-002
WHEN encoder profiling is enabled, the GoAI Metal recorder SHALL emit stable labels and GPU intervals with exact output parity, explicit omission counts, at most 2048 event pairs, and disabled-path benchmark overhead within two percent.

Rationale: Opt-in timestamp attribution must expose incomplete traces and remain performance-neutral when disabled, or it can mis-rank kernel work and perturb the production path it is intended to diagnose.

## API-METAL-PROFILING-CAPABILITY-001
WHEN recorder profiling capability is queried, the GoAI Metal backend SHALL return true only when stage-boundary timestamp sampling and a timestamp counter set are available through RecorderProfilingAvailable.

Rationale: Metal availability does not imply counter-sampling support; callers need a stable preflight boundary before entering the optional diagnostic path.

## GGUF-READFILE-MMAP-LIFETIME-001
WHEN ReadFile maps a regular GGUF file, the format/gguf reader SHALL call munmapFile only after readParsed returns and call Read with a buffered reader when mmapFileReadOnly fails.

Rationale: The file mapping replaces a model-sized staging allocation without leaking mapped lifetimes through the eager File API.

## M2-INCUMBENT-ATTRIBUTION-HARNESS-001 {applies: go:benchcompare.BenchmarkProdMetalProfiledDecodeGGUF}
WHEN comparing production M2 K-quant decode with llama.cpp before another leaf-kernel successor, the GoAI benchmark suite SHALL record immutable revisions, the identical model hash, matched semantics, five alternating samples, and both engines per-kernel Metal GPU distributions.

Rationale: An isolated leaf can sit near the memory roofline while the heterogeneous full token remains slower; matched in-situ attribution prevents selecting a locally fast but globally irrelevant kernel rewrite.

## METAL-MIXED-QUANT-QKV-001
WHILE mixed Q4_K/Q6_K QKV segments share K and fit the f16 cache budget, the Metal decoder SHALL use one exact combined f16 expansion and MPS GEMM at M>=24, preserve raw quant kernels below M24, and fuse RoPE with q/k/v scatter.

Rationale: Ten TinyLlama mixed layers improve 1.7378x at M64 and 1.2198x at M512; fused scatter preserves the end-to-end gain.

## MEASURED-METAL-SIMD-ACTIVATION-ROUTE-001 {applies: backend/metal/metal.go,backend/metal/activation_route_arm64simd.go,backend/metal/activation_route_default.go}
WHEN contiguous offset-zero F32 GELU or SiLU forward or backward executes on a darwin/arm64 SIMD build, the Metal SHALL use optimized CPU through 4,194,304 elements, with direct Metal retained elsewhere.

Rationale: ADR-01M0FYKCJMFRE: all 84 production-selector medians cleared 1.10x across three isolated count-7 campaigns, and full SIMD GPT training improved 1.038x.

## M2-CPU-QUANT-WHOLEMODEL-001
WHEN a post-kernel Q8_0 whole-model result is published, the GoAI CPU decode benchmark SHALL run BenchmarkQuantLlamaGenerate500 and BenchmarkLlamaGenerate500RowBuf from the same Go 1.26.6 binary, discard warmup, retain 10 samples, and report medians plus allocations.

## M2-CPU-QUANT-INCUMBENT-001
WHEN a CPU quantized-decode comparison against llama.cpp is published, the GoAI leadership matrix SHALL use the identical GGUF, hardware, thread count, prompt and generation lengths, batch, and forward-only boundary, and record model hash plus exact commits.

## M2-CPU-QUANT-LOSS-001
IF the matched llama.cpp CPU cell leads GoAI beyond measurement spread, THEN the GoAI performance roadmap SHALL publish the loss and profile GoAI before booking an implementation lever.

## CPU-SWIGLU-INPLACE-FUSION-001
WHEN an eager quantized SwiGLU executes on its projection backend and that backend advertises in-place fusion, the GoAI SHALL overwrite only the private gate projection with bit-identical SiLU(gate) multiplied by up and allocate zero activation or multiplication output tensors.

Rationale: This removes two hidden-width buffers and dispatches without changing public ownership.

## CPU-SWIGLU-INPLACE-FALLBACK-001
WHEN quantized SwiGLU recording is active or its projection backend lacks in-place fusion, the GoAI SHALL execute backend.OpSiLU followed by backend.OpMul without mutating their inputs.

Rationale: Autograd interception and unsupported backends must retain the established composition.

## PRIVATE-RESEARCH-SOURCES-ISOLATION-002
WHEN Git discovers files inside the repository-root .research-sources directory, the repository ignore configuration SHALL exclude 1 root-anchored .research-sources directory from tracking candidates while preserving every local file.

Rationale: The directory contains local research material, including commercial publications that must not be redistributed.

## CPU-QUANT-BENCHMARK-PINNING-001
WHEN TestProdCPUQuantDecodeGGUF runs while accelerator backends are registered, the GoAI SHALL make TestProdCPUQuantDecodeGGUF set backend.Preference to CPU before model construction and restore the prior preference on exit.

Rationale: QuantLinear selects backend.Default, so Context.WithBackend alone cannot prove CPU attribution.

## CPU-QUANT-BENCHMARK-PINNING-002
WHEN TestProdCPUQuantDecodeGGUF accepts an external GGUF leadership fixture, the GoAI SHALL report SHA-256 9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0 or the actual fixture hash plus runtime.Version outside timed decode.

Rationale: A performance result without artifact and toolchain identity is not reproducible.
