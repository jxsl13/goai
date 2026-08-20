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
