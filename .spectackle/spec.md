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

## PERF-ACCUM-RESIDENCY-001
IF an optimization trades register accumulators for a memory accumulator vector to cut passes over a matrix, THEN the implementer SHALL reject it unless measured faster than the register-blocked form on a working set too large for cache (see R-01KYPCPWZ9EDY).

## NUM-FUSED-PATH-FMA-001
IF a fused loop claims bit-exactness against a multi-op dispatch path, THEN the implementer SHALL round EVERY product explicitly, including ones assigned to a named local, because the compiler still inlines and contracts those (see R-01KYQ9CQ3XE1D).

## PROC-BENCH-CACHE-THRESHOLD-001
IF a benchmark validates a memory-access optimization (interchange, blocking, relayout), THEN the implementer SHALL report at least two sizes, one whose working set exceeds cache: SymEig measured 1.05x at n=64 (32KB, resident) and 1.89x at n=128.

## PERF-FUSED-PATH-CHAIN-001
IF a fused path must reproduce the bits of more than about five chained backend ops, THEN the implementer SHALL scope it down or record an ADR before starting, since every op's rounding must be replicated (see R-01KYQ9CQ3XE1D).

## NUM-SYMMETRY-NOT-EXACT-001
IF an optimization reads a symmetric matrix transposed to make access contiguous, THEN the implementer SHALL verify the symmetry is EXACT at that point, since an in-place sweep can leave one-ulp asymmetry that accumulates (see R-01KYQBJ7BREF4).

## PERF-SCANRULE-EMPTY-001
IF a new perfscan rule finds no genuine site once its soundness tests are added, THEN the implementer SHALL delete it rather than ship it, and record the negative result so it is not rebuilt (see R-01KYQFAFQVEN3).

## PROC-GODOC-DETACH-001
IF a new declaration is inserted immediately above an exported function, THEN the implementer SHALL place it above that function's doc-comment block, since inserting between comment and func silently detaches the godoc — caught three times by apicheck in one campaign.

## PROC-GODOC-DETACH-002
IF a new declaration is inserted immediately above an exported function, THEN the implementer SHALL place it above that function's doc-comment block and re-run internal/apicheck, which caught this detachment 3 times in one campaign.

## PROC-TASK-HOTNESS-001
IF a performance task is booked against a specific code site, THEN the implementer SHALL first panic-probe that a benchmark reaches the site and name that benchmark in the task (see T-01KYQGW5MNFWP, a cold fallback that cost an iteration).

## PROC-MERGE-ORPHAN-001
WHEN a merge resolves a hot file by taking the other side's version wholesale, the implementer SHALL grep the discarded side's helper names for surviving call sites, because PR #573 left four K-quant kernels dead with every test still green.

## PERF-COMPILER-ALREADY-001
IF an optimization removes source-level branches from a hot loop, THEN the implementer SHALL check the compiler does not already emit a conditional select, since a branchless Q3_K rewrite measured exactly 1.00x (see R-01KYQM4YDNFZ2).

## PROC-ID-RENUMBER-001
WHEN a globally-unique ID (a perfscan PSxxxx, a task ID) collides while rebasing or merging, the resolving agent SHALL list the remaining commits of the branch being replayed and reuse the replacement ID a later commit already assigns, instead of picking the next number free in the current tree.

Rationale: During a 180-commit rebase, main had claimed PS6005 for a different rule. The first resolution invented PS6014 from the IDs visible in the tree at that point. Six commits later a branch commit titled renumber PS6005 to PS6010 arrived, with PATTERNS.md and docs already citing PS6010, and the whole renumber had to be undone and redone. The rest of the todo already held the answer.

## PROC-RESOURCE-CLAIM-001
WHEN a change is justified by a resource property (goroutine count, peak memory, allocation residency) rather than by throughput, the implementing agent SHALL measure that resource directly, in the specific scenario the justification names, and record both numbers before writing the claim into a comment or commit message.

Rationale: A bounded-pool change was justified by nested callers: a parallel mixer calling a parallel matmul. Sampling peak live goroutines gave 37 either way, because the nested path took a different branch that was already pooled, so the named scenario never reached the changed code. The change was still worth keeping (22 against 98-101 peak goroutines under concurrent callers, at neutral throughput) but for a different reason than the one nearly committed. Throughput parity is not evidence for a resource claim, and a resource measurement of the wrong scenario is not either.

## PERF-STRIDE-REVISIT-001
WHEN a strided inner-loop walk is proposed for interchange or blocking (a PS6011 or PS4006 candidate), the implementing agent SHALL first establish whether the outer loop re-reads the same strided addresses; when it does, the footprint is the line count rather than the buffer size and the expected gain is nothing.

Rationale: The linalg triangular back-substitutions stride 4KB at n=512 over a 2MB buffer and were predicted at 2-5x with the mechanism called unambiguous. Measured 1.005x LU, 1.001x Lstsq, 0.994x Inverse. For a fixed column the pass touches the same n addresses on each of its n outer iterations: 32KB of footprint, resident after the first pass. Every strided walk that did pay this session (NSA 2.40x, KDA 1.75x, Sinkhorn 2.80x) had the strided index on the reduction axis, touched once per output and never revisited. Same AST shape, opposite locality.

## PERF-REWRITE-CROSSCHECK-001
WHEN two independent rewrites of the same loop each claim bit-identity with the original, the implementing agent SHALL digest the output of both arms (bitwise sum plus xor of every element) and require them equal before keeping either, rather than trusting the two claims separately.

Rationale: Main and this branch both rewrote the RetentionChunkwise output accumulations, one by interchange and one by 4-way blocking, each documenting bit-identity with the original. Two such claims are jointly a claim that the rewrites match each other, which is testable where each is only prose. The digests agreed exactly (sum 40b2bae2cf78b812, xor bf5a8e6dc0e400bb), so both claims held and the faster arm could be kept on performance alone - blocked beat interchanged 1.62x chunkwise and 1.68x recurrent. Had they disagreed, one comment was wrong and nothing would have caught it: the package tests were tolerance-based.

## PROC-SCANRULE-REPLAY-001
WHEN a new perfscan detector is validated by replaying it against the pre-fix source it was built from, the implementing agent SHALL also write a synthetic positive test for the SHAPE, because replay passing proves only that the rule finds that one instance and can hide a gap that one variable name wide would expose.

Rationale: PS6014 passed its replay against the real pre-fix rl/rl.go while carrying a real gap: names reachable only through len or cap counted as possible writes, so a New(F64, Shape{len(states), k}) between the two calls suppressed the finding. The real source passed because it happened to size its tensor from a different slice than it fed the forward. The synthetic positive test, which reused the same slice name, is what exposed it. Replay is necessary and not sufficient.

## PROC-SCANRULE-VOCAB-WIDEN-001
WHEN a new config-gated perfscan rule ships with a vocabulary of one or two names, the implementing agent SHALL run it once against a deliberately over-broad vocabulary and curate every hit before trusting the narrow result, then restore the sound list.

Rationale: PS6014 shipped keyed on calleeName, which collapses a qualified call to its last segment, so b.Wq.Forward(ctx, xn) and b.Wk.Forward(ctx, xn) - the three projections at the top of every attention block - compared as identical. The rule looked clean with its one configured name because that name is a package-level function with no receiver, so no receiver was ever compared. Widening the list past what is sound produced dozens of hits in nlp, all the same false positive, and that is what exposed the bug. A vocabulary of one hides receiver-shaped bugs by construction. Curating the hits is also what establishes which names are genuinely sound to declare: Forward was rejected because a Sequential.Forward with Dropout consumes RNG and a quantized decode Forward mutates its KV cache.

## PROC-SUPPRESSION-FLOOR-001
The implementing agent SHALL pair every scan-rule suppression test with a floor test asserting the rule still FIRES on the nearest positive, because a suppression test that expects zero passes whenever the input fails to match for any unrelated reason.

Rationale: A PS6014 suppression test was written with a method name the configured vocabulary did not contain, so nothing in it was a pure call at all and the expected zero was reached without exercising the suppression. The companion floor test - same receiver twice must still fire - failed and exposed it. A zero-expecting test cannot distinguish correct suppression from total non-matching on its own.

## PROC-SCANRULE-WRONG-TOOL-001
WHEN a required perfscan rule needs type information (interface-ness, struct size, aliasing) that go/ast cannot supply, the implementing agent SHALL document the tool that does see it (the compiler escape analysis via go build -gcflags=<pkg>=-m, or go vet) with the exact invocation, and build only the portion a parser can prove.

Rationale: The attrs-boxing defect has two forms. The inline-literal form is provable from the AST and shipped as PS6016. The already-hoisted-but-still-boxed-at-the-call-site form requires knowing a parameter is an interface type, which needs go/types; perfscan is deliberately go/ast-only. An approximation would either miss it or fire on correct code, and that second form is the one an earlier pass produced and then missed because hoisting the struct looks like the fix. Escape analysis names every escaping literal and is how both forms were actually found, so PATTERNS.md carries the invocation. A rule that cannot separate the good case from the bad one has no signal; naming the right tool does.

## PROC-BENCH-COVERAGE-NULL-001
WHEN an A/B reports allocation or byte counts that are bit-identical across both arms, the implementing agent SHALL treat that as evidence the benchmark does not execute the changed code and confirm coverage before recording a null result, because a real null moves the count by at least the run-to-run jitter.

Rationale: Batch 2 of the nlp attrs hoist measured identical allocs on both arms of three benchmarks. That read as no effect and actually meant no coverage: BenchmarkCohereDecode builds the float Cohere while every changed file was a quantized path. A genuine null still jitters by a few allocations between runs; an exactly repeated count is the signature of code that never ran. A purpose-built QuantCohereDecodeStep benchmark then showed 1436 to 1392 allocs, deterministic across three rounds.

## PROC-BENCH-ONE-SAMPLE-001
WHEN a ratio is about to be reported from a benchmark whose single iteration takes tens of milliseconds or more, the implementing agent SHALL raise benchtime until the within-arm spread is smaller than the effect being claimed, and only then report the ratio.

Rationale: MixtralPromptPrefill appeared 1.73x slower after the attrs hoist at benchtime=1x, and the two rounds agreed closely enough to look real. At benchtime=10x the arms were 77.98ms and 78.91ms, no regression at all. One iteration of an 80ms benchmark is one sample, and two such samples per arm can agree by coincidence.

## PROC-SCANRULE-SILENT-REGRESSION-001
WHEN a scan rule depends on a helper that can return an empty or unknown value for an input it must compare, the implementing agent SHALL make the positive test the guard for that helper, because an empty value that silences the whole rule leaves every zero-expecting suppression test green while the check reports nothing at all.

Rationale: PS6017 compares parameter types as rendered text. exprText has no StarExpr case and returns empty for every pointer. Handled as a placeholder it collapsed all pointer types and paired concat1D with an unrelated function; handled as unrenderable-and-skipped it dropped every candidate and the rule found zero across the whole tree. A suppression test written against the pointer case stayed green under both, because it asserts zero and the rule produces zero for the wrong reason. Swapping the renderer turns only the POSITIVE test red. A silent check is the more dangerous failure: it reads as a clean codebase.

## PROC-BENCH-NULL-KINDS-001
WHEN an A/B arm reports no change in allocations, the implementing agent SHALL distinguish the two kinds before recording it: an EXACTLY repeated count across arms means the benchmark never ran the changed code, while an overlapping jittery range means the effect is real but below that benchmark noise floor.

Rationale: A batch of 238 exec1-to-exec2 swaps showed no change on QuantLlamaGenerate500 (283397/283413/283397 against 283399/283405/283403) and a clear -1.8% on QuantCohereDecodeStep. The llama null is genuine: quant_llama.go got two swaps, both in Forward, and Generate steps the prompt through DecodeStep instead. Two batches earlier an identical-looking null was a coverage gap, and the signature there was counts repeating EXACTLY across arms because the code never executed. Both render as zero in a summary and they call for opposite responses - write a covering benchmark, or accept the effect is negligible there.

## PROC-BENCH-BIMODAL-001
WHEN a benchmark on this host shows a multi-fold spread WITHIN a single arm across rounds, the implementing agent SHALL make no timing claim from it in either direction and say so explicitly, using it only for allocation and byte counts which stay deterministic.

Rationale: MixtralPromptStepwise ranged 166ms to 519ms within one arm on the same host while its allocation counts held to five parts in 127720. A min-of-N over a bimodal distribution reports whichever arm happened to catch the fast mode, which is how an earlier round produced an apparent 1.73x regression that did not exist. The allocation axis remained usable throughout.

## PROC-BUILD-THE-INSTRUMENT-001
WHEN a second consecutive optimization batch cannot be verified because no benchmark executes the changed paths, the implementing agent SHALL stop applying rewrites and build the covering benchmark matrix first, then measure the accumulated batch against it.

Rationale: Two iterations of the nlp allocation sweep produced edits that could only be verified on the one or two paths that happened to have benchmarks; eleven of twelve quantized architectures had none on either Forward or DecodeStep. Writing 24 table-driven benchmarks over the existing GGUF fixtures took less effort than the optimization work it had been blocking, and immediately turned 135 pending swaps into a measured result on every architecture. The warning had already been cast as PROC-BENCH-COVERAGE-NULL-001 an iteration earlier and was read as a fact about one benchmark rather than a systemic gap, which cost a round.

## PROC-BENCH-MATRIX-DIAGNOSTIC-001
WHEN a benchmark matrix covering sibling implementations of one interface exists, the implementing agent SHALL read it across rows as data, normalizing for fixture geometry first, and profile any outlier of 2x or more before looking for new optimization targets by reading code.

Rationale: The twelve-architecture nlp matrix was built purely as a verification guard. Read as data it showed Gemma2 decode at 4862 allocations against MPT 1086 on identical two-layer dim-32 fixtures - a 4.5x spread geometry cannot explain - which pointed a profiler at one function holding 78 percent of the step and yielded 1.21x with 27.6 percent fewer allocations. Normalizing first is what made it readable: without confirming the fixtures share geometry the spread looks like model size. Sibling implementations of one interface are directly comparable in a way single benchmarks are not, so the matrix is a cheaper source of targets than reading code.

## PERF-MINOFN-NOT-A-DEFENSE-001
WHEN a min-of-N ratio is computed from benchmark rounds whose within-arm spread has not been checked, the implementing agent SHALL check the spread first and raise benchtime until it is smaller than the claimed effect, because min-of-N selects on the very tail a bimodal distribution produces and can invert the sign of a real result.

Rationale: Three instances this session. Low benchtime manufactured a 1.73x regression that did not exist on MixtralPromptPrefill; inflated a real Gemma2 gain to 1.90x where 1.21x was true; and on DeepSeekV2 one baseline round at 257us against its own 305-312us cluster made min-of-N report a real 1.12x gain as a 15 percent regression. Min-of-N is commonly treated as noise-robust, and against a bimodal distribution it is the opposite: it picks whichever arm happened to catch the fast mode.

## PERF-CACHE-MUTABLE-KEY-001
WHEN a computation derived from a TRAINABLE parameter is memoized across calls, the implementing agent SHALL key the cache on an exact copy of that parameter compared element by element, not on a generation counter or a checksum, and gate it on the inference context so a taped call always recomputes.

Rationale: Swin headBias is a pure function of the relative-position Table, which an optimizer step mutates. A stale bias changes only the numbers, never shapes or control flow, so the failure is silent. A generation counter needs something to increment it and nothing in the optimizer path does; a checksum collision is a wrong answer. The exact comparison costs (2M-1)^2*heads float compares against a matmul over M^4*(2M-1)^2 multiply-adds avoided, so it is orders of magnitude cheaper than what it saves while being a proof. Verified by mutation: replacing it with a cache-once key makes the invalidation test fail with the stale-value message. The tape gate is separate and equally necessary - a cached tensor from an earlier step carries the wrong gradient edges.

## PERF-GATE-IMPLIES-BLIND-SPOT-001
WHEN a code path is found excluded from an optimization for a stated reason (a covariance shape, a dtype, a backend), the implementing agent SHALL check whether that same path is also missing benchmark coverage, and add it before measuring the fix.

Rationale: GMM PredictProba parallel row scan was gated to GMMDiag because the full-covariance kernel used receiver scratch. Full covariance also had NO PredictProba benchmark at all - so the shape that could not be parallelized was the shape nothing measured, and a 5.8x sat unnoticed. The two gaps share a cause: whoever excludes a path from an optimization tends to also not benchmark it. Full-cov was where the work was, its solve being O(d^2) per component against the diagonal O(d).

## PERF-RECEIVER-SCRATCH-BLOCKS-PARALLEL-001
WHEN a PS6006 receiver-scratch-buffer finding is triaged, the implementing agent SHALL search the file for a parallelism gate that names that buffer as its reason, because the real cost is usually the foreclosed parallelism elsewhere rather than cache contention at the site.

Rationale: GMM logGaussianFullBatch read four solve buffers off the receiver. The cost was not contention in that kernel: it was a gate in PredictProba, dozens of lines away, excluding full covariance from a row-parallel scan and citing exactly those buffers. Passing them in as a parameter yielded 3.1x to 5.8x. A per-call temporary on shared state does not merely slow a call down, it forecloses parallelism, and the symptom appears as a conditional somewhere else in the file.

## PROC-UNROLL-TAIL-COVERAGE-001
WHEN a kernel that is unroll-and-jammed by N is made concurrent, or its scratch is moved off the receiver, the implementing agent SHALL test at a trip count NOT divisible by N so the scalar remainder path executes, because the tail is a separate code path that shares none of the fixes applied to the wide one.

Rationale: Parallelizing GMM full-covariance PredictProba moved logGaussianFullBatch four solve buffers off the receiver, but that kernel jams by four and finishes the remainder by calling the scalar logGaussian, which still read its own buffer off the receiver. The parallel scan therefore raced for any component count not a multiple of four. Every benchmark and test in that change used k=8, so the tail never executed - and the race detector cannot flag a line that does not run, nor can a parity test compare values on a path that is empty. The bug was one modulo away from the tested case. A k=5,6,7 test now produces four DATA RACE reports against the old tail.

## PROC-PROFILE-PARALLEL-CONDVAR-001
WHEN a CPU profile of a program that uses a worker pool shows a large share of time in pthread_cond_wait, pthread_cond_signal, usleep or similar park/unpark runtime frames, the implementing agent SHALL treat that share as pool IDLING rather than overhead, and settle the parallelism question with a wall-clock experiment (run at GOMAXPROCS=1 against the default) before changing any threshold or chunking.

Rationale: A CPU profile of a GBM fit showed 74 percent in cond_wait, cond_signal and usleep against 18 percent in the split search, which reads as textbook over-parallelization. A per-chunk work gate built on that reading measured 1.88x SLOWER on GBMFit and 1.20x slower on GBMHist_exact_80k - the parallelism was paying and the gate switched it off. A CPU profile sums time across all threads, so workers idling on a condition variable accumulate large flat percentages while consuming no wall-clock progress. A profile localizes work; only wall-clock arbitrates a parallelism decision, and GOMAXPROCS=1 against the default answers it in one command before any code is written.

## PROC-NO-RULE-FROM-UNVALIDATED-001
WHEN a candidate perfscan rule would encode a pattern whose harmfulness has not yet been measured, the implementing agent SHALL measure the fix on a real instance first and only write the rule if the measurement confirms the harm, because a scan rule asserts that a shape IS a defect and an unvalidated one institutionalizes a wrong belief across the whole codebase.

Rationale: A threshold that checks total work rather than per-chunk work looked like a defect in GBM feature fan-out and is perfectly AST-detectable. Building the rule first would have been natural under the standing mandate to generalize findings. The measurement then showed the fix was 1.88x SLOWER - the shape is not a defect for that caller and the borrowed crossover was doing its job. Detectability is not evidence; every rule in this set that earned its place cites a measured win.

## PERF-REDUCTION-AXIS-DECIDES-001
WHEN a reduction loop nest is considered for parallelization, the implementing agent SHALL identify which index carries the accumulation and parallelize over any OTHER index, because only that choice keeps each output element summing in its original order; parallelizing over the accumulation axis needs per-worker partials whose combination reassociates.

Rationale: The SoftmaxRegression Hessian Gram accumulates grams[q*mm + a*mAug + j] over exactly one axis, the sample index. Interchanging the feature row outside the sample loop and parallelizing over it is bit-identical - distinct feature rows write disjoint ranges and each element still sums over samples ascending - and measured 1.96x. Parallelizing the same loop over samples would have required per-worker partial Grams whose combination reassociates the sum. Same loop and same target, one axis legal and the other not.

## PERF-INTERCHANGE-SERIAL-COST-001
WHEN a loop interchange is introduced to enable parallelism, the implementing agent SHALL measure the interchanged nest at GOMAXPROCS=1 against the original and keep the original for the serial branch if it is faster, because the interchange usually trades locality for a parallelizable axis and a default-GOMAXPROCS A/B hides that regression entirely.

Rationale: Interchanging the SoftmaxRegression Gram nest to put the feature row outermost sweeps X once per feature row instead of once total, costing 23.8ms against the original 18.0ms at GOMAXPROCS=1 while delivering 1.96x at 12. The default-GOMAXPROCS A/B showed only the win. Keeping the original nest for the serial branch left single-core at parity. An optimization that regresses a single-core host is not one.

## PERF-PARITY-TARGET-SOLVER-001
WHEN a change alters values that a solver iteration reads to decide its control flow (SMO working-set selection, EM responsibilities, a line search), the implementing agent SHALL pin bit-identity on the FITTED MODEL rather than on the changed intermediate, because a divergence there does not stay small - the solver compounds it into a different iteration path and a different result set.

Rationale: Parallelizing SVC kernel-column evaluation changes values SMO reads to pick its maximal-violating pair. Comparing columns would have shown agreement without proving the fit agrees; a single differing entry would redirect the working-set selection and end at a different support vector set. Digests over the dual coefficients, the intercept and the support-vector count are what actually establish parity, and they reproduced exactly at both benchmark sizes.

## PROC-BENCH-BELOW-THRESHOLD-001
WHEN an A/B shows an unexpected result at one benchmark size while another size behaves as predicted, the implementing agent SHALL panic-probe whether the small size even reaches the changed code path before investigating the number, because a size below a work threshold runs identical code in both arms and its spread is pure machine noise.

Rationale: SVCFit/n1000_rbf showed 1.39ms in one baseline round against 1.93-2.01ms for the new arm, reading as a 1.4x regression. At d=20 that size puts the work at 20000 against a 32768 threshold, so both arms run identical serial code and the number carries no information about the change. A panic probe confirmed n=1000 never enters the parallel path and n=4000 always does - a structural answer in one command, against re-running until the noise settles.

## PERF-HOIST-VERSUS-FANOUT-001
WHEN a per-iteration scratch buffer is hoisted out of a loop to reduce allocations, the implementing agent SHALL ask in the same change whether that loop could otherwise be parallelized, and if so make the buffer per-worker rather than per-call so both wins are kept.

Rationale: Hoisting the forward-substitution buffer out of the linalg column loop was a real measured allocation win, and it coupled every right-hand-side column to every other, foreclosing a 7.87x parallel speedup on LUSolve. The same story had already played out twice in GMM with yScratch4 and yScratch. Allocation reduction and parallelizability pull in opposite directions on scratch buffers: one wants a single shared buffer, the other wants one per worker. Per-worker scratch is the form that satisfies both, and the cost of getting it wrong shows up nowhere near the buffer - as a serial gate or a flat GOMAXPROCS ratio elsewhere in the package.

## PROC-BENCH-TOTAL-TIME-001
WHEN a benchmark whose single iteration is under about 100 microseconds is A/B-ed, the implementing agent SHALL raise benchtime until total measured time per arm reaches at least tens of milliseconds before reading any ratio, because a handful of iterations on a microsecond benchmark measures scheduler noise.

Rationale: LUSolve_128x1 reported a 2.09x regression at benchtime=5x, which is 115 microseconds of total measurement on a 20us benchmark. At benchtime=20000x the arms are 20.1us against 19.4us with overlapping ranges - parity. This is the fourth distinct way a benchmark number misled in one session, alongside a phantom regression from a single sample, an inverted sign from a bimodal min-of-N, and a spread on a code path the change never reached.

## PROC-SWEEP-IS-HYPOTHESIS-001
WHEN a cheap diagnostic sweep (GOMAXPROCS ratios, dtype variants, size variants) produces a ranked table, the implementing agent SHALL treat each entry as a hypothesis and re-measure any entry before acting on it at a benchtime that gives tens of milliseconds per arm, acting only on the large clear signals.

Rationale: A GOMAXPROCS sweep run at benchtime=5x reported rl DQNLearn and PPORollout as slower at 12 cores than at 1, which was recorded as a finding worth investigating. Re-measured at 2000x both are flat and PPORollout is slightly faster parallel. The sweep cheapness is what invites running it at a low benchtime across dozens of benchmarks, and a sweep is an A/B in disguise whose arms are core counts. The same sweep correctly found three large signals - SoftmaxRegression 0.99x, SVC 0.97x, all of linalg at 1.00-1.10x - each confirmed on re-measurement and each yielding a real win, so the instrument is sound and only its marginal entries mislead.

## PROC-SWEEP-FIXTURE-SCALE-001
WHEN a GOMAXPROCS or other parallelism sweep is run against a benchmark suite, the implementing agent SHALL confirm the fixtures are large enough for the parallelism under test to engage before reading any flat result as a serial bottleneck, and prefer suites with production-like geometry.

Rationale: Sweeping the twelve-architecture quant matrix reported every path at 1.04-1.16x, reading as a package-wide serial bottleneck. Those fixtures are two layers at dim 32 by design - built to measure per-layer allocation counts, which are geometry-independent - so no backend op is large enough to fan out. Benchmarks with realistic geometry in the same package gave 4.20x and 5.63x. The flat numbers were perfectly reproducible; they measured the fixture rather than the code, which is why reproducibility alone does not validate a sweep entry.

## PERF-PER-STEP-PARALLEL-REGION-001
WHEN a parallel region is opened inside a loop that runs many times (an elimination step, a training step, a layer), the implementing agent SHALL route it through the shared bounded pool rather than spawning a worker set per call, and measure allocations alongside time, because per-call spawning multiplies by the outer loop count.

Rationale: Parallelizing the LU elimination step opens a region once per step - 768 of them at n=768. Spawning a worker set per step took Inverse/768 from 823 allocations to 17463, a 21x regression, while delivering the intended 1.74x wall-clock win. Routing through the shared pool kept the win at 1.73x with 1850 allocations. A wall-clock-only A/B reports this change as a clean success; only the allocation axis shows the cost, and the outer loop count is what makes it large.

## PERF-CLOSURE-ON-SERIAL-BRANCH-001
WHEN a parallel helper takes a body closure and is called from inside a hot loop, the implementing agent SHALL write the serial branch out inline at the call site instead of passing the closure to the helper, because a closure capturing loop state allocates once per call and most calls take the serial branch.

Rationale: The LU elimination helper was called once per step and most steps fall below the parallel threshold. Passing the closure regardless cost one heap allocation per step - 64 extra allocations at n=64, an entire factorization worth of garbage for no benefit. Writing the serial loop out at the call site returned small sizes and single-core runs to identical allocation counts. The same shape appeared in the SoftmaxRegression interchange, where the serial branch also had to be kept separate.

## PROC-PROFILE-WALL-SHARE-001
WHEN a CPU profile of a partly-parallel program is read to decide what to optimize next, the implementing agent SHALL convert each entry to its WALL share first: divide total CPU by wall clock to get average parallelism, then divide a serial block CPU share by that ratio while leaving parallel shares as they are.

Rationale: After the linalg solve phase was parallelized, LU Factor showed as 18% of a CPU profile, which reads as minor. Total CPU over wall clock gave about 3x average parallelism, so the serial 18% was roughly 60% of wall while the parallel 78% collapsed to about 20% - the opposite ranking. Parallelizing Factor then delivered 1.73x. Raw CPU percentages systematically understate serial blocks in a partly-parallel program, which is the same measurement trap as reading condvar wait time as overhead, seen from the other side.

## PROC-COVERAGE-GAP-BY-INPUT-001
WHEN a scan rule keeps reporting one stubborn candidate that looks like a deliberate exception, the implementing agent SHALL check whether the code path is selected by a property of the INPUT DATA rather than by a config flag, size or dtype, because such a branch cannot be reached by parameterizing an existing benchmark and needs a purpose-built fixture.

Rationale: Which of QuantDeepSeekV2 two per-head loops runs is decided by the loaded GGUF keys: split-form attn_k_b gives the absorbed operator, legacy unsplit attn_kv_b gives the reconstruction. No parameterization of the existing benchmarks could reach the second path, and the twelve-architecture matrix builds the split form throughout, so it had tests and zero benchmarks. Unlike a size threshold or a dtype, a branch on file content is invisible in a benchmark arguments and the symptom looks like an intentional exception. Building the fixture took most of the work; the fix was mechanical.

## PROC-PROFILE-BOTH-ALLOC-AXES-001
WHEN an allocation profile is taken to find an optimization target, the implementing agent SHALL read it on BOTH sample_index=alloc_objects and alloc_space, because few-and-large allocation sites never surface in a count profile and many-and-small ones never surface in a byte profile.

Rationale: Nine iterations of this sweep profiled alloc_objects only. One alloc_space profile put nlp rows2D at 206MB, about 38 percent of a prefill per-op footprint, having never appeared near the top of a count profile because it allocates one slice per row - few, large allocations. Consolidating them cut allocations 25 percent and bytes 3.4 percent. The two axes rank sites differently and asking for only one leaves a blind spot the size of the other.

## PROC-PROFILE-ONETIME-VS-PERROUND-001
WHEN a profile aggregated over benchmark iterations shows a large allocation site, the implementing agent SHALL read the call site to establish whether the cost is per-iteration setup or per-round work before proposing reuse, because a benchmark that constructs a fresh object each iteration makes a one-time cost look recurring.

Rationale: An alloc_space profile of a GBM fit ranked newGBMBuilder first at 53% of bytes and subsampleIdx second at 31%. newGBMBuilder is a one-time presort already hoisted outside the boosting loop, so there was no reuse to exploit - it looked recurring only because the benchmark builds a fresh model per iteration and the profile aggregates over iterations. subsampleIdx, ranked lower, is genuinely called once per round and yielded up to 55% fewer bytes. The profile ranking inverted the actionability ranking.

## PROC-RULE-FROM-N-INSTANCES-001
WHEN a scan rule is proposed on the grounds that a pattern has recurred N times, the implementing agent SHALL write down each instance separate declaration and fix, and confirm they are the SAME intervention before naming a shared shape, because distinct interventions on the same measurement axis read as one pattern.

Rationale: Three alloc_space wins were claimed to share the shape of a zero-value container grown by append under a known bound. They were in fact consolidation of many small allocations into one backing array (rows2D), reuse of a per-call buffer across rounds (subsampleIdx), and preallocation of a nil struct field to a known bound (knnHeap). Only the third matched. The rule built from the claimed shape found none of the three on replay while reporting 13 unrelated candidates, and was reverted. Replay-against-source and cite-a-measurement both caught the error; neither covers the pattern-recognition step between them, which is where it was made.

## PERF-APPEND-USUALLY-CONDITIONAL-001
WHEN a guarded append inside a loop is considered for capacity preallocation, the implementing agent SHALL estimate how often the guard FIRES, not whether one exists, and preallocate when it fires on most iterations.

Rationale: CORRECTED. The original treated a guard as disqualifying, which measurement contradicts: bpeInto appends inside an if-vocab-hit-else-if-unk guard and reserving capacity still cut encode allocations 65 percent and bytes 21 percent, because a vocab hit is the normal case. The six sites where preallocation is genuinely wrong - an entropy threshold, a mask test, an instruction-op switch, an early error return - have guards that fire on a small minority of iterations. The deciding property is the guard HIT RATE, a runtime fact, not the guard existing, a syntactic one. A capacity is an estimate: too large wastes a fraction of one allocation, too small falls back to doubling.

## PROC-CHECK-PREDICATE-FIRST-001
WHEN a scan rule proposal names a predicate to separate true instances from look-alikes, the implementing agent SHALL test that predicate by hand against every catalogued site, true and false, before writing any detector code.

Rationale: A PS6020 guard predicate was proposed on the belief that both measured true positives were unguarded and all six rejected sites guarded. Two greps showed the opposite: knnHeap appends inside a capacity check and not inside a loop at all in that function, and bpeInto appends inside a vocab-hit check. The predicate would have yielded zero true positives. The failure was not in the measurement or the detector but in the untested predicate between them, which is where the two earlier rule failures this session also landed.

## PERF-HOTNESS-IS-NOT-SYNTAX-001
WHEN a perfscan check is proposed to be refined by a syntactic proxy for how often the matched code runs, the implementing agent SHALL reject the refinement and leave the check advisory, since call frequency is cross-function and runtime.

Rationale: Two consecutive attempts failed identically. PS6020 hinged on a guard hit rate, not on whether a guard existed. PS6018 hinged on entry frequency: a loop-nesting proxy would have excluded all three shipped wins, whose movement clusters sit at function top level exactly like the rejected candidate. Perfscan output already instructs the reader to measure hotness; pushing hotness into the parser trades real recall for imagined precision.

## BENCH-ASSERT-REGIME-001
WHEN a benchmark chooses an input parameter that decides which code path executes, the author SHALL add a test asserting the resulting regime, so a benchmark that measures nothing fails instead of passing.

Rationale: BenchmarkDBSCANFit called itself the probe for the eps-neighbourhood search at eps=2, where measurement showed 0 clusters, 0 core points and all 4000 points noise: every neighbourhood was a singleton and the cluster-expansion flood fill never ran one iteration. It timed a tree build plus empty queries for as long as it existed, reporting healthy numbers throughout. Since every optimization here is validated solely by benchmark, a degenerate benchmark silently voids that gate.

## PROC-MUTATE-EVERY-CLAUSE-001
WHEN a scan rule ships with floors asserting it stays silent, the implementing agent SHALL break each predicate clause in turn and confirm exactly one floor turns red, then fix any clause no floor covers.

Rationale: PS6021 passed all nine tests on the first draft while two clauses were defective. A scratch-constructor exclusion was unreachable, since such a helper must pass its value to the callback, which then already carries a scratch parameter. And both post-fix floors were passing on parameter count alone, leaving the type check unexercised. Neither was visible from green tests; a floor can pass for a reason unrelated to the clause it appears to defend.

## BENCH-NULL-AB-FLOOR-001
WHEN an A/B reports a wall-clock delta under about five percent, the measuring agent SHALL run a null A/B with identical code in both arms and report the claimed delta against that measured floor.

Rationale: A broken stash left identical code in both arms of a vision A/B; it reported a geomean sec/op delta of -1.39 percent. The valid run of a real change reported -1.15 percent. Without the accidental null run the smaller number would have read as a modest win. Interleaving and min-of-N bound sampling error but say nothing about a benchmark that cannot resolve the effect size at all.

## BENCH-STASH-UNTRACKED-001
WHEN an A/B swaps arms with git stash and the change adds a new file, the measuring agent SHALL verify the base arm by grepping for pre-change text before timing it, since stash refuses untracked paths and leaves the new code in place.

Rationale: git stash push with an untracked pathspec fails, the subsequent pop reports no stash entries, and both loop iterations then time the new code. The run completed with plausible numbers and no visible error in the benchstat output. A one-line assertion that the base arm actually contains the old text turns a silent wrong answer into a loud failure.

## BENCH-REFERENCE-MIRRORS-SHAPE-001
WHEN a bit-identity test writes a reference implementation of the code under test, the test author SHALL transcribe the pre-change expression verbatim — same operand forms, same variable-versus-constant status — rather than rewriting it for readability.

Rationale: A softUpdate parity test computed its reference through Unravel and AtF64 into a separate slice with the blend factor declared constant, and failed by exactly 1 ulp while the change under test was innocent. Against a reference written in the implementation own flat-slice form with the factor as a variable, zero of 131841 elements differed. FMA contraction depends on the expression shape, so a readable rewrite measures the compiler instead of the change. Const folding was ruled out by direct comparison.

## PERF-SELECTION-NEEDS-INTROSELECT-001
WHEN a full sort is replaced by a quickselect, the implementing agent SHALL bound the partition count and finish with a sort, and measure on the real input distribution rather than synthetic random data.

Rationale: A beam-search selection beat pdqsort 61us to 1236us on varied candidates and LOST at 1442us against 1170us on a fixture where the model returns identical logits per prefix, making the array several copies of one smooth curve that defeats median-of-three. A 2*log2(n) budget with a sort fallback capped the bad case at 1201us while keeping the good case at 61us. Random test data would not have surfaced this.

## BENCH-ARM-MUST-BUILD-001
WHEN an A/B swaps arms by restoring files, the measuring agent SHALL assert the arm compiles before timing it, not merely that its source text is right.

Rationale: Restoring the base beam.go while deleting a new helper left a test file referencing the missing symbol, so the base arm failed to build and produced zero samples. Benchstat then printed a single-column report of the new arm alone with no error, which reads as a completed comparison. A source-text check passes in exactly this case; only a build check catches it.

## BENCH-CONTROL-BASELINE-MOVED-001
WHEN a change optimizes a benchmark that another comparison uses as its control, the implementing agent SHALL say so in the record and require the dependent comparison to re-establish its null floor rather than reusing the recorded one.

Rationale: The Frobenius-norm benchmark exists as the control for pseudoinverse A/Bs, chosen because it touches the same tensors without entering the accumulation loop. Devirtualizing its accessor made it 3.1x faster, so every recorded Pinv comparison that leaned on that control now rests on a stale baseline. A control is only a control while it is untouched.

## PERF-SUPPRESSION-DRIFTS-001
WHEN a scan-suppression comment is added or code is inserted near one, the implementing agent SHALL re-run the scanner and confirm the finding is actually gone, since a directive reaches only its own comment block and the following line.

Rationale: Three suppressions written in one session were separated from their targets by a later edit that inserted a selection block between comment and sort. All three read as deliberate and reasoned; none suppressed anything, and the findings they named went on reporting unnoticed. The analyzer now reports unused directives as PS0001, but the habit is what prevents writing them.

## BENCH-SERIAL-FRACTION-AT-TARGET-001
WHEN a parallelization is sized from a single-core profile, the estimating agent SHALL measure the candidate's share at the target core count instead, because shares shift when other parts of the workload already parallelize.

Rationale: A single-core profile put Swin window attention at 24.1 percent of the forward, 15.1 ms against a 17.2 ms twelve-core forward, implying it was nearly the entire twelve-core wall clock and predicting up to 3.3x. Measured gain was 1.11x. The large projections parallelize through the backend and shrink at twelve cores while this stayed serial, so the one-core shares did not describe the twelve-core mix.

## PERF-DELETE-THE-SUPERSEDED-ARM-001
WHEN a faster arm is added that returns before an existing branch, the implementing agent SHALL delete the branches the early return makes unreachable rather than leaving them as a second copy.

Rationale: After the Swin blockwise path returned early, its old in-loop branches became dead but textually identical. A mutation applied to the first textual match landed on the dead copy and the whole suite stayed green, which reads exactly like a test with no teeth. Two code paths computing one thing also let a property established for one go stale in the other.

## BENCH-NO-RATIO-ACROSS-SUBBENCH-001
WHEN two arms of a comparison run as sequential sub-benchmarks in one function, the measuring agent SHALL give each arm its own top-level benchmark and its own data before reading a ratio between them.

Rationale: A paired sub-benchmark over a shared slice reported the second arm 18 percent slower, which read as a real dispatch regression from carrying a chunk index in a task struct. A scale benchmark with independent arms per size put the two within 1.00x from 2^14 to 2^20. The second arm was inheriting a different cache and scheduler state.

## PERF-THRESHOLD-NEEDS-EVIDENCE-001
WHEN a parallel or fast-path threshold constant is introduced or copied from a sibling, the implementing agent SHALL cite the benchmark that located its crossover, and if none exists build one before relying on the constant.

Rationale: More than ten thresholds in this tree share the constant 1<<15, two describing it as measured, with no surviving artifact. A dispatch benchmark then showed the shared fan-out helper 15 percent SLOWER than serial at exactly that size for an elementwise body, with the real crossover between 2^15 and 2^16. A constant copied between call sites carries none of the original body's cost profile.

## PERF-BCE-NEEDS-LENGTH-RELATION-001
WHEN a hot loop is given a hoisted row slice to remove bounds checks, the implementing agent SHALL verify with the compiler's bounds-check debug output that they actually went away, and range over one slice with the others clamped to its length rather than over a separate count.

Rationale: An eigensolver loop already had the row-slice hoist and still carried both of its per-iteration checks, because ranging over a separate element count gives the prove pass no relation between that count and the slice length. Clamping the second slice to the first and ranging over it removed them, and the four checks removed this way were worth 15 percent at one size and 12 at another.

## PERF-BCE-PAYOFF-NEEDS-BRANCHLESS-001
WHEN bounds-check removal is estimated for a loop containing a data-dependent branch, the estimating agent SHALL discount the estimate sharply, because the mispredicted branch dominates and the check is not what the loop waits on.

Rationale: The same transformation was worth 15 percent in a Jacobi rotation - pure arithmetic, two loads, four multiplies, two adds, two stores, so the checks were two of thirteen uops - and 3.3 percent in a ball-tree distance kernel whose loop carries an early-exit compare that mispredicts on far points. Counting uops against issue width predicts the first correctly and overshoots the second by four times.

## PERF-MEASURE-UNLOCKS-THE-NEXT-001
WHEN a survey conditions a candidate on whether a loop turns out to be issue-bound, the implementing agent SHALL ship and measure the cheapest bit-identical candidate first, then revisit the conditioned ones with that answer in hand.

Rationale: An eigensolver survey estimated bounds-check removal at 1.03 to 1.10 times and framed it as the calibration deciding whether fusion and unrolling were worth attempting. It measured 1.18 times, establishing the loop was issue-bound, and the fusion it unlocked then added a further 3.5 and 7.2 percent. Dismissing the cheap candidate on its small estimate would have closed the expensive one too.

## PROC-AUDIT-EVERY-HIT-BEFORE-TRUSTING-001
WHEN a scan check accumulates dozens of hits without any being acted on, the implementing agent SHALL classify every hit and report the false-positive fraction before treating the check as a work queue.

Rationale: One check reached 110 hits and an audit found ZERO matched the shape its own message described: 57 were layer stacks, 35 were per-head or per-expert fan-outs. Its six genuine instances were all suppressed by another guard, so it had never once reported the class it was built for. A high hit count reads as high yield and can mean the opposite.

## PROC-RECOUNT-AFTER-FILTERING-001
WHEN a second precision fix is proposed for a check whose hit list has already been filtered, the implementing agent SHALL recount the second predicate against the SURVIVORS, not against the original population.

Rationale: An audit put a guard's blind spot at 36 of 110 hit functions, which read as a substantial second win. After a trip-count filter cut the check to 26, only 4 survivors carried any fused-path signal and only 1 carried the tight one. The two filters overlapped almost completely, so the second fix was worth a single hit at real recall cost rather than the 36 the pre-filter count implied.

## PERF-CHECK-BLIND-SPOT-IS-THE-HELPER-SET-001
WHEN a scan check reports a call only when a sibling of that shape exists and some call shapes go unreported, the implementing agent SHALL extend the helper set rather than the check, since declaring the missing sibling makes the check report those sites with no analyzer change.

Rationale: A variadic-call check could not see five per-iteration allocations because no four-input pooled sibling existed. Declaring one took its recall from 198 of 203 sibling-coverable sites to all 203 without touching the detector. The gap was in the code the check measures against, not in the check.

## PERF-WORK-A-QUEUE-BY-REACHABILITY-001
WHEN a sound scan check leaves dozens of hits whose individual payoff is one small allocation, the implementing agent SHALL convert only the sites a benchmark reaches, in that order, and declare the queue exhausted for measurable value when those run out.

Rationale: A check with perfect precision had 198 hits of which 148 were on no measured path. The first pass over its reachable sites gave six percent fewer allocations on two decode paths; the second gave one to three percent. The 119 that remain cannot be validated on this host, so further conversion is mechanical hygiene rather than optimization, and saying so prevents an unbounded sweep that looks like progress.

## BENCH-SPREAD-DECIDES-WHAT-YOU-MAY-CONCLUDE-001
WHEN a comparison reports no significant change, the measuring agent SHALL state the within-arm spread alongside it, because no-change on a one-percent-spread benchmark is a result while no-change on a fifty-percent-spread benchmark is an absence of resolution.

Rationale: In one comparison three tight arms held one percent spread and genuinely showed no time change, while two long generation benchmarks in the same run held fifteen to fifty-three percent and their timing columns supported no conclusion at all. Both printed as insignificant. Quoting the second as evidence of no regression would have been unfounded.

## PERF-ATTRIBUTE-BORROWED-NUMBERS-001
WHEN a scan check's message quotes a speedup measured at a different site, the author SHALL name the site it came from and the variables the check cannot see, rather than stating it as though it applies where the check fired.

Rationale: A triage of all 59 checks found six asserting a measured magnitude with no hedge, across 290 hits. That is the failure mode that made one check useless for months: a reader trusts the number, acts, and finds nothing. Attribution costs a clause and changes nothing about what fires.

## PROC-CORRECTNESS-SUPPRESSION-THRESHOLD-001
WHEN a narrowing would suppress a recommendation that is not merely useless but WRONG, the implementing agent SHALL ship it on high precision even at low recall, unlike a performance narrowing which needs enough instances to justify itself.

Rationale: A divide check recommended a reciprocal multiply that evaluates to zero on integer operands. Two proofs caught three of ten such sites with zero false positives. Suppressing a wrong recommendation costs nothing, and what is given up is a perf suggestion at sites where it would have produced zero. For a perf check the failure mode is low precision; for a correctness hazard it is low precision that must be avoided and low recall that is tolerable.

## PROC-CONFIRM-THE-MUTATION-APPLIED-001
WHEN a mutation is applied by string replacement to check whether a test has teeth, the implementing agent SHALL confirm it actually changed the file, by diff or match count, before reading the test result.

Rationale: A mutation intended to weaken a recall floor left the test green, which reads as a toothless test. It had silently no-oped on a shell escaping problem, and the floor was fine. Separately, a mutation earlier in the campaign landed on a dead duplicate of the target code and the suite stayed green for the same misleading reason. A green result after a mutation means nothing until the mutation is known to exist.

## PERF-CONVERSION-IS-NOT-A-CALL-001
WHEN an AST predicate treats the presence of a call expression as evidence a loop is bottlenecked, the implementing agent SHALL exclude predeclared conversions and builtins first, since Go models float64(x) as a call expression and len compiles to a field load.

Rationale: A floated narrowing for a register-blocking check would have removed 28 hits and lost ten of forty-five genuine sites. Seven of the casualties were the f32-widening twin of an f64 hit the same predicate kept - the same kernel in the same file, differing only by a conversion around each load. Excluding conversions and builtins turned the same idea into a 69 to 96 percent precision gain at 98 percent recall.

## TEST-THRESHOLD-GUARDED-PATH-UNCOVERED-001
IF a code path is entered only above a size or worker-count threshold, THEN the implementing agent SHALL add a gate whose 2 arms are one source selected by that threshold, and confirm it reddens.

Rationale: Observed three times in three packages within a few rounds, always reading as covered when it was not. linalg: every bit-identity golden ran at 24x16, 384 elements against a 65536-element parallel bound, so no test entered a parallel branch; perturbing the parallel apply by one ulp left the golden PASSING. nlp: every SnapKV test used 2 or 3 observation rows, below the group of 4, so the unrolled body was unreachable and regrouping its adds left all eight tests passing. nn: WandaPrune fans out only above two panels and the panel width is min(wandaPanel, cout), so the quickselect sweep capping cout at 40 and the smaller goldens all ran one panel serially — making the panel scratch SHARED across workers left all eleven Wanda tests passing, golden included, with the race detector firing. The gate must select its arms from ONE source via the threshold or an equivalent knob such as GOMAXPROCS, never from a separately written reference: a reference rewritten for readability contracts to FMA differently and fails by an ulp on arithmetic that never changed. Generalizes TEST-GOLDEN-BELOW-THRESHOLD-001 from linalg to every package.

## PROC-CONFLICT-LIST-TRUNCATED-001
IF a git merge reports conflicts and its output was piped through head or tail, THEN the implementing agent SHALL list unmerged paths with git diff --name-only --diff-filter=U and grep the tree for conflict markers before committing.

Rationale: A merge this session reported two conflicted files, the output was read through tail -2, and only the last one was seen. The other file was committed with its markers intact. The commit succeeded because git does not re-check content once the index is staged, and gofmt, build, vet and the package tests all failed AFTERWARD with parse errors pointing at the marker lines — by which time a broken merge was already in history and had to be amended. The mechanical guard is cheap and independent of how the merge output was read: git diff --name-only --diff-filter=U before staging, and a grep for a leading conflict marker across the tree before committing. Note that once the merge is committed, git checkout --theirs reports updating 0 paths because the conflict stages are gone, and the sides must be recovered from the merge commit parents instead.

## PERF-SLAB-IS-RESOURCE-NOT-SPEED-001
IF a loop allocates one slice per iteration into an index of a slice-of-slices, THEN the implementing agent SHALL collapse it to 1 slab with capped views and report the allocation count, not a wall-clock claim.

Rationale: Measured at two independent sites, both times the same split. linalg QR per-column reflectors: allocations per call fell 39.7% and the wall clock was indistinguishable from base at n>=512. classic GMM per-sample responsibility rows: allocations fell 62.2% on GMMFit, 5751 to 1753, with time p=0.501 and p=0.424 — no change. Bytes barely move either, up about 0.1%, since the slice-of-slices header remains and one slab rounds to a size class. The transform is bit-identical by construction because make zeroes and so does a fresh slab, and no value, order or association changes. It is worth doing: fewer allocations means less allocator work and less GC scan pressure on every machine, which is a portable resource improvement rather than a host-specific one. But it must be claimed as resource usage, never as throughput, and the payoff scales with the ITERATION count, so a loop bounded by a data size is worth converting where one bounded by a small constant is not. Preconditions the syntax cannot check: rows must not be appended to and must not be individually replaced later, because the views are valid only while the slab is. Jagged rows are a different transform needing per-row offsets. Detected by perfscan PS2008.
