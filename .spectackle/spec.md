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
