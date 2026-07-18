# ADR-0029 — the build-tagged test corpora have a verification boundary, and it is where the toolchain ends

- Status: Accepted, then **partially superseded the same day** — see "Update (T866)" below.
  The reasoning stands; one of its premises turned out to be wrong.
- Date: 2026-07-18
- Task: §T865 — from the `llamagpu` discovery sweep, which measured 117 test funcs in the
  package against 32 reachable by CI
- Related: ADR-0009 (cuda backend), ADR-0013 (tag-free Metal), §V23 (`go vet` compiles
  `_test.go`, §B45), §V26 (CI-skip fails open), §B77 (a guard written as a condition is not
  a guard — the same "looks like verification, verifies nothing" failure, one layer up)

## Context

A sweep of `llamagpu/` found that CI compiled neither the cuda nor the vulkan test corpus:

- `go vet ./...` skips both — wrong build tags.
- `go build -tags vulkan ./...` skips `_test.go` files entirely, by design.
- The vulkan lane's test step runs `./backend/vulkan/`, a different package.

So of 117 test funcs in `llamagpu`, CI reached 32. The other 85 — including every cuda
architecture `MatchesReference` and every Q8 `CloseToF32` — were neither run, nor compiled,
nor vet-checked. They could rot indefinitely behind a green pipeline.

The obvious remedy is to add `go vet -tags cuda ./llamagpu/` and `go vet -tags vulkan
./llamagpu/`. Only one of those is real.

## The trap

`go vet -tags cuda ./llamagpu/` **succeeds on darwin and verifies nothing.** The cuda test
files are:

```go
//go:build cuda && cgo && (linux || windows)
```

On darwin the OS constraint excludes every one of them, so the command compiles an empty
set and exits 0. This was confirmed the only way such a claim can be confirmed: by injecting
a syntax error into `cuda_bert_test.go` and watching the check pass anyway.

On ubuntu the same command *would* compile them — and would then need `backend/cuda`, hence
the CUDA toolkit headers, which no runner has. The step would fail for a reason unrelated to
the change under test, on a lane the parallel human worker also depends on.

So the naive fix is either vacuous or breaking, depending where it runs. There is no third
runner where it is both meaningful and green.

## Decision

**Add the vulkan check; decline the cuda one and write down why.**

`go vet -tags vulkan ./llamagpu/` goes into the vulkan lane, next to `go build -tags vulkan
./...`, because that lane already provisions the SDK. It was verified non-vacuous by the
same injected-syntax-error method: it catches the error that the cuda equivalent misses.

For cuda, the honest position is a boundary, not a gap to paper over:

| Rot class | Caught? | By what |
| --- | --- | --- |
| Syntax | yes | the tree-wide `gofmt -l .` step, which parses regardless of build tags |
| Type / API | **no** | nothing — and nothing can, without a CUDA toolchain |
| Logic inside a test body | **no** | needs execution on real silicon; CI has none |

That last row matters for expectation-setting. The sweep suggested this CI change "would
have caught" the inverted-NaN-guard bug found alongside it (§B78). It would not have.
Compiling a test is not running it, and no amount of vetting detects an assertion that
cannot fail. Conflating the two would have replaced a known gap with a false sense of
coverage — which is strictly worse, because it stops people looking.

## Consequences

- The vulkan test corpus can no longer rot silently; the cuda corpus still can, in the
  type/API dimension, and that is now a written-down limitation rather than an unnoticed one.
- Closing the cuda side requires either a CUDA-toolkit-provisioned runner (compile only) or
  real NVIDIA silicon (execution). Both are infrastructure decisions with cost, deferred
  rather than pretended away.
- The general lesson, which is the same one §B77 taught in a different guise: **a check that
  runs against an empty set is indistinguishable from a passing check.** Any build-tagged or
  conditionally-scoped verification must be proven non-vacuous — inject a fault and confirm
  it is caught — before it is trusted or cited as coverage.

## Update (T866) — the cuda row was closable after all

The decision above declined the cuda check on the grounds that "no runner has the CUDA
toolkit". That is a statement about the runner as configured, not about what is possible —
and treating a *configuration* as a *law of nature* is how a deferred gap becomes a
permanent one. The user asked the obvious question the ADR failed to ask itself: why not
install it?

Nothing blocks it. The build needs no GPU and no driver:

- Kernels are compiled at RUNTIME via nvrtc (`backend/cuda/cuda.go:341`), and the WMMA `.cu`
  files ship as a committed fatbin, so **nvcc is never invoked** by the Go build. Only
  headers and import libraries are required.
- `-lcuda` resolves against the driver STUB the toolkit ships in `lib64/stubs`, which exists
  precisely so CUDA software can be built on machines with no driver.

So a `cuda-compile` lane installs the dev subset (`cuda-cudart-dev`, `libcublas-dev`,
`cuda-nvrtc-dev`, `cuda-nvcc` for its crt/ headers — not the ~3 GB full toolkit) from NVIDIA's own apt repo, and runs
`go vet -tags cuda` over `backend/cuda` and `llamagpu`, then `go test -c -o /dev/null` to
LINK the test binaries. The link step matters: `vet` typechecks but does not resolve extern
symbols or validate `LDFLAGS`.

NVIDIA's apt repo is used directly rather than a third-party setup-cuda action: this repo
takes no dependencies it does not have to (§C1), and a build-time supply chain is still a
supply chain.

### What the revised boundary is

| Rot class | Before T866 | After T866 |
| --- | --- | --- |
| Syntax | caught (tree-wide gofmt) | caught |
| Type / API / cgo bindings / link symbols | **not caught** | **caught** |
| Logic inside a test body (a wrong number) | not caught | **still not caught** |

The third row is unchanged and is the honest residual: executing the cuda corpus needs real
NVIDIA silicon. The §B78 inverted-NaN-guard bug still would not be caught by this lane —
compiling a test is not running it. That caveat from the original decision survives intact;
only the middle row moved.

### The lesson worth keeping

The original ADR verified its *check* rigorously (fault injection proved the darwin cuda vet
vacuous) but did not challenge its *premise*. "The runner doesn't have it" invited "then add
it", and nobody asked until prompted. Verifying that a check works is not the same as
verifying that the constraint forcing the check was real.
