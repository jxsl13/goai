# ADR-0011 — NPU acceleration is a model-level, not op-level, target (§T44)

Status: accepted (2026-07-06). Closes the accelerator arc (ADR-0008..0010).

## Context

The user's priority is broad accelerator support (training AND inference). After
Metal (ADR-0009's sibling), CUDA (ADR-0009), and Vulkan (ADR-0010), the remaining
accelerator class is the NPU: Apple Neural Engine (ANE), Intel NPU, and the unit
Windows drives via DirectML. §C7 pre-declared NPU an explicit non-goal of the
first stage, to be re-evaluated after GPU. This ADR is that re-evaluation, backed
by current (2026) evidence (§R45).

GoAI's accel layer (L1b) is an **op-level** Kernel-dispatch interface: one
function per tensor op, selected by (op, dtype), validated against the Pure-Go
reference (§I4, §V3, §V9). The question is whether NPUs can be served at that
granularity from Go.

## Evidence (§R45, confirmed 2026)

- **Apple ANE:** reachable ONLY by compiling a whole CoreML model (.mlpackage /
  ML Program) and running prediction. No per-operation entry point; no public
  gradient/autodiff API; FP16/INT8 **inference-only**.
- **Intel NPU:** reached via OpenVINO — graph/whole-model, **inference-only**, no
  backward pass.
- **Windows DirectML:** DOES expose per-op dispatch (`IDMLDevice::CompileOperator`,
  `DML_GEMM`) and even training gradient operators (`DML_*_GRAD`) — but only
  through a native C++/nano-COM API bound to Direct3D12 (tensors are
  `ID3D12Resource` recorded into command lists). No official Go binding; wiring it
  in needs heavy cgo/COM interop far beyond a per-op matmul kernel.

## Decision

**Ship no NPU op-level backend. Document the honest non-goal** (the DoD's second
option for §T44). Concretely:

1. A `backend/npu` package that is pure-Go documentation plus `Available() bool`
   returning **false** — so feature-detection probes NPU support the same way as
   GPU backends and get an explicit "no", never a silent gap (§V4). Dual-audience
   godoc (professional + newcomer) per §V17, with a runnable example.
2. Record the real path: NPU acceleration belongs at the **model level** — export
   a whole graph to CoreML/ONNX/OpenVINO/DirectML and run it. That is a possible
   future **L5 "model runner"** track, explicitly NOT an L1b op backend, and it
   does not block or reshape the op-level architecture.

## Consequences

- **+** Honest, evidence-backed boundary; no fragile half-backend pretending to
  accelerate. The op-level architecture stays clean.
- **+** `backend/npu.Available()==false` gives callers a uniform, truthful probe.
- **−** No NPU acceleration today. Users wanting ANE/NPU inference must wait for
  the model-runner track (or export + run externally).
- **Revisit when:** (a) a maintained Go DirectML/D3D12 binding appears, making
  op-level DML dispatch practical on Windows; or (b) GoAI grows an L5 model-export
  runner, at which point CoreML/OpenVINO/DirectML NPU inference becomes reachable
  at the model level.

## Alternatives rejected

- **Force a CoreML 1-op model per matmul:** compiling an MLModel per call is
  absurd overhead and still gives no gradients. Rejected.
- **cgo/COM DirectML op backend now:** enormous interop surface, Windows-only,
  unverifiable here, for a backend the op interface barely fits. Rejected until a
  real Go binding exists.
