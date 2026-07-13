# ADR-0016 — Quantized matmul as an optional backend capability

**Status:** accepted (2026-07-07, §T142)

## Context

By T141 GoAI had in-kernel quantized matmul kernels on both GPU backends (Q8_0/Q4_K/Q6_K on
Metal + Vulkan) and a CPU reference (`gguf.QMatMul`), but **nothing used them**: the Llama
loader (`LlamaFromGGUF`) fully dequantizes every weight to F64 at load, so a "quantized" model
runs dense f32/f64 matmuls and keeps a full-precision copy in memory — defeating the point of
quantization. GPU quantized *inference* needs a model layer that keeps weights quantized and
dispatches the matmul to the accelerator.

Two forces conflict:

- **Layering.** `nn` (L3) must not import `backend/metal` or `backend/vulkan` (L1b) — it selects
  a backend only through the registry (`backend.Default()`), per ADR-0012.
- **The core op system is float-tensor-based.** `backend.Execute(Op, []*Tensor, Attrs)` moves
  `*tensor.Tensor`; a quantized weight is an opaque byte blob in a ggml block layout, not a
  float tensor. Forcing it through `OpMatMul`/`Attrs` would mean either a new quantized-tensor
  dtype threaded through L0–L2, or the dequant logic (which lives in `format/gguf`, L5) leaking
  into the backend layer (L1b→L5, backwards).

## Decision

Add an **optional capability interface** in `backend`, discovered by type-assertion off
`Default()` — exactly the pattern already used for `backend.Recorder` (autograd interception):

```go
type QuantMatMuler interface {
	QMatMul(x *tensor.Tensor, weight []byte, quantType uint32, n, k int) (*tensor.Tensor, error)
}
```

- `quantType` is the **ggml type code** (`8`=Q8_0, `12`=Q4_K, `14`=Q6_K), a plain `uint32` — so
  `backend` needs no dependency on `format/gguf` (which would be an L1→L5 cycle-risk). The GPU
  backends carry small private consts for the codes they accelerate.
- A backend returns the sentinel `backend.ErrQuantUnsupported` for a code it does not
  accelerate; any **other** error is a genuine fault and must propagate.
- `nn.QuantLinear{Weight, QT, In, Out}` holds the quantized weight bytes and `Forward` prefers
  the accelerator: it type-asserts `Default()` to `QuantMatMuler`, calls it with `uint32(QT)`,
  and falls back to the CPU `gguf.QMatMul` **only** on `ErrQuantUnsupported` (or when the backend
  is not a `QuantMatMuler` at all). Activations are converted to f32 for the GPU kernels; the CPU
  path keeps f64 accumulation. `nn` already depends on `gguf` (T132) and `backend`, so no new
  cross-layer edge is introduced.

## Consequences

- **GPU quantized inference is real**: `QuantLinear.Forward` runs the quantized GEMM on the
  active accelerator (Metal here) with the weight never materialized as f32; results match the
  f64 CPU reference within `crossTol` (V-CROSS). A backend/quant-type without a kernel degrades
  silently to identical CPU results.
- The core `Backend` interface is untouched (no quantized dtype, no L5 dependency in L1b). The
  capability is additive and backends opt in structurally.
- **Trade-off:** the quant weight bytes bypass the tape, so `QuantLinear` is inference-only (no
  autograd through the quantized weight) — correct, since quantized weights are frozen at
  inference. Training a quantized model uses the existing float layers (QLoRA etc.).
- **Follow-ups:** the remaining k-quants (Q5_K/Q2_K/Q3_K) get GPU kernels behind the same
  dispatcher (CPU fallback already covers them); wiring `QuantLinear` into a `Llama` variant that
  loads weights quantized (instead of dequantizing at load) is the next integration step;
  device-residency (upload the quantized weights once) remains the big perf lever (§B).
