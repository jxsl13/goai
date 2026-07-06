// Package npu documents GoAI's stance on NPU (neural-processing-unit)
// acceleration and provides an honest, always-false availability probe (§T44).
//
// # For the AI professional
//
// GoAI's accel layer (L1b) is an OP-LEVEL Kernel-dispatch interface: one function
// per tensor op (matmul, …) selected by (op, dtype) and validated against the
// Pure-Go reference (§I4, §V3, §V9). Every shipped accel backend — cpu-SIMD,
// Metal/MPS, CUDA/cuBLAS, Vulkan-compute — fits that per-op contract.
//
// NPUs do not fit it (confirmed 2026, §R45):
//   - Apple Neural Engine (ANE): reachable ONLY by compiling a whole CoreML model
//     (.mlpackage / ML Program) and running prediction. No per-operation entry
//     point, no public gradient/autodiff API; FP16/INT8 INFERENCE-ONLY.
//   - Intel NPU: reached via OpenVINO, which is graph/whole-model and
//     inference-only (no backward pass).
//   - Windows DirectML: does expose per-op dispatch (IDMLDevice::CompileOperator,
//     DML_GEMM) and even training gradient operators — but only through a native
//     C++/nano-COM API bound to Direct3D12 (tensors are ID3D12Resource recorded
//     into command lists). There is no official Go binding; wiring it up needs
//     heavy cgo/COM interop far beyond a per-op matmul kernel.
//
// So NPU acceleration belongs at the MODEL level (export a whole graph to CoreML/
// ONNX/OpenVINO/DirectML and run it), not behind GoAI's per-op backend. That is a
// possible future L5 "model runner" track — explicitly NOT an L1b op backend.
// This matches §C7 (NPU = explicit non-goal of the first stage) and keeps the
// promise honest rather than shipping a silent, broken fallback (§V4).
//
// # For the newcomer
//
// An "NPU" is a chip built to run trained neural networks fast and efficiently —
// Apple's Neural Engine, Intel's NPU, and the piece Windows drives via DirectML.
// They are fantastic at running a FINISHED model, but they don't let a program
// hand them one small math step (like a single matrix multiply) at a time, and
// most can't compute the gradients used to TRAIN a model at all. GoAI is built
// from those small steps, so it can't plug an NPU in the way it plugs in a GPU.
// The realistic way to use an NPU is to take a whole trained model and run it
// through Apple/Microsoft/Intel's own model runner — a separate feature GoAI may
// add later. Until then, GoAI runs on CPU and GPU, and this package's
// [Available] simply and honestly reports false.
package npu

// Available reports whether an NPU op-level backend is usable. It is always
// false: GoAI intentionally ships no NPU op backend (see the package doc and
// §R45 for why). It exists so feature-detection code can probe NPU support the
// same way it probes GPU backends (backend.Available) and get an explicit "no"
// rather than a silent gap (§V4).
func Available() bool { return false }
