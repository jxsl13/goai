// Package backend is layer L1: the compute abstraction and backend selection.
//
// # For the AI professional
//
// It defines the Backend and Kernel interfaces plus the registry and runtime
// feature detection. The Pure-Go reference backend (subpackage ref) is the source
// of numeric truth (§V9); accel backends are validated against it, never vice
// versa. The Backend interface fixes an execution/sync model so a later async GPU
// backend does not break API stability (§V14).
//
// Backend selection is automatic (§T46): build-tagged accel backends (cpu-SIMD,
// metal, cuda, vulkan) self-register in init() ONLY when their device is present,
// so registration doubles as detection. Default returns the highest-preference
// registered backend along a descending-performance order — cuda > metal >
// vulkan > cpu — with the reference as the guaranteed final fallback (validated
// vs ONNX Runtime EP priority, PyTorch cuda>mps>cpu, ggml's scored registry §R46,
// and on-host benchmark). Because NewContext and autograd.NewTape call Default,
// building with an accel tag makes real work run on the GPU with no code change.
// Override the order with SetPreference; pick a specific backend per call with
// Context.WithBackend. cgo/build tags are the only gating; everything else is
// detected at runtime (§C11).
//
// # For the newcomer
//
// This package decides WHERE your math runs — which chip does the work. You don't
// have to choose: if you built the library with GPU support and a supported GPU
// is present, GoAI finds it and uses it automatically; otherwise it uses the CPU.
// The rule is "fastest available first" (GPU before CPU), and there is always a
// plain-CPU fallback that works everywhere, so your program never fails just
// because a GPU is missing. If you want to force a particular choice, you can set
// your own order with [SetPreference] or pass one backend to a single call. The
// examples below show both the zero-effort automatic path and manual control.
//
// Further reading: Goto & van de Geijn 2008, "Anatomy of High-Performance Matrix Multiplication" (ACM TOMS), for what an optimized GEMM backend is up against; the ggml/llama.cpp source for the quantized-kernel state of the art this library measures itself on.
package backend
