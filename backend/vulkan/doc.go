// Package vulkan is the optional, portable Vulkan compute GPU backend (layer
// L1b, §T43).
//
// Unlike Metal (Apple-only, §T20) and CUDA (NVIDIA-only, §T42), Vulkan is
// vendor-neutral: the same SPIR-V compute shader runs on AMD, Intel, NVIDIA, and
// (via MoltenVK) Apple GPUs. There is NO built-in BLAS in Vulkan (§R39) — it is
// raw GPGPU, so the GEMM is a hand-written GLSL compute shader (shaders/matmul.comp)
// compiled to SPIR-V and embedded here.
//
// Strictly opt-in: build with `-tags vulkan` and cgo, with the Vulkan loader
// (-lvulkan) available. The default CGO_ENABLED=0 build contains no cgo (§V7,
// §I5) — without the tag this package is just this doc file, so `go build ./...`
// and every pure-Go platform stay green.
//
// BUILD STEP: the tagged build embeds shaders/matmul.spv, a build artifact
// produced from shaders/matmul.comp by `make vulkan-spv` (needs glslc from the
// Vulkan SDK) — exactly as `-tags cuda` needs the CUDA toolkit. Run that target
// before `go build -tags vulkan`.
//
// Kernels: MatMul f32 via the compute shader; every other op falls back to the
// Pure-Go backends through the dispatch layer (§I4). Training GEMMs reach the GPU
// through autograd.NewTapeOn(vulkanBackend) (forward + both backward matmuls of
// the VJP), so the backend serves inference AND training.
//
// Not host-verifiable on this arm64/macOS host without a Vulkan SDK — the backend
// and its §V3 cross-reference + §C3 gate benchmarks are CI-gated (§B36, mirrors
// the CUDA gate §B35 and archsimd §B13). Merged through the cgo gate (§C2/§C3).
package vulkan
