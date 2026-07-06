// Package ref is the Pure-Go scalar reference backend (layer L1).
//
// Every operation here is implemented for clarity and numeric correctness, not
// speed — it is the truth that optimized backends (cpu-simd, cuda, metal,
// vulkan) are validated against (§V3, §V9). It compiles and runs everywhere
// with CGO_ENABLED=0 (§I5).
package ref
