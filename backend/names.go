package backend

// Name is the typed identifier of a backend (ADR-0015). It is a small string enum
// used for registration, lookup (Get), and the selection-preference order, so
// backends are referred to by a checked constant — backend.Metal — instead of a
// bare "metal" string literal that a typo could silently break. The underlying
// type stays string so a Name is still a ready registry key and prints as itself.
type Name string

const (
	CPU    Name = "cpu"    // optimized Pure-Go SIMD backend (the usual default host backend)
	Ref    Name = "ref"    // Pure-Go reference backend: the numeric truth (§V9) and final fallback (§I4); see Reference()
	Metal  Name = "metal"  // Apple Metal / MetalPerformanceShaders GPU backend (darwin, cgo)
	CUDA   Name = "cuda"   // NVIDIA CUDA / cuBLAS GPU backend (cgo)
	Vulkan Name = "vulkan" // portable Vulkan-compute GPU backend (cgo)
)

// String returns the backend name as a plain string (its registry key and display
// form).
func (n Name) String() string { return string(n) }
