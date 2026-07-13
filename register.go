package goai

// Importing goai self-registers every backend usable on the current build, so a
// program needs no backend-selection code (§C11, §T47). Always registered: the
// Pure-Go reference (truth + fallback, §V9/§I4) and the optimized cpu backend.
// Accel GPU backends register through OS/tag-specific companion files:
//   - Metal (macOS): register_darwin.go — NO build tag, just cgo (§R47).
//   - CUDA:  register_cuda.go   — `-tags cuda`   (build-time link dep, §R47).
//   - Vulkan: register_vulkan.go — `-tags vulkan` (build-time link dep, §R47).
// Each accel backend registers itself only if its device is present, and
// backend.Default() auto-selects the fastest available (§V18).
import (
	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)
