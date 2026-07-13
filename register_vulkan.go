//go:build vulkan && cgo

package goai

// Building with `-tags vulkan` (and the Vulkan loader available) auto-registers
// the portable Vulkan compute backend. Like CUDA it is tag-gated today because
// `-lvulkan` is a build-time link dependency; the Vulkan loader is explicitly
// designed for runtime dlopen (§R47), so a future tag-free path is possible.
import _ "github.com/jxsl13/goai/backend/vulkan"
