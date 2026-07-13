//go:build cuda && cgo

package goai

// Building with `-tags cuda` (and the CUDA toolkit installed) auto-registers the
// NVIDIA cuBLAS backend. CUDA still needs an opt-in tag because libcublas is a
// build-time link dependency not present on a generic machine (§R47) — a future
// runtime-dlopen path could make it tag-free like Metal.
import _ "github.com/jxsl13/goai/backend/cuda"
