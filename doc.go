// Package goai is a Go-native, full-spectrum AI library built Pure-Go-first /
// cgo-last. See SPEC.md for goals (§G), constraints (§C), architecture
// invariants (§I), and verification invariants (§V).
//
// Layer model (§I):
//
//	L0  tensor          core: Tensor, Dtype, Device, Allocator, strides/views
//	L1  backend         Backend/Kernel interface + Pure-Go reference (truth)
//	L1b backend/*        swappable accel backends behind build tags + fallback
//	L2  autograd        tape/graph + VJP rules
//	L3  nn              layers, init, optimizer, loss, data
//	L4  (nlp, vision,   domains
//	     classic, rl)
//	L5  format          safetensors, GGUF, ONNX
//
// Invariant: higher layers never import backend internals; every op has a
// Pure-Go fallback; CGO_ENABLED=0 builds green on macOS, Windows, Linux.
package goai

// Version is the current library version. Pre-release: API unstable (§V8).
const Version = "0.0.0-dev"
