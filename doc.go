// Package goai is a Go-native, full-spectrum AI library built Pure-Go-first /
// cgo-last. The project's intent, architecture decisions and verifiable
// contracts live in the .spectackle/ spec bundle; read them with
// `spectackle call get '{"id":"."}'` or the /spectackle-state command.
//
// Layer model:
//
//	L0  tensor          core: Tensor, Dtype, strides/views
//	L1  backend         Backend/Kernel interface + Pure-Go reference (truth)
//	L1b backend/*        swappable accel backends (metal, vulkan, cuda) + fallback
//	L2  autograd        tape/graph + VJP rules
//	L3  nn, ops, linalg layers/optimizers/losses; eager ops; dense linear algebra
//	L4  nlp, vision,    domains
//	    classic, rl
//	L5  format          safetensors, GGUF, npy/npz
//	—   llamagpu        batched GPU decoding for GPT/Llama
//
// Invariant: higher layers never import backend internals; every op has a
// Pure-Go fallback; CGO_ENABLED=0 builds green on macOS, Windows, Linux.
// These are contracts ARCH-001, ARCH-002 and BUILD-001 in the spec bundle.
package goai

// Version is the current library version. Pre-release: API unstable (§V8).
const Version = "0.0.0-dev"
