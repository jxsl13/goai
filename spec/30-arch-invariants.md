## §I — architecture invariants

layer model (strict, top ⊥ know backend internals):

I.L0 core: `Tensor` (data, dtype, shape, strides, device), `Dtype`, `Device`, `Allocator`, views (reshape/slice/transpose = stride ops). ⊥ cgo in L0.

I.L1 compute: `Backend` + `Kernel` interface + Pure-Go reference backend = TRUTH. registry + feature-detection.

I.L1b accel: swappable backends (cpu-simd, cuda, metal, vulkan) behind same interface, feature-detect + fallback.

I.L2 autograd: tape/graph + `Variable` + VJP rules per op.

I.L3 nn: layers, init, optimizer, loss, data pipeline.

I.L4 domains: transformer/LLM, CV, classic-ML, RL, probabilistic.

I.L5 io: safetensors, GGUF, ONNX.

INVARIANTS:

I1: higher layer ⊥ import backend internals. public API backend-agnostic.

I2: ∀ op → Pure-Go fallback exists.

I3: ⊥ cgo in L0.

I4: accel backend selected via feature-detection @ runtime, ⊥ compile-time-only lock; missing accel → fallback.

I5: `CGO_ENABLED=0` build green ∀ {macOS, Windows, Linux}.

I6: op parameters = TYPED. `backend.Attrs` is a SEALED interface (`interface{ opAttrs() }`), one struct per op (AttnAttrs, NormAttrs, RoPEAttrs, …); ⊥ `map[string]any`/stringly-typed keys. construct = checked struct literal; read = comma-ok assert + `WithDefaults()` (defaults single-sourced, kernel+VJP share). Execute/Kernel/Record/VJP signatures unchanged (param stays `Attrs`). [ADR-0014, §C14, §V20]
