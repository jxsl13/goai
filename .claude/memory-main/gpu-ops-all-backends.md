---
name: gpu-ops-all-backends
description: "GPU ops must be implemented+tested on ALL host-supported backends (metal, vulkan, cpu), never metal-only."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 56de00c6-6c80-4734-a46b-f5e03083b2b4
---

When adding a GPU/accelerator op, implement AND test it on every backend the
current host supports — metal, vulkan, and cpu — not just Metal. Vulkan must
explicitly NOT be skipped.

**Why:** On 2026-07-06 the user said "vulkan, metal und cpu sind alle auf dem
aktuellen host supportet. also implementiere und teste alle diese varianten.
vulkan soll explizit nicht ausgelassen werden." Iterations T85/T86/T88 had added
attention/conv to Metal only, leaving Vulkan stuck at matmul.

**How to apply:** This host (Apple Silicon) runs Vulkan via MoltenVK — `glslc` is in
PATH, `make vulkan-test` runs the tag-gated cgo tests with
`VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json`. Adding a
Vulkan op = write a `shaders/<op>.comp` GLSL kernel, `glslc`→`.spv` (add to the
`vulkan-spv` Makefile target), commit + `//go:embed` it, add an op-specific
`vk_<op>_f32` in `vk_bridge.c` (mirror `vk_matmul_f32`/`vk_mha_f32`: storage
buffers + push-constant block + dispatch), wire the Go kernel in `vulkan.go`.
Metal uses inline MSL in `metal_bridge.m`; cpu usually routes to ref unless a SIMD
kernel exists. Verify each with V-CROSS vs the reference (§V3/§V11) on-device.

**Confirmed actionable (2026-07-10):** BOTH `go test -tags vulkan` and `-tags metal`
BUILD AND RUN green on this host — GPU is the user's #1 priority and is NOT blocked
(only T11b archsimd-amd64 is). Don't skip GPU work thinking it's blocked. The generic
`vk_dispatch(spv, nbuf, lens, data, up/down, push, gx,gy,gz)` helper NOW EXISTS in
`vk_bridge.c` — a new Vulkan op's C wrapper is ~15 lines (see `vk_rmsnorm_f32`). Metal
adds an MSL `NSString` + `ensure_<op>` pipeline + `mtl_<op>` wrapper (~70 lines, mirror
`mtl_qmatmul_q8_0`). GPU-accelerated so far: matmul, qmatmul(Q2K–Q8), MHA(+bwd),
FlashAttn(fwd), Conv2D(+bwd), Retention(+bwd), RMSNorm (§T324, one-thread-per-row), and
RoPE (§T325, one-thread-per-pair; host-precomputes RoPEFreqs, GPU does only the rotation),
Softmax (§T326, per-row max-shift), LayerNorm (§T327, per-row two-pass mean/var + β), and
9 unary elementwise ops (§T328, one generic `switch(op)` kernel: Neg/Exp/Log/Tanh/ReLU/
Sigmoid/SiLU/Sqrt/Abs), and 6 same-shape binary ops (§T329, one generic kernel: Add/Sub/Mul/
Div/Max/Min; broadcasting→ref fallback). GELU stays on ref (exact erf → no GLSL primitive).
(metal kernel named `binaryop` defensively.) With §T324-329 a whole transformer layer
(norm+rope+attn+ffn+residual) runs GPU-resident for INFERENCE.

GPU TRAINING (§T330+): the RoPE/RMSNorm/LayerNorm VJPs were hand-rolled Go loops IGNORING the
tape ctx → CPU-pinned. Fix = the mhaVJP pattern: add OpXxxBackward (enum+name+ref kernel), make
the VJP `backend.Execute(ctx, OpXxxBackward, ...)` so it runs on the active backend, add GPU
kernels. Done for RoPE backward (§T330; inverse rotation = forward with 2 output lines flipped;
vulkan REUSES vk_rope_f32 with rope_bwd.spv) and RMSNorm backward (§T331; per-row dx + CROSS-ROW dγ
via float atomics — GLSL `atomicAdd`+`#extension GL_EXT_shader_atomic_float`, metal `atomic_float`+
`atomic_fetch_add_explicit`; DGamma uploaded zero-init as the accumulator; vulkan gated by
vk_atomic_float()→ -7 ⇒ ref fallback). NEXT: LayerNorm backward (same pattern + dβ=Σ_rows g);
softmax/CE backward. §R35 = LayerNorm-bwd formula. Done LayerNorm-bwd too (§T332; 2 atomic
accumulators dγ+dβ + mean-sub; op takes (x,gamma,g) since β's value is unused). GPU TRAINING now:
matmul/MHA/FlashAttn/Conv2D/Retention/RoPE/RMSNorm/LayerNorm/CrossEntropy/Embed backward. §T334
(embed-bwd, scatter-add via float atomics, one thread per (token,dim)) was the LAST CPU-pinned VJP →
a FULL transformer training step now runs GPU-resident end-to-end on both backends. NEXT: a
GPU-resident training-step E2E test (fwd+bwd, weights updated on-device); softmax standalone backward;
GELU erf-poly for vulkan. Pattern for a new backward op = enum+name + ref kernel (move loop out of
VJP) + VJP→backend.Execute(ctx, OpXxxBackward,...) + GPU kernels; cross-row reductions/scatter → float
atomics (accumulator uploaded zero-init; vulkan gated by vk_atomic_float()→-7⇒ref).
GOTCHA: `half` is a GLSL reserved word AND a Metal built-in type — name that var `halfd`.
Shader-compile failures surface as return code -6. Each op ~1 iteration: GLSL→glslc→.spv
(+Makefile) + `vk_<op>` C wrapper over `vk_dispatch` + Go dispatch; metal MSL NSString +
`ensure_<op>` + `mtl_<op>` + Go; V-CROSS parity vs ref (rtol=1e-6·√dim for reductions, 1e-4/1e-5
for cos-sin/exp). NEXT GPU targets: elementwise BINARY Mul/Add (closes the FFN's SiLU⊙up → fully-resident FFN);
GELU via erf polynomial; then RMSNorm/LayerNorm/attention BACKWARD for GPU training (§R35 = LayerNorm-bwd).
See [[goai-autonomous-loop]].
