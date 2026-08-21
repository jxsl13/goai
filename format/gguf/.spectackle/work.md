---
schema: v1
---

## P-01M0JBW0SVETY8QC1HZV6G561V Open quantized GGUF files through retained read-only mappings
kind: proposal
state: active
created: 2026-08-21
grilled: 2026-08-21 open=0
targets: go:gguf.ReadRaw, go:gguf.parseMapped, format/gguf/mmap_unix.go, format/gguf/mmap_other.go

Context: T-01KYJNDV2VFRFR5RDBSPKJBRVK shipped eager ReadFile mmap and ARM64 eager-dequant NEON, but quantized inference still opens a file and calls ReadRaw, which allocates and copies the full encoded section before keeping read-only tensor views. Add an explicit closable file API that maps a regular non-empty GGUF, parses metadata and tensor extents in place, retains the mapping for QuantTensor lifetime, and releases it exactly once on Close. Unsupported platforms and mmap failures use the existing buffered ReadRaw semantics. Existing ReadRaw and ReadFile remain source- and behavior-compatible. Correctness gates: mapped/buffered metadata, tensor shape/type/bytes and dequantized values match; tensor capacities remain clamped; malformed input releases the mapping; Close is idempotent; non-Unix cross-compile and CGO-disabled tests pass. Performance gate: fresh-process, order-alternated Apple M2 real TinyLlama Q4_K_M A/B against buffered ReadRaw-file loading, at least 1.25x lower median latency and at least one model-sized heap allocation removed, with complete pinned evidence. API naming must signal ownership and Close lifetime; no finalizer may unmap while tensor slices survive.
