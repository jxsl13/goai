---
schema: v1
---

## T-01KYJ977Z3FE6BHXW1HFRM6CBS Cut the residual serial cost in gguf ReadFile
kind: task
state: approved
created: 2026-07-27
targets: format/gguf

---
schema: v1
---







After the parallel decode landed, ReadFile of the 1.1B model is about 323ms, and the residual is the serial 668MB read-into-heap plus parse. The dequantization itself now parallelizes nearly fully and is bandwidth-bound on the 4.4GB f32 output.

Three independent levers, each a dedicated fresh-context piece of work rather than a tail-end add-on, since each is assembly, platform syscall, or architectural:

(1) Zero-copy mmap: ReadFile maps the file (syscall.Mmap on unix, CreateFileMapping on windows, behind build-tagged mmap_unix.go / mmap_windows.go / mmap_other.go) and parse and decode read directly from the mapped bytes rather than an io.ReadAll heap copy. Requires restructuring the parser to reference the mapped data section rather than copy it, and munmap after decode. decodeTensor already copies into its own tensor storage, so no long-lived mapping is needed. Saves the 668MB allocation and copy plus the peak memory.

(2) SIMD Q4_K and Q6_K dequantization: the scalar path is already invariant-hoisted and destination-direct (measured 745 MB/s for Q4_K), but the inner nibble-unpack and FMA loop is NEON-vectorizable using the established word-encoded Plan9 assembly pattern, arch-gated and f32-parity-tested. Would approach llama.cpp's dequantization speed.

(3) Deferred dequantization: return quantized tensors rather than eager f32, avoiding the 4.4GB write and the 4x memory, matching llama.cpp's quantized-inference model. Architectural, touches model load.

Migrated from cavekit SPEC.md T914.
