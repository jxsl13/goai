// Package format is layer L5: model interop and serialization. Pure-Go, no cgo.
//
// Subpackages: safetensors (read/write, the checkpoint format — GPT.Safetensors
// round-trips bit-identically, §T435), gguf (read/write incl. quantized ggml
// tensor blocks and quantized matmul; LlamaFromGGUF/LlamaToGGUF round-trip,
// hostile-input hardened), and npy/npz (NumPy array interop, fuzz-tested).
// Every reader is validated against files produced by the reference tooling and
// survives §V15 round-trip + hostile/fuzz tests.
//
// In plain terms: reading and writing the file formats models are shipped in, so weights trained anywhere can be loaded here and vice versa.
//
// Further reading: the safetensors and GGUF format specifications (huggingface/safetensors, ggml-org) — the defining references for this layer (file formats have no paper, SPEC V16).
package format
