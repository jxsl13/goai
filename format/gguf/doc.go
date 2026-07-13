// Package gguf reads the GGUF file format — the model container used by
// llama.cpp / ggml for quantized large language models.
//
// # For AI practitioners
//
// A GGUF file has three parts: a small header, a block of metadata key/value
// pairs (architecture, hyperparameters, tokenizer, and arbitrary typed scalars and
// arrays), and the tensor data. Tensor dimensions are stored fastest-varying
// first, so this reader reverses them to the row-major [Shape] the rest of GoAI
// uses. [Read]/[ReadFile] parse a stream into a [File] holding the version, the
// metadata map, and every tensor.
//
// Weights are usually quantized. On read, F16 is widened and the block-quantized
// types Q8_0 and Q4_0 are dequantized to F32, so [File.Tensors] always hands back
// ready-to-compute F32 tensors — verified block-for-block against the ggml
// reference (§V1, §R19/§R21). For inference that must stay compressed, [QMatMul]
// multiplies by a quantized weight while dequantizing ONE ROW at a time, so a
// quantized model runs without ever materializing the full-precision weight matrix
// (§T39) — the memory payoff of quantization (~3.8× for Q8_0, ~7.1× for Q4_0).
//
// The parser is written to be safe on hostile input (§V15): string and array
// lengths are bounded, metadata array nesting is depth-capped so a crafted file
// cannot blow the stack, and every malformed stream returns an error rather than
// panicking. [Write]/[WriteFile] serialize a [File] back to GGUF, writing tensors as
// F32 (the lossless in-memory form Read produces) with metadata types and alignment
// preserved, so Read→Write→Read round-trips exactly and is fuzzed (§V15). The
// definitional source of truth is the ggml/gguf reference implementation, not a paper.
//
// # For everyone else
//
// A large language model is gigabytes of numbers. To make it fit on a laptop those
// numbers are "quantized" — stored at low precision (as little as 4 bits each)
// with a scale factor per small block, like saving a photo as a smaller JPEG.
// GGUF is the standard file that packages such a compressed model together with a
// description of it (which architecture, how big, what tokenizer). This package
// opens those files: it reads the description, unpacks the compressed numbers back
// into ordinary decimals, and hands them over ready to use — while refusing to
// crash on a corrupt or malicious file.
//
// The runnable Example functions below open the small test model shipped with the
// package and read its metadata and tensors.
package gguf
