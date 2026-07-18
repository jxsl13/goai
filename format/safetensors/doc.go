// Package safetensors reads and writes the safetensors file format — the storage
// container Hugging Face uses for model weights.
//
// # For AI practitioners
//
// A safetensors file is a tiny JSON header followed by a flat block of raw tensor
// bytes. The header maps each tensor name to its dtype, shape, and a [start,end)
// byte range in the block; there is no pickling and no code execution, which is
// the format's whole point — loading an untrusted checkpoint cannot run arbitrary
// code the way a PyTorch .bin (Python pickle) can. The layout also makes loads
// cheap: names and shapes are read from the small header, and tensor data is a
// contiguous slice that can be memory-mapped or copied directly.
//
// This package implements both directions and is verified against the official
// Python library (§V1): [Load]/[LoadFile] parse a file into a name→tensor map plus
// the string metadata, and [Save]/[SaveFile] write one back. The writer is
// deterministic (§V13) — tensor names are sorted, data is laid out at contiguous
// offsets, and the header is padded to an 8-byte boundary with spaces — so the
// same tensors always produce byte-identical output. Round-trip and
// malformed-input fuzzing (§V15) guarantee that Save∘Load is the identity and that
// no hostile byte stream can panic (every error is returned, never thrown).
//
// Read and write support f32, f64, f16 and bf16 — the 16-bit floats pass
// through as their raw bits, round-tripping bit-exactly with no widening.
// Loading additionally accepts the remaining dtypes of the official spec
// (FP8, bool, and the integer families) by widening them to f32/f64 (§T577);
// shapes are arbitrary. Higher layers build on
// this: nlp.FromSafetensors loads a GPT straight from a checkpoint's tensor map.
//
// Checkpoints above Hugging Face's ~5 GB shard budget ship as several numbered
// files plus a model.safetensors.index.json table of contents; the package
// covers both directions of that form too. [LoadSharded] reads such a
// checkpoint through its index and returns the same map [LoadFile] would have
// for a single file, and [SaveSharded] writes one, splitting exactly as
// huggingface_hub's split_torch_state_dict_into_shards does (same 5 GB decimal
// default, same shard naming, same index shape, same plain-single-file
// fallback when everything fits in one shard) — so a GoAI export is
// indistinguishable from one written by transformers save_pretrained.
//
// # For everyone else
//
// A trained model is just a big pile of numbers ("weights"). Those numbers have to
// be saved to disk and loaded back exactly, on any machine. safetensors is a
// deliberately boring, safe box for them: a short table of contents (which numbers
// are where, and their shape) followed by the numbers themselves. "Safe" is
// literal — older formats could smuggle executable code into a weights file, so
// merely opening a downloaded model could run a program on your computer;
// safetensors cannot, because it stores only numbers and a description of them.
//
// The runnable Example functions below show saving weights to an in-memory buffer,
// loading them back, and confirming the numbers survive the round trip unchanged.
package safetensors
