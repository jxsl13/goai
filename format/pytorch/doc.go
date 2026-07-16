// Package pytorch loads PyTorch checkpoints (the .pt / .bin files produced by
// torch.save of a state_dict) into GoAI tensors — WITHOUT executing code.
//
// # For the AI professional
//
// A .pt file is a ZIP archive holding a Python pickle (data.pkl) that describes
// the tensor structure, plus the raw storage bytes (data/N). Python's
// torch.load runs that pickle through a full unpickler, a well-known
// arbitrary-code-execution vector: any GLOBAL/REDUCE opcode can invoke an
// arbitrary callable, so loading an untrusted checkpoint can run code. This
// loader instead interprets only the bounded opcode subset a state_dict emits
// and resolves GLOBAL / REDUCE against a strict whitelist — the tensor-rebuild
// function torch._utils._rebuild_tensor_v2, the typed storage classes, and
// collections.OrderedDict. Any other global, reduce, or opcode is an error, so a
// hostile checkpoint cannot execute code (verified by a fuzz target and a
// code-execution-rejection test), matching the safetensors guarantee.
//
// [Load] / [LoadFile] return map[string]*tensor.Tensor keyed by the checkpoint's
// parameter names (nested modules flattened with '.') — the same shape the model
// builders in package nlp consume, so a Hugging Face checkpoint saved as .bin
// flows straight into e.g. nlp.GPT2FromHF. Storage dtypes map exactly as in
// package safetensors: float types are verbatim (F32/F64/F16/BF16), integer and
// bool storages widen (to F32, or F64 for 32/64-bit ints). The common
// contiguous, offset-0 state_dict layout is supported; non-contiguous views and
// shared-storage aliasing are reported as errors rather than mis-read.
//
// # For everyone else
//
// PyTorch is the most common way models are shared, and its save format (.pt or
// .bin) is a little archive with the model's numbers inside — but opening one the
// normal way can secretly run a program hidden in the file, which is a real
// security risk. This package opens those files the safe way: it reads the
// numbers and refuses anything that looks like hidden code. So you can load a
// downloaded PyTorch model into GoAI without trusting whoever made it.
//
// Further reading: the PyTorch serialization docs (torch.save/torch.load) and
// the pickle format specification; the safety approach mirrors the "weights-only"
// loading motivation behind safetensors (§R7).
//
// In plain terms: load PyTorch model files into GoAI without the code-execution
// risk of torch.load.
package pytorch
