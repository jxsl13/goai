// Package ops is the eager, functional API over GoAI's tensor operations: each
// function runs one kernel immediately and returns a new tensor, with no
// autograd tape involved.
//
// # For AI practitioners
//
// Every function here is a thin wrapper that dispatches an op on the active
// [backend] (auto-selected: the fastest registered accelerator, reference on CPU
// as the guaranteed fallback) and returns (result, error). It is the imperative
// counterpart to the autograd path: use ops for inference, data preprocessing,
// metrics, and quick numerical work where you do not need gradients; use an
// autograd.Tape when you do. The two share the exact same kernels, so results are
// bit-identical.
//
// The surface mirrors the kernel set: elementwise binary ([Add], [Sub], [Mul],
// [Div]) and unary ([Neg], [Exp], [Log], [Tanh], [ReLU], [GELU], [Sigmoid],
// [SiLU]); reductions ([Sum], [Mean], [Max], [Min], [ArgMax]); linear algebra
// ([MatMul], [Dot], [Nrm2], [Axpy], [AddBias]); the neural-network primitives
// ([Softmax], [LayerNorm], [RMSNorm], [RoPE]); and 2-D vision ops ([Conv2D],
// [MaxPool2D], [AvgPool2D]). Reductions accumulate in f64 (§V10); each op carries
// the same golden-tested numerical guarantees as the underlying kernel.
//
// # For everyone else
//
// This package is the calculator layer: it does one math operation at a time on
// arrays of numbers — add them, multiply two matrices, run the "softmax" that
// turns scores into probabilities, and so on — and hands back the answer right
// away. It is the plain, do-it-now interface, as opposed to the mode that also
// records how to compute gradients for training. Both compute the same numbers.
//
// The runnable Example functions below add two vectors, multiply two matrices,
// take a softmax, and chain a few ops into a tiny neural-network layer.
package ops
