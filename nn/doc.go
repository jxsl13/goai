// Package nn is layer L3: neural-network building blocks.
//
// Layers, parameter initialization (xavier/kaiming), losses, optimizers
// (SGD, Adam), and the data pipeline. Built on autograd (L2) and backend (L1);
// it never imports backend internals (§I1).
package nn
