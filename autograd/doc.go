// Package autograd is layer L2: reverse-mode automatic differentiation.
//
// It provides the tape/graph, Variable, and Backward(), intercepting the op
// dispatch exposed by package backend so L1 ops need no rewrite (§T13). Each
// differentiable op supplies a VJP rule validated by numeric gradient checking
// (§V2).
package autograd
