package ops

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Einsum evaluates the Einstein-summation contraction described by spec over the
// operands (numpy.einsum). spec is a comma-separated list of per-operand index
// subscripts, optionally followed by "->" and the output subscript, e.g.
// "ij,jk->ik" (matmul), "bij,bjk->bik" (batched matmul), "ij->ji" (transpose),
// "ii->" (trace), "i,j->ij" (outer), "ij->i" (row sum), "ij,ij->" (Frobenius dot).
// Each subscript letter is an index; a letter repeated across operands (or within
// one, as in "ii") is contracted/aliased, and any index not in the output is summed
// over. With no "->", the output is the indices that appear exactly once, in
// alphabetical order (numpy's implicit mode). Indices are single lowercase letters;
// ellipsis (...) is not supported.
//
// It is differentiable for contractions, reductions (e.g. "ij->i", "ij->", a
// summed-over index carries a broadcast gradient) AND single-operand diagonals/traces
// ("ii->i", "ii->", whose gradient scatters back onto the diagonal). Only a repeated
// index shared across MULTIPLE operands, or a repeated OUTPUT index (which
// diagonalizes the output), are not yet differentiable and error on the backward
// pass; the forward handles them.
func Einsum(spec string, operands ...*tensor.Tensor) (*tensor.Tensor, error) {
	return eagerA(backend.OpEinsum, backend.EinsumAttrs{Spec: spec}, operands...)
}
