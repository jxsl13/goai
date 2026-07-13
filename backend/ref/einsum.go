package ref

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// einsumKernel evaluates the Einstein-summation contraction of its operands per the
// spec in attrs (numpy.einsum). Pure tensor contraction on the reference/cpu (the
// GPU backends fall back, §I4).
func einsumKernel(_ *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) == 0 {
		return nil, fmt.Errorf("ref: einsum needs ≥1 operand")
	}
	pa, _ := attrs.(backend.EinsumAttrs)
	inSubs, outSub, err := backend.ParseEinsum(pa.Spec, len(in))
	if err != nil {
		return nil, err
	}
	out, err := backend.EinsumContract(inSubs, outSub, in)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpEinsum, tensor.F32, einsumKernel)
	std.add(backend.OpEinsum, tensor.F64, einsumKernel)
}
