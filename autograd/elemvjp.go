package autograd

import "github.com/jxsl13/goai/tensor"

// elemVJP runs step over n elements with CONTIGUOUS f64 slices, taking the tensors' own storage
// when every one of them is already F64 and materializing scratch through the accessors when any
// is not.
//
// The preference-optimization backwards are all the same shape — read one value from each input
// at i, write one to each output — and every one of them went through AtF64/SetF64, a shape walk
// and a storage-type switch per element, which on the flat tensors these rules use is the whole
// cost of the access.
//
// STEP GETS SLICES, NOT ELEMENTS, and that is the point. An earlier version called step once per
// element with the values in scratch arrays; it worked and it cost half the win — CPO measured
// 1888.6 to 1208.8 microseconds that way against 1888.6 to 817.2 with the rule writing its own
// loop over the slices. The per-element closure and the two inner index loops are not free.
//
// THE ACCESSOR ARM IS NOT OPTIONAL: an output tensor takes the dtype of its input, so a rule
// called with F32 operands must still work, and its result is stored as float32. The two arms
// are therefore "the same float64 computation rounded once", not equal bits.
func elemVJP(n int, ins, outs []*tensor.Tensor, step func(in, out [][]float64, n int)) {
	is := make([][]float64, len(ins))
	os := make([][]float64, len(outs))
	typed := true
	for k, t := range ins {
		if t.Dtype() != tensor.F64 {
			typed = false
			break
		}
		is[k] = t.Contiguous().Storage().F64()
	}
	if typed {
		for k, t := range outs {
			if t.Dtype() != tensor.F64 {
				typed = false
				break
			}
			os[k] = t.Storage().F64()
		}
	}
	if typed {
		step(is, os, n)
		return
	}
	// Fallback: materialize through the accessors, run the SAME step, scatter back. One step
	// implementation serves both arms, so they cannot drift apart.
	for k, t := range ins {
		buf := make([]float64, n)
		for i := range n {
			buf[i] = t.AtF64(i)
		}
		is[k] = buf
	}
	for k := range outs {
		os[k] = make([]float64, n)
	}
	step(is, os, n)
	for k, t := range outs {
		for i := range n {
			t.SetF64(os[k][i], i)
		}
	}
}
