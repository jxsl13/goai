package backend

import (
	"fmt"
	"slices"
	"strings"

	"github.com/jxsl13/goai/tensor"
)

// EinsumAttrs parameterises the einsum op (OpEinsum): the subscript spec, e.g.
// "ij,jk->ik".
type EinsumAttrs struct {
	Spec string // Einstein-summation subscripts (inputs comma-separated, optional "->" output)
}

func (EinsumAttrs) opAttrs() {}

// ParseEinsum splits an einsum spec into per-operand input subscripts and the
// output subscript, deriving the implicit output (indices appearing exactly once,
// alphabetical) when "->" is absent. Indices must be single lowercase letters;
// ellipsis is rejected. Shared by the einsum kernel and its VJP.
func ParseEinsum(spec string, nOperands int) (inSubs [][]byte, outSub []byte, err error) {
	spec = strings.ReplaceAll(spec, " ", "")
	if strings.Contains(spec, "...") {
		return nil, nil, fmt.Errorf("backend: einsum ellipsis (...) not supported")
	}
	lhs, rhs, explicit := strings.Cut(spec, "->")
	parts := strings.Split(lhs, ",")
	if len(parts) != nOperands {
		return nil, nil, fmt.Errorf("backend: einsum spec has %d input subscripts but %d operands", len(parts), nOperands)
	}
	counts := map[byte]int{}
	for _, p := range parts {
		for i := range len(p) {
			c := p[i]
			if c < 'a' || c > 'z' {
				return nil, nil, fmt.Errorf("backend: einsum index %q must be a lowercase letter", string(c))
			}
			counts[c]++
		}
		inSubs = append(inSubs, []byte(p))
	}
	if explicit {
		for i := range len(rhs) {
			if rhs[i] < 'a' || rhs[i] > 'z' {
				return nil, nil, fmt.Errorf("backend: einsum output index %q must be a lowercase letter", string(rhs[i]))
			}
		}
		return inSubs, []byte(rhs), nil
	}
	var once []byte
	for c, n := range counts {
		if n == 1 {
			once = append(once, c)
		}
	}
	slices.Sort(once)
	return inSubs, once, nil
}

// EinsumContract evaluates the parsed einsum contraction over the operands: for
// every distinct-index assignment it accumulates the product of the operands'
// elements into the output position given by outSub. Index sizes are validated for
// consistency across positions. This is the shared engine used by the forward
// kernel and (via constructed gradient specs) the VJP.
func EinsumContract(inSubs [][]byte, outSub []byte, operands []*tensor.Tensor) (*tensor.Tensor, error) {
	// Subscript indices are BYTES, so a fixed [256]int is a dense, exact replacement
	// for the map — no hashing, no allocation, and the bound is the key type itself.
	// sizeSet distinguishes "index absent" from "size 0", which the map's comma-ok
	// gave for free.
	var size [256]int
	var sizeSet [256]bool
	for k, sub := range inSubs {
		if len(sub) != operands[k].Ndim() {
			return nil, fmt.Errorf("backend: einsum subscript %q has %d indices but operand %d is %dD", sub, len(sub), k, operands[k].Ndim())
		}
		for pos := range len(sub) {
			d := operands[k].Shape()[pos]
			if have, ok := size[sub[pos]], sizeSet[sub[pos]]; ok && have != d {
				return nil, fmt.Errorf("backend: einsum index %q has inconsistent sizes %d and %d", string(sub[pos]), have, d)
			}
			size[sub[pos]], sizeSet[sub[pos]] = d, true
		}
	}
	for i := range len(outSub) {
		if !sizeSet[outSub[i]] {
			return nil, fmt.Errorf("backend: einsum output index %q does not appear in any operand", string(outSub[i]))
		}
	}
	// order: output indices first, then the summed ones
	order := append([]byte(nil), outSub...)
	seen := map[byte]bool{}
	for i := range len(outSub) {
		seen[outSub[i]] = true
	}
	for _, sub := range inSubs {
		for i := range len(sub) {
			if !seen[sub[i]] {
				seen[sub[i]] = true
				order = append(order, sub[i])
			}
		}
	}
	outShape := make(tensor.Shape, len(outSub))
	for i := range len(outSub) {
		outShape[i] = size[outSub[i]]
	}
	out := tensor.New(operands[0].Dtype(), outShape)
	total := 1
	for _, ix := range order {
		total *= size[ix]
	}
	// Same treatment for the per-combination index assignment: read and written once
	// per index per combination inside the O(total) contraction loop, so it was the
	// hottest map in the engine.
	var val [256]int
	// Typed fast path: when every operand and the output are F64, index the
	// contiguous backing slices with precomputed row-major strides instead of the
	// per-combination variadic AtF64/SetF64 dispatch below (each of which recomputes
	// the flat offset and bounds-checks). Same ascending-combo accumulation order and
	// identical products, so the result is bit-identical to the generic path.
	allF64 := out.Dtype() == tensor.F64
	for _, op := range operands {
		if op.Dtype() != tensor.F64 {
			allF64 = false
			break
		}
	}
	if allF64 {
		outData := out.Storage().F64()
		outStride := tensor.RowMajorStrides(outShape)
		opData := make([][]float64, len(inSubs))
		opStride := make([]tensor.Strides, len(inSubs))
		for k := range inSubs {
			c := operands[k].Contiguous()
			opData[k] = c.Storage().F64()
			opStride[k] = tensor.RowMajorStrides(c.Shape())
		}
		for combo := range total {
			rem := combo
			for _, ix := range order {
				val[ix] = rem % size[ix]
				rem /= size[ix]
			}
			prod := 1.0
			for k, sub := range inSubs {
				st := opStride[k]
				off := 0
				for pos := range len(sub) {
					off += val[sub[pos]] * st[pos]
				}
				prod *= opData[k][off]
			}
			of := 0
			for i := range len(outSub) {
				of += val[outSub[i]] * outStride[i]
			}
			outData[of] += prod
		}
		return out, nil
	}
	coords := make([][]int, len(inSubs))
	for k := range inSubs {
		coords[k] = make([]int, len(inSubs[k]))
	}
	outCoord := make([]int, len(outSub))
	for combo := range total {
		rem := combo
		for _, ix := range order {
			val[ix] = rem % size[ix]
			rem /= size[ix]
		}
		prod := 1.0
		for k, sub := range inSubs {
			for pos := range len(sub) {
				coords[k][pos] = val[sub[pos]]
			}
			prod *= operands[k].AtF64(coords[k]...)
		}
		for i := range len(outSub) {
			outCoord[i] = val[outSub[i]]
		}
		out.SetF64(out.AtF64(outCoord...)+prod, outCoord...)
	}
	return out, nil
}
