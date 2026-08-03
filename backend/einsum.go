package backend

import (
	"fmt"
	"github.com/jxsl13/goai/internal/parallel"
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
	// The contraction is a sum over the "summed" indices for EACH independent output position
	// (order lists output indices first, so combo = outPos + outTotal·sumPos). Output positions
	// therefore parallelize over disjoint writes; the summed loop stays serial per output to keep
	// the ascending-combo accumulation order — bit-identical to the serial contraction.
	outTotal := 1
	for i := range outShape {
		outTotal *= outShape[i]
	}
	if total == 0 || outTotal == 0 {
		return out, nil
	}
	sumTotal := total / outTotal
	sumOrder := order[len(outSub):]
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
		parallel.Rows(outTotal, func(olo, ohi int) {
			var val [256]int
			for oc := olo; oc < ohi; oc++ {
				rem := oc
				of := 0
				for i := range outSub {
					v := rem % size[outSub[i]]
					rem /= size[outSub[i]]
					val[outSub[i]] = v
					of += v * outStride[i]
				}
				var acc float64
				for sc := 0; sc < sumTotal; sc++ {
					r := sc
					for _, ix := range sumOrder {
						val[ix] = r % size[ix]
						r /= size[ix]
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
					acc += prod
				}
				outData[of] = acc
			}
		})
		return out, nil
	}
	// F32 twin of the fast path above: F32 operands otherwise fall through to the
	// variadic AtF64/SetF64 loop (common on inference). The generic path reads each
	// operand via AtF64 (exact widen to f64), accumulates prod in f64, and writes the
	// output with SetF64 — which NARROWS to f32 after every combination. Reproduce that
	// exactly: prod in f64 from f64(opF32), and out[of] = f32(f64(out[of]) + prod) per
	// combo (same ascending order, same per-combo narrowing) → bit-identical.
	allF32 := out.Dtype() == tensor.F32
	for _, op := range operands {
		if op.Dtype() != tensor.F32 {
			allF32 = false
			break
		}
	}
	if allF32 {
		outData := out.Storage().F32()
		outStride := tensor.RowMajorStrides(outShape)
		opData := make([][]float32, len(inSubs))
		opStride := make([]tensor.Strides, len(inSubs))
		for k := range inSubs {
			c := operands[k].Contiguous()
			opData[k] = c.Storage().F32()
			opStride[k] = tensor.RowMajorStrides(c.Shape())
		}
		parallel.Rows(outTotal, func(olo, ohi int) {
			var val [256]int
			for oc := olo; oc < ohi; oc++ {
				rem := oc
				of := 0
				for i := range outSub {
					v := rem % size[outSub[i]]
					rem /= size[outSub[i]]
					val[outSub[i]] = v
					of += v * outStride[i]
				}
				var acc float32
				for sc := 0; sc < sumTotal; sc++ {
					r := sc
					for _, ix := range sumOrder {
						val[ix] = r % size[ix]
						r /= size[ix]
					}
					prod := 1.0
					for k, sub := range inSubs {
						st := opStride[k]
						off := 0
						for pos := range len(sub) {
							off += val[sub[pos]] * st[pos]
						}
						prod *= float64(opData[k][off])
					}
					acc = float32(float64(acc) + prod)
				}
				outData[of] = acc
			}
		})
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
