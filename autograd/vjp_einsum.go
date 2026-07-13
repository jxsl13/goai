package autograd

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Einsum VJP (the einsum-swap rule): for C = einsum(s₁,…,sₙ → s_O), the gradient
// w.r.t. operand k is itself an einsum with the OTHER operands and the upstream
// gradient g (taking the output subscript), contracted down to operand k's
// subscript — dAₖ = einsum(s₁,…(skip k)…,sₙ,s_O → sₖ; A…(skip k)…, g).
//
// Two subscript shapes need extra handling, both exact and definitional (§R143):
//   - a REDUCTION index of sₖ (present in k alone, summed out — the "j" of "ij->i")
//     carries a gradient that is constant (broadcast) along it, reconstructed by
//     injecting a ones vector of the right size/subscript so the swap-rule
//     contraction yields dAₖ[…,c,…]=g[…]·1[c];
//   - a REPEATED letter in sₖ (a diagonal/trace like "ii->i" or "ii->") reads A_k on
//     its generalized diagonal, so the gradient is the collapsed (de-duplicated)
//     gradient SCATTERED back onto that diagonal (zero off it).
//
// Still unsupported (error on backward, forward works): a repeated index shared
// across MULTIPLE operands, and a repeated OUTPUT index (which diagonalizes the
// output) — both would need a cross-operand diagonal scatter.
func init() {
	RegisterVJP(backend.OpEinsum, func(_ *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		pa, _ := attrs.(backend.EinsumAttrs)
		inSubs, outSub, err := backend.ParseEinsum(pa.Spec, len(in))
		if err != nil {
			return nil, err
		}
		if hasDupLetters(outSub) {
			return nil, fmt.Errorf("autograd: einsum gradient with a repeated output index (%q) not supported", outSub)
		}
		nDup := 0
		for _, s := range inSubs {
			if hasDupLetters(s) {
				nDup++
			}
		}
		if nDup > 0 && len(in) > 1 {
			return nil, fmt.Errorf("autograd: einsum gradient with a repeated index shared across operands (%q) not supported", pa.Spec)
		}
		grads := make([]*tensor.Tensor, len(in))
		for k := range in {
			// dedupK collapses repeated letters of sₖ to their first occurrence; the
			// swap rule is computed over dedupK, then scattered onto the diagonal.
			dedupK, firstPos := dedupSub(inSubs[k])
			gInSubs := make([][]byte, 0, len(in)+len(dedupK))
			gOperands := make([]*tensor.Tensor, 0, len(in)+len(dedupK))
			avail := map[byte]bool{}
			for m := range in {
				if m == k {
					continue
				}
				gInSubs = append(gInSubs, inSubs[m])
				gOperands = append(gOperands, in[m])
				for _, c := range inSubs[m] {
					avail[c] = true
				}
			}
			gInSubs = append(gInSubs, outSub)
			gOperands = append(gOperands, g)
			for _, c := range outSub {
				avail[c] = true
			}
			// Reduction indices of dedupK (summed out) get a broadcast gradient: inject
			// a ones vector of the right size and subscript.
			for _, c := range dedupK {
				if !avail[c] {
					ones := tensor.Ones(in[k].Dtype(), tensor.Shape{in[k].Shape()[firstPos[c]]})
					gInSubs = append(gInSubs, []byte{c})
					gOperands = append(gOperands, ones)
					avail[c] = true
				}
			}
			gDedup, err := backend.EinsumContract(gInSubs, dedupK, gOperands)
			if err != nil {
				return nil, err
			}
			if len(dedupK) == len(inSubs[k]) {
				grads[k] = gDedup // no repeat: dedupK == sₖ
			} else {
				grads[k] = scatterDiag(gDedup, inSubs[k], in[k].Shape(), dedupK, firstPos, in[k].Dtype())
			}
		}
		return grads, nil
	})
}

// dedupSub returns the distinct letters of s in first-appearance order and a map
// from each letter to its first position in s.
func dedupSub(s []byte) (dedup []byte, firstPos map[byte]int) {
	firstPos = make(map[byte]int, len(s))
	for pos, c := range s {
		if _, ok := firstPos[c]; !ok {
			firstPos[c] = pos
			dedup = append(dedup, c)
		}
	}
	return dedup, firstPos
}

// scatterDiag builds dA (shape akShape) from the collapsed gradient gDedup (over
// dedupK): dA[idx] = gDedup[collapsed(idx)] when idx lies on the generalized diagonal
// (every position sharing a letter holds the same coordinate), else 0.
func scatterDiag(gDedup *tensor.Tensor, sk []byte, akShape tensor.Shape, dedupK []byte, firstPos map[byte]int, dt tensor.Dtype) *tensor.Tensor {
	dA := tensor.New(dt, akShape)
	coord := make([]int, len(dedupK))
	for pos := range akShape.Numel() {
		idx := tensor.Unravel(pos, akShape)
		onDiag := true
		for p, c := range sk {
			if idx[p] != idx[firstPos[c]] { // occurrences of c must share a coordinate
				onDiag = false
				break
			}
		}
		if !onDiag {
			continue // off the diagonal ⇒ zero gradient
		}
		for di, c := range dedupK {
			coord[di] = idx[firstPos[c]]
		}
		dA.SetF64(gDedup.AtF64(coord...), idx...)
	}
	return dA
}

func hasDupLetters(s []byte) bool {
	var seen [26]bool
	for _, c := range s {
		if c >= 'a' && c <= 'z' {
			if seen[c-'a'] {
				return true
			}
			seen[c-'a'] = true
		}
	}
	return false
}
