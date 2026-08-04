package nn

import (
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// State-Space Duality (SSD) — the core of Mamba-2 (Dao & Gu 2024, "Transformers
// are SSMs", arXiv:2405.21060, §T553). Mamba-2 restricts Mamba's selective SSM to
// a SCALAR per-step decay a_t, which makes the sequence transformation exactly a
// masked "attention" with a 1-semiseparable mask:
//
//	h_t = a_t·h_{t−1} + B_t·x_tᵀ            (linear-time recurrence, state [N,d])
//	y_t = C_tᵀ·h_t
//	  ⇔
//	y = M·X,  M_ij = (C_i·B_j)·Π_{k=j+1..t=i} a_k   for j ≤ i, else 0
//
// SSDRecurrent computes the left side, SSDQuadratic the right; the duality
// theorem says they are the SAME map — pinned by the parity test. a_t = 1
// degenerates to unmasked causal linear attention, a_t = 0 to a purely local
// (diagonal) map. Host f64 utilities in the RetentionRecurrent/WKV mold
// (single head; Mamba-2's multi-head form applies this per head).

// SSDRecurrent runs the linear-time scan: x[T,d], a[T] scalar decays, B[T,N],
// C[T,N] → y[T,d] with O(N·d) state.
func SSDRecurrent(x, a, b, c *tensor.Tensor) (*tensor.Tensor, error) {
	if err := ssdCheck(x, a, b, c); err != nil {
		return nil, err
	}
	T, d := x.Shape()[0], x.Shape()[1]
	n := b.Shape()[1]
	h := make([]float64, n*d) // state [N,d]
	y := tensor.New(x.Dtype(), x.Shape())
	// x_t and c_t are read across the other dimension in the O(n·d) inner loops
	// (x[t,j] once per state row i, c[t,i] once per channel j), so hoist each row once per
	// step into a contiguous buffer instead of re-dispatching AtF64. Bit-identical.
	xrow := make([]float64, d)
	crow := make([]float64, n)
	yrow := make([]float64, d) // out row accumulator (reused by the interchanged path)
	// The output y_t = Cᵀ_t·h can read the [N,d] state two ways. The natural j-outer form
	// dots down a STRIDED column h[i*d+j]; the i-outer interchange streams contiguous rows
	// but writes the accumulator row N times per step. Interchange only pays once the state
	// N·d is large enough that the strided walk thrashes cache (below ~L1 the state is
	// resident and the strided reads are free, so the extra writes make it a net loss —
	// measured regression at N·d=1024, ~1.4-1.7× win at N·d≥8192). Both paths keep the
	// ascending-i sum, so the result is bit-identical either way; this only picks the layout.
	interleave := n*d >= 4096
	for t := range T {
		at := a.AtF64(t)
		for j := range d {
			xrow[j] = x.AtF64(t, j)
		}
		for i := range n {
			crow[i] = c.AtF64(t, i)
		}
		for i := range n {
			bi := b.AtF64(t, i)
			base := i * d
			for j := range d {
				//perfscan:ignore PS3084 two-product recurrence, documented non-jammable (FMA drift)
				h[base+j] = at*h[base+j] + bi*xrow[j]
			}
		}
		if interleave {
			// i-outer: stream h's CONTIGUOUS row i into the accumulator row (AXPY, no
			// reduction dependency). Same ascending-i sum → bit-identical.
			for j := range d {
				yrow[j] = 0
			}
			// EIGHT STATE ROWS PER PASS. yrow does not vary with i, so this loop loaded and stored
			// the whole output row once per state row for one multiply-add each; eight at a time
			// hold yrow[j] in a register across eight of them. BIT-IDENTICAL: yrow[j] still sums i
			// ascending, into an EXPLICIT LOCAL rather than a compound assignment over a sum of
			// eight products, which would associate differently (T1183).
			//
			// THE STATE UPDATE ABOVE IS NOT JAMMABLE THE SAME WAY, and it was tried here before
			// being ruled out: at*h + b_i*x ADDS TWO PRODUCTS, so two fused-multiply-add
			// contractions are legal and the compiler's choice does not survive the restructuring —
			// all five digests moved. RetNet showed the same thing in T1189, where the drift
			// appeared even at a size where the jammed body never runs. This loop has ONE multiply,
			// so only one contraction is legal and the jam is safe.
			ib := 0
			//perfscan:ignore PS3066 already hand-blocked 8-way optimal output loop
			for ; ib+7 < n; ib += 8 {
				c0 := crow[ib+0]
				c1 := crow[ib+1]
				c2 := crow[ib+2]
				c3 := crow[ib+3]
				c4 := crow[ib+4]
				c5 := crow[ib+5]
				c6 := crow[ib+6]
				c7 := crow[ib+7]
				r0 := h[(ib+0)*d : (ib+0)*d+d]
				r1 := h[(ib+1)*d : (ib+1)*d+d]
				r2 := h[(ib+2)*d : (ib+2)*d+d]
				r3 := h[(ib+3)*d : (ib+3)*d+d]
				r4 := h[(ib+4)*d : (ib+4)*d+d]
				r5 := h[(ib+5)*d : (ib+5)*d+d]
				r6 := h[(ib+6)*d : (ib+6)*d+d]
				r7 := h[(ib+7)*d : (ib+7)*d+d]
				for j := range d {
					acc := yrow[j]
					acc += c0 * r0[j]
					acc += c1 * r1[j]
					acc += c2 * r2[j]
					acc += c3 * r3[j]
					acc += c4 * r4[j]
					acc += c5 * r5[j]
					acc += c6 * r6[j]
					acc += c7 * r7[j]
					yrow[j] = acc
				}
			}
			for ; ib < n; ib++ {
				ci := crow[ib]
				hrow := h[ib*d : ib*d+d]
				for j := range d {
					yrow[j] += ci * hrow[j]
				}
			}
			for j := range d {
				y.SetF64(yrow[j], t, j)
			}
		} else {
			// j-outer: small state is cache-resident, so the strided dot wins.
			//perfscan:ignore PS1006 deliberate small-state j-outer path (interchange regresses <N·d 4096)
			for j := range d {
				var s float64
				//perfscan:ignore PS3010 deliberate small-state dot, tol-gated multi-acc
				for i := range n {
					//perfscan:ignore PS6011 deliberate small-state strided path (documented)
					s += crow[i] * h[i*d+j]
				}
				y.SetF64(s, t, j)
			}
		}
	}
	return y, nil
}

// SSDQuadratic materializes the dual attention form: the [T,T] 1-semiseparable
// mask M_ij = (C_i·B_j)·Π_{k=j+1..i} a_k applied to x. O(T²) — the "transformers
// are SSMs" side of the duality; useful for verification and short sequences.
func SSDQuadratic(x, a, b, c *tensor.Tensor) (*tensor.Tensor, error) {
	if err := ssdCheck(x, a, b, c); err != nil {
		return nil, err
	}
	T, d := x.Shape()[0], x.Shape()[1]
	n := b.Shape()[1]
	y := tensor.New(x.Dtype(), x.Shape())
	// Hoist the per-element dispatches out of the O(T²)–O(T³) loops: a_t drives the decay
	// product (read O(T³) times via a.AtF64), c_i is re-read for every earlier position j,
	// and the output was a y.AtF64+SetF64 read-modify-write per (i,j,d). arow/crow are
	// contiguous rows; yrow accumulates the row locally and is written once per i. Values
	// and the ascending-j / ascending-k accumulation order are unchanged (bit-identical).
	arow := make([]float64, T)
	for t := range T {
		arow[t] = a.AtF64(t)
	}
	crow := make([]float64, n)
	yrow := make([]float64, d)
	// decayrow[j] = arow[j+1]*..*arow[i] (suffix product, rebuilt once per i in O(i))
	// so the inner (i,j) loop reads decay in O(1) instead of recomputing an O(i-j)
	// serial-dependent product T²/2 times (the dominant O(T³) un-vectorizable term).
	decayrow := make([]float64, T)
	// b_j and x_j are read inside the O(T²) (i,j) loop, so their AtF64 dispatch — ~20× a
	// bare multiply — dominates even the O(T³) (already-typed) decay product. Walk their
	// contiguous rows directly on the F64 fast path (a/c/y stay AtF64: O(T)/O(T·n)/O(T·d),
	// cheap). Bit-identical: same values, same ascending-k / ascending-dd accumulation.
	if bs, xs := flatF64(b), flatF64(x); bs != nil && xs != nil {
		for i := range T {
			for k := range n {
				crow[k] = c.AtF64(i, k)
			}
			for dd := range d {
				yrow[dd] = y.AtF64(i, dd)
			}
			decayrow[i] = 1.0
			for j := i - 1; j >= 0; j-- {
				decayrow[j] = decayrow[j+1] * arow[j+1]
			}
			//perfscan:ignore PS1007 already the yrow AXPY accumulator (documented)
			for j := 0; j <= i; j++ {
				brow := bs[j*n : j*n+n : j*n+n]
				var cb float64
				//perfscan:ignore PS3010,PS4012 O(n) C·B dot, tol-gated reassociation | false-positive: decay scale not quant dequant
				for k := range n {
					cb += crow[k] * brow[k]
				}
				m := cb * decayrow[j]
				xrow := xs[j*d : j*d+d : j*d+d]
				for dd := range d {
					//perfscan:ignore PS3075 yrow is the AXPY accumulator row, inherent
					yrow[dd] += m * xrow[dd]
				}
			}
			for dd := range d {
				y.SetF64(yrow[dd], i, dd)
			}
		}
		return y, nil
	}
	for i := range T { // exotic-dtype fallback: b/x via AtF64
		for k := range n {
			crow[k] = c.AtF64(i, k)
		}
		for dd := range d {
			yrow[dd] = y.AtF64(i, dd)
		}
		decayrow[i] = 1.0
		for j := i - 1; j >= 0; j-- {
			decayrow[j] = decayrow[j+1] * arow[j+1]
		}
		//perfscan:ignore PS1007 exotic-dtype fallback branch (f64 takes fast path)
		for j := 0; j <= i; j++ {
			var cb float64
			//perfscan:ignore PS3010 exotic-dtype fallback branch dot
			for k := range n {
				cb += crow[k] * b.AtF64(j, k)
			}
			m := cb * decayrow[j]
			for dd := range d {
				//perfscan:ignore PS3075 exotic-dtype fallback branch
				yrow[dd] += m * x.AtF64(j, dd)
			}
		}
		for dd := range d {
			y.SetF64(yrow[dd], i, dd)
		}
	}
	return y, nil
}

func ssdCheck(x, a, b, c *tensor.Tensor) error {
	if x.Ndim() != 2 || b.Ndim() != 2 || c.Ndim() != 2 || a.Ndim() != 1 {
		return fmt.Errorf("nn: SSD wants x[T,d], a[T], B[T,N], C[T,N]")
	}
	T := x.Shape()[0]
	if a.Shape()[0] != T || b.Shape()[0] != T || c.Shape()[0] != T {
		return fmt.Errorf("nn: SSD sequence lengths differ")
	}
	if b.Shape()[1] != c.Shape()[1] {
		return fmt.Errorf("nn: SSD B/C state dims differ: %d vs %d", b.Shape()[1], c.Shape()[1])
	}
	return nil
}
