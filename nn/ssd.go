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
				h[base+j] = at*h[base+j] + bi*xrow[j]
			}
		}
		for j := range d {
			var s float64
			for i := range n {
				s += crow[i] * h[i*d+j]
			}
			y.SetF64(s, t, j)
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
	for i := range T {
		for k := range n {
			crow[k] = c.AtF64(i, k)
		}
		for dd := range d {
			yrow[dd] = y.AtF64(i, dd)
		}
		for j := 0; j <= i; j++ {
			var cb float64
			for k := range n {
				cb += crow[k] * b.AtF64(j, k)
			}
			decay := 1.0
			for k := j + 1; k <= i; k++ {
				decay *= arow[k]
			}
			m := cb * decay
			for dd := range d {
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
