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
	for t := range T {
		at := a.AtF64(t)
		for i := range n {
			bi := b.AtF64(t, i)
			for j := range d {
				h[i*d+j] = at*h[i*d+j] + bi*x.AtF64(t, j)
			}
		}
		for j := range d {
			var s float64
			for i := range n {
				s += c.AtF64(t, i) * h[i*d+j]
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
	for i := range T {
		for j := 0; j <= i; j++ {
			var cb float64
			for k := range n {
				cb += c.AtF64(i, k) * b.AtF64(j, k)
			}
			decay := 1.0
			for k := j + 1; k <= i; k++ {
				decay *= a.AtF64(k)
			}
			m := cb * decay
			for dd := range d {
				y.SetF64(y.AtF64(i, dd)+m*x.AtF64(j, dd), i, dd)
			}
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
