package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/ops"
	"github.com/jxsl13/goai/tensor"
)

// NewPiSSA builds a PiSSA adapter over a base weight (Meng, Wang & Zhang 2024, "PiSSA: Principal
// Singular Values and Singular Vectors Adaptation of Large Language Models", NeurIPS,
// arXiv:2404.02948). PiSSA is LoRA with a smarter initialization: instead of a random A and a
// zero B (which start ΔW=0 and adapt the RESIDUAL of the weight), it factorizes the pretrained
// weight by SVD, W = U·diag(σ)·Vᵀ, and initializes the two trainable low-rank matrices from the
// TOP-r PRINCIPAL components
//
//	A = U[:,:r]·diag(σ[:r])^½ ,   B = diag(σ[:r])^½·V[:,:r]ᵀ   ⇒   A·B = U_r·diag(σ_r)·Vᵀ_r
//
// so A·B is the rank-r principal reconstruction of W. The FROZEN base is the residual
// W_res = W − A·B, so the output is unchanged at initialization (W_res + A·B = W) yet training now
// updates the high-singular-value directions that carry the most of the weight — which the paper
// shows converges faster and to a better optimum than vanilla LoRA (§R27), which adapts the tail.
//
// It returns a *LoRALinear (in this library's [in,out] row convention, A[in,r], B[r,out]) with the
// residual as its frozen base and scale 1 (Alpha=r ⇒ α/r=1, no LoRA rescaling), so the standard
// LoRALinear.Forward y = x·W_res + (x·A)·B and Params (only A,B train) apply unchanged.
func NewPiSSA(w *tensor.Tensor, r int) (*LoRALinear, error) {
	if w.Ndim() != 2 {
		return nil, fmt.Errorf("nn: PiSSA base must be [in,out], got %v", w.Shape())
	}
	in, out := w.Shape()[0], w.Shape()[1]
	if r <= 0 || r > in || r > out {
		return nil, fmt.Errorf("nn: PiSSA rank %d out of range for [%d,%d]", r, in, out)
	}
	dt := w.Dtype()

	// ops.SVD needs a tall/square matrix (rows ≥ cols). For in≥out decompose W directly; for
	// in<out decompose Wᵀ and swap the factor roles (W = V·diag(σ)·Uᵀ from Wᵀ = U·diag(σ)·Vᵀ).
	// aFac[i,k] and bFac give A[i,k]=aFac[i,k]·√σ_k, B[k,j]=√σ_k·bFac[j,k].
	var uFac, vFac *tensor.Tensor // A drawn from uFac (rows=in), B from vFac (rows=out)
	var sig *tensor.Tensor
	if in >= out {
		u, s, v, err := ops.SVD(w) // U[in,out], σ[out], V[out,out]
		if err != nil {
			return nil, fmt.Errorf("nn: PiSSA SVD: %w", err)
		}
		uFac, sig, vFac = u, s, v
	} else {
		wt := transpose2D(w) // [out,in], out>in
		u, s, v, err := ops.SVD(wt)
		if err != nil {
			return nil, fmt.Errorf("nn: PiSSA SVD: %w", err)
		}
		// Wᵀ = U·diag(σ)·Vᵀ ⇒ W = V·diag(σ)·Uᵀ: A drawn from V (rows=in), B from U (rows=out).
		uFac, sig, vFac = v, s, u
	}

	a := tensor.New(dt, tensor.Shape{in, r})
	b := tensor.New(dt, tensor.Shape{r, out})
	for k := 0; k < r; k++ {
		root := math.Sqrt(math.Max(0, sig.AtF64(k)))
		for i := 0; i < in; i++ {
			a.SetF64(uFac.AtF64(i, k)*root, i, k)
		}
		for j := 0; j < out; j++ {
			b.SetF64(root*vFac.AtF64(j, k), k, j)
		}
	}

	// residual base W_res = W − A·B (frozen).
	wres := tensor.New(dt, tensor.Shape{in, out})
	for i := 0; i < in; i++ {
		for j := 0; j < out; j++ {
			var ab float64
			for k := 0; k < r; k++ {
				ab += a.AtF64(i, k) * b.AtF64(k, j)
			}
			wres.SetF64(w.AtF64(i, j)-ab, i, j)
		}
	}
	return &LoRALinear{W: wres, A: a, B: b, R: r, Alpha: float64(r)}, nil // Alpha=r ⇒ scale 1
}

// transpose2D returns a fresh [n,m] transpose of the [m,n] matrix t.
func transpose2D(t *tensor.Tensor) *tensor.Tensor {
	m, n := t.Shape()[0], t.Shape()[1]
	out := tensor.New(t.Dtype(), tensor.Shape{n, m})
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			out.SetF64(t.AtF64(i, j), j, i)
		}
	}
	return out
}
