package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// NystromAttention is the linear-complexity self-attention of Xiong, Zeng,
// Chakraborty, Tan, Fung, Li & Singh (2021, "Nyströmformer: A Nyström-based
// Algorithm for Approximating Self-Attention", arXiv:2102.03902). It approximates
// the L×L softmax attention matrix with a Nyström low-rank reconstruction built
// from m LANDMARK points, so the cost is O(L·m) instead of the O(L²) of a full
// softmax — with m ≪ L this turns quadratic attention into linear attention.
//
// # The mechanism (paper §3)
//
// Standard attention forms S = softmax(Q Kᵀ/√d) then O = S·V, materializing the
// full L×L matrix S. Nyströmformer never builds S. It first picks m LANDMARK
// queries Q̃ and m landmark keys K̃ by SEGMENT-MEANS — split the L rows of Q (and
// of K) into m contiguous groups and average each group, giving Q̃,K̃ ∈ ℝ^{m×d}
// (per head). It then approximates S by three SMALL softmax blocks bridged by a
// pseudo-inverse:
//
//	S ≈ F₁ · Ã⁺ · F₃
//	F₁ = softmax(Q  K̃ᵀ/√d)   [L,m]   (each token vs the m landmark keys)
//	Ã  = softmax(Q̃ K̃ᵀ/√d)   [m,m]   (landmark queries vs landmark keys)
//	F₃ = softmax(Q̃ Kᵀ/√d)    [m,L]   (landmark queries vs every key)
//	O  ≈ F₁ · (Ã⁺ · (F₃ · V))          [L,d]
//
// where Ã⁺ is the Moore–Penrose pseudo-inverse of the m×m landmark kernel Ã. None
// of the three softmax blocks is L×L — they are [L,m], [m,m] and [m,L], so the
// whole layer is O(L·m·d). Multiplying right-to-left (F₃·V first, [m,d]) keeps
// every intermediate O(L·m) or O(m²); the L×L matrix is never formed.
//
// # The pseudo-inverse (paper Algorithm 1)
//
// Ã⁺ is computed by the paper's ITERATIVE Newton–Schulz approximation — no linear
// solve, just matmuls, so it is differentiable through the unrolled iteration:
//
//	Z₀   = Ãᵀ / (‖Ã‖₁ · ‖Ã‖_∞)                 (guarantees ‖I − ÃZ₀‖ < 1)
//	Zⱼ₊₁ = Zⱼ (2I − Ã Zⱼ)                       (quadratic convergence to Ã⁺)
//
// ~6 iterations suffice in the paper; more iterations tighten Z toward the exact
// pseudo-inverse (WithNystromPinvIters, §C21). Because Z is built only from
// matmul / sub / scale, gradients flow through it end to end.
//
// # Landmarks, complexity and the accuracy knob (§C21)
//
// m (numLandmarks) trades accuracy for speed: the Nyström reconstruction error
// shrinks as m→L, and at m=L with each token as its own landmark the three-block
// product telescopes back to exact softmax attention (F₁·Ã⁺·F₃ = softmax(QKᵀ)
// when Ã is invertible — the §V16 collapse anchor). Segment-means require the L
// rows to be split into m groups; when L is not divisible by m the first (L mod m)
// groups take one extra row (near-equal groups), so any 1 ≤ m ≤ L is accepted. The
// pseudo-inverse iteration count is the second knob: too few iterations leave Z a
// crude inverse and inflate the approximation error independently of m.
//
// # Bidirectional only
//
// This layer is NON-CAUSAL (bidirectional): every query attends to every key. The
// paper describes a causal variant, but its landmark segments and the pinv must be
// restricted to the visible prefix, which is involved; a causal option is out of
// scope here (§C21). The optional depthwise-conv-on-V skip connection from the
// paper is likewise not implemented — it is an accuracy add-on, and omitting it
// keeps the full-landmark collapse to standard attention exact and clean.
//
// # Why no new kernel is needed
//
// Every step — OpMatMul (projections, the three score products, V aggregation and
// the pinv matmuls), OpMul (scale), OpSoftmax (the three blocks), OpSum/OpMax
// (the pinv init norms), OpDiv/OpSub (the pinv iteration), OpSlice/OpConcat (heads)
// — is an existing dispatched op with a first-order VJP, so the whole layer trains
// end to end with NO backend change (§C18).
//
// FURTHER READING (§C18): Xiong et al. 2021, arXiv:2102.03902 (this layer);
// Vaswani et al. 2017, arXiv:1706.03762 (the softmax attention it approximates);
// Wang et al. 2020, arXiv:2006.04768 (Linformer — the low-rank sibling in this
// package that projects K,V instead of using Nyström landmarks).
type NystromAttention struct {
	Dim          int // model / embedding width (columns of X)
	Heads        int // number of attention heads
	HeadDim      int // per-head dimension d = Dim/Heads
	NumLandmarks int // m: number of Nyström landmark points (1 ≤ m ≤ L at Forward)

	QProj *Linear // query projection  [Dim → Dim]
	KProj *Linear // key projection    [Dim → Dim]
	VProj *Linear // value projection  [Dim → Dim]
	OProj *Linear // output projection [Dim → Dim]

	pinvIters         int  // Newton–Schulz iterations for the m×m pseudo-inverse
	identityLandmarks bool // test seam: require L==m and use identity landmarks (Q̃=Q,K̃=K)
	seed              uint64
}

// NystromAttentionOption configures a NystromAttention layer (functional-options
// idiom, §C12).
type NystromAttentionOption func(*NystromAttention)

// WithNystromHeads sets the number of attention heads (default 1). The model
// width Dim must be divisible by the head count; each head runs its own Nyström
// approximation over the shared m landmarks.
func WithNystromHeads(heads int) NystromAttentionOption {
	return func(n *NystromAttention) { n.Heads = heads }
}

// WithNystromPinvIters sets the number of Newton–Schulz iterations used to
// approximate the m×m landmark-kernel pseudo-inverse (default 6, the paper's
// value). More iterations drive Z closer to the exact Moore–Penrose inverse —
// tightening both the pseudo-inverse identities (ÃZÃ≈Ã, ZÃZ≈Z) and the
// full-landmark collapse to standard attention — at a linear compute cost in the
// iteration count (§C21). Must be ≥ 1.
func WithNystromPinvIters(iters int) NystromAttentionOption {
	return func(n *NystromAttention) { n.pinvIters = iters }
}

// WithNystromSeed sets the deterministic seed for the four Xavier-uniform
// projections (they use seed, seed+1, seed+2, seed+3). Default 0.
func WithNystromSeed(seed uint64) NystromAttentionOption {
	return func(n *NystromAttention) { n.seed = seed }
}

// WithNystromIdentityLandmarks makes each token its own landmark (Q̃=Q, K̃=K)
// instead of segment-means. It REQUIRES the sequence length L to equal m at
// Forward (one landmark per position); otherwise Forward errors. This is the test
// seam for the §V16 full-landmark collapse: with identity landmarks and m=L the
// three softmax blocks are all softmax(QKᵀ/√d), so F₁·Ã⁺·F₃ = softmax(QKᵀ) and
// the layer reduces to exact bidirectional softmax attention. Not for normal use
// (it defeats the O(L·m) point); default off (segment-means).
func WithNystromIdentityLandmarks() NystromAttentionOption {
	return func(n *NystromAttention) { n.identityLandmarks = true }
}

// NewNystromAttention builds a Nyströmformer attention layer over model width
// dModel with numLandmarks Nyström landmark points m. The four Q/K/V/output
// projections are Xavier-uniform Linears (Glorot 2010) with zero bias — the
// standard transformer init. Options set the head count (default 1), the
// pseudo-inverse iteration count (default 6) and the seed (default 0). Errors if
// dModel ≤ 0, numLandmarks < 1, or Dim is not divisible by the head count.
func NewNystromAttention(dtype tensor.Dtype, dModel, numLandmarks int, opts ...NystromAttentionOption) (*NystromAttention, error) {
	if dModel <= 0 {
		return nil, fmt.Errorf("nn: NystromAttention dModel %d must be positive", dModel)
	}
	if numLandmarks < 1 {
		return nil, fmt.Errorf("nn: NystromAttention numLandmarks %d must be ≥ 1", numLandmarks)
	}
	n := &NystromAttention{
		Dim:          dModel,
		Heads:        1,
		NumLandmarks: numLandmarks,
		pinvIters:    6,
	}
	for _, o := range opts {
		o(n)
	}
	if n.Heads <= 0 || dModel%n.Heads != 0 {
		return nil, fmt.Errorf("nn: NystromAttention dModel %d not divisible by heads %d", dModel, n.Heads)
	}
	if n.pinvIters < 1 {
		return nil, fmt.Errorf("nn: NystromAttention pinvIters %d must be ≥ 1", n.pinvIters)
	}
	n.HeadDim = dModel / n.Heads
	n.QProj = NewLinear(dtype, dModel, dModel, n.seed)
	n.KProj = NewLinear(dtype, dModel, dModel, n.seed+1)
	n.VProj = NewLinear(dtype, dModel, dModel, n.seed+2)
	n.OProj = NewLinear(dtype, dModel, dModel, n.seed+3)
	return n, nil
}

func (n *NystromAttention) exec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// nystromHeadDebug carries per-head intermediates for white-box tests: the three
// softmax block shapes and head 0's landmark kernel Ã together with its computed
// pseudo-inverse Z (for the ÃZÃ≈Ã / ZÃZ≈Z checks and the shape assertions).
type nystromHeadDebug struct {
	b1Shape, midShape, b3Shape tensor.Shape
	mid0, z0                   *tensor.Tensor
}

// Forward runs bidirectional Nyström attention on x[L, Dim] → [L, Dim].
func (n *NystromAttention) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	out, _, err := n.forward(ctx, x)
	return out, err
}

// forward is Forward exposing head-0 intermediates for tests (Forward returns only
// the output). Fully differentiable: the projections, the three softmax blocks and
// the unrolled pseudo-inverse are all dispatched ops with VJPs, so gradients reach
// Wq/Wk/Wv/Wo and x.
func (n *NystromAttention) forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, *nystromHeadDebug, error) {
	if x.Ndim() != 2 || x.Shape()[1] != n.Dim {
		return nil, nil, fmt.Errorf("nn: NystromAttention expects x [L,%d], got %v", n.Dim, x.Shape())
	}
	L := x.Shape()[0]
	m := n.NumLandmarks
	if m > L {
		return nil, nil, fmt.Errorf("nn: NystromAttention numLandmarks %d exceeds sequence length %d", m, L)
	}
	if n.identityLandmarks && L != m {
		return nil, nil, fmt.Errorf("nn: NystromAttention identity landmarks require L==m, got L=%d m=%d", L, m)
	}

	q, err := n.QProj.Forward(ctx, x)
	if err != nil {
		return nil, nil, err
	}
	k, err := n.KProj.Forward(ctx, x)
	if err != nil {
		return nil, nil, err
	}
	v, err := n.VProj.Forward(ctx, x)
	if err != nil {
		return nil, nil, err
	}

	invSqrtD := scalarTensor(x.Dtype(), 1/math.Sqrt(float64(n.HeadDim)))
	// P[m,L] averages each contiguous token group into one landmark row; at m=L it
	// is the identity, so identity landmarks and full-length segment-means coincide.
	pMat := nystromLandmarkMatrix(x.Dtype(), m, L)

	slice := func(t *tensor.Tensor, h int) (*tensor.Tensor, error) {
		lo, hi := h*n.HeadDim, (h+1)*n.HeadDim
		return n.exec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: lo, End: hi}, t)
	}
	// scaled softmax(a·bᵀ/√d) with rows over the columns of b.
	softScores := func(a, b *tensor.Tensor) (*tensor.Tensor, error) {
		bT, err := n.exec(ctx, backend.OpTranspose, nil, b)
		if err != nil {
			return nil, err
		}
		s, err := n.exec(ctx, backend.OpMatMul, nil, a, bT)
		if err != nil {
			return nil, err
		}
		if s, err = n.exec(ctx, backend.OpMul, nil, s, invSqrtD); err != nil {
			return nil, err
		}
		return n.exec(ctx, backend.OpSoftmax, nil, s)
	}

	var dbg *nystromHeadDebug
	headsOut := make([]*tensor.Tensor, n.Heads)
	for h := range n.Heads {
		qh, err := slice(q, h) // [L,d]
		if err != nil {
			return nil, nil, err
		}
		kh, err := slice(k, h)
		if err != nil {
			return nil, nil, err
		}
		vh, err := slice(v, h)
		if err != nil {
			return nil, nil, err
		}
		qTilde, err := n.exec(ctx, backend.OpMatMul, nil, pMat, qh) // [m,d] segment-mean landmarks
		if err != nil {
			return nil, nil, err
		}
		kTilde, err := n.exec(ctx, backend.OpMatMul, nil, pMat, kh) // [m,d]
		if err != nil {
			return nil, nil, err
		}

		f1, err := softScores(qh, kTilde) // [L,m]
		if err != nil {
			return nil, nil, err
		}
		mid, err := softScores(qTilde, kTilde) // [m,m]
		if err != nil {
			return nil, nil, err
		}
		f3, err := softScores(qTilde, kh) // [m,L]
		if err != nil {
			return nil, nil, err
		}
		z, err := nystromPinv(ctx, n.exec, mid, n.pinvIters) // Ã⁺ ≈ Z [m,m]
		if err != nil {
			return nil, nil, err
		}

		// O_h = F₁·(Z·(F₃·V)) — right-to-left keeps every intermediate O(L·m)/O(m²).
		fv, err := n.exec(ctx, backend.OpMatMul, nil, f3, vh) // [m,d]
		if err != nil {
			return nil, nil, err
		}
		zfv, err := n.exec(ctx, backend.OpMatMul, nil, z, fv) // [m,d]
		if err != nil {
			return nil, nil, err
		}
		headsOut[h], err = n.exec(ctx, backend.OpMatMul, nil, f1, zfv) // [L,d]
		if err != nil {
			return nil, nil, err
		}
		if h == 0 {
			dbg = &nystromHeadDebug{
				b1Shape:  f1.Shape(),
				midShape: mid.Shape(),
				b3Shape:  f3.Shape(),
				mid0:     mid,
				z0:       z,
			}
		}
	}

	concat, err := n.exec(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, headsOut...) // [L,Dim]
	if err != nil {
		return nil, nil, err
	}
	out, err := n.OProj.Forward(ctx, concat)
	if err != nil {
		return nil, nil, err
	}
	return out, dbg, nil
}

// Params returns every trainable tensor — the Q, K, V and output projection
// weights W_q, W_k, W_v, W_o (each Linear's W and its zero-initialized bias, 8
// tensors). The Nyström landmark selection and the pseudo-inverse add NO
// parameters of their own. Feed this to an optimizer.
func (n *NystromAttention) Params() []*tensor.Tensor {
	var ps []*tensor.Tensor
	for _, l := range []*Linear{n.QProj, n.KProj, n.VProj, n.OProj} {
		ps = append(ps, l.Params()...)
	}
	return ps
}

// execFn is the single-output exec signature both the layer method and the
// free-standing pseudo-inverse share.
type execFn func(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error)

// nystromPinv approximates the Moore–Penrose pseudo-inverse of the m×m matrix A by
// the paper's Newton–Schulz iteration (Algorithm 1, §C18): Z₀ = Aᵀ/(‖A‖₁·‖A‖_∞)
// then Zⱼ₊₁ = Zⱼ(2I − A Zⱼ) for iters steps. The init scaling by the product of
// the max column sum (‖·‖₁) and max row sum (‖·‖_∞) guarantees the spectral
// condition ‖I − A Z₀‖ < 1 so the iteration converges (quadratically). Every step
// is a matmul / sub / div, so Z is differentiable in A. More iters → tighter
// pseudo-inverse (§C21).
func nystromPinv(ctx *backend.Context, exec execFn, a *tensor.Tensor, iters int) (*tensor.Tensor, error) {
	m := a.Shape()[0]
	dtype := a.Dtype()

	aT, err := exec(ctx, backend.OpTranspose, nil, a)
	if err != nil {
		return nil, err
	}
	colSums, err := exec(ctx, backend.OpSum, backend.ReduceAttrs{Axes: []int{0}, KeepDims: true}, a) // [1,m]
	if err != nil {
		return nil, err
	}
	n1, err := exec(ctx, backend.OpMax, backend.ReduceAttrs{Axes: []int{0, 1}, KeepDims: true}, colSums) // ‖A‖₁ [1,1]
	if err != nil {
		return nil, err
	}
	rowSums, err := exec(ctx, backend.OpSum, backend.ReduceAttrs{Axes: []int{1}, KeepDims: true}, a) // [m,1]
	if err != nil {
		return nil, err
	}
	nInf, err := exec(ctx, backend.OpMax, backend.ReduceAttrs{Axes: []int{0, 1}, KeepDims: true}, rowSums) // ‖A‖_∞ [1,1]
	if err != nil {
		return nil, err
	}
	denom, err := exec(ctx, backend.OpMul, nil, n1, nInf) // [1,1]
	if err != nil {
		return nil, err
	}
	z, err := exec(ctx, backend.OpDiv, nil, aT, denom) // Z₀ = Aᵀ/denom, broadcast [m,m]/[1,1]
	if err != nil {
		return nil, err
	}

	twoEye := nystromScaledEye(dtype, m, 2) // 2I
	for range iters {
		az, err := exec(ctx, backend.OpMatMul, nil, a, z) // A Zⱼ
		if err != nil {
			return nil, err
		}
		t, err := exec(ctx, backend.OpSub, nil, twoEye, az) // 2I − A Zⱼ
		if err != nil {
			return nil, err
		}
		if z, err = exec(ctx, backend.OpMatMul, nil, z, t); err != nil { // Zⱼ(2I − A Zⱼ)
			return nil, err
		}
	}
	return z, nil
}

// nystromLandmarkMatrix builds the [m,L] averaging matrix P whose g-th row is
// 1/|group g| on the columns of contiguous token group g and 0 elsewhere, so P·M
// replaces the L rows of M with m segment-means (the Nyström landmarks, paper §3).
// When L is not divisible by m the first (L mod m) groups take one extra token, so
// groups differ in size by at most one; at m=L every group is a single token and P
// is the identity (identity landmarks).
func nystromLandmarkMatrix(dtype tensor.Dtype, m, L int) *tensor.Tensor {
	p := tensor.New(dtype, tensor.Shape{m, L})
	base, rem := L/m, L%m
	col := 0
	for g := range m {
		size := base
		if g < rem {
			size++
		}
		inv := 1.0 / float64(size)
		for c := col; c < col+size; c++ {
			p.SetF64(inv, g, c)
		}
		col += size
	}
	return p
}

// nystromScaledEye returns the n×n diagonal matrix with s on the diagonal.
func nystromScaledEye(dtype tensor.Dtype, n int, s float64) *tensor.Tensor {
	e := tensor.New(dtype, tensor.Shape{n, n})
	for i := range n {
		e.SetF64(s, i, i)
	}
	return e
}
