package nn

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// SpinRotation is a SpinQuant orthogonal rotation for outlier-free low-bit quantization
// (Liu, Li, Chen, Chandra, Tulloch, Blankevoort, Kumar, Farhadi, Blalock et al. 2024,
// "SpinQuant: LLM quantization with learned rotations", arXiv:2405.16406). LLM activations
// carry a few systematic OUTLIER channels whose huge magnitude forces a coarse quantization
// grid on every other channel, wrecking low-bit accuracy. SpinQuant inserts an orthogonal
// rotation R into the network so that a linear y = x·W becomes (x·R)·(Rᵀ·W) — the rotation
// and its inverse cancel, leaving the FULL-PRECISION output UNCHANGED (the computational
// invariance, §V16(1)), while the rotated activations x·R spread each outlier's energy
// across all channels, shrinking the per-tensor max and kurtosis so the same bit-width
// quantizes with far less error (§V16(2)). The rotation folds into the adjacent weights at
// deploy time, so it costs nothing at inference.
//
// Two constructors mirror the paper's two rotation families:
//   - [NewHadamardRotation] — a randomized Walsh-Hadamard-Rademacher rotation R = (1/√n)·H·D
//     (training-free, the SpinQuant-had baseline; requires a power-of-two dimension).
//   - [NewLearnableRotation] — an orthogonal matrix parameterized as the Q factor of QR(W)
//     of a trainable W, a Stiefel-manifold retraction differentiable end-to-end (§V16(3))
//     so R can be OPTIMIZED to minimize the post-quantization loss, the paper's key win over
//     random Hadamard.
//
// # For the AI professional
//
// R is n×n orthogonal (RᵀR = I to machine precision). [SpinRotation.Apply] computes x·R,
// [SpinRotation.FoldWeight] computes Rᵀ·W; composed on a linear they reproduce x·W exactly
// (fold the rotation into weights offline). [SpinRotation.QuantizeActivations] is the
// deploy-time path: rotate → per-tensor symmetric fake-quant → unrotate, whose reconstruction
// error is below quantizing the un-rotated activations directly whenever they carry outliers.
// The learnable variant's Q = qr(W).Q is differentiable via the QR VJP, so W trains under any
// optimizer through [SpinRotation.Params]; the Hadamard variant has no parameters.
//
// # For the newcomer
//
// Quantizing a neural network means storing its numbers in very few bits. A handful of
// "outlier" activations are thousands of times larger than the rest, and they force a
// coarse rounding that ruins the small numbers. SpinQuant multiplies the activations by a
// carefully chosen rotation — a reversible spin of the coordinate space — that smears those
// giant values out evenly so every number becomes easy to round. Because the same rotation
// is absorbed into the next weight matrix, the network computes exactly the same thing; only
// the quantization gets easier.
//
// Further reading: Liu et al. 2024, "SpinQuant", arXiv:2405.16406; compare the training-free
// [SmoothQuant] outlier-migration transform and the random Hadamard KV-cache rotation in
// package nlp (TurboQuant).
//
// In plain terms: spin the activations so their outliers spread out and low-bit rounding
// stops hurting, then hide the spin inside the weights so the model's answer is unchanged.
type SpinRotation struct {
	dim   int
	r     *tensor.Tensor // n×n orthogonal rotation R (materialized for the Hadamard variant)
	rt    *tensor.Tensor // Rᵀ (the inverse), for the Hadamard variant
	w     *tensor.Tensor // trainable pre-image of the learnable rotation; nil for the Hadamard variant
	dtype tensor.Dtype
	// Hadamard variant only (w == nil): the Rademacher diagonal D and 1/√n scale, so the rotation
	// R = (1/√n)·H·D can be applied as an O(n log n) FWHT (x·R = inv·D∘fwht(x_row)) instead of the
	// dense O(n²) matmul against r — the same rotation the materialized r encodes.
	signs []float64
	inv   float64
}

// spinCfg holds the resolved options for a SpinRotation constructor.
type spinCfg struct{ seed uint64 }

// SpinOption configures a [SpinRotation] constructor (functional-options idiom, §C12).
type SpinOption func(*spinCfg)

// WithSpinSeed sets the random seed for the rotation — the Rademacher sign flips of a Hadamard
// rotation, or the initial trainable matrix of a learnable one. Fixing it makes construction
// reproducible (§V10).
//
// In plain terms: the same seed always builds the same rotation, so results are repeatable.
// Default 0.
func WithSpinSeed(seed uint64) SpinOption { return func(c *spinCfg) { c.seed = seed } }

// NewHadamardRotation builds the training-free SpinQuant-had rotation for dim-dimensional
// activations: R = (1/√n)·H·D with H the n×n Walsh-Hadamard matrix and D a random ±1
// (Rademacher) diagonal, n = dim. It is exactly orthogonal, and the Hadamard structure spreads
// outliers into near-uniform coordinates. dim must be a power of two (the Walsh-Hadamard matrix
// exists only for those sizes; use [NewLearnableRotation] for arbitrary dims). dtype selects the
// working precision (F32 or F64).
func NewHadamardRotation(dtype tensor.Dtype, dim int, opts ...SpinOption) (*SpinRotation, error) {
	if dim < 1 {
		return nil, fmt.Errorf("nn: NewHadamardRotation needs dim ≥ 1, got %d", dim)
	}
	if dim&(dim-1) != 0 {
		return nil, fmt.Errorf("nn: NewHadamardRotation needs a power-of-two dim, got %d (use NewLearnableRotation for arbitrary dims)", dim)
	}
	cfg := spinCfg{}
	for _, o := range opts {
		o(&cfg)
	}
	rng := rand.New(rand.NewPCG(cfg.seed, 0x14057b7ef767814f))
	signs := make([]float64, dim)
	for i := range signs {
		if rng.Uint64()&1 == 0 {
			signs[i] = 1
		} else {
			signs[i] = -1
		}
	}
	// column j of the (unnormalized) Hadamard matrix is fwht(e_j); H is symmetric so rows = cols.
	inv := 1 / math.Sqrt(float64(dim))
	r := tensor.New(dtype, tensor.Shape{dim, dim})
	rt := tensor.New(dtype, tensor.Shape{dim, dim})
	col := make([]float64, dim)
	for j := range dim {
		for i := range col {
			col[i] = 0
		}
		col[j] = 1
		fwhtInPlace(col)
		for i := range dim {
			// R = (1/√n)·H·D ⇒ column j scaled by the Rademacher sign D[j].
			v := inv * col[i] * signs[j]
			r.SetF64(v, i, j)
			rt.SetF64(v, j, i) // Rᵀ
		}
	}
	return &SpinRotation{dim: dim, r: r, rt: rt, dtype: dtype, signs: signs, inv: inv}, nil
}

// NewLearnableRotation builds a trainable SpinQuant rotation for dim-dimensional activations:
// the orthogonal matrix R = Q of the QR decomposition of a trainable dim×dim matrix W, a
// standard Stiefel-manifold retraction that stays exactly orthogonal for every value of W and
// is differentiable through the QR VJP. W is initialized to the identity plus small Gaussian
// noise (so R starts near I) and is returned by [SpinRotation.Params] for an optimizer to learn.
// dim ≥ 1 (any size); dtype selects the working precision.
func NewLearnableRotation(dtype tensor.Dtype, dim int, opts ...SpinOption) (*SpinRotation, error) {
	if dim < 1 {
		return nil, fmt.Errorf("nn: NewLearnableRotation needs dim ≥ 1, got %d", dim)
	}
	cfg := spinCfg{}
	for _, o := range opts {
		o(&cfg)
	}
	rng := rand.New(rand.NewPCG(cfg.seed, 0x2545f4914f6cdd1d))
	w := tensor.New(dtype, tensor.Shape{dim, dim})
	for i := range dim {
		for j := range dim {
			v := 0.02 * rng.NormFloat64()
			if i == j {
				v += 1 // identity + small noise ⇒ QR(W).Q starts near I
			}
			w.SetF64(v, i, j)
		}
	}
	return &SpinRotation{dim: dim, w: w, dtype: dtype}, nil
}

// Dim returns the rotation dimension n (activations and weights are rotated along an n-sized axis).
func (s *SpinRotation) Dim() int { return s.dim }

// Params returns the trainable tensors: the pre-image matrix W of a learnable rotation (for an
// optimizer to update), or nil for a training-free Hadamard rotation.
func (s *SpinRotation) Params() []*tensor.Tensor {
	if s.w == nil {
		return nil
	}
	return []*tensor.Tensor{s.w}
}

// Matrix returns the n×n orthogonal rotation R through ctx. For a Hadamard rotation it is the
// fixed materialized matrix; for a learnable rotation it is Q = qr(W).Q recomputed on ctx, so
// passing a recording context (autograd.Tape.Context()) lets gradients flow back to W.
func (s *SpinRotation) Matrix(ctx *backend.Context) (*tensor.Tensor, error) {
	if s.w == nil {
		return s.r, nil
	}
	out, err := backend.Execute(ctx, backend.OpQR, []*tensor.Tensor{s.w}, nil)
	if err != nil {
		return nil, err
	}
	return out[0], nil // Q
}

// Apply returns the rotated activations x·R (x is [rows, n]; the last axis is rotated). This is
// the SpinQuant activation rotation; because R is orthogonal it changes neither norms nor inner
// products, only the coordinate frame — spreading outliers so the result quantizes cleanly.
// Runs through ctx, so a learnable rotation stays differentiable end-to-end.
func (s *SpinRotation) Apply(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != s.dim {
		return nil, fmt.Errorf("nn: SpinRotation.Apply wants x [rows,%d], got %v", s.dim, x.Shape())
	}
	r, err := s.Matrix(ctx)
	if err != nil {
		return nil, err
	}
	out, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{x, r}, nil)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// FoldWeight returns Rᵀ·W, the rotation folded into a following linear's weight W [n, out] (the
// repo's [in,out] layout, in = n). Composed with [SpinRotation.Apply] it reproduces the original
// linear exactly: (x·R)·(Rᵀ·W) = x·W (§V16(1) computational invariance). Fold offline once so the
// rotation is free at inference. Runs through ctx.
func (s *SpinRotation) FoldWeight(ctx *backend.Context, w *tensor.Tensor) (*tensor.Tensor, error) {
	if w.Ndim() != 2 || w.Shape()[0] != s.dim {
		return nil, fmt.Errorf("nn: SpinRotation.FoldWeight wants W [%d,out], got %v", s.dim, w.Shape())
	}
	r, err := s.Matrix(ctx)
	if err != nil {
		return nil, err
	}
	rt, err := backend.Execute(ctx, backend.OpTranspose, []*tensor.Tensor{r}, nil)
	if err != nil {
		return nil, err
	}
	out, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{rt[0], w}, nil)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// QuantizeActivations is the SpinQuant deploy-time path: rotate x by R, symmetric per-tensor
// fake-quantize the rotated activations to `levels` uniform steps, then rotate back by Rᵀ,
// returning the dequantized approximation of x (same shape). Because the rotation spreads
// outliers, the rotated tensor's max-abs is smaller, giving a finer step and lower
// reconstruction error than fake-quantizing x directly (§V16(2)). levels ≥ 2. This is a
// non-differentiable inference transform (rounding), so it always runs eagerly.
func (s *SpinRotation) QuantizeActivations(x *tensor.Tensor, levels int) (*tensor.Tensor, error) {
	if levels < 2 {
		return nil, fmt.Errorf("nn: SpinRotation.QuantizeActivations needs levels ≥ 2, got %d", levels)
	}
	if x.Ndim() != 2 || x.Shape()[1] != s.dim {
		return nil, fmt.Errorf("nn: SpinRotation.QuantizeActivations wants x [rows,%d], got %v", s.dim, x.Shape())
	}
	if s.w == nil {
		return s.quantizeActivationsFWHT(x, levels) // O(rows·n log n) Hadamard fast path
	}
	ctx := backend.NewContext()
	rot, err := s.Apply(ctx, x)
	if err != nil {
		return nil, err
	}
	// per-tensor symmetric absmax range → uniform mid-rise quantizer (reuses UniformQuantizer).
	var absmax float64
	rows := rot.Shape()[0]
	for i := range rows {
		for j := range s.dim {
			if v := math.Abs(rot.AtF64(i, j)); v > absmax {
				absmax = v
			}
		}
	}
	if absmax == 0 {
		absmax = 1 // all-zero activations: nothing to quantize, keep a finite step
	}
	q := UniformQuantizer(levels, -absmax, absmax)
	for i := range rows {
		for j := range s.dim {
			rot.SetF64(q(rot.AtF64(i, j)), i, j)
		}
	}
	// unrotate: (quantized rotated) · Rᵀ.
	r, err := s.Matrix(ctx)
	if err != nil {
		return nil, err
	}
	rt, err := backend.Execute(ctx, backend.OpTranspose, []*tensor.Tensor{r}, nil)
	if err != nil {
		return nil, err
	}
	out, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{rot, rt[0]}, nil)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// quantizeActivationsFWHT is QuantizeActivations for the Hadamard variant: it applies the rotation
// R = (1/√n)·H·D and its inverse Rᵀ = (1/√n)·D·H via the O(n log n) FWHT — rotate is inv·D∘fwht(row),
// unrotate is inv·fwht(D∘row) (H symmetric ⇒ x·H = fwht(x)) — replacing the two dense O(rows·n²)
// matmuls against the materialized r/rt with row-wise butterflies. Same rotation, so the result
// matches the matmul path within FWHT reassociation tolerance (the incumbent-standard SpinQuant apply).
func (s *SpinRotation) quantizeActivationsFWHT(x *tensor.Tensor, levels int) (*tensor.Tensor, error) {
	rows, n := x.Shape()[0], s.dim
	rot := tensor.New(s.dtype, x.Shape())
	buf := make([]float64, n)
	var absmax float64
	for i := 0; i < rows; i++ { // rotate: rot[i][j] = inv·D[j]·fwht(x[i])[j]; track per-tensor absmax
		for j := 0; j < n; j++ {
			buf[j] = x.AtF64(i, j)
		}
		fwhtInPlace(buf)
		for j := 0; j < n; j++ {
			v := s.inv * s.signs[j] * buf[j]
			rot.SetF64(v, i, j)
			if a := math.Abs(v); a > absmax {
				absmax = a
			}
		}
	}
	if absmax == 0 {
		absmax = 1
	}
	q := UniformQuantizer(levels, -absmax, absmax)
	out := tensor.New(s.dtype, x.Shape())
	for i := 0; i < rows; i++ { // fake-quant the rotated row, then unrotate: out[i][k] = inv·fwht(D∘q(rot[i]))[k]
		for j := 0; j < n; j++ {
			buf[j] = q(rot.AtF64(i, j)) * s.signs[j]
		}
		fwhtInPlace(buf)
		for k := 0; k < n; k++ {
			out.SetF64(s.inv*buf[k], i, k)
		}
	}
	return out, nil
}

// fwhtInPlace applies the unnormalized in-place Fast Walsh-Hadamard Transform (len(a) a power of
// two): the O(n log n) butterfly for H·a. Used once at construction to materialize the columns of
// the Walsh-Hadamard matrix.
func fwhtInPlace(a []float64) {
	n := len(a)
	for h := 1; h < n; h <<= 1 {
		for i := 0; i < n; i += h << 1 {
			for j := i; j < i+h; j++ {
				x, y := a[j], a[j+h]
				a[j], a[j+h] = x+y, x-y
			}
		}
	}
}
