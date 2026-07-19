package nn

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/jxsl13/goai/linalg"
	"github.com/jxsl13/goai/tensor"
)

// Parameter initializers (§T15). Deterministic: callers pass an explicit seed
// (§V13 reproducibility). Formulas follow the original papers.

// XavierUniform fills t with U(-a, a), a = √(6/(fanIn+fanOut)) — Glorot &
// Bengio (2010), "Understanding the difficulty of training deep feedforward
// neural networks". Keeps forward/backward variance constant for tanh-like
// activations.
func XavierUniform(t *tensor.Tensor, fanIn, fanOut int, seed uint64) {
	a := math.Sqrt(6.0 / float64(fanIn+fanOut))
	fillUniform(t, -a, a, seed)
}

// KaimingNormal fills t with N(0, √(2/fanIn)) — He et al. (2015), "Delving Deep
// into Rectifiers". The variance-preserving choice for ReLU activations.
func KaimingNormal(t *tensor.Tensor, fanIn int, seed uint64) {
	std := math.Sqrt(2.0 / float64(fanIn))
	rng := rand.New(rand.NewPCG(seed, 0x6b79a2c3d4e5f601))
	fillGen(t, func() float64 { return rng.NormFloat64() * std })
}

// KaimingUniform fills t with U(-a, a), a = √(6/fanIn) (He et al. 2015, uniform
// variant).
func KaimingUniform(t *tensor.Tensor, fanIn int, seed uint64) {
	a := math.Sqrt(6.0 / float64(fanIn))
	fillUniform(t, -a, a, seed)
}

// Zeros fills t with 0 (the conventional bias init).
func Zeros(t *tensor.Tensor) {
	fillGen(t, func() float64 { return 0 })
}

func fillUniform(t *tensor.Tensor, lo, hi float64, seed uint64) {
	rng := rand.New(rand.NewPCG(seed, 0x6b79a2c3d4e5f601))
	fillGen(t, func() float64 { return lo + rng.Float64()*(hi-lo) })
}

// fillGen writes gen() into every element of t in row-major (flat) order.
//
// For the overwhelmingly common case — a freshly allocated, contiguous,
// offset-0 tensor — it writes straight into the typed backing slice, one dtype
// decision for the whole tensor instead of the per-element SetF64(Unravel(i)…)
// path, which allocated an index slice and ran a strided dispatch for EVERY
// element (a real cost when initializing large weight matrices; it showed up as
// ~11% of a decode benchmark's CPU profile — all of it model setup). The values
// are bit-identical to the slow path: gen() is called in the same flat order and
// the f64→dtype conversion mirrors storage.setF64 exactly (float32() for F32,
// f32ToF16/BF16 for the half floats). Non-contiguous or offset views fall back to
// the general path, so a fill through a transposed view still behaves correctly.
func fillGen(t *tensor.Tensor, gen func() float64) {
	n := t.Numel()
	if t.IsContiguous() && t.Offset() == 0 {
		switch t.Dtype() {
		case tensor.F64:
			d := t.Storage().F64()
			for i := range n {
				d[i] = gen()
			}
			return
		case tensor.F32:
			d := t.Storage().F32()
			for i := range n {
				d[i] = float32(gen())
			}
			return
		}
		// F16/BF16 and any future dtype fall through to the general path, whose
		// SetF64 carries the correct half-float rounding.
	}
	shape := t.Shape()
	for i := range n {
		t.SetF64(gen(), tensor.Unravel(i, shape)...)
	}
}

// --- Truncated normal (§T15 extension) ---

// truncNormalCfg holds the resolved TruncNormal parameters.
type truncNormalCfg struct{ mean, std, a, b float64 }

// TruncNormalOption configures TruncNormal.
type TruncNormalOption func(*truncNormalCfg)

// WithTruncMean sets the mean of the underlying (untruncated) normal. Default 0.
func WithTruncMean(mean float64) TruncNormalOption {
	return func(c *truncNormalCfg) { c.mean = mean }
}

// WithTruncStd sets the standard deviation of the underlying (untruncated)
// normal. Must be > 0. Default 1.
//
// Note this is the std of the PARENT normal, not of the truncated result: the
// values you get back have a strictly smaller spread (for the default ±2σ
// window, ≈0.8796·std).
func WithTruncStd(std float64) TruncNormalOption {
	return func(c *truncNormalCfg) { c.std = std }
}

// WithTruncBounds sets the truncation window [a, b] in ABSOLUTE units (not
// multiples of std), matching torch.nn.init.trunc_normal_. Requires a < b.
// Default [-2, 2], which is ±2σ only at the default std of 1.
func WithTruncBounds(a, b float64) TruncNormalOption {
	return func(c *truncNormalCfg) { c.a, c.b = a, b }
}

// TruncNormal fills t with samples from a normal N(mean, std²) truncated to
// [a, b] — the initializer ViT, DeiT and most modern vision transformers use for
// patch embeddings, class tokens and position embeddings (Dosovitskiy et al.
// 2021), where an untruncated normal's tail draws destabilize early training.
// Reach for it wherever you want Gaussian-shaped weights with a hard guarantee
// that no outlier lands in the tensor; reach for KaimingNormal instead when you
// want exact variance preservation through a ReLU stack.
//
// Method: INVERSE-CDF (torch's _no_grad_trunc_normal_), not rejection sampling.
// It draws u ~ U(2·Φ((a−mean)/std) − 1, 2·Φ((b−mean)/std) − 1), maps it through
// erfinv, scales by std·√2 and shifts by mean. This consumes exactly one uniform
// per element (rejection sampling would consume a variable number and thread the
// RNG differently), and reproduces torch's distribution bit-for-bit in method —
// the two approaches yield the SAME distribution but different sample sequences
// for a given stream, so do not expect element-wise agreement with torch for a
// shared seed.
//
// Failure modes, stated honestly:
//   - std ≤ 0 or a ≥ b return an error; t is left untouched.
//   - If [a, b] sits far out in the parent normal's tail (torch warns at >2 std
//     from the mean), Φ(a) and Φ(b) collapse to the same float and every sample
//     degenerates toward one endpoint. This is a property of the method, not a
//     bug; it is NOT rejected, so validate your bounds.
//   - A final clamp to [a, b] guards the erfinv(±1) = ±Inf edge, so the [a, b]
//     containment guarantee is absolute even under rounding.
func TruncNormal(t *tensor.Tensor, seed uint64, opts ...TruncNormalOption) error {
	cfg := truncNormalCfg{mean: 0, std: 1, a: -2, b: 2}
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.std <= 0 {
		return fmt.Errorf("nn: TruncNormal std must be > 0, got %v", cfg.std)
	}
	if !(cfg.a < cfg.b) {
		return fmt.Errorf("nn: TruncNormal needs a < b, got [%v, %v]", cfg.a, cfg.b)
	}
	// Uniform window in erfinv-space: [2Φ(α)−1, 2Φ(β)−1] with Φ the standard
	// normal CDF, α = (a−mean)/std, β = (b−mean)/std.
	lo := 2*normCDF((cfg.a-cfg.mean)/cfg.std) - 1
	hi := 2*normCDF((cfg.b-cfg.mean)/cfg.std) - 1

	rng := rand.New(rand.NewPCG(seed, 0x6b79a2c3d4e5f601))
	for i := range t.Numel() {
		u := lo + rng.Float64()*(hi-lo)
		v := math.Erfinv(u)*cfg.std*math.Sqrt2 + cfg.mean
		v = math.Min(math.Max(v, cfg.a), cfg.b)
		t.SetF64(v, tensor.Unravel(i, t.Shape())...)
	}
	return nil
}

// normCDF is the standard normal cumulative distribution function Φ.
func normCDF(x float64) float64 { return 0.5 * (1 + math.Erf(x/math.Sqrt2)) }

// --- Orthogonal (§T15 extension) ---

// orthogonalCfg holds the resolved Orthogonal parameters.
type orthogonalCfg struct{ gain float64 }

// OrthogonalOption configures Orthogonal.
type OrthogonalOption func(*orthogonalCfg)

// WithOrthogonalGain scales the orthogonal factor by gain. Default 1.
//
// The conventional choices mirror torch.nn.init.calculate_gain: 1 for linear or
// tanh-ish stacks, √2 for ReLU. Note gain ≠ 1 means the result is a SCALED
// orthogonal matrix, so WᵀW = gain²·I, not I.
func WithOrthogonalGain(gain float64) OrthogonalOption {
	return func(c *orthogonalCfg) { c.gain = gain }
}

// Orthogonal fills t with a (semi-)orthogonal matrix times gain — Saxe et al.
// (2014), "Exact solutions to the nonlinear dynamics of learning in deep linear
// networks". A random Gaussian matrix is orthonormalized by QR, so every
// singular value is exactly gain: the map neither amplifies nor attenuates any
// direction, which is why it is the default recurrent-weight init for RNN/LSTM/
// GRU stacks and a common choice for deep plain-linear networks where Xavier
// still lets gradients drift over many layers. Reach for Xavier/Kaiming instead
// in ordinary feed-forward CNN/MLP stacks — orthogonality buys little there and
// costs a QR.
//
// Non-square handling follows torch: t is treated as a rows×cols matrix with
// rows = t.Shape()[0] and cols = Numel/rows (all trailing dims flattened into
// the columns), so a Conv2D kernel [outC, inC, kh, kw] is orthogonalized as
// [outC, inC·kh·kw]. When rows < cols the matrix is transposed before QR and
// transposed back after, giving ROW-orthonormality (W·Wᵀ = gain²·I) in the wide
// case and COLUMN-orthonormality (Wᵀ·W = gain²·I) in the tall/square case. Only
// min(rows, cols) directions can be orthogonal, so one of the two always holds
// and the other cannot.
//
// Reuses the repo's backward-stable Householder QR (linalg.QR) rather than
// Gram-Schmidt. Sign fixing: the Q columns are multiplied by the sign of the
// corresponding diagonal entry of R, which makes the result a deterministic
// function of the Gaussian draw. DEVIATION from torch: torch uses sign(0) = 0,
// which would zero out a column and silently break orthogonality; a zero
// diagonal is measure-zero for a Gaussian draw, but this implementation maps it
// to +1 so the WᵀW ≈ I guarantee holds unconditionally.
//
// Failure modes: t must be rank ≥ 2 with a non-empty shape, else an error is
// returned and t is untouched. Rank-1 tensors (biases) have no orthogonality to
// impose and are rejected rather than silently reshaped.
func Orthogonal(t *tensor.Tensor, seed uint64, opts ...OrthogonalOption) error {
	cfg := orthogonalCfg{gain: 1}
	for _, o := range opts {
		o(&cfg)
	}
	if t.Ndim() < 2 {
		return fmt.Errorf("nn: Orthogonal needs a rank-≥2 tensor, got %v", t.Shape())
	}
	rows := t.Shape()[0]
	if rows <= 0 || t.Numel() <= 0 {
		return fmt.Errorf("nn: Orthogonal needs a non-empty tensor, got %v", t.Shape())
	}
	cols := t.Numel() / rows

	// Gaussian draw, laid out row-major as rows×cols.
	rng := rand.New(rand.NewPCG(seed, 0x6b79a2c3d4e5f601))
	flat := make([]float64, rows*cols)
	for i := range flat {
		flat[i] = rng.NormFloat64()
	}

	// linalg.QR requires m ≥ n, so transpose the wide case (torch does the same).
	m, n := rows, cols
	transposed := rows < cols
	if transposed {
		t2 := make([]float64, rows*cols)
		for i := range rows {
			for j := range cols {
				t2[j*rows+i] = flat[i*cols+j]
			}
		}
		flat, m, n = t2, cols, rows
	}

	q, r, err := linalg.QR(tensor.FromFloat64(tensor.Shape{m, n}, flat))
	if err != nil {
		return fmt.Errorf("nn: Orthogonal QR failed: %w", err)
	}

	// Sign-fix Q's columns by sign(diag(R)); 0 → +1 (see doc).
	qm := make([]float64, m*n)
	for j := range n {
		s := 1.0
		if r.AtF64(j, j) < 0 {
			s = -1
		}
		for i := range m {
			qm[i*n+j] = q.AtF64(i, j) * s * cfg.gain
		}
	}

	// Undo the transpose, then write back over t's flattened row-major layout.
	if transposed {
		t2 := make([]float64, rows*cols)
		for i := range m {
			for j := range n {
				t2[j*m+i] = qm[i*n+j]
			}
		}
		qm = t2
	}
	for i := range t.Numel() {
		t.SetF64(qm[i], tensor.Unravel(i, t.Shape())...)
	}
	return nil
}
