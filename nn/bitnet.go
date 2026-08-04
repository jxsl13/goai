package nn

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// parallelRange splits [0,n) across GOMAXPROCS workers, running body on each disjoint
// [lo,hi) sub-range concurrently. Bodies must write only within their own range (no shared
// writes / no cross-range reduction), so the result is identical to the serial loop. Runs
// serial below a small work threshold to avoid fork/join overhead on tiny tensors.
//
//perfscan:ignore PS3048,PS3061 QAT-fill fan-out helper; fill is fraction of matmul, worker-floor tuning resource-only | same helper; fanout-s
func parallelRange(n int, body func(lo, hi int)) {
	nw := runtime.GOMAXPROCS(0)
	if nw <= 1 || n < 1<<14 {
		body(0, n)
		return
	}
	var wg sync.WaitGroup
	//perfscan:ignore PS3011 static-chunk barrier; E-core imbalance <1pct of matmul-dominated forward
	chunk := (n + nw - 1) / nw
	for lo := 0; lo < n; lo += chunk {
		hi := lo + chunk
		if hi > n {
			hi = n
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			body(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}

// parallelRows splits n independent rows across GOMAXPROCS workers, gated on the TOTAL work
// n·workPerItem (not n) so a small row count with heavy rows (e.g. 512 rows of 4096) still
// fans out. body must touch only rows in its [lo,hi) range, so the result is identical to
// the serial loop.
//
//perfscan:ignore PS3048,PS3061 parallelRows fan-out; act-fill fraction of matmul, tuning resource-only | same helper; fanout worker-count tun
func parallelRows(n, workPerItem int, body func(lo, hi int)) {
	nw := runtime.GOMAXPROCS(0)
	if nw <= 1 || n < 2 || n*workPerItem < 1<<14 {
		body(0, n)
		return
	}
	if nw > n {
		nw = n
	}
	var wg sync.WaitGroup
	//perfscan:ignore PS3011 static-chunk barrier in parallelRows; low-leverage training fill
	chunk := (n + nw - 1) / nw
	for lo := 0; lo < n; lo += chunk {
		hi := lo + chunk
		if hi > n {
			hi = n
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			body(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}

// BitLinear is the BitNet b1.58 quantization-aware-training linear layer (Ma, Wang,
// Ma, Wang, Wang, Huang, Dong, Wang, Xue & Wei 2024, "The Era of 1-bit LLMs: All
// Large Language Models are in 1.58 Bits", arXiv:2402.17764; architecture from Wang
// et al. 2023, "BitNet: Scaling 1-bit Transformers for Large Language Models",
// arXiv:2310.11453). It trains with TERNARY {−1, 0, +1} weights and 8-bit
// activations: every forward pass fake-quantizes the FULL-PRECISION latent ("master")
// weight W with the paper's absmean quantizer,
//
//	γ  = mean(|W|)                       (scalar over the whole matrix, detached)
//	W̃  = clamp(round(W/γ), −1, +1)       ∈ {−1, 0, +1}
//	Wq = γ·W̃                             (the effective weight used by the matmul)
//
// and the (pre-normalized) input x with the per-token absmax 8-bit quantizer
//
//	s  = 127 / max(|x_row|)              (one scale per row/token, detached)
//	x̃  = clamp(round(x·s), −128, 127)/s ,
//
// both through the straight-through estimator (STE): the composition is
// v + StopGradient(quant(v) − v), so the FORWARD value equals the quantized tensor
// while the BACKWARD gradient passes through to v as identity (exactly the
// round-with-STE idiom of LSQQuantize and FSQ here, and the
// `w + (quant(w) − w).detach()` of the official BitNet code). The optimizer therefore
// updates the full-precision master weights (returned by Params), which drift across
// ternary decision boundaries as training progresses — quantization-aware training
// from scratch, not a post-hoc conversion.
//
// Following the paper, a normalization layer (its SubLN slot; RMSNorm here, as in
// LLaMA and the paper's reference implementation) runs before activation
// quantization.
//
// Contrast with this package's POST-training quantizers — AWQ (AWQScaleSearch),
// GPTQ (GPTQQuantize) and HQQ (HQQQuantize) quantize an ALREADY-TRAINED
// full-precision model, trading accuracy for size after the fact — BitNet instead
// TRAINS the network under ternary constraints so it adapts to them. Contrast with
// LSQ (LSQQuantize), which is also QAT but learns a continuous step size for a
// multi-bit grid by gradient descent; BitNet b1.58 uses a fixed ternary grid whose
// scale γ is recomputed analytically from the weights each forward, never learned.
//
// Storage: a ternary weight carries log2(3) ≈ 1.58 bits of information
// (BitNetBitsPerWeight) versus 32/16 bits for float weights — a >20× storage/memory
// reduction at INFERENCE time (and matmuls that need only additions, since every
// effective weight is ±γ or 0). During training the masters stay full-precision;
// the win is realized when the trained model is exported ternary.
type BitLinear struct {
	// W is the full-precision latent ("master") weight [in, out]. The optimizer
	// updates W; the forward pass uses its ternary fake-quantization γ·W̃.
	W *tensor.Tensor
	// B is the bias [out]; nil (WithBitLinearBias(false)) disables it.
	B *tensor.Tensor
	// Norm is the pre-quantization normalization (the paper's SubLN slot,
	// RMSNorm here); nil (WithBitLinearNorm(false)) disables it.
	Norm   *RMSNorm
	quantW bool // ternary weight quantization on (default true)
	quantA bool // 8-bit activation quantization on (default true)
}

// BitLinearOption configures a BitLinear at construction.
type BitLinearOption func(*BitLinear)

// WithBitLinearBias enables or disables the additive bias (default true).
func WithBitLinearBias(enabled bool) BitLinearOption {
	return func(l *BitLinear) {
		if !enabled {
			l.B = nil
		}
	}
}

// WithBitLinearNorm enables or disables the pre-quantization RMSNorm (default
// true, per the paper's norm-before-quantization).
func WithBitLinearNorm(enabled bool) BitLinearOption {
	return func(l *BitLinear) {
		if !enabled {
			l.Norm = nil
		}
	}
}

// WithBitLinear8BitAct enables or disables the 8-bit activation quantizer
// (default true, the b1.58 configuration).
func WithBitLinear8BitAct(enabled bool) BitLinearOption {
	return func(l *BitLinear) { l.quantA = enabled }
}

// WithBitLinearWeightQuant enables or disables the ternary weight quantizer
// (default true). With both quantizers and the norm disabled, BitLinear reduces
// exactly to a plain Linear on the same weights — useful as a full-precision
// baseline and for wiring tests.
func WithBitLinearWeightQuant(enabled bool) BitLinearOption {
	return func(l *BitLinear) { l.quantW = enabled }
}

// NewBitLinear builds a BitLinear layer with Xavier-uniform full-precision master
// W [in, out], zero bias and a fresh RMSNorm(in), mirroring NewLinear.
// Deterministic via seed.
func NewBitLinear(dtype tensor.Dtype, in, out int, seed uint64, opts ...BitLinearOption) *BitLinear {
	w := tensor.New(dtype, tensor.Shape{in, out})
	XavierUniform(w, in, out, seed)
	b := tensor.New(dtype, tensor.Shape{out})
	Zeros(b)
	l := &BitLinear{W: w, B: b, Norm: NewRMSNorm(dtype, in), quantW: true, quantA: true}
	for _, o := range opts {
		o(l)
	}
	return l
}

// Forward computes norm → 8-bit activation fake-quant → matmul with the ternary
// effective weight γ·W̃ (+ bias if present) for x [batch, in], each stage subject
// to its option. Gradients reach the full-precision masters via the STE.
func (l *BitLinear) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 {
		return nil, fmt.Errorf("nn: BitLinear expects x rank-2 [batch,in], got %v", x.Shape())
	}
	if x.Shape()[1] != l.W.Shape()[0] {
		return nil, fmt.Errorf("nn: BitLinear in-dim %d != x cols %d", l.W.Shape()[0], x.Shape()[1])
	}
	h := x
	var err error
	if l.Norm != nil {
		if h, err = l.Norm.Forward(ctx, h); err != nil {
			return nil, err
		}
	}
	if l.quantA {
		if h, err = BitQuantizeAct8(ctx, h); err != nil {
			return nil, err
		}
	}
	weff := l.W
	if l.quantW {
		if weff, _, err = BitQuantizeWeight(ctx, l.W); err != nil {
			return nil, err
		}
	}
	//perfscan:ignore PS3024 variadic-pack alloc, resource-only; one matmul dispatch per forward
	y, err := bitExec(ctx, backend.OpMatMul, nil, h, weff)
	if err != nil {
		return nil, err
	}
	if l.B != nil {
		//perfscan:ignore PS3024 variadic-pack alloc, resource-only; one bias dispatch per forward
		return bitExec(ctx, backend.OpAddBias, nil, y, l.B)
	}
	return y, nil
}

// Params returns the trainable FULL-PRECISION tensors — the latent master weight
// W, the bias (if present) and the norm gain (if present). These are what the
// optimizer updates; the ternary weights exist only inside the forward pass.
func (l *BitLinear) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{l.W}
	if l.B != nil {
		ps = append(ps, l.B)
	}
	if l.Norm != nil {
		ps = append(ps, l.Norm.Gamma)
	}
	return ps
}

// bitnetScaleFloor guards the quantizer scales against division by ~zero (the
// official BitNet code's clamp(min=1e-5)).
const bitnetScaleFloor = 1e-5

// BitQuantizeWeight fake-quantizes w to the BitNet b1.58 ternary grid via the
// absmean quantizer: γ = mean(|w|) (detached, returned), effective weight
// γ·clamp(round(w/γ), −1, +1) ∈ {−γ, 0, +γ}. The round/clamp is composed with
// backend.OpStopGradient as w + sg(quant(w) − w), so the returned tensor equals the
// quantized weight in the forward pass while ∂effective/∂w = 1 (straight-through) —
// gradients reach the full-precision master unchanged.
func BitQuantizeWeight(ctx *backend.Context, w *tensor.Tensor) (*tensor.Tensor, float64, error) {
	if w.Numel() == 0 {
		return nil, 0, fmt.Errorf("nn: BitQuantizeWeight got an empty weight tensor")
	}
	q := tensor.New(w.Dtype(), w.Shape())
	gamma := bitTernaryFill(w, q)
	weff, err := bitSTE(ctx, w, q)
	if err != nil {
		return nil, 0, err
	}
	return weff, gamma, nil
}

// bitTernaryFill writes the BitNet absmean ternary fake-quant of w into q (same
// shape/dtype, freshly allocated) and returns γ = mean(|w|). Both the absmean
// reduction and the ternary map sweep the FULL weight matrix on every QAT
// forward, so for a contiguous, offset-0 w it reads/writes the typed backing
// slices once (one dtype decision for the whole matrix) instead of the
// per-element AtF64(Unravel)/SetF64 dispatch that dominated the loop. The output
// is bit-identical to the general path below — the f64→dtype narrowing mirrors
// storage.setF64 (float32() for F32) and γ is accumulated in the same flat order
// (see the slowBitTernaryFill oracle). F16/BF16 fall through to the general path.
func bitTernaryFill(w, q *tensor.Tensor) float64 {
	n := w.Numel()
	if w.IsContiguous() && w.Offset() == 0 {
		switch w.Dtype() {
		case tensor.F64:
			wd, qd := w.Storage().F64(), q.Storage().F64()
			var sum float64
			//perfscan:ignore PS3010 absmean sum must match serial oracle; reassoc breaks parity, low-leverage
			for i := range n {
				sum += math.Abs(wd[i])
			}
			gamma := sum / float64(n)
			scale := math.Max(gamma, bitnetScaleFloor)
			// Pass 2 is embarrassingly parallel — each qd[i] depends only on wd[i] and the
			// shared scale, with no cross-element reduction — so a disjoint-range fan-out is
			// bit-identical to the serial loop (pass 1's gamma stays serial for an exact sum).
			parallelRange(n, func(lo, hi int) {
				//perfscan:ignore PS5001 divide feeds math.Round quantizer; reciprocal-hoist unsafe per PS5001 rule
				for i := lo; i < hi; i++ {
					t := math.Round(wd[i] / scale)
					//perfscan:ignore PS3077,PS3082 ternary clamp, fraction of matmul; builtin min/max signed-zero parity risk | same clamp; low-leverage, bit-exa
					qd[i] = scale * math.Max(-1, math.Min(1, t))
				}
			})
			return gamma
		case tensor.F32:
			wd, qd := w.Storage().F32(), q.Storage().F32()
			var sum float64
			//perfscan:ignore PS3010 F32 absmean sum, oracle-exact serial; small fraction of matmul
			for i := range n {
				sum += math.Abs(float64(wd[i]))
			}
			gamma := sum / float64(n)
			scale := math.Max(gamma, bitnetScaleFloor)
			parallelRange(n, func(lo, hi int) {
				//perfscan:ignore PS5001 F32 divide feeds math.Round; hoist unsafe per PS5001 rule
				for i := lo; i < hi; i++ {
					t := math.Round(float64(wd[i]) / scale)
					//perfscan:ignore PS3077,PS3082 F32 ternary clamp; low-leverage, parity-constrained | same F32 clamp; low-leverage/parity
					qd[i] = float32(scale * math.Max(-1, math.Min(1, t)))
				}
			})
			return gamma
		}
	}
	var sum float64
	for i := range n {
		sum += math.Abs(w.AtF64(tensor.Unravel(i, w.Shape())...))
	}
	gamma := sum / float64(n)
	scale := math.Max(gamma, bitnetScaleFloor)
	for i := range n {
		c := tensor.Unravel(i, w.Shape())
		t := math.Round(w.AtF64(c...) / scale)
		//perfscan:ignore PS3077,PS3082 non-contiguous fallback; AtF64/SetF64-dispatch-dominated, minmax negligible | same fallback path; dispatch-dom
		q.SetF64(scale*math.Max(-1, math.Min(1, t)), c...)
	}
	return gamma
}

// BitQuantizeAct8 fake-quantizes x [batch, d] to 8 bits with the BitNet b1.58
// per-token absmax quantizer: per row, s = 127/max(|x_row|) (detached), quantized
// clamp(round(x·s), −128, 127)/s. Composed with backend.OpStopGradient as
// x + sg(quant(x) − x): forward equals the quantized activations, backward is the
// identity straight-through to x.
func BitQuantizeAct8(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 {
		return nil, fmt.Errorf("nn: BitQuantizeAct8 wants rank-2 x[batch,d], got %v", x.Shape())
	}
	q := tensor.New(x.Dtype(), x.Shape())
	bitAct8Fill(x, q)
	return bitSTE(ctx, x, q)
}

// bitAct8Fill writes the per-token absmax 8-bit fake-quant of x[rows,cols] into q
// (same shape/dtype, freshly allocated). For a contiguous, offset-0 x it walks
// the typed backing slice by flat row-major index (r*cols+c) instead of the
// per-element AtF64/SetF64 dispatch — this runs over the activation matrix on
// every forward. Bit-identical to the general path (see the slowBitAct8Fill
// oracle); F16/BF16 fall through.
func bitAct8Fill(x, q *tensor.Tensor) {
	rows, cols := x.Shape()[0], x.Shape()[1]
	if x.IsContiguous() && x.Offset() == 0 {
		switch x.Dtype() {
		case tensor.F64:
			xd, qd := x.Storage().F64(), q.Storage().F64()
			// Each row's absmax → scale → quantize is self-contained (no cross-row reduction),
			// so a disjoint row-range fan-out is bit-identical to the serial loop.
			parallelRows(rows, cols, func(lo, hi int) {
				for r := lo; r < hi; r++ {
					base := r * cols
					absmax := 0.0
					for c := range cols {
						if a := math.Abs(xd[base+c]); a > absmax {
							absmax = a
						}
					}
					//perfscan:ignore PS3082 per-row scale math.Max, once per row, negligible
					s := 127 / math.Max(absmax, bitnetScaleFloor)
					//perfscan:ignore PS5001 v/s must match serial oracle; act-fill fraction of matmul
					for c := range cols {
						//perfscan:ignore PS3077,PS3082 act clamp; low-leverage, bit-exact-oracle constrained | same act clamp; low-leverage/parity
						v := math.Max(-128, math.Min(127, math.Round(xd[base+c]*s)))
						qd[base+c] = v / s
					}
				}
			})
			return
		case tensor.F32:
			xd, qd := x.Storage().F32(), q.Storage().F32()
			parallelRows(rows, cols, func(lo, hi int) {
				for r := lo; r < hi; r++ {
					base := r * cols
					absmax := 0.0
					for c := range cols {
						if a := math.Abs(float64(xd[base+c])); a > absmax {
							absmax = a
						}
					}
					//perfscan:ignore PS3082 F32 per-row scale math.Max, once per row, negligible
					s := 127 / math.Max(absmax, bitnetScaleFloor)
					//perfscan:ignore PS5001 F32 v/s must match oracle; act-fill fraction of matmul
					for c := range cols {
						//perfscan:ignore PS3077,PS3082 F32 act clamp; low-leverage/parity | same F32 clamp; low-leverage/parity
						v := math.Max(-128, math.Min(127, math.Round(float64(xd[base+c])*s)))
						qd[base+c] = float32(v / s)
					}
				}
			})
			return
		}
	}
	for r := range rows {
		absmax := 0.0
		for c := range cols {
			if a := math.Abs(x.AtF64(r, c)); a > absmax {
				absmax = a
			}
		}
		//perfscan:ignore PS3082 non-contiguous fallback; AtF64-dispatch-dominated
		s := 127 / math.Max(absmax, bitnetScaleFloor)
		for c := range cols {
			//perfscan:ignore PS3077,PS3082 fallback clamp; per-element dispatch-dominated | same fallback clamp; dispatch-dominated
			v := math.Max(-128, math.Min(127, math.Round(x.AtF64(r, c)*s)))
			q.SetF64(v/s, r, c)
		}
	}
}

// BitNetBitsPerWeight returns the information content of one ternary weight,
// log2(3) ≈ 1.58 bits — the "1.58" of BitNet b1.58. Against 32-bit floats this is
// a >20× storage reduction at inference/export time; during training the master
// weights stay full-precision (the QAT trade: train big, deploy tiny).
func BitNetBitsPerWeight() float64 { return math.Log2(3) }

// bitSTE composes the straight-through estimator v + sg(q − v): forward value q,
// backward identity into v (OpStopGradient has a zero-grad VJP), the same idiom as
// LSQQuantize and FSQ.Quantize.
func bitSTE(ctx *backend.Context, v, q *tensor.Tensor) (*tensor.Tensor, error) {
	//perfscan:ignore PS3024 bitSTE variadic-pack alloc, resource-only, one dispatch/forward
	delta, err := bitExec(ctx, backend.OpSub, nil, q, v)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 variadic-pack alloc, resource-only, one dispatch/forward
	deltaSG, err := bitExec(ctx, backend.OpStopGradient, nil, delta)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 variadic-pack alloc, resource-only, one dispatch/forward
	return bitExec(ctx, backend.OpAdd, nil, v, deltaSG)
}

func bitExec(ctx *backend.Context, op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, in, at)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}
