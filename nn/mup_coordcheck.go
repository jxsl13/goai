package nn

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// CoordCheck is the μP "coordinate check" of Yang, Hu et al. (arXiv:2011.14522,
// "Tensor Programs V") — the standard diagnostic that VALIDATES a Maximal-Update
// Parametrization (μP) implementation.
//
// For practitioners: it answers "is my μP actually μP?". It builds the same MLP at a
// sweep of widths, runs a fixed input through it (optionally after a few optimizer
// steps), and records each layer's activation "coordinate scale" (the RMS of the
// pre-activations). Under CORRECT μP those scales stay ~width-INVARIANT as the model
// grows wider; under Standard Parametrization (SP) they blow up or vanish with width.
// A flat coordinate check across widths is the visual/numeric proof that μP works and
// that hyperparameters tuned on a small proxy width will transfer to a large one
// (μTransfer). Run once with muP=true and once with muP=false and compare the tables
// via [CoordCheckStable]: the μP ratios should be small and near 1, the SP ratios much
// larger — that contrast is the whole point of the check.
//
// For the mechanics: the built-in model is a Depth-hidden-layer MLP width→width with a
// zero-initialized scalar readout, ReLU activations, standard fan-in ([KaimingNormal])
// hidden init, trained on a fixed synthetic regression target. The ONLY differences
// between the μP and SP runs are the ones the parametrization dictates, all taken from
// the [MuP] rules (not reimplemented here): the readout output multiplier
// ([MuP.ReadoutMult] = 1/m under μP, 1 under SP) and the per-group learning rates
// ([MuP.InputLRScale] for the input layer, [MuP.HiddenLRScale] for hidden layers,
// [MuP.ReadoutLRScale] for the readout; all 1 under SP), where m = width/BaseWidth. Init
// is identical for both, so the divergence is a training-time phenomenon: SP's un-scaled
// readout LR/multiplier make the coordinates grow with width, μP's 1/m scaling holds them
// steady. Attention scaling ([MuP.AttentionScale]) has no analog in a plain MLP and is
// not exercised here.
//
// The [MuP] LR rules are the Adam column of Yang & Hu Table 8 ("for the Adam optimizer"),
// so the check trains with [Adam] (SGD-μP would need the distinct SGD-column rules, which
// [MuP] does not expose). This mirrors the standard microsoft/mup coordinate check.
//
// The zero value is not usable; set the fields and call [CoordCheck.Run].
type CoordCheck struct {
	// Widths is the width sweep, e.g. {64,128,256,512,1024}. Must hold ≥2 distinct
	// positive widths — a single width proves nothing about width-invariance.
	Widths []int
	// BaseWidth is the μP base (proxy) width; every μP scaling is relative to it via the
	// width multiplier m = width/BaseWidth. Must be ≥1. Typically the smallest swept width.
	BaseWidth int
	// Depth is the number of hidden layers L (each width→width, the first mapping InDim→width).
	// Must be ≥1. The result table has Depth+1 columns: one pre-activation per hidden layer
	// plus the output logit.
	Depth int
	// InDim is the fixed input feature dimension (held constant across the sweep, so the
	// input layer's coordinate scale is width-independent by construction). Must be ≥1.
	InDim int
	// Batch is the number of rows in the fixed probe input. Must be ≥1.
	Batch int
	// Steps is how many SGD steps to take before measuring. 0 measures at initialization
	// (where μP and SP nearly coincide); a handful of steps (e.g. 5–10) is what surfaces the
	// μP-vs-SP contrast, since the divergence is driven by the per-group learning rates.
	// Must be ≥0.
	Steps int
	// LR is the base learning rate, before μP per-group scaling. Must be >0.
	LR float64
	// Seed makes the whole check deterministic: the same CoordCheck value yields a bit-identical
	// table on every run (§V13). Weights are seeded per (Seed, width, layer); the probe input
	// and target are seeded from Seed alone, so every width sees the SAME fixed input.
	Seed uint64
}

// CoordTable is the coordinate-check result for one parametrization: for each swept width,
// the per-layer coordinate scale (activation RMS). Produced by [CoordCheck.Run] and consumed
// by [CoordCheckStable].
type CoordTable struct {
	// Widths mirrors the CoordCheck.Widths that produced this table, in order.
	Widths []int
	// RMS is indexed [widthIndex][layer]. Each row has Depth+1 entries: layers 0..Depth-1 are
	// the hidden-layer pre-activation RMS (layer 0 = the first, InDim→width, layer), and the
	// last entry (index Depth) is the output-logit RMS — μP's headline width-invariant quantity.
	RMS [][]float64
}

// Run executes the coordinate check for one parametrization: μP when muP is true, Standard
// Parametrization when false. It builds the built-in MLP at each width in Widths, initializes
// it deterministically, takes Steps SGD updates on a fixed input/target, and records the RMS
// of every hidden pre-activation and of the output logit. The returned [CoordTable] has one
// row per width. Run reports an error for a malformed configuration (see the field docs for the
// bounds). It never mutates the receiver, so μP and SP runs from the same value are independent.
func (c CoordCheck) Run(muP bool) (*CoordTable, error) {
	switch {
	case len(c.Widths) < 2:
		return nil, fmt.Errorf("nn: CoordCheck needs ≥2 widths, got %d", len(c.Widths))
	case c.BaseWidth < 1:
		return nil, fmt.Errorf("nn: CoordCheck BaseWidth must be ≥1, got %d", c.BaseWidth)
	case c.Depth < 1:
		return nil, fmt.Errorf("nn: CoordCheck Depth must be ≥1, got %d", c.Depth)
	case c.InDim < 1:
		return nil, fmt.Errorf("nn: CoordCheck InDim must be ≥1, got %d", c.InDim)
	case c.Batch < 1:
		return nil, fmt.Errorf("nn: CoordCheck Batch must be ≥1, got %d", c.Batch)
	case c.Steps < 0:
		return nil, fmt.Errorf("nn: CoordCheck Steps must be ≥0, got %d", c.Steps)
	case !(c.LR > 0):
		return nil, fmt.Errorf("nn: CoordCheck LR must be >0, got %g", c.LR)
	}
	seen := make(map[int]bool, len(c.Widths))
	for _, w := range c.Widths {
		if w < 1 {
			return nil, fmt.Errorf("nn: CoordCheck widths must be ≥1, got %d", w)
		}
		if seen[w] {
			return nil, fmt.Errorf("nn: CoordCheck widths must be distinct, %d repeats", w)
		}
		seen[w] = true
	}

	p := NewMuP(c.BaseWidth)
	tbl := &CoordTable{Widths: append([]int(nil), c.Widths...), RMS: make([][]float64, len(c.Widths))}
	for wi, w := range c.Widths {
		row, err := c.runWidth(p, w, muP)
		if err != nil {
			return nil, err
		}
		tbl.RMS[wi] = row
	}
	return tbl, nil
}

// runWidth builds, trains and measures the model at a single width, returning the Depth+1
// per-layer RMS row (hidden pre-activations followed by the output logit).
func (c CoordCheck) runWidth(p *MuP, width int, muP bool) ([]float64, error) {
	m := &coordMLP{}
	// Layer 0 maps the fixed input dim → width (the μP "input" weight); layers 1..Depth-1 are
	// width→width hidden weights; the readout maps width→1. Standard fan-in (Kaiming) init on all,
	// biases dropped so the parametrization is the sole μP/SP difference.
	for i := 0; i < c.Depth; i++ {
		in := width
		if i == 0 {
			in = c.InDim
		}
		l := NewLinear(tensor.F64, in, width, 0)
		l.B = nil
		KaimingNormal(l.W, in, mixSeed(c.Seed, uint64(width), uint64(i)))
		m.layers = append(m.layers, l)
	}
	m.readout = NewLinear(tensor.F64, width, 1, 0)
	m.readout.B = nil
	// Readout zero-init (the standard coordinate-check convention, microsoft/mup
	// readout_zero_init): the output logit starts at exactly 0 for every width, so the
	// measured output RMS is purely the FEATURE-LEARNED signal, undisturbed by an init
	// term that under μP would otherwise vanish as 1/m and pollute the width ratio.
	Zeros(m.readout.W)

	// Readout output multiplier: μP scales the logits by 1/m (MuP.ReadoutMult), SP by 1. Applied
	// as an elementwise OpMul against a constant tensor (NOTE B60: OpMul, not OpAXPY, for a scalar
	// that may be 1). The multiplier is never a trained parameter.
	mult := 1.0
	if muP {
		mult = p.ReadoutMult(width)
	}
	m.mult = tensor.New(tensor.F64, tensor.Shape{c.Batch, 1})
	fill(m.mult, mult)

	// Fixed probe input and target, seeded from Seed alone so every width sees the same data.
	x := randTensor(tensor.Shape{c.Batch, c.InDim}, 1, mixSeed(c.Seed, seedInput, 0))
	tgt := randTensor(tensor.Shape{c.Batch, 1}, 1, mixSeed(c.Seed, seedTarget, 0))

	// Per-group learning rates from the MuP rules (all ×base LR). Under SP every scale is 1.
	inLR, hidLR, outLR := c.LR, c.LR, c.LR
	if muP {
		inLR = c.LR * p.InputLRScale()
		hidLR = c.LR * p.HiddenLRScale(width)
		outLR = c.LR * p.ReadoutLRScale(width)
	}
	opts := make([]Optimizer, 0, c.Depth+1)
	for i, l := range m.layers {
		lr := hidLR
		if i == 0 {
			lr = inLR // the input layer keeps the un-scaled LR under μP
		}
		opts = append(opts, NewAdam(l.Params(), lr))
	}
	opts = append(opts, NewAdam(m.readout.Params(), outLR))

	for step := 0; step < c.Steps; step++ {
		tape := autograd.NewTape()
		_, logit, err := m.forward(tape.Context(), x)
		if err != nil {
			return nil, err
		}
		loss, err := mseLoss(tape.Context(), logit, tgt)
		if err != nil {
			return nil, err
		}
		if err := tape.Backward(loss); err != nil {
			return nil, err
		}
		grad := tape.Grad
		for _, o := range opts {
			if err := o.Step(grad); err != nil {
				return nil, err
			}
		}
	}

	// Measure: forward once more (no tape) and take the RMS of each pre-activation and the logit.
	pre, logit, err := m.forward(backend.NewContext(), x)
	if err != nil {
		return nil, err
	}
	row := make([]float64, 0, c.Depth+1)
	for _, z := range pre {
		row = append(row, rms(z))
	}
	row = append(row, rms(logit))
	return row, nil
}

// seed selectors for the fixed probe input and target (distinct from the per-width weight seeds).
const (
	seedInput  uint64 = 0x1000
	seedTarget uint64 = 0x2000
)

// coordMLP is the built-in coordinate-check model: Depth hidden Linear layers with ReLU, plus a
// scalar readout whose output is scaled by a constant multiplier.
type coordMLP struct {
	layers  []*Linear
	readout *Linear
	mult    *tensor.Tensor
}

// forward runs the MLP through ctx, returning each hidden layer's pre-activation (before ReLU)
// and the scaled output logits.
func (m *coordMLP) forward(ctx *backend.Context, x *tensor.Tensor) (pre []*tensor.Tensor, logit *tensor.Tensor, err error) {
	h := x
	for _, l := range m.layers {
		z, err := l.Forward(ctx, h)
		if err != nil {
			return nil, nil, err
		}
		pre = append(pre, z)
		act, err := backend.Execute(ctx, backend.OpReLU, []*tensor.Tensor{z}, nil)
		if err != nil {
			return nil, nil, err
		}
		h = act[0]
	}
	raw, err := m.readout.Forward(ctx, h)
	if err != nil {
		return nil, nil, err
	}
	out, err := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{raw, m.mult}, nil)
	if err != nil {
		return nil, nil, err
	}
	return pre, out[0], nil
}

// mseLoss is the sum of squared errors between logit and target, differentiable through ctx.
func mseLoss(ctx *backend.Context, logit, tgt *tensor.Tensor) (*tensor.Tensor, error) {
	diff, err := backend.Execute(ctx, backend.OpSub, []*tensor.Tensor{logit, tgt}, nil)
	if err != nil {
		return nil, err
	}
	sq, err := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{diff[0], diff[0]}, nil)
	if err != nil {
		return nil, err
	}
	s, err := backend.Execute(ctx, backend.OpSum, []*tensor.Tensor{sq[0]}, nil)
	if err != nil {
		return nil, err
	}
	return s[0], nil
}

// CoordCheckStable reduces a [CoordTable] to the per-layer max/min RMS ratio across the width
// sweep — the coordinate check's numeric verdict. Result index i is the ratio for layer column i
// (layers 0..Depth-1 are hidden pre-activations, the last is the output logit). A ratio near 1
// means that layer's coordinate scale is width-invariant (μP working as intended); a large ratio
// means it grows or shrinks with width (the SP failure mode). Compare a μP table's ratios against
// an SP table's: the μP ratios should be small and much less than SP's, especially for the output
// column. Returns an error if the table is empty, ragged, or holds a non-positive RMS (which would
// make a ratio meaningless).
func CoordCheckStable(t *CoordTable) ([]float64, error) {
	if t == nil || len(t.RMS) == 0 {
		return nil, fmt.Errorf("nn: CoordCheckStable needs a non-empty table")
	}
	if len(t.RMS) < 2 {
		return nil, fmt.Errorf("nn: CoordCheckStable needs ≥2 widths to form a ratio, got %d", len(t.RMS))
	}
	layers := len(t.RMS[0])
	if layers == 0 {
		return nil, fmt.Errorf("nn: CoordCheckStable table has no layers")
	}
	mins := make([]float64, layers)
	maxs := make([]float64, layers)
	for l := range mins {
		mins[l] = math.Inf(1)
		maxs[l] = math.Inf(-1)
	}
	for wi, row := range t.RMS {
		if len(row) != layers {
			return nil, fmt.Errorf("nn: CoordCheckStable ragged table: row %d has %d layers, want %d", wi, len(row), layers)
		}
		for l, v := range row {
			if !(v > 0) || math.IsInf(v, 0) || math.IsNaN(v) {
				return nil, fmt.Errorf("nn: CoordCheckStable non-positive RMS %g at [%d][%d]", v, wi, l)
			}
			mins[l] = math.Min(mins[l], v)
			maxs[l] = math.Max(maxs[l], v)
		}
	}
	ratio := make([]float64, layers)
	for l := range ratio {
		ratio[l] = maxs[l] / mins[l]
	}
	return ratio, nil
}

// rms returns sqrt(mean(x²)) over all elements of t — the coordinate scale the check tracks.
func rms(t *tensor.Tensor) float64 {
	n := t.Numel()
	var s float64
	for i := 0; i < n; i++ {
		v := t.AtF64(tensor.Unravel(i, t.Shape())...)
		s += v * v
	}
	return math.Sqrt(s / float64(n))
}

// fill sets every element of t to v.
func fill(t *tensor.Tensor, v float64) {
	for i := 0; i < t.Numel(); i++ {
		t.SetF64(v, tensor.Unravel(i, t.Shape())...)
	}
}

// randTensor returns a shape tensor of N(0, std²) samples, seeded deterministically.
func randTensor(shape tensor.Shape, std float64, seed uint64) *tensor.Tensor {
	t := tensor.New(tensor.F64, shape)
	rng := rand.New(rand.NewPCG(seed, 0x9e3779b97f4a7c15))
	for i := 0; i < t.Numel(); i++ {
		t.SetF64(rng.NormFloat64()*std, tensor.Unravel(i, shape)...)
	}
	return t
}

// mixSeed combines a base seed with two selectors into one deterministic stream seed.
func mixSeed(base, a, b uint64) uint64 {
	h := base ^ 0x9e3779b97f4a7c15
	h = (h^a)*0xff51afd7ed558ccd + 1
	h = (h^b)*0xc4ceb9fe1a85ec53 + 1
	return h
}
