package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// Robustness class-audit (hardening, cf. format/*/hostile_test.go): the recently
// added exp/softmax/log-space nn/ algorithms — sigmoid/selective/multi-token/
// stick-breaking/CoPE attention, Aaren, and the mLSTM/RG-LRU/HGRN recurrent cells —
// must NOT emit NaN/Inf or panic on adversarial-but-valid degenerate inputs
// (single-token sequences, all-equal / all-zero inputs, fully-masked or fully-
// decayed rows, ±1e4 magnitudes), and must produce FINITE gradients where a
// backward is defined. Each case below is either a passing hardening asset (the
// algorithm was already robust) or pins a fix made in the algorithm's own file:
//
//   - HGRN (nn/hgrn.go): with the default γ=0 a saturated forget gate underflows to
//     exactly 0, so log f = −∞ made the parallel decay matrix NaN (−∞−(−∞)) — while
//     the sequential scan stayed finite, so the NaN also broke the parallel≡
//     sequential duality. FIX: floor f at ε (hgrnForgetFloor) before the log.
//   - RG-LRU (nn/griffin.go): a saturated recurrence gate gives decay a=1, so the
//     input normalizer √(1−a²)=√0 has an infinite VJP g/(2·√·) → NaN gradient. FIX:
//     floor the radicand at ε (rgLRUVarFloor) before the √.
//
// All other algorithm×case pairs were already robust; their passing tests are the
// hardening assets.

// hostileAssertFinite scans every element of x and fails if any is NaN or ±Inf.
func hostileAssertFinite(t *testing.T, name string, x *tensor.Tensor) {
	t.Helper()
	for i := range x.Numel() {
		coord := tensor.Unravel(i, x.Shape())
		v := x.AtF64(coord...)
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("%s: non-finite output at %v: %v", name, coord, v)
		}
	}
}

// hostileAssertGradFinite runs fwd on x, sums to a scalar loss, backprops, and
// asserts every non-nil parameter gradient (and the forward output) is finite. It
// is the degenerate-input finite-gradient guarantee (no NaN grad from a 0/0 or
// log(0)/√0 in a VJP).
func hostileAssertGradFinite(t *testing.T, name string, fwd func(*backend.Context, *tensor.Tensor) (*tensor.Tensor, error), params []*tensor.Tensor, x *tensor.Tensor) {
	t.Helper()
	tape := autograd.NewTape()
	out, err := fwd(tape.Context(), x)
	if err != nil {
		t.Fatalf("%s: forward error: %v", name, err)
	}
	hostileAssertFinite(t, name+" [fwd]", out)
	loss, err := sumAll(tape.Context(), out)
	if err != nil {
		t.Fatalf("%s: loss error: %v", name, err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatalf("%s: backward error: %v", name, err)
	}
	for pi, p := range params {
		g := tape.Grad(p)
		if g == nil {
			continue
		}
		for i := range g.Numel() {
			coord := tensor.Unravel(i, g.Shape())
			v := g.AtF64(coord...)
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("%s: non-finite gradient, param %d at %v: %v", name, pi, coord, v)
			}
		}
	}
}

// hostileConst builds a [seq,dim] matrix with every element equal to val — covers
// the single-token (seq=1), all-zero (val=0), all-equal, and extreme-magnitude
// (val=±1e4) degenerate inputs from one helper. A constant input makes Q, K and V
// per-row identical, the "Q=K=V=constant" adversarial case.
func hostileConst(seq, dim int, val float64) *tensor.Tensor {
	x := tensor.New(tensor.F64, tensor.Shape{seq, dim})
	for i := range seq {
		for j := range dim {
			x.SetF64(val, i, j)
		}
	}
	return x
}

// hostileCases is the shared degenerate-input battery (seq length × fill value).
var hostileCases = []struct {
	name string
	seq  int
	val  float64
}{
	{"single-token", 1, 1.0},      // seq len 1: softmax/recurrence over one step
	{"all-zero", 5, 0.0},          // zero Q/K/V and gates
	{"all-equal", 5, 3.0},         // every token identical
	{"single-token-zero", 1, 0.0}, // one step, zero input
	{"extreme-pos", 5, 1e4},       // huge scores/gates — must clamp, not overflow
	{"extreme-neg", 5, -1e4},      // huge-negative — must not underflow to NaN
	{"extreme-single", 1, 1e4},    // one step at extreme magnitude
}

func TestHostileSigmoidAttention(t *testing.T) {
	const dim, heads = 4, 2
	for _, c := range hostileCases {
		for _, causal := range []bool{false, true} {
			opts := []nn.SigmoidAttentionOption{nn.WithLearnableSigmoidBias()}
			if causal {
				opts = append(opts, nn.WithSigmoidAttnCausal())
			}
			s, err := nn.NewSigmoidAttention(tensor.F64, dim, heads, 1, opts...)
			if err != nil {
				t.Fatal(err)
			}
			hostileAssertGradFinite(t, "sigmoid "+c.name, s.Forward, s.Params(), hostileConst(c.seq, dim, c.val))
		}
	}
	// Fully-"masked" limits: bias → −∞ drives every weight to 0 (attend to nothing,
	// output 0); bias → +∞ drives every weight to 1. Both must be finite, no 0/0.
	for _, b := range []float64{-1e30, 1e30} {
		s, err := nn.NewSigmoidAttention(tensor.F64, dim, heads, 1, nn.WithSigmoidAttnCausal(), nn.WithSigmoidAttnBias(b))
		if err != nil {
			t.Fatal(err)
		}
		out, err := s.Forward(backend.NewContext(), hostileConst(5, dim, 2.0))
		if err != nil {
			t.Fatal(err)
		}
		hostileAssertFinite(t, "sigmoid saturated-bias", out)
	}
}

func TestHostileSelectiveAttention(t *testing.T) {
	const dim, heads = 4, 2
	for _, c := range hostileCases {
		s, err := nn.NewSelectiveAttention(tensor.F64, dim, heads, 1)
		if err != nil {
			t.Fatal(err)
		}
		hostileAssertGradFinite(t, "selective "+c.name, s.Forward, s.Params(), hostileConst(c.seq, dim, c.val))
	}
}

func TestHostileMultiTokenAttention(t *testing.T) {
	const dim, heads = 4, 2
	for _, c := range hostileCases {
		for _, bidir := range []bool{false, true} {
			opts := []nn.MultiTokenAttentionOption{nn.WithMTAKeyQueryKernel(3, 3)}
			if bidir {
				opts = append(opts, nn.WithMTABidirectional())
			}
			m, err := nn.NewMultiTokenAttention(tensor.F64, dim, heads, 1, opts...)
			if err != nil {
				t.Fatal(err)
			}
			hostileAssertGradFinite(t, "mta "+c.name, m.Forward, m.Params(), hostileConst(c.seq, dim, c.val))
		}
	}
	// Head convolution engaged (ch=heads) under extreme magnitudes.
	m, err := nn.NewMultiTokenAttention(tensor.F64, dim, heads, 1, nn.WithMTAKeyQueryKernel(3, 3), nn.WithMTAHeadKernel(heads))
	if err != nil {
		t.Fatal(err)
	}
	hostileAssertGradFinite(t, "mta head-conv extreme", m.Forward, m.Params(), hostileConst(5, dim, 1e4))
}

func TestHostileStickBreakingAttention(t *testing.T) {
	const dim, heads = 4, 2
	for _, c := range hostileCases {
		// With the learnable per-head break bias so the bias path is exercised too.
		s, err := nn.NewStickBreakingAttention(tensor.F64, dim, heads, 1, nn.WithStickBreakingBias())
		if err != nil {
			t.Fatal(err)
		}
		// extreme-neg drives every break probability β→0 (a fully-un-allocated stick,
		// row sum → 0, "attend to nothing"); extreme-pos drives β→1. Both stay finite
		// through the log-space reverse-cumsum — no 0/0 or log(0).
		hostileAssertGradFinite(t, "stickbreak "+c.name, s.Forward, s.Params(), hostileConst(c.seq, dim, c.val))
	}
}

func TestHostileCoPEAttention(t *testing.T) {
	const dim, heads = 4, 2
	for _, c := range hostileCases {
		cope, err := nn.NewCoPEAttention(tensor.F64, dim, heads, 1, nn.WithCoPEMaxPos(4))
		if err != nil {
			t.Fatal(err)
		}
		hostileAssertGradFinite(t, "cope "+c.name, cope.Forward, cope.Params(), hostileConst(c.seq, dim, c.val))
	}
}

func TestHostileMLSTM(t *testing.T) {
	const dim = 4
	for _, expForget := range []bool{false, true} {
		for _, heads := range []int{1, 2} {
			for _, c := range hostileCases {
				opts := []nn.MLSTMOption{nn.WithMLSTMHeads(heads)}
				if expForget {
					opts = append(opts, nn.WithMLSTMExpForget())
				}
				m, err := nn.NewMLSTM(tensor.F64, dim, 1, opts...)
				if err != nil {
					t.Fatal(err)
				}
				name := "mLSTM"
				if expForget {
					name = "mLSTM-expF"
				}
				hostileAssertGradFinite(t, name+" "+c.name, m.Forward, m.Params(), hostileConst(c.seq, dim, c.val))
			}
		}
	}
}

func TestHostileRGLRU(t *testing.T) {
	const dim = 4
	for _, c := range hostileCases {
		r, err := nn.NewRGLRU(tensor.F64, dim, nn.WithRGLRUSeed(1))
		if err != nil {
			t.Fatal(err)
		}
		x := hostileConst(c.seq, dim, c.val)
		// FIX ANCHOR: extreme inputs saturate the recurrence gate so a=1 and √(1−a²)=√0
		// — the forward is finite but the √ VJP divides by zero without the ε floor.
		hostileAssertGradFinite(t, "RGLRU parallel "+c.name, r.Forward, r.Params(), x)
		hostileAssertGradFinite(t, "RGLRU sequential "+c.name, r.ForwardSequential, r.Params(), x)
		hostileRecurrentDuality(t, "RGLRU "+c.name, r.Forward, r.ForwardSequential, x)
	}
}

func TestHostileHGRN(t *testing.T) {
	const dim = 4
	// γ=0 (default) is the failure surface: a saturated forget gate underflows to 0
	// so log f = −∞ made the parallel decay matrix NaN (fixed by hgrnForgetFloor).
	// Also exercise a mid γ and the output gate.
	for _, gamma := range []float64{0.0, 0.5} {
		for _, outGate := range []bool{false, true} {
			for _, c := range hostileCases {
				opts := []nn.HGRNOption{nn.WithHGRNSeed(1), nn.WithHGRNLowerBound(gamma)}
				if outGate {
					opts = append(opts, nn.WithHGRNOutputGate())
				}
				h, err := nn.NewHGRN(tensor.F64, dim, opts...)
				if err != nil {
					t.Fatal(err)
				}
				x := hostileConst(c.seq, dim, c.val)
				hostileAssertGradFinite(t, "HGRN parallel "+c.name, h.Forward, h.Params(), x)
				hostileAssertGradFinite(t, "HGRN sequential "+c.name, h.ForwardSequential, h.Params(), x)
				hostileRecurrentDuality(t, "HGRN "+c.name, h.Forward, h.ForwardSequential, x)
			}
		}
	}
}

func TestHostileAaren(t *testing.T) {
	const dim, heads = 4, 2
	for _, c := range hostileCases {
		for _, bidir := range []bool{false, true} {
			opts := []nn.AarenOption{nn.WithAarenHeads(heads), nn.WithAarenSeed(1)}
			if bidir {
				opts = append(opts, nn.WithAarenBidirectional())
			}
			a, err := nn.NewAaren(tensor.F64, dim, opts...)
			if err != nil {
				t.Fatal(err)
			}
			hostileAssertGradFinite(t, "aaren "+c.name, a.Forward, a.Params(), hostileConst(c.seq, dim, c.val))
		}
		// O(1) recurrent/streaming path (inference-only, no tape): folding extreme
		// tokens into the (m,n,d) state must stay finite (the running-max stabilizer).
		a, err := nn.NewAaren(tensor.F64, dim, nn.WithAarenHeads(heads), nn.WithAarenSeed(1))
		if err != nil {
			t.Fatal(err)
		}
		x := hostileConst(c.seq, dim, c.val)
		row := func(i int) *tensor.Tensor {
			r := tensor.New(tensor.F64, tensor.Shape{1, dim})
			for j := range dim {
				r.SetF64(x.AtF64(i, j), 0, j)
			}
			return r
		}
		st, err := a.NewAarenState(backend.NewContext(), row(c.seq-1))
		if err != nil {
			t.Fatal(err)
		}
		var out *tensor.Tensor
		for i := range c.seq {
			if out, err = a.StepRecurrent(st, row(i)); err != nil {
				t.Fatal(err)
			}
		}
		hostileAssertFinite(t, "aaren StepRecurrent "+c.name, out)
	}
}

// hostileRecurrentDuality asserts the parallel and sequential forms agree even on
// degenerate inputs — the fixes must not silently diverge the two paths (the §V16
// duality is the strongest correctness anchor for these gated linear recurrences).
func hostileRecurrentDuality(t *testing.T, name string, par, seq func(*backend.Context, *tensor.Tensor) (*tensor.Tensor, error), x *tensor.Tensor) {
	t.Helper()
	p, err := par(backend.NewContext(), x)
	if err != nil {
		t.Fatalf("%s: parallel error: %v", name, err)
	}
	s, err := seq(backend.NewContext(), x)
	if err != nil {
		t.Fatalf("%s: sequential error: %v", name, err)
	}
	if d := maxAbsDiff(p, s); d > 1e-8 {
		t.Errorf("%s: parallel≢sequential on degenerate input, max abs diff %.3g", name, d)
	}
}
