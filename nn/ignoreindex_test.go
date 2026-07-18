package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/npz"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

const ceIgnore = -100

// ceFixtureCases enumerates the golden's cross-product tags in the same order
// scripts/golden_ignoreindex.py writes them.
func ceFixtureCases() []struct {
	tag  string
	red  backend.Reduction
	ii   bool
	ls   float64
	name string
} {
	var out []struct {
		tag  string
		red  backend.Reduction
		ii   bool
		ls   float64
		name string
	}
	for _, r := range []struct {
		tag string
		red backend.Reduction
	}{{"mean", backend.ReductionMean}, {"sum", backend.ReductionSum}, {"none", backend.ReductionNone}} {
		for _, m := range []struct {
			tag string
			ii  bool
		}{{"noii", false}, {"ii", true}} {
			for _, l := range []struct {
				tag string
				ls  float64
			}{{"ls0", 0}, {"ls01", 0.1}} {
				out = append(out, struct {
					tag  string
					red  backend.Reduction
					ii   bool
					ls   float64
					name string
				}{
					tag:  fmt.Sprintf("%s_%s_%s", r.tag, m.tag, l.tag),
					red:  r.red,
					ii:   m.ii,
					ls:   l.ls,
					name: fmt.Sprintf("%s/%s/%s", r.tag, m.tag, l.tag),
				})
			}
		}
	}
	return out
}

// ceOpts builds the option list for a fixture case.
func ceOpts(red backend.Reduction, ii bool, ls float64) []nn.CEOption {
	opts := []nn.CEOption{nn.WithReduction(red)}
	if ii {
		opts = append(opts, nn.WithIgnoreIndex(ceIgnore))
	}
	if ls != 0 {
		opts = append(opts, nn.WithLabelSmoothing(ls))
	}
	return opts
}

// ceForwardBackward runs CrossEntropy on a fresh tape and returns (loss, dLogits).
// A non-scalar loss (ReductionNone) is backpropped as sum(loss), which seeds every
// row's cotangent with 1 — exactly what the golden's loss.sum().backward() does.
func ceForwardBackward(t *testing.T, z, tg *tensor.Tensor, opts ...nn.CEOption) (*tensor.Tensor, *tensor.Tensor) {
	t.Helper()
	tape := autograd.NewTape()
	ctx := tape.Context()
	zc := ceCopy(z)
	loss, err := nn.CrossEntropy(ctx, zc, tg, opts...)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(zc)
	if g == nil {
		t.Fatal("nil logits gradient")
	}
	return loss, g
}

// cmpF64 compares two f64 tensors elementwise with NaN treated as equal to NaN
// (the all-ignored mean case is legitimately NaN on both sides).
func cmpF64(t *testing.T, name string, got, want *tensor.Tensor, tol float64) {
	t.Helper()
	if !got.Shape().Equal(want.Shape()) {
		t.Fatalf("%s: shape %v, torch %v", name, got.Shape(), want.Shape())
	}
	gs, ws := got.Contiguous().Storage().F64(), want.Contiguous().Storage().F64()
	for i := range gs {
		if math.IsNaN(gs[i]) && math.IsNaN(ws[i]) {
			continue
		}
		if d := math.Abs(gs[i] - ws[i]); d > tol || math.IsNaN(d) {
			t.Errorf("%s elem %d: got %.17g, torch %.17g (|Δ| %.3g)", name, i, gs[i], ws[i], d)
			return
		}
	}
}

// TestCrossEntropyIgnoreIndexTorchParity is the §V16 tier-1 anchor: nn.CrossEntropy's
// WithIgnoreIndex/WithReduction options must reproduce torch.nn.functional.cross_entropy's
// forward loss AND logits gradient across the full
// reduction{mean,sum,none} × ignore_index{off,-100} × label_smoothing{0,0.1} cross-product
// to ≤ 1e-12 (f64 both sides; scripts/golden_ignoreindex.py).
//
// The smoothing × masking × mean intersection is the point of the sweep: torch smooths
// only the surviving rows and divides the mean by the surviving-row count.
func TestCrossEntropyIgnoreIndexTorchParity(t *testing.T) {
	fix, err := npz.LoadFile("testdata/ignoreindex_torch.npz")
	if err != nil {
		t.Fatal(err)
	}
	z, tg := fix["z"], fix["t"]
	if z == nil || tg == nil {
		t.Fatal("fixture missing z/t")
	}
	const tol = 1e-12
	for _, tc := range ceFixtureCases() {
		t.Run(tc.name, func(t *testing.T) {
			wantLoss, wantGrad := fix["loss_"+tc.tag], fix["grad_"+tc.tag]
			if wantLoss == nil || wantGrad == nil {
				t.Fatalf("fixture missing case %q", tc.tag)
			}
			// The "off" arm's targets are the surviving rows only (the -100s are
			// illegal without a mask), so feed the same filtered batch the golden did.
			zi, ti := z, tg
			if !tc.ii {
				zi, ti = ceFilter(t, z, tg)
			}
			loss, grad := ceForwardBackward(t, zi, ti, ceOpts(tc.red, tc.ii, tc.ls)...)
			cmpF64(t, "loss", loss, wantLoss, tol)
			if !tc.ii {
				grad = ceScatter(t, grad, tg, z.Shape()[0])
			}
			cmpF64(t, "grad", grad, wantGrad, tol)
		})
	}
}

// TestCrossEntropyAllIgnoredMatchesTorch pins the degenerate batch in which EVERY row is
// masked. torch's behaviour here was confirmed empirically, not assumed: mean → NaN (0/0,
// since the divisor is the non-ignored count), sum → 0, none → all zeros — with exactly
// zero gradients in all three cases.
func TestCrossEntropyAllIgnoredMatchesTorch(t *testing.T) {
	fix, err := npz.LoadFile("testdata/ignoreindex_torch.npz")
	if err != nil {
		t.Fatal(err)
	}
	z, tg := fix["zall"], fix["tall"]
	for _, tc := range []struct {
		tag string
		red backend.Reduction
	}{{"mean", backend.ReductionMean}, {"sum", backend.ReductionSum}, {"none", backend.ReductionNone}} {
		t.Run(tc.tag, func(t *testing.T) {
			loss, grad := ceForwardBackward(t, z, tg,
				nn.WithReduction(tc.red), nn.WithIgnoreIndex(ceIgnore))
			cmpF64(t, "loss", loss, fix["loss_all_"+tc.tag], 1e-12)
			cmpF64(t, "grad", grad, fix["grad_all_"+tc.tag], 0) // exactly zero
			// Belt and braces on the shape/NaN split, independent of the fixture.
			switch tc.red {
			case backend.ReductionMean:
				if !math.IsNaN(loss.AtF64()) {
					t.Errorf("all-ignored mean: got %g, want NaN", loss.AtF64())
				}
			case backend.ReductionSum:
				if loss.AtF64() != 0 {
					t.Errorf("all-ignored sum: got %g, want 0", loss.AtF64())
				}
			case backend.ReductionNone:
				if loss.Shape()[0] != z.Shape()[0] {
					t.Errorf("all-ignored none: shape %v", loss.Shape())
				}
			}
		})
	}
}

// ceFilter drops every row whose target is ceIgnore, returning the surviving batch.
func ceFilter(t *testing.T, z, tg *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor) {
	t.Helper()
	b, c := z.Shape()[0], z.Shape()[1]
	var keep []int
	for i := range b {
		if int(tg.AtF64(i)) != ceIgnore {
			keep = append(keep, i)
		}
	}
	zf := tensor.New(z.Dtype(), tensor.Shape{len(keep), c})
	tf := tensor.New(tg.Dtype(), tensor.Shape{len(keep)})
	for r, i := range keep {
		for j := range c {
			zf.SetF64(z.AtF64(i, j), r, j)
		}
		tf.SetF64(tg.AtF64(i), r)
	}
	return zf, tf
}

// ceScatter expands a filtered-batch gradient back to the full batch, leaving 0 in the
// rows that were dropped — the layout the masked kernel must produce.
func ceScatter(t *testing.T, g, tg *tensor.Tensor, b int) *tensor.Tensor {
	t.Helper()
	c := g.Shape()[1]
	out := tensor.New(g.Dtype(), tensor.Shape{b, c})
	r := 0
	for i := range b {
		if int(tg.AtF64(i)) == ceIgnore {
			continue
		}
		for j := range c {
			out.SetF64(g.AtF64(r, j), i, j)
		}
		r++
	}
	return out
}

// TestCrossEntropyMaskedEqualsFilteredBitExact is the venv-free property gate (no fixture,
// CI-safe): running a batch with masked rows through the MASKED kernel must be BIT-EXACT
// with running the hand-filtered batch through the UNMASKED kernel — for loss and for the
// surviving rows' gradients — across every reduction × label-smoothing combination. It is
// the invariant that makes ignore_index meaningful, and it is exactly the equivalence
// torch satisfies (verified when the golden was generated).
func TestCrossEntropyMaskedEqualsFilteredBitExact(t *testing.T) {
	z, tg := ceSynthetic(6, 5, []int{1, ceIgnore, 3, ceIgnore, 0, 2})
	zf, tf := ceFilter(t, z, tg)
	for _, red := range []backend.Reduction{backend.ReductionMean, backend.ReductionSum, backend.ReductionNone} {
		for _, ls := range []float64{0, 0.1} {
			t.Run(fmt.Sprintf("%s/ls%g", red, ls), func(t *testing.T) {
				maskedLoss, maskedGrad := ceForwardBackward(t, z, tg, ceOpts(red, true, ls)...)
				plainLoss, plainGrad := ceForwardBackward(t, zf, tf, ceOpts(red, false, ls)...)

				if red == backend.ReductionNone {
					// "none" keeps the ignored slots in place, so compare the surviving
					// entries against the filtered vector position by position.
					if maskedLoss.Shape()[0] != z.Shape()[0] {
						t.Fatalf("none: loss shape %v, want [%d]", maskedLoss.Shape(), z.Shape()[0])
					}
					r := 0
					for i := range z.Shape()[0] {
						if int(tg.AtF64(i)) == ceIgnore {
							if maskedLoss.AtF64(i) != 0 {
								t.Errorf("ignored row %d: loss %g, want exactly 0", i, maskedLoss.AtF64(i))
							}
							continue
						}
						if maskedLoss.AtF64(i) != plainLoss.AtF64(r) {
							t.Errorf("row %d: masked %.17g != filtered %.17g", i, maskedLoss.AtF64(i), plainLoss.AtF64(r))
						}
						r++
					}
				} else if maskedLoss.AtF64() != plainLoss.AtF64() {
					t.Errorf("loss: masked %.17g != filtered %.17g (not bit-exact)",
						maskedLoss.AtF64(), plainLoss.AtF64())
				}

				cmpF64(t, "grad", maskedGrad, ceScatter(t, plainGrad, tg, z.Shape()[0]), 0)
			})
		}
	}
}

// TestCrossEntropyIgnoredRowsExactlyZeroGrad pins the hard guarantee separately from any
// comparison: a masked row's gradient is EXACTLY zero (not merely small), under every
// reduction and with label smoothing and z-loss both engaged — the two features that write
// a non-zero value into every column of an unmasked row.
func TestCrossEntropyIgnoredRowsExactlyZeroGrad(t *testing.T) {
	z, tg := ceSynthetic(6, 5, []int{1, ceIgnore, 3, ceIgnore, 0, 2})
	for _, red := range []backend.Reduction{backend.ReductionMean, backend.ReductionSum, backend.ReductionNone} {
		t.Run(red.String(), func(t *testing.T) {
			_, g := ceForwardBackward(t, z, tg,
				nn.WithReduction(red), nn.WithIgnoreIndex(ceIgnore),
				nn.WithLabelSmoothing(0.1), nn.WithZLoss(1e-3))
			for i := range z.Shape()[0] {
				if int(tg.AtF64(i)) != ceIgnore {
					continue
				}
				for j := range z.Shape()[1] {
					if v := g.AtF64(i, j); v != 0 {
						t.Errorf("ignored row %d col %d: grad %g, want exactly 0", i, j, v)
					}
				}
			}
		})
	}
}

// TestCrossEntropyReductionNoneShape pins reduction="none"'s contract: a [batch] tensor
// (one loss per input row, ignored slots included as 0), not a filtered-length vector —
// so the result stays aligned with the input batch and can be weighted per example.
func TestCrossEntropyReductionNoneShape(t *testing.T) {
	z, tg := ceSynthetic(7, 4, []int{0, 1, ceIgnore, 2, 3, ceIgnore, 1})
	for _, ii := range []bool{false, true} {
		opts := []nn.CEOption{nn.WithReduction(backend.ReductionNone)}
		zi, ti := z, tg
		if ii {
			opts = append(opts, nn.WithIgnoreIndex(ceIgnore))
		} else {
			zi, ti = ceFilter(t, z, tg)
		}
		loss, err := nn.CrossEntropy(backend.NewContext(), zi, ti, opts...)
		if err != nil {
			t.Fatal(err)
		}
		want := zi.Shape()[0]
		if loss.Ndim() != 1 || loss.Shape()[0] != want {
			t.Errorf("ignore=%v: shape %v, want [%d]", ii, loss.Shape(), want)
		}
	}
}

// TestCrossEntropyMaskedGradCheck runs the EXPORTED autograd.GradCheck (§T839) on the
// masked path: the analytic gradient of the masked, smoothed, z-lossed cross-entropy must
// match central finite differences of its own forward. This proves the backward's
// non-ignored-count divisor is the true derivative of the forward's divisor — a mismatch
// there is invisible to any single-reduction comparison.
func TestCrossEntropyMaskedGradCheck(t *testing.T) {
	z, tg := ceSynthetic(6, 4, []int{1, ceIgnore, 3, ceIgnore, 0, 2})
	for _, tc := range []struct {
		name string
		opts []nn.CEOption
	}{
		{"mean", []nn.CEOption{nn.WithIgnoreIndex(ceIgnore)}},
		{"sum", []nn.CEOption{nn.WithIgnoreIndex(ceIgnore), nn.WithReduction(backend.ReductionSum)}},
		{"none", []nn.CEOption{nn.WithIgnoreIndex(ceIgnore), nn.WithReduction(backend.ReductionNone)}},
		{"mean+smooth+zloss", []nn.CEOption{
			nn.WithIgnoreIndex(ceIgnore), nn.WithLabelSmoothing(0.1), nn.WithZLoss(1e-3)}},
		{"none+smooth", []nn.CEOption{
			nn.WithIgnoreIndex(ceIgnore), nn.WithReduction(backend.ReductionNone), nn.WithLabelSmoothing(0.1)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := autograd.GradCheck(func(ctx *backend.Context, in []*tensor.Tensor) (*tensor.Tensor, error) {
				return nn.CrossEntropy(ctx, in[0], tg, tc.opts...)
			}, []*tensor.Tensor{ceCopy(z)})
			if err != nil {
				t.Error(err)
			}
		})
	}
}

// TestCrossEntropyDefaultsUnchanged is the regression gate on the option plumbing: the
// two-argument call, and every explicit spelling of the defaults, must produce the exact
// same loss as before the fields existed.
func TestCrossEntropyDefaultsUnchanged(t *testing.T) {
	z, tg := ceSynthetic(5, 6, []int{0, 3, 1, 5, 2})
	ctx := backend.NewContext()
	base, err := nn.CrossEntropy(ctx, z, tg)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		opts []nn.CEOption
	}{
		{"explicit mean", []nn.CEOption{nn.WithReduction(backend.ReductionMean)}},
		{"zero smoothing", []nn.CEOption{nn.WithLabelSmoothing(0)}},
		{"unmatched ignore index", []nn.CEOption{nn.WithIgnoreIndex(ceIgnore)}}, // no row carries -100
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := nn.CrossEntropy(ctx, z, tg, tc.opts...)
			if err != nil {
				t.Fatal(err)
			}
			if got.AtF64() != base.AtF64() {
				t.Errorf("got %.17g, plain CrossEntropy %.17g", got.AtF64(), base.AtF64())
			}
		})
	}
	// The option form of the existing helpers must equal the helpers themselves.
	sm, err := nn.CrossEntropySmooth(ctx, z, tg, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	smOpt, err := nn.CrossEntropy(ctx, z, tg, nn.WithLabelSmoothing(0.1))
	if err != nil {
		t.Fatal(err)
	}
	if sm.AtF64() != smOpt.AtF64() {
		t.Errorf("WithLabelSmoothing %.17g != CrossEntropySmooth %.17g", smOpt.AtF64(), sm.AtF64())
	}
	zl, err := nn.CrossEntropyZLoss(ctx, z, tg, 1e-3)
	if err != nil {
		t.Fatal(err)
	}
	zlOpt, err := nn.CrossEntropy(ctx, z, tg, nn.WithZLoss(1e-3))
	if err != nil {
		t.Fatal(err)
	}
	if zl.AtF64() != zlOpt.AtF64() {
		t.Errorf("WithZLoss %.17g != CrossEntropyZLoss %.17g", zlOpt.AtF64(), zl.AtF64())
	}
}

// ceCopy returns an independent contiguous copy (tape leaves are keyed by pointer, so
// each run needs its own logits tensor).
func ceCopy(x *tensor.Tensor) *tensor.Tensor {
	out := tensor.New(x.Dtype(), x.Shape().Clone())
	src, dst := x.Contiguous().Storage().F64(), out.Storage().F64()
	copy(dst, src)
	return out
}

// ceSynthetic builds a deterministic f64 logits/targets pair (no RNG dependency, so the
// venv-free property gates are reproducible across platforms).
func ceSynthetic(b, c int, targets []int) (*tensor.Tensor, *tensor.Tensor) {
	z := tensor.New(tensor.F64, tensor.Shape{b, c})
	for i := range b {
		for j := range c {
			z.SetF64(math.Sin(float64(i*c+j)*0.7)*1.5+0.25*float64(j), i, j)
		}
	}
	tg := tensor.New(tensor.F64, tensor.Shape{b})
	for i, v := range targets {
		tg.SetF64(float64(v), i)
	}
	return z, tg
}
