package llamagpu_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// encoderMaxAbs compares a GPU encoder's flattened [seq,dim] hidden states against a reference
// element-wise. It returns the largest absolute difference, or an error naming the first element
// that is NaN or further than tol from the reference.
//
// The NaN test lives in the SAME branch as the tolerance test and that branch is the FAILING one —
// the form used by TestDecoderMatchesReference (llamagpu_test.go:63-65, §T428) and by every other
// IsNaN site in this package. The tempting alternative — folding the NaN test into a running
// maximum, `if math.IsNaN(got) || d > maxAbs { maxAbs = d }` — is not a guard at all but an
// ASSIGNMENT: it sets maxAbs to NaN, the deferred `maxAbs > tol` check is then false (NaN loses
// every ordered comparison), and the NaN is sticky, so every later `d > maxAbs` is false too and
// real divergence after the first NaN is invisible as well. TestEncoderMaxAbsRejectsNaN runs both
// forms side by side and pins that difference.
//
// Encoders (BERT/RoBERTa/DistilBERT/T5) compare a whole [seq,dim] hidden state rather than one
// logit row, so they share this helper instead of open-coding the loop per test. want is the
// reference element accessor — pass a loaded tensor's AtF64 directly.
func encoderMaxAbs(got []float32, seq, dim int, tol float64, want func(idx ...int) float64) (float64, error) {
	if len(got) != seq*dim {
		return 0, fmt.Errorf("got %d values, want seq·dim %d", len(got), seq*dim)
	}
	var maxAbs float64
	for i := 0; i < seq; i++ {
		for j := 0; j < dim; j++ {
			g := float64(got[i*dim+j])
			w := want(i, j)
			d := math.Abs(g - w)
			if math.IsNaN(g) || d > tol {
				return maxAbs, fmt.Errorf("hidden state [%d,%d] = %v, want %v (abs diff %.3e > tol %.1e)", i, j, g, w, d, tol)
			}
			if d > maxAbs {
				maxAbs = d
			}
		}
	}
	return maxAbs, nil
}

// TestEncoderMaxAbsRejectsNaN is the executable before/after for the inverted NaN guard that made
// TestCUDABertMatchesReference, TestCUDARobertaMatchesReference, TestCUDADistilBertMatchesReference
// and TestCUDAT5MatchesReference unable to fail. Those tests need CUDA hardware, so the regression
// they missed is pinned here instead, on the shared comparison helper, where it runs everywhere.
//
// Each case runs BOTH forms over the same data: "old" is the shipped `if IsNaN(g) || d > maxAbs {
// maxAbs = d }` accumulate-then-check, "new" is encoderMaxAbs. The assertions on the old form are
// the point — they document that it could not fail, which is why a green run of the four encoder
// tests proved nothing.
func TestEncoderMaxAbsRejectsNaN(t *testing.T) {
	const tol = 2e-3
	nan := float32(math.NaN())

	// oldForm is a verbatim copy of the shipped guard (cuda_bert_test.go:60, :112,
	// cuda_t5_test.go:62 before the fix). It returns the verdict those tests computed.
	oldForm := func(got []float32, seq, dim int, want func(idx ...int) float64) (maxAbs float64, failed bool) {
		for i := 0; i < seq; i++ {
			for j := 0; j < dim; j++ {
				d := math.Abs(float64(got[i*dim+j]) - want(i, j))
				if math.IsNaN(float64(got[i*dim+j])) || d > maxAbs {
					maxAbs = d
				}
			}
		}
		return maxAbs, maxAbs > tol
	}

	zero := func(idx ...int) float64 { return 0 }

	cases := []struct {
		name string
		got  []float32
		want func(idx ...int) float64
		// oldFailed is what the shipped form concluded; every NaN case is false — the bug.
		oldFailed bool
	}{
		{"all NaN", []float32{nan, nan, nan, nan}, zero, false},
		{"single NaN among clean values", []float32{0, 0, nan, 0}, zero, false},
		{"NaN then 996-magnitude divergence (sticky)", []float32{nan, 996, 0, 0}, zero, false},
		{"NaN in the reference-matching tail", []float32{0, 0, 0, nan}, zero, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			oldMax, oldFailed := oldForm(c.got, 2, 2, c.want)
			if oldFailed != c.oldFailed {
				t.Fatalf("old form: failed = %v, want %v (maxAbs %v)", oldFailed, c.oldFailed, oldMax)
			}
			if oldFailed {
				t.Fatalf("old form unexpectedly failed; this case is meant to show it could not")
			}
			// The old maximum is itself NaN — poisoned for every later comparison.
			if !math.IsNaN(oldMax) {
				t.Errorf("old form: maxAbs = %v, want NaN (the assignment poisons the accumulator)", oldMax)
			}
			gotMax, err := encoderMaxAbs(c.got, 2, 2, tol, c.want)
			if err == nil {
				t.Fatalf("encoderMaxAbs: no error for %q (maxAbs %v); an injected NaN must fail", c.name, gotMax)
			}
			t.Logf("old form: PASS (maxAbs %v) | encoderMaxAbs: FAIL (%v)", oldMax, err)
		})
	}

	// The 996-magnitude divergence the sticky NaN hid is caught on its own too.
	if _, err := encoderMaxAbs([]float32{0, 996, 0, 0}, 2, 2, tol, zero); err == nil {
		t.Error("encoderMaxAbs: no error for a 996-magnitude divergence")
	}

	// Control: clean data within tolerance still passes and reports the true maximum.
	max, err := encoderMaxAbs([]float32{0, 1e-4, -5e-4, 0}, 2, 2, tol, zero)
	if err != nil {
		t.Fatalf("encoderMaxAbs: clean data must pass: %v", err)
	}
	if math.Abs(max-5e-4) > 1e-9 {
		t.Errorf("encoderMaxAbs: maxAbs = %v, want 5e-4", max)
	}

	// A wrong-length result is caught rather than indexed past the end.
	if _, err := encoderMaxAbs([]float32{0, 0}, 2, 2, tol, zero); err == nil {
		t.Error("encoderMaxAbs: no error for a short result slice")
	}

	// The four encoder tests pass a loaded reference tensor's AtF64 straight through as want.
	// Exercising that exact shape here type-checks it on darwin, where the cuda-tagged encoder
	// tests are excluded by their `linux || windows` build constraint and so never compile.
	ref := tensor.New(tensor.F64, tensor.Shape{2, 2})
	ref.SetF64(0.5, 1, 1)
	if _, err := encoderMaxAbs([]float32{0, 0, 0, 0.5}, 2, 2, tol, ref.AtF64); err != nil {
		t.Fatalf("encoderMaxAbs against a tensor reference: %v", err)
	}
	if _, err := encoderMaxAbs([]float32{0, 0, 0, nan}, 2, 2, tol, ref.AtF64); err == nil {
		t.Error("encoderMaxAbs against a tensor reference: no error for an injected NaN")
	}
}
