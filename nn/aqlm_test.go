package nn_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// randWeight returns a deterministic rows×cols f64 weight matrix with mildly correlated
// group structure (a couple of latent directions) so additive codebooks have something to
// exploit — closer to real weight statistics than pure white noise.
func randWeight(rows, cols int, seed uint64) *tensor.Tensor {
	rng := rand.New(rand.NewPCG(seed, seed^0xABCDEF))
	w := tensor.New(tensor.F64, tensor.Shape{rows, cols})
	for r := range rows {
		a, b := rng.NormFloat64(), rng.NormFloat64()
		for c := range cols {
			// low-rank-ish signal + noise → group vectors cluster, rewarding quantization.
			v := a*math.Sin(0.3*float64(c)) + b*math.Cos(0.17*float64(c)) + 0.15*rng.NormFloat64()
			w.SetF64(v, r, c)
		}
	}
	return w
}

// mseAQLM returns the mean squared error between a weight tensor and an AQLM reconstruction.
func mseAQLM(w, wHat *tensor.Tensor) float64 {
	rows, cols := w.Shape()[0], w.Shape()[1]
	var s float64
	for r := range rows {
		for c := range cols {
			d := w.AtF64(r, c) - wHat.AtF64(r, c)
			s += d * d
		}
	}
	return s / float64(rows*cols)
}

// §V16(1) reconstruction value: MSE must DECREASE as the number of codebooks M grows (rate
// rising), and the run reports the rate-distortion curve (bits/weight vs MSE).
func TestAQLMReconstructionMSEDecreasesWithM(t *testing.T) {
	w := randWeight(24, 32, 1)
	const bits, g = 4, 4
	var prevMSE = math.Inf(1)
	for _, m := range []int{1, 2, 3, 4} {
		q, err := nn.EncodeAQLM(w,
			nn.WithAQLMCodebooks(m), nn.WithAQLMBits(bits), nn.WithAQLMGroupSize(g),
			nn.WithAQLMIters(12), nn.WithAQLMSeed(7))
		if err != nil {
			t.Fatalf("M=%d: %v", m, err)
		}
		mse := mseAQLM(w, q.Reconstruct())
		t.Logf("rate-distortion: M=%d bits/weight=%.3f MSE=%.6g", m, q.BitsPerWeight(), mse)
		if !(mse < prevMSE) {
			t.Errorf("M=%d MSE %.6g not < previous %.6g (adding a codebook must not worsen the fit)", m, mse, prevMSE)
		}
		prevMSE = mse
	}
}

// §V16(1) reconstruction value: MSE must also DECREASE as the bit-width B grows (richer
// codebooks), the other rate axis.
func TestAQLMReconstructionMSEDecreasesWithBits(t *testing.T) {
	w := randWeight(24, 32, 2)
	const m, g = 2, 4
	var prevMSE = math.Inf(1)
	for _, bits := range []int{1, 2, 3, 4, 5} {
		q, err := nn.EncodeAQLM(w,
			nn.WithAQLMCodebooks(m), nn.WithAQLMBits(bits), nn.WithAQLMGroupSize(g),
			nn.WithAQLMIters(12), nn.WithAQLMSeed(9))
		if err != nil {
			t.Fatalf("B=%d: %v", bits, err)
		}
		mse := mseAQLM(w, q.Reconstruct())
		t.Logf("rate-distortion: B=%d bits/weight=%.3f MSE=%.6g", bits, q.BitsPerWeight(), mse)
		if !(mse < prevMSE) {
			t.Errorf("B=%d MSE %.6g not < previous %.6g (more entries must not worsen the fit)", bits, mse, prevMSE)
		}
		prevMSE = mse
	}
}

// §V16(1) bits/weight formula: BitsPerWeight = M·B/g exactly.
func TestAQLMBitsPerWeight(t *testing.T) {
	w := randWeight(8, 16, 3)
	for _, tc := range []struct {
		m, b, g int
		want    float64
	}{
		{1, 8, 8, 1.0}, {2, 8, 8, 2.0}, {2, 4, 4, 2.0}, {3, 2, 8, 0.75},
	} {
		q, err := nn.EncodeAQLM(w, nn.WithAQLMCodebooks(tc.m), nn.WithAQLMBits(tc.b), nn.WithAQLMGroupSize(tc.g), nn.WithAQLMIters(1))
		if err != nil {
			t.Fatal(err)
		}
		if got := q.BitsPerWeight(); math.Abs(got-tc.want) > 1e-12 {
			t.Errorf("M=%d B=%d g=%d: bits/weight=%g want %g", tc.m, tc.b, tc.g, got, tc.want)
		}
	}
}

// §V16(2) round-trip determinism + reproducibility: the same weight and seed produce
// bit-identical codes and codebooks across two independent encodes.
func TestAQLMDeterministic(t *testing.T) {
	w := randWeight(16, 24, 4)
	opts := []nn.AQLMOption{nn.WithAQLMCodebooks(2), nn.WithAQLMBits(4), nn.WithAQLMGroupSize(4), nn.WithAQLMIters(8), nn.WithAQLMSeed(42)}
	q1, err := nn.EncodeAQLM(w, opts...)
	if err != nil {
		t.Fatal(err)
	}
	q2, err := nn.EncodeAQLM(w, opts...)
	if err != nil {
		t.Fatal(err)
	}
	if len(q1.Codes) != len(q2.Codes) {
		t.Fatalf("code count differs: %d vs %d", len(q1.Codes), len(q2.Codes))
	}
	for i := range q1.Codes {
		if q1.Codes[i] != q2.Codes[i] {
			t.Fatalf("codes[%d] differ: %d vs %d (encode not reproducible)", i, q1.Codes[i], q2.Codes[i])
		}
	}
	for p := range q1.Codebooks {
		for t2 := range q1.Codebooks[p] {
			if q1.Codebooks[p][t2] != q2.Codebooks[p][t2] {
				t.Fatalf("codebook[%d][%d] differ: %g vs %g", p, t2, q1.Codebooks[p][t2], q2.Codebooks[p][t2])
			}
		}
	}
	// round-trip determinism: reconstruction is identical too.
	if mseAQLM(q1.Reconstruct(), q2.Reconstruct()) != 0 {
		t.Errorf("reconstructions differ across identical encodes")
	}
}

// §V16(2) known-behavior anchor: M=1 is EXACTLY plain vector quantization — every group is
// reconstructed as a single codebook entry, namely its nearest one.
func TestAQLMSingleCodebookIsVectorQuantization(t *testing.T) {
	w := randWeight(20, 16, 5)
	const bits, g = 4, 4
	q, err := nn.EncodeAQLM(w, nn.WithAQLMCodebooks(1), nn.WithAQLMBits(bits), nn.WithAQLMGroupSize(g), nn.WithAQLMIters(10), nn.WithAQLMSeed(3))
	if err != nil {
		t.Fatal(err)
	}
	wHat := q.Reconstruct()
	gpr := q.Cols / g
	for r := range q.Rows {
		for gc := range gpr {
			gi := r*gpr + gc
			code := q.Codes[gi] // M=1 ⇒ one code per group
			entry := q.Codebooks[code]
			// reconstruction of this group equals the codebook entry it selected...
			for ti := range g {
				if got, want := wHat.AtF64(r, gc*g+ti), entry[ti]; math.Abs(got-want) > 1e-9 {
					t.Fatalf("group (%d,%d)[%d]: recon %g != codebook entry %g", r, gc, ti, got, want)
				}
			}
			// ...and that selected entry is the NEAREST codebook entry to the group (plain VQ).
			grp := make([]float64, g)
			for ti := range g {
				grp[ti] = w.AtF64(r, gc*g+ti)
			}
			best, bd := 0, math.Inf(1)
			for j := range q.Codebooks {
				var d float64
				for ti := range g {
					diff := grp[ti] - q.Codebooks[j][ti]
					d += diff * diff
				}
				if d < bd {
					bd, best = d, j
				}
			}
			// distance to the chosen entry must equal the min distance (ties allowed).
			var dsel float64
			for ti := range g {
				diff := grp[ti] - entry[ti]
				dsel += diff * diff
			}
			if math.Abs(dsel-bd) > 1e-9 {
				t.Errorf("group (%d,%d): chosen entry dist %g != nearest %g (entry %d)", r, gc, dsel, bd, best)
			}
		}
	}
}

// §V16(3) forward parity: AQLM.Forward with the reconstructed weights (a) EXACTLY equals a
// dense nn.Linear built from Reconstruct(), and (b) approximates a dense Linear on the ORIGINAL
// weights within the reconstruction tolerance.
func TestAQLMForwardParity(t *testing.T) {
	const rows, cols = 12, 16 // Out=12, In=16
	w := randWeight(rows, cols, 6)
	q, err := nn.EncodeAQLM(w, nn.WithAQLMCodebooks(3), nn.WithAQLMBits(4), nn.WithAQLMGroupSize(4), nn.WithAQLMIters(12), nn.WithAQLMSeed(11))
	if err != nil {
		t.Fatal(err)
	}
	ctx := backend.NewContext()

	// inputs x[batch, In]
	const batch = 5
	x := tensor.New(tensor.F64, tensor.Shape{batch, cols})
	rng := rand.New(rand.NewPCG(100, 200))
	for i := range batch {
		for j := range cols {
			x.SetF64(rng.NormFloat64(), i, j)
		}
	}

	y, err := q.Forward(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	if y.Shape()[0] != batch || y.Shape()[1] != rows {
		t.Fatalf("Forward shape %v, want [%d,%d]", y.Shape(), batch, rows)
	}

	// (a) EXACT parity with a dense Linear built from the reconstructed weights (layout [In,Out]).
	wHat := q.Reconstruct()
	wT := tensor.New(tensor.F64, tensor.Shape{cols, rows})
	for r := range rows {
		for c := range cols {
			wT.SetF64(wHat.AtF64(r, c), c, r)
		}
	}
	dense := &nn.Linear{W: wT}
	yDense, err := dense.Forward(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	for i := range batch {
		for j := range rows {
			if got, want := y.AtF64(i, j), yDense.AtF64(i, j); math.Abs(got-want) > 1e-9 {
				t.Fatalf("forward[%d,%d] %g != dense(reconstructed) %g", i, j, got, want)
			}
		}
	}

	// (b) approximate parity with a dense Linear on the ORIGINAL weights, bounded by the
	// reconstruction error propagated through x: ‖y − y_true‖ ≤ ‖Ŵ − W‖·‖x‖.
	wTrueT := tensor.New(tensor.F64, tensor.Shape{cols, rows})
	for r := range rows {
		for c := range cols {
			wTrueT.SetF64(w.AtF64(r, c), c, r)
		}
	}
	yTrue, err := (&nn.Linear{W: wTrueT}).Forward(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	// crude per-element bound: |Δy| ≤ Σ_k |Δw||x|; assert it's small relative to the signal.
	var num, den float64
	for i := range batch {
		for j := range rows {
			d := y.AtF64(i, j) - yTrue.AtF64(i, j)
			num += d * d
			den += yTrue.AtF64(i, j) * yTrue.AtF64(i, j)
		}
	}
	rel := math.Sqrt(num / den)
	t.Logf("forward relative error to dense(original) = %.4g", rel)
	if rel > 0.5 {
		t.Errorf("forward relative error %.4g too large — reconstruction not tracking dense output", rel)
	}
}

// §V16(4) sanity / shape errors.
func TestAQLMErrors(t *testing.T) {
	w := randWeight(4, 12, 7)
	if _, err := nn.EncodeAQLM(tensor.New(tensor.F64, tensor.Shape{4})); err == nil {
		t.Error("expected error for rank-1 weight")
	}
	if _, err := nn.EncodeAQLM(w, nn.WithAQLMGroupSize(5)); err == nil {
		t.Error("expected error: cols 12 not divisible by group size 5")
	}
	if _, err := nn.EncodeAQLM(w, nn.WithAQLMCodebooks(0)); err == nil {
		t.Error("expected error for M=0")
	}
	if _, err := nn.EncodeAQLM(w, nn.WithAQLMBits(0)); err == nil {
		t.Error("expected error for B=0")
	}
	// Forward shape mismatch.
	q, err := nn.EncodeAQLM(w, nn.WithAQLMCodebooks(1), nn.WithAQLMBits(3), nn.WithAQLMGroupSize(3), nn.WithAQLMIters(1))
	if err != nil {
		t.Fatal(err)
	}
	bad := tensor.New(tensor.F64, tensor.Shape{2, 7}) // In should be 12
	if _, err := q.Forward(backend.NewContext(), bad); err == nil {
		t.Error("expected Forward error for wrong input width")
	}
}

func ExampleEncodeAQLM() {
	// A 4×8 weight, quantized additively: with 1 codebook (plain vector quantization) versus 2
	// codebooks (an additive correction on top), 2-bit codes over groups of 4. Adding the
	// second codebook lowers the reconstruction error — the AQLM rate-distortion lever.
	w := tensor.New(tensor.F64, tensor.Shape{4, 8})
	rng := rand.New(rand.NewPCG(7, 7))
	for i := range 4 {
		for j := range 8 {
			w.SetF64(rng.NormFloat64(), i, j)
		}
	}
	mse := func(q *nn.AQLM) float64 {
		wh := q.Reconstruct()
		var s float64
		for i := range 4 {
			for j := range 8 {
				d := w.AtF64(i, j) - wh.AtF64(i, j)
				s += d * d
			}
		}
		return s / 32
	}
	q1, _ := nn.EncodeAQLM(w, nn.WithAQLMCodebooks(1), nn.WithAQLMBits(2), nn.WithAQLMGroupSize(4), nn.WithAQLMSeed(1))
	q2, _ := nn.EncodeAQLM(w, nn.WithAQLMCodebooks(2), nn.WithAQLMBits(2), nn.WithAQLMGroupSize(4), nn.WithAQLMSeed(1))
	var _ *nn.AQLM = q2 // the fitted representation (codebooks + codes)

	// Forward applies the quantized weights like a dense linear layer.
	x := tensor.New(tensor.F64, tensor.Shape{1, 8})
	if _, err := q2.Forward(backend.NewContext(), x); err != nil {
		fmt.Println(err)
	}
	fmt.Printf("bits/weight: 1cb=%.2f 2cb=%.2f  mse drops: %v\n",
		q1.BitsPerWeight(), q2.BitsPerWeight(), mse(q2) < mse(q1))
	// Output:
	// bits/weight: 1cb=0.50 2cb=1.00  mse drops: true
}
