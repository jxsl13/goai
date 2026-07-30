package classic

import (
	"math"
	"math/rand/v2"
	"runtime"
	"strings"
	"testing"
)

// TestEStepParallelBitIdentical pins the two properties the parallel eStep rests on, neither of
// which had a gate.
//
// FIRST, THE PER-WORKER SCRATCH. eStep previously read m.yScratch4 and m.yScratch off the
// receiver. Shared across workers those race, so the fan-out gives each worker its own; this test
// is also what the race detector runs over.
//
// The two arms are the SAME source selected by GOMAXPROCS — eStep runs its serial branch when it
// sees a single processor — so any difference is attributable to the partition rather than to a
// separately written reference. llHistory is compared per iteration rather than only at the end,
// because a divergence that later gets absorbed by convergence would otherwise be invisible.
//
// SECOND, THE REDUCTION ORDER, and it needs a DIFFERENT oracle than the arm comparison above.
// eStep used to accumulate `total += norm` inside the sample loop, which fixes the summation
// order; the fan-out parks each contribution in llBuf and sums it afterwards, and that pass must
// run ASCENDING to reproduce the serial sequence. The two-arm comparison cannot see this: both
// arms share the same final reduction, so reversing it moves them together and they still agree —
// verified, reversing the loop left both this test and the whole package green
// (TEST-ORACLE-SHARES-THE-SUBJECT-001). TestEStepReductionOrderGolden below pins the value
// instead.
func TestEStepParallelBitIdentical(t *testing.T) {
	if runtime.NumCPU() < 2 {
		t.Skip("single-CPU host: eStep cannot take its parallel branch")
	}
	for _, cov := range []GMMCovariance{GMMFull, GMMDiag} {
		for _, k := range []int{3, 6, 8} {
			// n chosen so the SMALLEST k in the sweep still clears eStep's n*k >= 1<<11 gate:
			// 512 was tried first and left k=3 on the serial branch in both arms, which the guard
			// below caught rather than letting the case pass vacuously.
			const n, d = 1024, 12
			if n*k < 1<<11 {
				t.Fatalf("geometry no longer reaches eStep's parallel branch: n*k = %d", n*k)
			}
			rng := rand.New(rand.NewPCG(uint64(k), 77))
			x := make([][]float64, n)
			for i := range x {
				row := make([]float64, d)
				for j := range row {
					row[j] = rng.NormFloat64() + float64(i%k)*2.5 // cluster structure
				}
				x[i] = row
			}

			fit := func(procs int) ([]float64, []float64) {
				defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(procs))
				m := NewGaussianMixture(WithGMMComponents(k), WithGMMCovariance(cov),
					WithGMMSeed(int64(k)*31+7), WithGMMMaxIter(12))
				if err := m.Fit(x); err != nil {
					t.Fatalf("cov=%v k=%d procs=%d: %v", cov, k, procs, err)
				}
				params := append([]float64(nil), m.Weights...)
				for c := range k {
					params = append(params, m.Means[c]...)
				}
				return append([]float64(nil), m.llHistory...), params
			}

			serLL, serP := fit(1)
			parLL, parP := fit(runtime.NumCPU())

			if len(serLL) != len(parLL) {
				t.Fatalf("cov=%v k=%d: %d EM iterations serial, %d parallel — the log-likelihood "+
					"diverged enough to change convergence", cov, k, len(serLL), len(parLL))
			}
			for i := range serLL {
				if math.Float64bits(serLL[i]) != math.Float64bits(parLL[i]) {
					t.Fatalf("cov=%v k=%d iteration %d: mean log-likelihood serial %v (%016x) != "+
						"parallel %v (%016x) — the reduction was reassociated", cov, k, i,
						serLL[i], math.Float64bits(serLL[i]), parLL[i], math.Float64bits(parLL[i]))
				}
			}
			if len(serP) != len(parP) {
				t.Fatalf("cov=%v k=%d: %d parameters serial, %d parallel", cov, k, len(serP), len(parP))
			}
			for i := range serP {
				if math.Float64bits(serP[i]) != math.Float64bits(parP[i]) {
					t.Fatalf("cov=%v k=%d parameter %d: serial %v (%016x) != parallel %v (%016x)",
						cov, k, i, serP[i], math.Float64bits(serP[i]), parP[i], math.Float64bits(parP[i]))
				}
			}
		}
	}
}

// TestEStepReductionOrderGolden freezes the per-iteration mean log-likelihood, which is the only
// thing that catches a reassociated final sum: the arm comparison above shares that code with both
// arms, so it is blind to it. The hash was captured from the ascending reduction, which reproduces
// the in-loop accumulation the serial implementation performed.
//
// A moved hash means the log-likelihood sequence changed. That is not cosmetic — EM's convergence
// test compares it against the previous iteration, so a reassociation can change the iteration
// count and therefore the fitted parameters.
func TestEStepReductionOrderGolden(t *testing.T) {
	const n, d, k = 1024, 12, 6
	const wantIters = 2
	const wantHash uint64 = 0x92db33bf801fcc73

	rng := rand.New(rand.NewPCG(uint64(k), 77))
	x := make([][]float64, n)
	for i := range x {
		row := make([]float64, d)
		for j := range row {
			row[j] = rng.NormFloat64() + float64(i%k)*2.5
		}
		x[i] = row
	}
	m := NewGaussianMixture(WithGMMComponents(k), WithGMMCovariance(GMMFull),
		WithGMMSeed(int64(k)*31+7), WithGMMMaxIter(12))
	if err := m.Fit(x); err != nil {
		t.Fatal(err)
	}
	if len(m.llHistory) != wantIters {
		t.Fatalf("%d EM iterations, want %d — the log-likelihood sequence changed enough to move "+
			"convergence", len(m.llHistory), wantIters)
	}
	var h uint64 = 14695981039346656037
	for _, v := range m.llHistory {
		h = (h ^ math.Float64bits(v)) * 1099511628211
	}
	if h != wantHash {
		t.Fatalf("llHistory hash %#x, want %#x — the mean log-likelihood is no longer the ascending "+
			"sum of the per-sample contributions", h, wantHash)
	}
}

// TestMStepErrorIsLowestComponent pins the error DETERMINISM the M-step fan-out had to preserve.
//
// The serial loop returned on the first non-positive-definite covariance, so it always named the
// lowest failing component. Fanned out over components, whichever worker finishes first would
// otherwise decide the message, and with 12 workers that is a coin toss. The implementation
// collects one error per component and scans ascending after the fan-out; this asserts the
// resulting index is stable and is the lowest, comparing the SERIAL and PARALLEL arms selected by
// GOMAXPROCS so a difference is attributable to the partition.
//
// regCovar is zeroed and each component gets fewer samples than dimensions, which makes every
// accumulated covariance rank-deficient — so more than one component fails and the choice among
// them is observable rather than incidental.
func TestMStepErrorIsLowestComponent(t *testing.T) {
	if runtime.NumCPU() < 2 {
		t.Skip("single-CPU host: mStep cannot take its parallel branch")
	}
	const d, k = 8, 4
	n := k * 2
	x := make([][]float64, n)
	for i := range x {
		row := make([]float64, d)
		for j := range row {
			row[j] = float64((i%k)*10 + j)
		}
		x[i] = row
	}
	run := func(procs int) string {
		defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(procs))
		m := NewGaussianMixture(WithGMMComponents(k), WithGMMCovariance(GMMFull),
			WithGMMSeed(5), WithGMMMaxIter(3), WithGMMRegCovar(0))
		err := m.Fit(x)
		if err == nil {
			t.Fatalf("procs=%d: expected a non-positive-definite covariance; the fixture no longer "+
				"produces one and this test would pass vacuously", procs)
		}
		return err.Error()
	}
	ser, par := run(1), run(runtime.NumCPU())
	if ser != par {
		t.Fatalf("the reported component depends on the partition:\n serial:   %s\n parallel: %s", ser, par)
	}
	if !strings.Contains(ser, "component 0 ") {
		t.Fatalf("expected the LOWEST failing component to be reported, got: %s", ser)
	}
}
