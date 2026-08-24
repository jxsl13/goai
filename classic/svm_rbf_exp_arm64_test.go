//go:build arm64 && goexperiment.simd

package classic

import (
	"math"
	"reflect"
	"testing"
)

func TestSVCRBFExpSIMDConvergence(t *testing.T) {
	x, y := svmBenchData(4000, 20)
	fit := func(forceScalar bool) (*SVC, int) {
		m := NewSVC(WithSVMKernel(SVMKernelRBF))
		m.forceScalarRBF = forceScalar
		steps, err := m.fit(x, y)
		if err != nil {
			t.Fatal(err)
		}
		return m, steps
	}

	scalar, scalarSteps := fit(true)
	vector, vectorSteps := fit(false)
	repeated, repeatedSteps := fit(false)
	if vectorSteps != scalarSteps {
		t.Fatalf("SIMD fit used %d SMO steps, scalar control used %d", vectorSteps, scalarSteps)
	}
	if vectorSteps != repeatedSteps || !reflect.DeepEqual(vector, repeated) {
		t.Fatal("SIMD RBF fit is not bit-deterministic")
	}
	if len(vector.sv) != len(scalar.sv) {
		t.Fatalf("SIMD fit retained %d support vectors, scalar control retained %d", len(vector.sv), len(scalar.sv))
	}

	scalarDecision, err := scalar.DecisionFunction(x)
	if err != nil {
		t.Fatal(err)
	}
	vectorDecision, err := vector.DecisionFunction(x)
	if err != nil {
		t.Fatal(err)
	}
	var maxAbs float64
	for i := range scalarDecision {
		delta := math.Abs(vectorDecision[i] - scalarDecision[i])
		maxAbs = max(maxAbs, delta)
		if (vectorDecision[i] >= 0) != (scalarDecision[i] >= 0) {
			t.Fatalf("decision sign differs at row %d: SIMD=%g scalar=%g", i, vectorDecision[i], scalarDecision[i])
		}
	}
	if maxAbs > 1e-10 {
		t.Fatalf("maximum SIMD/scalar decision delta %g exceeds 1e-10", maxAbs)
	}
	t.Logf("SMO steps=%d support_vectors=%d max_abs_decision_delta=%g", vectorSteps, len(vector.sv), maxAbs)
}

func benchmarkSVCFitRBFExp(b *testing.B, forceScalar bool) {
	x, y := svmBenchData(4000, 20)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		m := NewSVC(WithSVMKernel(SVMKernelRBF))
		m.forceScalarRBF = forceScalar
		if err := m.Fit(x, y); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSVCFitRBFExpScalarControl(b *testing.B) { benchmarkSVCFitRBFExp(b, true) }
func BenchmarkSVCFitRBFExpSIMD(b *testing.B)          { benchmarkSVCFitRBFExp(b, false) }
