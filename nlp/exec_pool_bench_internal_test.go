package nlp

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// PS6017 reports 118 convertible exec1 call sites, 97 of them in this package, and its registry entry
// carries no measurement. Before converting any of them the premise has to be checked directly: does
// routing through the pooled sibling actually remove an allocation?
//
// The claim under test is narrow. exec1 takes `ins ...*tensor.Tensor`, so each call materializes a
// slice; exec1a and exec2 borrow a fixed-size slice from a sync.Pool instead, and fall back to exec1
// when a Recorder is attached (because Execute retains the slice through Record). These benchmarks run
// with a nil Recorder, which is the pooled path, on the cheapest real op available so the slice
// handling is as large a share of the measurement as it can be.
func benchExecInputs(b *testing.B) (*backend.Context, *tensor.Tensor, *tensor.Tensor) {
	b.Helper()
	a := tensor.FromFloat64(tensor.Shape{4, 4}, make([]float64, 16))
	c := tensor.FromFloat64(tensor.Shape{4, 4}, make([]float64, 16))
	ctx := backend.NewContext()
	if ctx.Recorder != nil {
		b.Fatal("a fresh context must have no Recorder, or this measures the exec1 fallback twice")
	}
	return ctx, a, c
}

func BenchmarkExec1Variadic1(b *testing.B) {
	ctx, a, _ := benchExecInputs(b)
	b.ReportAllocs()
	for b.Loop() {
		if _, err := exec1(ctx, backend.OpNeg, nil, a); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkExec1aPooled1(b *testing.B) {
	ctx, a, _ := benchExecInputs(b)
	b.ReportAllocs()
	for b.Loop() {
		if _, err := exec1a(ctx, backend.OpNeg, nil, a); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkExec1Variadic2(b *testing.B) {
	ctx, a, c := benchExecInputs(b)
	b.ReportAllocs()
	for b.Loop() {
		if _, err := exec1(ctx, backend.OpAdd, nil, a, c); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkExec2Pooled2(b *testing.B) {
	ctx, a, c := benchExecInputs(b)
	b.ReportAllocs()
	for b.Loop() {
		if _, err := exec2(ctx, backend.OpAdd, nil, a, c); err != nil {
			b.Fatal(err)
		}
	}
}
