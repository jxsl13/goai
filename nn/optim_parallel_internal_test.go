package nn

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// TestParStepParallelMatchesSerial proves each parallelized optimizer update (parStep fanning a param
// above parStepMinElems across the worker pool) is BIT-IDENTICAL to the serial inline path. It runs the
// SAME optimizer code twice on identical inputs — once with the threshold forced high (serial), once
// forced low (fully parallel) — so there is no hand-written reference to drift: any difference would be
// a partitioning bug (overlap / lost element / race), which disjoint per-element writes preclude. Covers
// Adam/SGD-momentum/Lion, F64 and F32, a ragged (non-chunk-multiple) length, and several steps so the
// accumulated momentum/moment state is checked too.
func TestParStepParallelMatchesSerial(t *testing.T) {
	const n = 1<<18 + 12345 // large enough to fan out; not a chunk multiple (ragged tail)
	const steps = 4
	orig := parStepMinElems
	defer func() { parStepMinElems = orig }()

	opts := []struct {
		name   string
		newOpt func([]*tensor.Tensor) Optimizer
	}{
		{"Adam", func(p []*tensor.Tensor) Optimizer { return NewAdam(p, 1e-3) }},
		{"AdamW", func(p []*tensor.Tensor) Optimizer { return NewAdamW(p, 1e-3, 0.01) }},
		{"SGD-momentum", func(p []*tensor.Tensor) Optimizer { return NewSGD(p, 1e-2, 0.9) }},
		{"Lion", func(p []*tensor.Tensor) Optimizer { return NewLion(p, 1e-4) }},
		{"Sophia", func(p []*tensor.Tensor) Optimizer { return NewSophia(p, 1e-3) }},
		{"MARS", func(p []*tensor.Tensor) Optimizer { return NewMARS(p, 1e-3) }},
	}
	for _, oc := range opts {
		for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
			init := make([]float64, n)
			grad := make([]float64, n)
			for i := range init {
				init[i] = math.Sin(float64(i)*0.0013)*0.5 + 0.1
				grad[i] = math.Cos(float64(i)*0.0007) * 1e-2
			}
			run := func(threshold int) []float64 {
				parStepMinElems = threshold
				p := tensor.New(dt, tensor.Shape{n})
				g := tensor.New(tensor.F64, tensor.Shape{n})
				if dt == tensor.F64 {
					copy(p.Storage().F64(), init)
				} else {
					pf := p.Storage().F32()
					for i, x := range init {
						pf[i] = float32(x)
					}
				}
				copy(g.Storage().F64(), grad)
				gc := g.Cast(dt)
				opt := oc.newOpt([]*tensor.Tensor{p})
				for s := 0; s < steps; s++ {
					if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return gc }); err != nil {
						t.Fatal(err)
					}
				}
				out := make([]float64, n)
				if dt == tensor.F64 {
					copy(out, p.Storage().F64())
				} else {
					for i, x := range p.Storage().F32() {
						out[i] = float64(x)
					}
				}
				return out
			}
			serial := run(1 << 40)  // threshold above n ⇒ inline body(0,n)
			parallel := run(1 << 4) // threshold below n ⇒ parallel.Rows
			for i := 0; i < n; i++ {
				if serial[i] != parallel[i] {
					t.Fatalf("%s %v index %d: serial %v != parallel %v (bit)", oc.name, dt, i, serial[i], parallel[i])
				}
			}
			t.Logf("%s %v: parallel step bit-identical to serial over %d elems × %d steps", oc.name, dt, n, steps)
		}
	}
}

// benchStepLargeThreshold times an optimizer Step over ONE DRAM-resident matrix (33.5M f64 params,
// ~1 GB of param+state+grad) with the parallel threshold forced — the same-binary A/B for the parStep
// win across optimizers of different arithmetic intensity (Adam sqrt/div = compute-bound; SGD/Lion
// lighter = more memory-bandwidth-bound).
func benchStepLargeThreshold(b *testing.B, threshold int, newOpt func([]*tensor.Tensor) Optimizer) {
	orig := parStepMinElems
	parStepMinElems = threshold
	defer func() { parStepMinElems = orig }()
	const rows, cols = 8192, 4096
	p := tensor.New(tensor.F64, tensor.Shape{rows, cols})
	g := tensor.New(tensor.F64, tensor.Shape{rows, cols})
	gd := g.Storage().F64()
	for i := range gd {
		gd[i] = float64(i%17) * 1e-3
	}
	opt := newOpt([]*tensor.Tensor{p})
	gfn := func(*tensor.Tensor) *tensor.Tensor { return g }
	b.ResetTimer()
	for b.Loop() {
		if err := opt.Step(gfn); err != nil {
			b.Fatal(err)
		}
	}
}

func adamCtor(p []*tensor.Tensor) Optimizer { return NewAdam(p, 1e-3) }
func sgdCtor(p []*tensor.Tensor) Optimizer  { return NewSGD(p, 1e-2, 0.9) }
func lionCtor(p []*tensor.Tensor) Optimizer { return NewLion(p, 1e-4) }

func BenchmarkAdamStepLargeSerial(b *testing.B)   { benchStepLargeThreshold(b, 1<<40, adamCtor) }
func BenchmarkAdamStepLargeParallel(b *testing.B) { benchStepLargeThreshold(b, 1<<4, adamCtor) }
func BenchmarkSGDStepLargeSerial(b *testing.B)    { benchStepLargeThreshold(b, 1<<40, sgdCtor) }
func BenchmarkSGDStepLargeParallel(b *testing.B)  { benchStepLargeThreshold(b, 1<<4, sgdCtor) }
func BenchmarkLionStepLargeSerial(b *testing.B)   { benchStepLargeThreshold(b, 1<<40, lionCtor) }
func BenchmarkLionStepLargeParallel(b *testing.B) { benchStepLargeThreshold(b, 1<<4, lionCtor) }

// TestClipGradNormParallel exercises the large-grad parallel path of ClipGradNorm (sumsq via
// parallel.SumF64 + parStep scaling): the reduction is a REASSOCIATION so it is not bit-identical to
// serial, but it must be (a) DETERMINISTIC (fixed partition + fixed combine order ⇒ reproducible),
// (b) within a tight tolerance of the serial norm, and (c) correctness-preserving (post-clip global
// norm == maxNorm).
func TestClipGradNormParallel(t *testing.T) {
	const n = 1<<18 + 9999
	orig := parStepMinElems
	defer func() { parStepMinElems = orig }()
	g := tensor.New(tensor.F64, tensor.Shape{n})
	gd := g.Storage().F64()
	for i := range gd {
		gd[i] = math.Sin(float64(i)*0.001)*0.3 + 0.05 // global norm >> 1 ⇒ clipping triggers
	}
	p := tensor.New(tensor.F64, tensor.Shape{n})
	gfn := func(*tensor.Tensor) *tensor.Tensor { return g }
	run := func(threshold int) (float64, []float64) {
		parStepMinElems = threshold
		clip, norm := ClipGradNorm([]*tensor.Tensor{p}, gfn, 1.0)
		out := clip(p).Storage().F64()
		return norm, append([]float64(nil), out...)
	}
	nP1, oP1 := run(1 << 4)
	nP2, oP2 := run(1 << 4)
	if nP1 != nP2 {
		t.Fatalf("parallel norm nondeterministic: %v vs %v", nP1, nP2)
	}
	for i := range oP1 {
		if oP1[i] != oP2[i] {
			t.Fatalf("parallel scaled grad nondeterministic at %d", i)
		}
	}
	nS, _ := run(1 << 40)
	if rel := math.Abs(nP1-nS) / nS; rel > 1e-12 {
		t.Fatalf("parallel norm %.15g vs serial %.15g: rel %g > 1e-12", nP1, nS, rel)
	}
	// post-clip global norm must equal maxNorm (1.0) — correctness preserved
	var sq float64
	for _, v := range oP1 {
		sq += v * v
	}
	if pc := math.Sqrt(sq); math.Abs(pc-1) > 1e-9 {
		t.Fatalf("post-clip norm %v, want 1", pc)
	}
	t.Logf("ClipGradNorm parallel: deterministic, norm rel-diff %g vs serial, post-clip norm ok", math.Abs(nP1-nS)/nS)
}

func benchClipGradNorm(b *testing.B, threshold int) {
	orig := parStepMinElems
	parStepMinElems = threshold
	defer func() { parStepMinElems = orig }()
	const n = 8192 * 4096 // 33.5M
	g := tensor.New(tensor.F64, tensor.Shape{n})
	gd := g.Storage().F64()
	for i := range gd {
		gd[i] = float64(i%17)*1e-2 + 0.1 // norm >> maxNorm ⇒ scaling closure materializes
	}
	p := tensor.New(tensor.F64, tensor.Shape{n})
	gfn := func(*tensor.Tensor) *tensor.Tensor { return g }
	b.ResetTimer()
	for b.Loop() {
		clip, _ := ClipGradNorm([]*tensor.Tensor{p}, gfn, 1.0)
		_ = clip(p)
	}
}

func BenchmarkClipGradNormSerial(b *testing.B)   { benchClipGradNorm(b, 1<<40) }
func BenchmarkClipGradNormParallel(b *testing.B) { benchClipGradNorm(b, 1<<4) }

// TestLAMBParStepMatchesSerial checks LAMB's parallelized Step against its serial inline path via the
// same threshold toggle. Unlike Adam/SGD/Lion, LAMB's per-param trust ratio divides two Σ-of-squares
// (wNorm2/uNorm2) computed with parSumSq, which REASSOCIATES the reduction — so the final param is
// tol-close (~ULP), not bit-identical; the moment state m/v (pure per-index parStep writes) IS bit-exact.
func TestLAMBParStepMatchesSerial(t *testing.T) {
	const n = 1<<18 + 4321 // ragged tail; above parStepMinElems so the parallel path engages
	const steps = 3
	orig := parStepMinElems
	defer func() { parStepMinElems = orig }()
	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		init := make([]float64, n)
		grad := make([]float64, n)
		for i := range init {
			init[i] = math.Sin(float64(i)*0.0011)*0.7 - 0.2
			grad[i] = math.Cos(float64(i)*0.0009) * 2e-2
		}
		run := func(threshold int) (param, mom []float64) {
			parStepMinElems = threshold
			p := tensor.New(dt, tensor.Shape{n})
			g := tensor.New(tensor.F64, tensor.Shape{n})
			if dt == tensor.F64 {
				copy(p.Storage().F64(), init)
			} else {
				pf := p.Storage().F32()
				for i, x := range init {
					pf[i] = float32(x)
				}
			}
			copy(g.Storage().F64(), grad)
			gc := g.Cast(dt)
			opt := NewLAMB([]*tensor.Tensor{p}, 1e-3)
			for s := 0; s < steps; s++ {
				if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return gc }); err != nil {
					t.Fatal(err)
				}
			}
			param = make([]float64, n)
			if dt == tensor.F64 {
				copy(param, p.Storage().F64())
			} else {
				for i, x := range p.Storage().F32() {
					param[i] = float64(x)
				}
			}
			mom = append([]float64(nil), opt.m[0]...) // moment m — pure parStep writes, must be bit-exact
			return
		}
		sp, sm := run(1 << 40)
		pp, pm := run(1 << 4)
		for i := 0; i < n; i++ {
			if sm[i] != pm[i] {
				t.Fatalf("%v m[%d]: serial %v != parallel %v (moment must be bit-exact)", dt, i, sm[i], pm[i])
			}
			if d := math.Abs(sp[i] - pp[i]); d > 1e-9*math.Abs(sp[i])+1e-11 {
				t.Fatalf("%v param[%d]: serial %v vs parallel %v (Δ %g > tol)", dt, i, sp[i], pp[i], d)
			}
		}
		t.Logf("%v: LAMB parallel matches serial (m bit-exact, param within trust-ratio ULP tol) over %d elems", dt, n)
	}
}

func benchLAMBStep(b *testing.B, n, threshold int) {
	orig := parStepMinElems
	parStepMinElems = threshold
	defer func() { parStepMinElems = orig }()
	p := tensor.New(tensor.F32, tensor.Shape{n})
	pf := p.Storage().F32()
	for i := range pf {
		pf[i] = float32(i%1000)/500 - 1
	}
	g := tensor.New(tensor.F32, tensor.Shape{n})
	gf := g.Storage().F32()
	for i := range gf {
		gf[i] = float32(i%777) * 1e-4
	}
	opt := NewLAMB([]*tensor.Tensor{p}, 1e-3)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		opt.Step(func(*tensor.Tensor) *tensor.Tensor { return g })
	}
}

func BenchmarkLAMBStep_1M_parallel(b *testing.B) { benchLAMBStep(b, 1<<20, 1<<8) }
func BenchmarkLAMBStep_1M_serial(b *testing.B)   { benchLAMBStep(b, 1<<20, 1<<40) }

func benchSophiaStep(b *testing.B, n, threshold int) {
	orig := parStepMinElems
	parStepMinElems = threshold
	defer func() { parStepMinElems = orig }()
	p := tensor.New(tensor.F32, tensor.Shape{n})
	pf := p.Storage().F32()
	for i := range pf {
		pf[i] = float32(i%1000)/500 - 1
	}
	g := tensor.New(tensor.F32, tensor.Shape{n})
	gf := g.Storage().F32()
	for i := range gf {
		gf[i] = float32(i%777) * 1e-4
	}
	opt := NewSophia([]*tensor.Tensor{p}, 1e-3)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		opt.Step(func(*tensor.Tensor) *tensor.Tensor { return g })
	}
}

func BenchmarkSophiaStep_1M_parallel(b *testing.B) { benchSophiaStep(b, 1<<20, 1<<8) }
func BenchmarkSophiaStep_1M_serial(b *testing.B)   { benchSophiaStep(b, 1<<20, 1<<40) }

func benchMARSStep(b *testing.B, n, threshold int) {
	orig := parStepMinElems
	parStepMinElems = threshold
	defer func() { parStepMinElems = orig }()
	p := tensor.New(tensor.F32, tensor.Shape{n})
	pf := p.Storage().F32()
	for i := range pf {
		pf[i] = float32(i%1000)/500 - 1
	}
	g := tensor.New(tensor.F32, tensor.Shape{n})
	gf := g.Storage().F32()
	for i := range gf {
		gf[i] = float32(i%777) * 1e-4
	}
	opt := NewMARS([]*tensor.Tensor{p}, 1e-3)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		opt.Step(func(*tensor.Tensor) *tensor.Tensor { return g })
	}
}

func BenchmarkMARSStep_1M_parallel(b *testing.B) { benchMARSStep(b, 1<<20, 1<<8) }
func BenchmarkMARSStep_1M_serial(b *testing.B)   { benchMARSStep(b, 1<<20, 1<<40) }
