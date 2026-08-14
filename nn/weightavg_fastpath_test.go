package nn

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// T870 follow-up: SWA.Update / EMA.Update / NewEMA read a full parameter tensor
// on every call (SWA at cycle boundaries, EMA every step, NewEMA once at
// construction) via AtF64(Unravel(i)...) — the read-side twin of the
// SetF64(Unravel(i)...) write anti-pattern fillGen replaced. There is no
// existing fillGen-shaped helper for reads (fillGen only writes), so this
// follow-up adds readGen (see nn/init.go): identical contiguous/offset-0 guard
// and dtype switch, mirrored for reads. materialize() is a pure write and reuses
// fillGen directly, exactly like TruncNormal/Orthogonal.

// wavgParams builds n contiguous tensors of the given shape/dtype, filled with
// distinct deterministic values via the plain SetF64(Unravel(i)...) path
// (deliberately NOT fillGen/readGen, to keep fixture construction independent of
// the code under test).
func wavgParams(dtype tensor.Dtype, shape tensor.Shape, n int, seed uint64) []*tensor.Tensor {
	out := make([]*tensor.Tensor, n)
	for i := range out {
		t := tensor.New(dtype, shape)
		wavgMutate(t, seed+uint64(i))
		out[i] = t
	}
	return out
}

// wavgMutate overwrites every element of t in place with a fresh deterministic
// value via the plain SetF64(Unravel(i)...) path.
func wavgMutate(t *tensor.Tensor, seed uint64) {
	rng := rand.New(rand.NewPCG(seed, 0x1234567890abcdef))
	shape := t.Shape()
	for i := range t.Numel() {
		t.SetF64(rng.NormFloat64(), tensor.Unravel(i, shape)...)
	}
}

// --- readGen itself: bit-identity + non-contiguous fallback ---

// slowReadGen is the pre-conversion per-element read path, the oracle for readGen.
func slowReadGen(t *tensor.Tensor, visit func(float64)) {
	shape := t.Shape()
	for i := range t.Numel() {
		visit(t.AtF64(tensor.Unravel(i, shape)...))
	}
}

func TestReadGenBitIdenticalToSlowPath(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, shape := range []tensor.Shape{{1}, {7}, {4, 5}, {2, 3, 4}, {512, 512}} {
			src := tensor.New(dt, shape)
			wavgMutate(src, 3)
			var fastVals, slowVals []float64
			readGen(src, func(v float64) { fastVals = append(fastVals, v) })
			slowReadGen(src, func(v float64) { slowVals = append(slowVals, v) })
			if len(fastVals) != len(slowVals) {
				t.Fatalf("dt=%v shape=%v: fast produced %d values, slow %d", dt, shape, len(fastVals), len(slowVals))
			}
			for i := range fastVals {
				if math.Float64bits(fastVals[i]) != math.Float64bits(slowVals[i]) {
					t.Fatalf("dt=%v shape=%v elem %d: fast %v != slow %v (not bit-identical)", dt, shape, i, fastVals[i], slowVals[i])
				}
			}
		}
	}
}

// A non-contiguous view must still read correctly, in the view's own flat order,
// via the readGen fallback. Verified against an independently computed expected
// sequence (not the oracle above, since both would take the identical fallback
// branch on a non-contiguous input and so wouldn't differentiate a bug there).
func TestReadGenNonContiguousView(t *testing.T) {
	base := tensor.Zeros(tensor.F64, tensor.Shape{4, 3})
	for i := range 4 {
		for j := range 3 {
			base.SetF64(float64(i*3+j), i, j)
		}
	}
	view, err := base.Transpose(0, 1) // [3,4], non-contiguous
	if err != nil {
		t.Fatal(err)
	}
	var got []float64
	readGen(view, func(v float64) { got = append(got, v) })
	want := make([]float64, 0, 12)
	for j := range 3 {
		for i := range 4 {
			want = append(want, float64(i*3+j))
		}
	}
	if len(got) != len(want) {
		t.Fatalf("got %d values, want %d", len(got), len(want))
	}
	for k := range want {
		if got[k] != want[k] {
			t.Fatalf("elem %d: got %v want %v", k, got[k], want[k])
		}
	}
}

// --- SWA.Update ---

// slowSWAUpdate is the verbatim pre-conversion per-element path for SWA.Update.
func slowSWAUpdate(s *SWA) error {
	for pi, p := range s.Params {
		if p.Numel() != len(s.avg[pi]) {
			return fmt.Errorf("nn: SWA param %d size changed: %d != %d", pi, p.Numel(), len(s.avg[pi]))
		}
		for i := range p.Numel() {
			w := p.AtF64(tensor.Unravel(i, p.Shape())...)
			s.avg[pi][i] += (w - s.avg[pi][i]) / float64(s.n+1)
		}
	}
	s.n++
	return nil
}

func TestSWAUpdateBitIdenticalToSlowPath(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, shape := range []tensor.Shape{{7}, {4, 5}, {33, 17}} {
			params := wavgParams(dt, shape, 3, 1)
			fast := NewSWA(params)
			slow := NewSWA(params)
			for round := 0; round < 3; round++ {
				for _, p := range params {
					wavgMutate(p, uint64(round)+50)
				}
				if err := fast.Update(); err != nil {
					t.Fatalf("fast.Update: %v", err)
				}
				if err := slowSWAUpdate(slow); err != nil {
					t.Fatalf("slowSWAUpdate: %v", err)
				}
			}
			for pi := range fast.avg {
				for i := range fast.avg[pi] {
					a, b := fast.avg[pi][i], slow.avg[pi][i]
					if math.Float64bits(a) != math.Float64bits(b) {
						t.Fatalf("dt=%v shape=%v param %d elem %d: fast %v != slow %v (not bit-identical)", dt, shape, pi, i, a, b)
					}
				}
			}
		}
	}
}

// SWA.Update fed a non-contiguous parameter (e.g. a transposed weight-tying
// view) must still match the slow oracle exactly.
func TestSWAUpdateNonContiguousParam(t *testing.T) {
	base := tensor.Zeros(tensor.F64, tensor.Shape{4, 3})
	for i := range 4 {
		for j := range 3 {
			base.SetF64(float64(i*3+j)+0.5, i, j)
		}
	}
	view, err := base.Transpose(0, 1) // [3,4], non-contiguous
	if err != nil {
		t.Fatal(err)
	}
	fast := NewSWA([]*tensor.Tensor{view})
	slow := NewSWA([]*tensor.Tensor{view})
	if err := fast.Update(); err != nil {
		t.Fatalf("fast.Update: %v", err)
	}
	if err := slowSWAUpdate(slow); err != nil {
		t.Fatalf("slowSWAUpdate: %v", err)
	}
	for i := range fast.avg[0] {
		a, b := fast.avg[0][i], slow.avg[0][i]
		if math.Float64bits(a) != math.Float64bits(b) {
			t.Fatalf("elem %d: fast %v != slow %v (not bit-identical)", i, a, b)
		}
	}
}

func BenchmarkSWAUpdateFast(b *testing.B) {
	params := wavgParams(tensor.F32, tensor.Shape{512, 512}, 4, 1)
	s := NewSWA(params)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = s.Update()
	}
}

func BenchmarkSWAUpdateSlow(b *testing.B) {
	params := wavgParams(tensor.F32, tensor.Shape{512, 512}, 4, 1)
	s := NewSWA(params)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = slowSWAUpdate(s)
	}
}

// --- EMA.Update / NewEMA ---

func slowNewEMA(params []*tensor.Tensor, decay float64) *EMA {
	e := &EMA{Params: params, Decay: decay, avg: make([][]float64, len(params))}
	for i, p := range params {
		e.avg[i] = make([]float64, p.Numel())
		for j := range p.Numel() {
			e.avg[i][j] = p.AtF64(tensor.Unravel(j, p.Shape())...)
		}
	}
	return e
}

// slowEMAUpdate is the verbatim pre-conversion per-element path for EMA.Update.
func slowEMAUpdate(e *EMA) error {
	for pi, p := range e.Params {
		if p.Numel() != len(e.avg[pi]) {
			return fmt.Errorf("nn: EMA param %d size changed: %d != %d", pi, p.Numel(), len(e.avg[pi]))
		}
		for i := range p.Numel() {
			w := p.AtF64(tensor.Unravel(i, p.Shape())...)
			// math.FMA, matching EMA.Update — see the FMA-contraction note there. A
			// bare a*b+c*d here would let the compiler fuse whichever product it likes,
			// so this reference would drift from the very code it guards.
			e.avg[pi][i] = math.FMA(e.Decay, e.avg[pi][i], (1-e.Decay)*w)
		}
	}
	return nil
}

func TestNewEMABitIdenticalToSlowPath(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, shape := range []tensor.Shape{{7}, {4, 5}, {33, 17}} {
			params := wavgParams(dt, shape, 3, 5)
			fast := NewEMA(params, 0.99)
			slow := slowNewEMA(params, 0.99)
			for pi := range fast.avg {
				for i := range fast.avg[pi] {
					a, b := fast.avg[pi][i], slow.avg[pi][i]
					if math.Float64bits(a) != math.Float64bits(b) {
						t.Fatalf("dt=%v shape=%v param %d elem %d: fast %v != slow %v (not bit-identical)", dt, shape, pi, i, a, b)
					}
				}
			}
		}
	}
}

func TestEMAUpdateBitIdenticalToSlowPath(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, shape := range []tensor.Shape{{7}, {4, 5}, {33, 17}} {
			params := wavgParams(dt, shape, 3, 9)
			fast := NewEMA(params, 0.9)
			slow := slowNewEMA(params, 0.9)
			for round := 0; round < 3; round++ {
				for _, p := range params {
					wavgMutate(p, uint64(round)+100)
				}
				if err := fast.Update(); err != nil {
					t.Fatalf("fast.Update: %v", err)
				}
				if err := slowEMAUpdate(slow); err != nil {
					t.Fatalf("slowEMAUpdate: %v", err)
				}
			}
			for pi := range fast.avg {
				for i := range fast.avg[pi] {
					a, b := fast.avg[pi][i], slow.avg[pi][i]
					if math.Float64bits(a) != math.Float64bits(b) {
						t.Fatalf("dt=%v shape=%v param %d elem %d: fast %v != slow %v (not bit-identical)", dt, shape, pi, i, a, b)
					}
				}
			}
		}
	}
}

func BenchmarkNewEMAFast(b *testing.B) {
	params := wavgParams(tensor.F32, tensor.Shape{512, 512}, 4, 1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = NewEMA(params, 0.99)
	}
}

func BenchmarkNewEMASlow(b *testing.B) {
	params := wavgParams(tensor.F32, tensor.Shape{512, 512}, 4, 1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = slowNewEMA(params, 0.99)
	}
}

func BenchmarkEMAUpdateFast(b *testing.B) {
	params := wavgParams(tensor.F32, tensor.Shape{512, 512}, 4, 1)
	e := NewEMA(params, 0.99)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = e.Update()
	}
}

func BenchmarkEMAUpdateSlow(b *testing.B) {
	params := wavgParams(tensor.F32, tensor.Shape{512, 512}, 4, 1)
	e := NewEMA(params, 0.99)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = slowEMAUpdate(e)
	}
}

// --- materialize ---

// slowMaterialize is the verbatim pre-conversion per-element path for materialize.
func slowMaterialize(ref []*tensor.Tensor, avg [][]float64) []*tensor.Tensor {
	out := make([]*tensor.Tensor, len(ref))
	for pi, p := range ref {
		t := tensor.New(p.Dtype(), p.Shape())
		for i := range p.Numel() {
			t.SetF64(avg[pi][i], tensor.Unravel(i, p.Shape())...)
		}
		out[pi] = t
	}
	return out
}

func TestMaterializeBitIdenticalToSlowPath(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, shape := range []tensor.Shape{{1}, {7}, {4, 5}, {33, 17}} {
			ref := wavgParams(dt, shape, 2, 21)
			avg := make([][]float64, len(ref))
			rng := rand.New(rand.NewPCG(42, 0))
			for i, p := range ref {
				avg[i] = make([]float64, p.Numel())
				for j := range avg[i] {
					avg[i][j] = rng.NormFloat64()
				}
			}
			fast := materialize(ref, avg)
			slow := slowMaterialize(ref, avg)
			for pi := range fast {
				n := fast[pi].Numel()
				for i := range n {
					idx := tensor.Unravel(i, shape)
					a, b := fast[pi].AtF64(idx...), slow[pi].AtF64(idx...)
					if math.Float64bits(a) != math.Float64bits(b) {
						t.Fatalf("dt=%v shape=%v param %d elem %d: fast %v != slow %v (not bit-identical)", dt, shape, pi, i, a, b)
					}
				}
			}
		}
	}
}

func BenchmarkMaterializeFast(b *testing.B) {
	ref := wavgParams(tensor.F32, tensor.Shape{512, 512}, 4, 1)
	avg := make([][]float64, len(ref))
	for i, p := range ref {
		avg[i] = make([]float64, p.Numel())
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = materialize(ref, avg)
	}
}

func BenchmarkMaterializeSlow(b *testing.B) {
	ref := wavgParams(tensor.F32, tensor.Shape{512, 512}, 4, 1)
	avg := make([][]float64, len(ref))
	for i, p := range ref {
		avg[i] = make([]float64, p.Numel())
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = slowMaterialize(ref, avg)
	}
}
