package cpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// Grind regression tests: the broadcast suffix fast path, SIMD addbias, and the
// scratch-free softmax must stay bit-identical (§V3) / within-ulps (§V9) with
// backend/ref across every shape class the new code paths distinguish.

// TestCPUBroadcastMatchesRef cross-checks all four binary ops against ref for
// every broadcast shape class: suffix-periodic (vector, scalar, multi-dim
// suffix, size-1 mid dims), NON-periodic fallback (column, mid-dim), and
// both-operands-broadcast. Both operand orders are exercised — Sub/Div are not
// commutative, so a swapped fast path that flips arguments would fail here.
func TestCPUBroadcastMatchesRef(t *testing.T) {
	cpu := cpuBackend(t)
	ref, _ := backend.Get(backend.Ref)
	ops := []backend.Op{backend.OpAdd, backend.OpSub, backend.OpMul, backend.OpDiv}
	shapes := [][2]tensor.Shape{
		{{4, 7}, {7}},             // bias vector (fast path, small)
		{{4, 7}, {1}},             // scalar (fast path, tiled)
		{{2, 3, 5}, {3, 5}},       // multi-dim suffix (fast path)
		{{2, 3, 5}, {1, 5}},       // size-1 dim left of suffix run (fast path)
		{{4, 3, 1, 5}, {3, 1, 5}}, // size-1 INSIDE the matching run (fast path)
		{{4, 7}, {4, 1}},          // column broadcast (prefix-block fast path)
		{{4, 5000}, {4, 1}},       // column broadcast, block > bcastTile (tile stride loop)
		{{3, 4, 5}, {3, 1, 1}},    // multi-dim block (prefix-block fast path)
		{{2, 4, 5}, {2, 1, 5}},    // mid-dim broadcast (fallback: neither suffix nor prefix)
		{{4, 1}, {1, 7}},          // both operands broadcast (fallback)
		{{64, 600}, {600}},        // parallel path (38400 > parThreshold), fast path
		{{64, 600}, {1}},          // parallel path, scalar tile repeat
		{{64, 600}, {64, 1}},      // parallel path, prefix-block
	}
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		mk := func(sh tensor.Shape, seed uint64) *tensor.Tensor {
			if dt == tensor.F64 {
				return bench.RandF64(sh, seed)
			}
			return bench.RandF32(sh, seed)
		}
		for _, sp := range shapes {
			a, b := mk(sp[0], 11), mk(sp[1], 12)
			for _, op := range ops {
				// big operand left, small right
				gc, gr := run(t, cpu, op, a, b), run(t, ref, op, a, b)
				assertEqualExact(t, gc, gr, op.String()+"/"+dt.String()+"/big-left")
				// small operand left, big right (Sub/Div argument order!)
				gc, gr = run(t, cpu, op, b, a), run(t, ref, op, b, a)
				assertEqualExact(t, gc, gr, op.String()+"/"+dt.String()+"/small-left")
			}
		}
	}
}

// TestCPUBroadcastSignedZero: the prefix-block fast path splats each block's
// scalar into a reused tile; a value-equality refill skip would conflate -0.0
// with +0.0 (they compare equal) and flip Mul result signs. Locks sign
// propagation bit-exactly against ref.
func TestCPUBroadcastSignedZero(t *testing.T) {
	cpu := cpuBackend(t)
	ref, _ := backend.Get(backend.Ref)
	x := tensor.New(tensor.F32, tensor.Shape{2, 4})
	for i := range x.Storage().F32() {
		x.Storage().F32()[i] = 1
	}
	col := tensor.New(tensor.F32, tensor.Shape{2, 1})
	copy(col.Storage().F32(), []float32{0, float32(math.Copysign(0, -1))})
	gc, gr := run(t, cpu, backend.OpMul, x, col), run(t, ref, backend.OpMul, x, col)
	c, r := gc.Storage().F32(), gr.Storage().F32()
	for i := range c {
		if math.Signbit(float64(c[i])) != math.Signbit(float64(r[i])) || c[i] != r[i] {
			t.Errorf("mul signed-zero [%d]: cpu %v ref %v", i, c[i], r[i])
		}
	}
}

// TestCPUBroadcastNonContiguous: the fast path must materialize non-contiguous
// operands before slicing raw storage (a transposed view's storage order does
// NOT match its logical order).
func TestCPUBroadcastNonContiguous(t *testing.T) {
	cpu := cpuBackend(t)
	ref, _ := backend.Get(backend.Ref)
	m := bench.RandF32(tensor.Shape{5, 3}, 21)
	mt, _ := m.Transpose(0, 1) // logical [3,5], non-contiguous
	v := bench.RandF32(tensor.Shape{5}, 22)
	for _, op := range []backend.Op{backend.OpAdd, backend.OpSub, backend.OpDiv} {
		gc, gr := run(t, cpu, op, mt, v), run(t, ref, op, mt, v)
		assertEqualExact(t, gc, gr, op.String()+"/noncontig-bcast")
	}
}

// TestCPUAddBiasMatchesRef locks the SIMD-per-row addbias against ref
// bit-exactly, across serial and parallel row counts and both dtypes,
// including a non-contiguous x.
func TestCPUAddBiasMatchesRef(t *testing.T) {
	cpu := cpuBackend(t)
	ref, _ := backend.Get(backend.Ref)
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		mk := func(sh tensor.Shape, seed uint64) *tensor.Tensor {
			if dt == tensor.F64 {
				return bench.RandF64(sh, seed)
			}
			return bench.RandF32(sh, seed)
		}
		for _, sh := range []tensor.Shape{{3, 17}, {257, 129}, {130, 512}} { // serial + parallel
			x, b := mk(sh, 31), mk(tensor.Shape{sh[1]}, 32)
			gc, gr := run(t, cpu, backend.OpAddBias, x, b), run(t, ref, backend.OpAddBias, x, b)
			assertEqualExact(t, gc, gr, "addbias/"+dt.String())
		}
		// non-contiguous x via transpose
		x := mk(tensor.Shape{9, 6}, 33)
		xt, _ := x.Transpose(0, 1)
		b := mk(tensor.Shape{9}, 34)
		gc, gr := run(t, cpu, backend.OpAddBias, xt, b), run(t, ref, backend.OpAddBias, xt, b)
		assertEqualExact(t, gc, gr, "addbias-noncontig/"+dt.String())
	}
}

// TestCPUAddBiasBackwardMatchesRef locks the parallel column-owned addbias
// backward (dbias[j] = Σ_i g[i,j]) against ref BIT-EXACTLY, both dtypes.
// Column ownership preserves each column's row-order f64 accumulation, so the
// sum sequence is identical to ref's serial loop (§V3). Shapes cover the
// serial path (below parThreshold), both profiled training shapes ([256,2048],
// [256,512] — parallel, chunked over columns), a non-multiple-of-workers width
// (uneven last chunk), and a non-contiguous g via transpose.
func TestCPUAddBiasBackwardMatchesRef(t *testing.T) {
	cpu := cpuBackend(t)
	ref, _ := backend.Get(backend.Ref)
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		mk := func(sh tensor.Shape, seed uint64) *tensor.Tensor {
			if dt == tensor.F64 {
				return bench.RandF64(sh, seed)
			}
			return bench.RandF32(sh, seed)
		}
		for _, sh := range []tensor.Shape{{3, 17}, {256, 2048}, {256, 512}, {130, 129}} {
			g := mk(sh, 41)
			gc, gr := run(t, cpu, backend.OpAddBiasBackward, g), run(t, ref, backend.OpAddBiasBackward, g)
			assertEqualExact(t, gc, gr, "addbias-backward/"+dt.String())
		}
		// non-contiguous g via transpose
		g := mk(tensor.Shape{9, 6}, 42)
		gt, _ := g.Transpose(0, 1)
		gc, gr := run(t, cpu, backend.OpAddBiasBackward, gt), run(t, ref, backend.OpAddBiasBackward, gt)
		assertEqualExact(t, gc, gr, "addbias-backward-noncontig/"+dt.String())
	}
}

// TestCPUSoftmaxEdge locks the scratch-free softmax's edge behavior to ref:
// +Inf rows (NaN everywhere: Inf−Inf), −Inf rows (exp(NaN)), NaN propagation,
// huge-magnitude rows (max-shift prevents overflow), and a tiny d=1 axis.
// Ref semantics are the contract; comparisons are NaN-aware and exact.
func TestCPUSoftmaxEdge(t *testing.T) {
	cpu := cpuBackend(t)
	ref, _ := backend.Get(backend.Ref)
	inf, nan := math.Inf(1), math.NaN()
	rows := [][]float32{
		{1, 2, 3, 4},
		{float32(inf), 1, 2, 3}, // Inf−Inf → NaN row
		{float32(-inf), float32(-inf), float32(-inf), float32(-inf)}, // all −Inf → NaN row
		{float32(nan), 1, 2, 3}, // NaN poisons the row
		{3e38, 3e38, 1, -3e38},  // near-F32-max: stable shift, no overflow
		{-1e4, 0, 1e4, 2e4},     // large spread: underflow to 0 on the left
	}
	d := len(rows[0])
	x := tensor.New(tensor.F32, tensor.Shape{len(rows), d})
	xs := x.Storage().F32()
	for i, r := range rows {
		copy(xs[i*d:], r)
	}
	gc, gr := run(t, cpu, backend.OpSoftmax, x), run(t, ref, backend.OpSoftmax, x)
	c, r := gc.Storage().F32(), gr.Storage().F32()
	for i := range c {
		cf, rf := float64(c[i]), float64(r[i])
		if math.IsNaN(cf) != math.IsNaN(rf) {
			t.Errorf("softmax edge [%d]: cpu=%v ref=%v (NaN mismatch)", i, c[i], r[i])
		} else if !math.IsNaN(cf) && math.Abs(cf-rf) > 5e-5*math.Max(1, math.Abs(rf)) {
			t.Errorf("softmax edge [%d]: cpu=%v ref=%v", i, c[i], r[i])
		}
	}

	// Wide path (intra-row parallel, rows==1, d ≥ 8192): must match ref within
	// the same ulps budget as the row-parallel path. Also covers a d just over
	// the gate and a chunk-ragged d (not a multiple of GOMAXPROCS).
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, sh := range []tensor.Shape{{1, 16384}, {1, 9001}, {1, 8193}} {
			w := tensor.Randn(dt, 41, sh)
			gw, grw := run(t, cpu, backend.OpSoftmax, w), run(t, ref, backend.OpSoftmax, w)
			assertCloseUlps(t, gw, grw, "softmax-wide/"+dt.String())
		}
	}

	// d=1: every row normalizes to exactly 1.
	one := tensor.New(tensor.F32, tensor.Shape{3, 1})
	copy(one.Storage().F32(), []float32{-5, 0, 7})
	g := run(t, cpu, backend.OpSoftmax, one)
	for i, v := range g.Storage().F32() {
		if v != 1 {
			t.Errorf("softmax d=1 [%d] = %v, want 1", i, v)
		}
	}
}
