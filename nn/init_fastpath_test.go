package nn

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/linalg"
	"github.com/jxsl13/goai/tensor"
)

// slowFillUniform is the verbatim pre-T870 per-element path, kept here as the
// bit-identity oracle: the fast path must reproduce it exactly (seeded, §V13).
func slowFillUniform(t *tensor.Tensor, lo, hi float64, seed uint64) {
	rng := rand.New(rand.NewPCG(seed, 0x6b79a2c3d4e5f601))
	for i := range t.Numel() {
		t.SetF64(lo+rng.Float64()*(hi-lo), tensor.Unravel(i, t.Shape())...)
	}
}

func TestFillGenBitIdenticalToSlowPath(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, shape := range []tensor.Shape{{1}, {7}, {4, 5}, {2, 3, 4}, {512, 512}} {
			fast := tensor.Zeros(dt, shape)
			slow := tensor.Zeros(dt, shape)
			fillUniform(fast, -0.3, 0.7, 42)
			slowFillUniform(slow, -0.3, 0.7, 42)
			n := fast.Numel()
			for i := range n {
				idx := tensor.Unravel(i, shape)
				a, b := fast.AtF64(idx...), slow.AtF64(idx...)
				if math.Float64bits(a) != math.Float64bits(b) {
					t.Fatalf("dt=%v shape=%v elem %d: fast %v != slow %v (not bit-identical)", dt, shape, i, a, b)
				}
			}
		}
	}
}

// A non-contiguous view must still fill correctly via the fallback.
func TestFillGenNonContiguousView(t *testing.T) {
	base := tensor.Zeros(tensor.F64, tensor.Shape{4, 3})
	view, err := base.Transpose(0, 1) // [3,4], non-contiguous
	if err != nil {
		t.Fatal(err)
	}
	fillUniform(view, 1.0, 1.0, 7) // constant 1.0 (lo==hi): every elem must be 1
	n := view.Numel()
	for i := range n {
		if got := view.AtF64(tensor.Unravel(i, view.Shape())...); got != 1.0 {
			t.Fatalf("non-contiguous fill elem %d = %v, want 1.0", i, got)
		}
	}
}

func BenchmarkFillUniformFast(b *testing.B) {
	t := tensor.Zeros(tensor.F32, tensor.Shape{2048, 2048})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fillUniform(t, -0.1, 0.1, uint64(i))
	}
}

func BenchmarkFillUniformSlow(b *testing.B) {
	t := tensor.Zeros(tensor.F32, tensor.Shape{2048, 2048})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		slowFillUniform(t, -0.1, 0.1, uint64(i))
	}
}

func slowZeros(t *tensor.Tensor) {
	for i := range t.Numel() {
		t.SetF64(0, tensor.Unravel(i, t.Shape())...)
	}
}

func BenchmarkZerosFast(b *testing.B) {
	t := tensor.Zeros(tensor.F32, tensor.Shape{2048, 2048})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Zeros(t)
	}
}

func BenchmarkZerosSlow(b *testing.B) {
	t := tensor.Zeros(tensor.F32, tensor.Shape{2048, 2048})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		slowZeros(t)
	}
}

// NOTE: TruncNormal was evaluated for this same conversion (fillGen, gen()
// returning the erfinv-mapped sample) and measured only ~1.09x end-to-end
// (2048x2048 F32, RNG+erfinv-dominated) — under the 1.1x bar — so it was
// reverted; see the comment on TruncNormal's loop in init.go. No test lives
// here for it since there is no conversion left to gate.

// --- Orthogonal (T870 follow-up: converts the tail write-back to fillGen) ---

// slowOrthogonal is the verbatim pre-conversion per-element path for Orthogonal's
// final write-back, kept as the bit-identity oracle. Identical to Orthogonal
// except the tail loop; everything upstream of it (Gaussian draw, QR, sign-fix,
// transpose undo) is untouched by the T870 follow-up so is reproduced verbatim.
func slowOrthogonal(t *tensor.Tensor, seed uint64, opts ...OrthogonalOption) error {
	cfg := orthogonalCfg{gain: 1}
	for _, o := range opts {
		o(&cfg)
	}
	if t.Ndim() < 2 {
		return fmt.Errorf("nn: Orthogonal needs a rank-≥2 tensor, got %v", t.Shape())
	}
	rows := t.Shape()[0]
	if rows <= 0 || t.Numel() <= 0 {
		return fmt.Errorf("nn: Orthogonal needs a non-empty tensor, got %v", t.Shape())
	}
	cols := t.Numel() / rows

	rng := rand.New(rand.NewPCG(seed, 0x6b79a2c3d4e5f601))
	flat := make([]float64, rows*cols)
	for i := range flat {
		flat[i] = rng.NormFloat64()
	}

	m, n := rows, cols
	transposed := rows < cols
	if transposed {
		t2 := make([]float64, rows*cols)
		for i := range rows {
			for j := range cols {
				t2[j*rows+i] = flat[i*cols+j]
			}
		}
		flat, m, n = t2, cols, rows
	}

	q, r, err := linalg.QR(tensor.FromFloat64(tensor.Shape{m, n}, flat))
	if err != nil {
		return fmt.Errorf("nn: Orthogonal QR failed: %w", err)
	}

	qm := make([]float64, m*n)
	for j := range n {
		s := 1.0
		if r.AtF64(j, j) < 0 {
			s = -1
		}
		for i := range m {
			qm[i*n+j] = q.AtF64(i, j) * s * cfg.gain
		}
	}

	if transposed {
		t2 := make([]float64, rows*cols)
		for i := range m {
			for j := range n {
				t2[j*m+i] = qm[i*n+j]
			}
		}
		qm = t2
	}
	for i := range t.Numel() {
		t.SetF64(qm[i], tensor.Unravel(i, t.Shape())...)
	}
	return nil
}

func TestOrthogonalBitIdenticalToSlowPath(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		// {3,7} exercises the transposed (wide, rows<cols) branch; {2,3,4} exercises
		// rank-3 (cols = Numel/rows, trailing dims flattened).
		for _, shape := range []tensor.Shape{{4, 4}, {3, 7}, {9, 3}, {2, 3, 4}} {
			fast := tensor.Zeros(dt, shape)
			slow := tensor.Zeros(dt, shape)
			if err := Orthogonal(fast, 77, WithOrthogonalGain(1.3)); err != nil {
				t.Fatalf("shape=%v Orthogonal: %v", shape, err)
			}
			if err := slowOrthogonal(slow, 77, WithOrthogonalGain(1.3)); err != nil {
				t.Fatalf("shape=%v slowOrthogonal: %v", shape, err)
			}
			n := fast.Numel()
			for i := range n {
				idx := tensor.Unravel(i, shape)
				a, b := fast.AtF64(idx...), slow.AtF64(idx...)
				if math.Float64bits(a) != math.Float64bits(b) {
					t.Fatalf("dt=%v shape=%v elem %d: fast %v != slow %v (not bit-identical)", dt, shape, i, a, b)
				}
			}
		}
	}
}

// A non-contiguous view must still fill correctly (and match the slow oracle
// element-for-element) via the fillGen fallback.
func TestOrthogonalNonContiguousView(t *testing.T) {
	baseFast := tensor.Zeros(tensor.F64, tensor.Shape{4, 4})
	baseSlow := tensor.Zeros(tensor.F64, tensor.Shape{4, 4})
	viewFast, err := baseFast.Transpose(0, 1) // [4,4], non-contiguous
	if err != nil {
		t.Fatal(err)
	}
	viewSlow, err := baseSlow.Transpose(0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := Orthogonal(viewFast, 99); err != nil {
		t.Fatal(err)
	}
	if err := slowOrthogonal(viewSlow, 99); err != nil {
		t.Fatal(err)
	}
	n := viewFast.Numel()
	for i := range n {
		idx := tensor.Unravel(i, viewFast.Shape())
		a, b := viewFast.AtF64(idx...), viewSlow.AtF64(idx...)
		if math.Float64bits(a) != math.Float64bits(b) {
			t.Fatalf("non-contiguous Orthogonal elem %d: fast %v != slow %v", i, a, b)
		}
	}
}

// BenchmarkOrthogonalFillFast/Slow isolate the CONVERTED step (writing a
// precomputed qm slice into t) from the unrelated, unchanged QR/Gaussian-draw
// cost that dominates whole-Orthogonal-call timing for any non-trivial matrix —
// this is the true apples-to-apples "per conversion" comparison.
func BenchmarkOrthogonalFillFast(b *testing.B) {
	shape := tensor.Shape{512, 512}
	qm := make([]float64, shape.Numel())
	for i := range qm {
		qm[i] = float64(i)
	}
	t := tensor.Zeros(tensor.F64, shape)
	b.ResetTimer()
	for k := 0; k < b.N; k++ {
		idx := 0
		fillGen(t, func() float64 {
			v := qm[idx]
			idx++
			return v
		})
	}
}

func BenchmarkOrthogonalFillSlow(b *testing.B) {
	shape := tensor.Shape{512, 512}
	qm := make([]float64, shape.Numel())
	for i := range qm {
		qm[i] = float64(i)
	}
	t := tensor.Zeros(tensor.F64, shape)
	b.ResetTimer()
	for k := 0; k < b.N; k++ {
		for i := range t.Numel() {
			t.SetF64(qm[i], tensor.Unravel(i, t.Shape())...)
		}
	}
}

// BenchmarkOrthogonalFast/Slow additionally report the whole-call cost for
// context; QR dominates here (O(n^3) Householder vs the O(n^2) fill), so this
// ratio is expected to sit much closer to 1.0x than the isolated fill above.
func BenchmarkOrthogonalFast(b *testing.B) {
	t := tensor.Zeros(tensor.F64, tensor.Shape{256, 256})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Orthogonal(t, uint64(i))
	}
}

func BenchmarkOrthogonalSlow(b *testing.B) {
	t := tensor.Zeros(tensor.F64, tensor.Shape{256, 256})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = slowOrthogonal(t, uint64(i))
	}
}
