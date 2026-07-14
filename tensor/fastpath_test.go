package tensor

import (
	"math"
	"testing"
)

// Cross-reference tests pinning the devirtualized Cast/Contiguous fast paths
// (§base-perf) to the generic widen-through-f64 reference semantics: for every
// dtype pair and layout, the fast path must produce bit-identical results to
// setF64(atF64(x)) in row-major traversal order.

// refCast is the reference cast: per-element widen-through-f64 in row-major
// order, exactly what the pre-devirt generic loop did.
func refCast(t *Tensor, dtype Dtype) *Tensor {
	out := NewOn(t.Device(), dtype, t.Shape())
	n := t.Numel()
	for i := 0; i < n; i++ {
		idx := Unravel(i, t.Shape())
		out.storage.setF64(i, t.AtF64(idx...))
	}
	return out
}

// bitsOf reads element i of a contiguous tensor as raw comparable bits so NaN
// patterns compare exactly.
func bitsOf(t *Tensor, i int) uint64 {
	switch t.Dtype() {
	case F32:
		return uint64(math.Float32bits(t.Storage().F32()[i]))
	case F64:
		return math.Float64bits(t.Storage().F64()[i])
	case F16, BF16:
		return uint64(t.Storage().U16()[i])
	}
	panic("bitsOf: bad dtype")
}

// interestingF64 mixes normals, denormal-range values, huge values (f16
// overflow), negatives, zeros, and NaN/Inf so every rounding branch is hit.
var interestingF64 = []float64{
	0, math.Copysign(0, -1), 1, -1, 0.5, 1.0009765625, 65504, 65519, 65520, 65536,
	-65520, 3.14159265358979, 1e-8, -1e-8, 6.103515625e-05, 5.960464477539063e-08,
	2.980232238769531e-08, 1e300, -1e300, 1e-300, math.Inf(1), math.Inf(-1), math.NaN(),
}

// buildSrc creates a (4, len(interestingF64)) tensor of dtype dt whose rows
// cycle through the interesting values with different signs/scales.
func buildSrc(t *testing.T, dt Dtype) *Tensor {
	t.Helper()
	cols := len(interestingF64)
	src := New(dt, Shape{4, cols})
	for r := 0; r < 4; r++ {
		for c := 0; c < cols; c++ {
			v := interestingF64[c]
			if r%2 == 1 {
				v = -v
			}
			if r >= 2 {
				v *= 3
			}
			src.SetF64(v, r, c)
		}
	}
	return src
}

var allDtypes = []Dtype{F32, F64, F16, BF16}

// TestCastMatchesReferenceAllPairs cross-checks Cast for every (src,dst) dtype
// pair over four layouts: contiguous, contiguous-with-offset (dim-0 slice),
// transposed (strided), and inner-row slice (innermost stride 1).
func TestCastMatchesReferenceAllPairs(t *testing.T) {
	for _, sd := range allDtypes {
		src := buildSrc(t, sd)
		views := map[string]*Tensor{"contig": src}
		if v, err := src.Slice(0, 1, 3); err == nil {
			views["offset"] = v
		} else {
			t.Fatal(err)
		}
		if v, err := src.Transpose(0, 1); err == nil {
			views["transposed"] = v
		} else {
			t.Fatal(err)
		}
		if v, err := src.Slice(1, 2, len(interestingF64)-3); err == nil {
			views["innerRows"] = v
		} else {
			t.Fatal(err)
		}
		for name, v := range views {
			for _, dd := range allDtypes {
				got := v.Cast(dd)
				want := refCast(v, dd)
				if !got.Shape().Equal(want.Shape()) || got.Dtype() != dd {
					t.Fatalf("%v→%v %s: shape/dtype mismatch", sd, dd, name)
				}
				for i := 0; i < got.Numel(); i++ {
					g, w := bitsOf(got, i), bitsOf(want, i)
					if g != w {
						t.Errorf("%v→%v %s: elem %d bits 0x%X want 0x%X", sd, dd, name, i, g, w)
					}
				}
			}
		}
	}
}

// TestContiguousMatchesReferenceAllLayouts cross-checks Contiguous over the
// same layouts for all dtypes: values (as raw bits) must equal the row-major
// element order of the view.
func TestContiguousMatchesReferenceAllLayouts(t *testing.T) {
	for _, sd := range allDtypes {
		src := buildSrc(t, sd)
		views := map[string]*Tensor{}
		var err error
		if views["offset"], err = src.Slice(0, 1, 3); err != nil {
			t.Fatal(err)
		}
		if views["transposed"], err = src.Transpose(0, 1); err != nil {
			t.Fatal(err)
		}
		if views["innerRows"], err = src.Slice(1, 2, len(interestingF64)-3); err != nil {
			t.Fatal(err)
		}
		// A 3-D permuted view with innermost stride 1 exercises gatherRows with
		// a multi-dim outer walk.
		r3 := New(sd, Shape{3, 4, 5})
		for i := 0; i < r3.Numel(); i++ {
			r3.Storage().setF64(i, float64(i)*0.25-7)
		}
		if views["perm3d"], err = r3.Permute(1, 0, 2); err != nil {
			t.Fatal(err)
		}
		for name, v := range views {
			got := v.Contiguous()
			if got == v || got.Offset() != 0 || !got.IsContiguous() {
				t.Fatalf("%v %s: result must be a fresh contiguous offset-0 tensor", sd, name)
			}
			want := refCast(v, v.Dtype()) // reference row-major materialization
			for i := 0; i < got.Numel(); i++ {
				if g, w := bitsOf(got, i), bitsOf(want, i); g != w {
					t.Errorf("%v %s: elem %d bits 0x%X want 0x%X", sd, name, i, g, w)
				}
			}
			// Fresh copy, not an alias: writing the copy must not touch the view.
			if got.Numel() > 0 {
				before := v.AtF64(make([]int, v.Ndim())...)
				got.SetF64(before+1, make([]int, got.Ndim())...)
				if v.AtF64(make([]int, v.Ndim())...) != before {
					t.Fatalf("%v %s: Contiguous result aliases the source", sd, name)
				}
			}
		}
	}
}

// TestContiguousEmptyViews: zero-element strided views must not touch storage
// (a dim-0 slice at the end of storage has offset == len(data); a flat-copy
// fast path that sliced [off:off+0] before the n==0 check would still be legal,
// but [off:off+n] with a stale n would panic — lock the behavior).
func TestContiguousEmptyViews(t *testing.T) {
	x := New(F32, Shape{4, 3})
	v, err := x.Slice(0, 4, 4) // empty, offset = 12 = len(data)
	if err != nil {
		t.Fatal(err)
	}
	c := v.Contiguous()
	if c.Numel() != 0 || !c.Shape().Equal(Shape{0, 3}) {
		t.Fatalf("empty contiguous wrong: %v", c.Shape())
	}
	if got := v.Cast(F64); got.Numel() != 0 {
		t.Fatal("empty cast wrong")
	}
}

// TestFillHalfMatchesSetF64 pins the devirtualized F16/BF16 fill paths in the
// constructors to the setF64 narrowing semantics.
func TestFillHalfMatchesSetF64(t *testing.T) {
	for _, dt := range []Dtype{F16, BF16} {
		got := Full(dt, Shape{3, 2}, 3.14159265358979)
		want := New(dt, Shape{3, 2})
		for i := 0; i < 6; i++ {
			want.storage.setF64(i, 3.14159265358979)
		}
		for i := 0; i < 6; i++ {
			if bitsOf(got, i) != bitsOf(want, i) {
				t.Errorf("%v Full elem %d: bits 0x%X want 0x%X", dt, i, bitsOf(got, i), bitsOf(want, i))
			}
		}
	}
}

// TestEyeAllDtypes pins the diagonal-only Eye against the definition, including
// the half dtypes that now skip the generator entirely.
func TestEyeAllDtypes(t *testing.T) {
	for _, dt := range allDtypes {
		e := Eye(dt, 5)
		for i := 0; i < 5; i++ {
			for j := 0; j < 5; j++ {
				want := 0.0
				if i == j {
					want = 1.0
				}
				if got := e.AtF64(i, j); got != want {
					t.Errorf("%v Eye[%d,%d] = %g, want %g", dt, i, j, got, want)
				}
			}
		}
	}
	if e := Eye(F32, 0); e.Numel() != 0 {
		t.Error("Eye(0) must be empty")
	}
}

// TestPoolGrowWithinClassZeroed: regression for the Free-without-reslice
// optimization. Free stores the buffer with its handed-out LENGTH (5), not its
// capacity (8); a subsequent Alloc of a LARGER n in the same class must extend
// via capacity and still return fully zeroed memory.
func TestPoolGrowWithinClassZeroed(t *testing.T) {
	p := NewPool()
	b1 := p.Alloc(F32, 5).([]float32) // class 3, cap 8
	full := b1[:cap(b1)]
	for i := range full {
		full[i] = 99 // dirty the whole backing array, beyond len too
	}
	p.Free(b1)
	b2 := p.Alloc(F32, 7).([]float32) // same class, larger n than stored len
	if len(b2) != 7 {
		t.Fatalf("len %d, want 7", len(b2))
	}
	for i, v := range b2 {
		if v != 0 {
			t.Fatalf("grown pooled alloc not zeroed at %d: %g", i, v)
		}
	}
	// And the f64 side.
	c1 := p.Alloc(F64, 3).([]float64) // class 2, cap 4
	for i := range c1[:cap(c1)] {
		c1[:cap(c1)][i] = -1
	}
	p.Free(c1)
	c2 := p.Alloc(F64, 4).([]float64)
	for i, v := range c2 {
		if v != 0 {
			t.Fatalf("f64 grown pooled alloc not zeroed at %d: %g", i, v)
		}
	}
}

// TestF16BoundaryRounding locks the f16 overflow/underflow rounding boundaries
// (RNE): 65520 is the exact tie between max-finite 65504 and 2^16 and must
// round to +Inf (even); 2^-25 is the exact tie between 0 and the smallest
// subnormal and must round to 0 (even); anything strictly above 2^-25 must
// round up to the smallest subnormal 2^-24.
func TestF16BoundaryRounding(t *testing.T) {
	if got := f32ToF16(65520); got != 0x7C00 {
		t.Errorf("f32ToF16(65520) = 0x%04X, want 0x7C00 (+Inf, RNE tie)", got)
	}
	if got := f32ToF16(65519.996); got != 0x7BFF {
		t.Errorf("f32ToF16(65519.996) = 0x%04X, want 0x7BFF (max finite)", got)
	}
	tie := float32(math.Exp2(-25))
	if got := f32ToF16(tie); got != 0x0000 {
		t.Errorf("f32ToF16(2^-25) = 0x%04X, want 0x0000 (tie to even zero)", got)
	}
	above := math.Nextafter32(tie, 1)
	if got := f32ToF16(above); got != 0x0001 {
		t.Errorf("f32ToF16(2^-25+ulp) = 0x%04X, want 0x0001 (smallest subnormal)", got)
	}
	if got := f32ToF16(float32(math.Exp2(-24))); got != 0x0001 {
		t.Errorf("f32ToF16(2^-24) = 0x%04X, want 0x0001", got)
	}
	// Negative zero keeps its sign.
	if got := f32ToF16(float32(math.Copysign(0, -1))); got != 0x8000 {
		t.Errorf("f32ToF16(-0) = 0x%04X, want 0x8000", got)
	}
}

// TestBF16NaNLowMantissa: a float32 NaN whose payload lives ONLY in the low 16
// mantissa bits (0x7F800001) would truncate to the +Inf bit pattern; f32ToBF16
// must still produce a NaN.
func TestBF16NaNLowMantissa(t *testing.T) {
	nan := math.Float32frombits(0x7F800001)
	bits := f32ToBF16(nan)
	if !math.IsNaN(float64(bf16ToF32(bits))) {
		t.Errorf("f32ToBF16(low-mantissa NaN) = 0x%04X, not NaN", bits)
	}
	neg := math.Float32frombits(0xFF800001)
	if !math.IsNaN(float64(bf16ToF32(f32ToBF16(neg)))) {
		t.Error("negative low-mantissa NaN lost")
	}
}

// TestUnravelOffsetRoundTrip: Unravel is the inverse of Offset with row-major
// strides and base 0, for every position, including a shape with a size-1 dim.
func TestUnravelOffsetRoundTrip(t *testing.T) {
	for _, sh := range []Shape{{7}, {3, 4}, {2, 1, 5}, {2, 3, 2, 2}} {
		st := RowMajorStrides(sh)
		for pos := 0; pos < sh.Numel(); pos++ {
			idx := Unravel(pos, sh)
			if back := Offset(0, st, idx); back != pos {
				t.Fatalf("shape %v: Offset(Unravel(%d)) = %d", sh, pos, back)
			}
		}
	}
}

// TestCloneShapeStridesIndependence: the single-allocation shape+strides pair
// must behave exactly like independent slices — appending to the shape must
// never overwrite the strides (full-capacity slicing).
func TestCloneShapeStridesIndependence(t *testing.T) {
	x := New(F32, Shape{2, 3})
	sh := x.Shape()
	st := x.Strides()
	if !sh.Equal(Shape{2, 3}) || st[0] != 3 || st[1] != 1 {
		t.Fatalf("shape/strides wrong: %v %v", sh, st)
	}
	_ = append(sh, 99) // must reallocate, not clobber st
	if st[0] != 3 || st[1] != 1 {
		t.Fatal("append to Shape() clobbered Strides() — shared backing not capacity-limited")
	}
	// nil-shape parity with the old Clone()+RowMajorStrides path
	shN, stN := cloneShapeStrides(nil)
	if shN != nil || stN == nil || len(stN) != 0 {
		t.Fatalf("nil shape: got (%v,%v), want (nil, empty)", shN, stN)
	}
	shE, stE := cloneShapeStrides(Shape{})
	if shE == nil || len(shE) != 0 || stE == nil {
		t.Fatalf("empty shape: got (%v,%v)", shE, stE)
	}
}
