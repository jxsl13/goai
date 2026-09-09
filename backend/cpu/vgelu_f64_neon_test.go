//go:build arm64 && goexperiment.simd

package cpu

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func geluF64Oracle(x float64) float64 {
	return 0.5 * x * (1 + math.Erf(x/math.Sqrt2))
}

func geluGradF64Oracle(x, g float64) float64 {
	phi := 0.5 * (1 + math.Erf(x/math.Sqrt2))
	pdf := 0.3989422804014327 * math.Exp(-0.5*x*x)
	return g * (phi + x*pdf)
}

func assertGELUF64Value(t *testing.T, got, want float64) {
	t.Helper()
	if math.IsNaN(want) {
		if !math.IsNaN(got) {
			t.Fatalf("got %g, want NaN", got)
		}
		return
	}
	if math.Float64bits(got) != math.Float64bits(want) {
		t.Fatalf("got %.17g (%016x), want %.17g (%016x)", got, math.Float64bits(got), want, math.Float64bits(want))
	}
}

func assertGELUF64Close(t *testing.T, got, want float64) {
	t.Helper()
	if math.IsNaN(want) {
		if !math.IsNaN(got) {
			t.Fatalf("got %g, want NaN", got)
		}
		return
	}
	if math.IsNaN(got) || math.IsInf(got, 0) || math.IsInf(want, 0) {
		if math.Float64bits(got) != math.Float64bits(want) {
			t.Fatalf("got %g (%016x), want %g (%016x)", got, math.Float64bits(got), want, math.Float64bits(want))
		}
		return
	}
	if want == 0 {
		if math.Float64bits(got) != math.Float64bits(want) {
			t.Fatalf("zero sign differs: got %016x, want %016x", math.Float64bits(got), math.Float64bits(want))
		}
		return
	}
	if math.Abs(got-want) > 1e-12*math.Max(1, math.Abs(want)) {
		t.Fatalf("got %.17g, want %.17g", got, want)
	}
}

func TestVGELUF64NeonBaselineNumerics(t *testing.T) {
	values := []float64{
		0, math.Copysign(0, -1), math.SmallestNonzeroFloat64, -math.SmallestNonzeroFloat64,
		math.Inf(1), math.Inf(-1), math.NaN(), math.MaxFloat64, -math.MaxFloat64,
		1e-300, -1e-300, 1e150, -1e150,
	}
	for i := -4096; i <= 4096; i++ {
		values = append(values, float64(i)/128)
	}
	for _, boundary := range []float64{math.Sqrt2, -math.Sqrt2, 6 * math.Sqrt2, -6 * math.Sqrt2, 32, -32} {
		values = append(values, math.Nextafter(boundary, math.Inf(-1)), boundary, math.Nextafter(boundary, math.Inf(1)))
	}
	for k := 0; k <= 64; k++ {
		x := math.Sqrt(2 * (float64(k) + 0.5) * math.Ln2)
		values = append(values, x, -x)
	}
	g := make([]float64, len(values))
	for i := range g {
		g[i] = -2 + 4*float64((i*37)%257)/256
	}
	xBefore, gBefore := slices.Clone(values), slices.Clone(g)
	forward, backward := make([]float64, len(values)), make([]float64, len(values))
	vgeluF64(forward, values)
	vgeluGradF64(backward, values, g)
	if !equalF64Bits(values, xBefore) || !equalF64Bits(g, gBefore) {
		t.Fatal("GELU leaf changed an input")
	}
	for i, x := range values {
		assertGELUF64Value(t, forward[i], geluF64Oracle(x))
		assertGELUF64Value(t, backward[i], geluGradF64Oracle(x, g[i]))
	}
}

func equalF64Bits(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float64bits(a[i]) != math.Float64bits(b[i]) {
			return false
		}
	}
	return true
}

func TestVGELUF64NeonLengthsAliasesAndBodyTail(t *testing.T) {
	for _, n := range []int{0, 1, 2, 3, 7, 17, 2049} {
		x := make([]float64, n)
		g := make([]float64, n)
		for i := range x {
			x[i] = -1 + 2*float64(i+1)/float64(n+1)
			g[i] = -2 + 4*float64((i*19)%101)/100
		}
		wantF, wantB := make([]float64, n), make([]float64, n)
		for i := range x {
			wantF[i] = geluF64Oracle(x[i])
			wantB[i] = geluGradF64Oracle(x[i], g[i])
		}
		gotF, gotB := make([]float64, n), make([]float64, n)
		vgeluF64(gotF, x)
		vgeluGradF64(gotB, x, g)
		xAlias := slices.Clone(x)
		vgeluF64(xAlias, xAlias)
		xBackAlias := slices.Clone(x)
		vgeluGradF64(xBackAlias, xBackAlias, g)
		gAlias := slices.Clone(g)
		vgeluGradF64(gAlias, x, gAlias)
		for i := range x {
			assertGELUF64Close(t, gotF[i], wantF[i])
			assertGELUF64Value(t, xAlias[i], gotF[i])
			assertGELUF64Close(t, gotB[i], wantB[i])
			assertGELUF64Value(t, xBackAlias[i], gotB[i])
			assertGELUF64Value(t, gAlias[i], gotB[i])
		}
	}
}

func TestVGELUF64NeonEligibleAccuracyAndBodyTail(t *testing.T) {
	values := []float64{0, math.Copysign(0, -1), math.SmallestNonzeroFloat64, -math.SmallestNonzeroFloat64}
	for i := -4096; i <= 4096; i++ {
		values = append(values, float64(i)/128)
	}
	for _, y := range []float64{-6, -1, 1, 6} {
		x := y * math.Sqrt2
		values = append(values, math.Nextafter(x, math.Inf(-1)), x, math.Nextafter(x, math.Inf(1)))
	}
	values = append(values, math.Nextafter(-32, 0), math.Nextafter(32, 0))
	for k := 0; k <= 738; k++ {
		x := math.Sqrt(2 * (float64(k) + 0.5) * math.Ln2)
		for _, v := range []float64{x, -x} {
			if math.Abs(v) <= 32 {
				values = append(values, math.Nextafter(v, math.Inf(-1)), v, math.Nextafter(v, math.Inf(1)))
			}
		}
	}
	state := uint64(0xd1b54a32d192ed03)
	for range 4096 {
		state = state*6364136223846793005 + 1442695040888963407
		values = append(values, 64*(float64(state>>11)*(1.0/(1<<53)))-32)
	}
	g := make([]float64, len(values))
	for i := range g {
		g[i] = -2 + 4*float64((i*37)%257)/256
	}
	for i, v := range []float64{-8, -1e-150, 0, 1e-150, 8, math.Nextafter(-8, 0), math.Nextafter(-1e-150, math.Inf(-1)), math.Nextafter(1e-150, math.Inf(1)), math.Nextafter(8, 0)} {
		g[i] = v
	}
	f, d := make([]float64, len(values)), make([]float64, len(values))
	vgeluF64(f, values)
	vgeluGradF64(d, values, g)
	var maxAbs, maxNorm float64
	for i, x := range values {
		wantF, wantD := geluF64Oracle(x), geluGradF64Oracle(x, g[i])
		assertGELUF64Close(t, f[i], wantF)
		assertGELUF64Close(t, d[i], wantD)
		for _, pair := range [][2]float64{{f[i], wantF}, {d[i], wantD}} {
			err := math.Abs(pair[0] - pair[1])
			maxAbs = math.Max(maxAbs, err)
			maxNorm = math.Max(maxNorm, err/math.Max(1, math.Abs(pair[1])))
		}
		pairX := []float64{x, 0.25}
		pairG := []float64{g[i], 0.5}
		pairF, pairD := make([]float64, 2), make([]float64, 2)
		tailF, tailD := make([]float64, 1), make([]float64, 1)
		vgeluF64(pairF, pairX)
		vgeluF64(tailF, pairX[:1])
		vgeluGradF64(pairD, pairX, pairG)
		vgeluGradF64(tailD, pairX[:1], pairG[:1])
		assertGELUF64Value(t, pairF[0], tailF[0])
		assertGELUF64Value(t, pairD[0], tailD[0])
	}
	t.Logf("max_abs_error=%g max_normalized_error=%g", maxAbs, maxNorm)
}

func TestVGELUF64NeonExactFallbackWholeCall(t *testing.T) {
	badX := []float64{math.Nextafter(32, math.Inf(1)), math.Nextafter(-32, math.Inf(-1)), math.Inf(1), math.Inf(-1), math.NaN()}
	for _, n := range []int{3, 4} {
		for badIndex := 0; badIndex < n; badIndex++ {
			for badCase, bad := range badX {
				t.Run(fmt.Sprintf("badx/n%d/index%d/case%d", n, badIndex, badCase), func(t *testing.T) {
					x, g := make([]float64, n), make([]float64, n)
					for i := range x {
						x[i], g[i] = -0.75+float64(i)/3, -1+float64(i)
					}
					x[badIndex] = bad
					xBefore, gBefore := slices.Clone(x), slices.Clone(g)
					wantF, wantD := make([]float64, n), make([]float64, n)
					for i := range x {
						wantF[i], wantD[i] = geluF64Oracle(x[i]), geluGradF64Oracle(x[i], g[i])
					}
					gotF, gotD := make([]float64, n), make([]float64, n)
					vgeluF64(gotF, x)
					vgeluGradF64(gotD, x, g)
					if !equalF64Bits(x, xBefore) || !equalF64Bits(g, gBefore) {
						t.Fatal("fallback changed input")
					}
					xForwardAlias, xBackwardAlias, gBackwardAlias := slices.Clone(x), slices.Clone(x), slices.Clone(g)
					vgeluF64(xForwardAlias, xForwardAlias)
					vgeluGradF64(xBackwardAlias, xBackwardAlias, g)
					vgeluGradF64(gBackwardAlias, x, gBackwardAlias)
					for i := range x {
						for _, got := range []float64{gotF[i], xForwardAlias[i]} {
							assertGELUF64Value(t, got, wantF[i])
						}
						for _, got := range []float64{gotD[i], xBackwardAlias[i], gBackwardAlias[i]} {
							assertGELUF64Value(t, got, wantD[i])
						}
					}
				})
			}
		}
	}
	badG := []float64{math.Inf(1), math.Inf(-1), math.NaN(), math.Nextafter(1e-150, 0), math.Nextafter(-1e-150, 0), math.Nextafter(8, math.Inf(1)), math.Nextafter(-8, math.Inf(-1))}
	for _, n := range []int{3, 4} {
		for badIndex := 0; badIndex < n; badIndex++ {
			for badCase, bad := range badG {
				t.Run(fmt.Sprintf("badg/n%d/index%d/case%d", n, badIndex, badCase), func(t *testing.T) {
					x, g := make([]float64, n), make([]float64, n)
					for i := range x {
						x[i], g[i] = -0.75+float64(i)/3, -1+float64(i)
					}
					g[badIndex] = bad
					xBefore, gBefore := slices.Clone(x), slices.Clone(g)
					want, got := make([]float64, n), make([]float64, n)
					for i := range x {
						want[i] = geluGradF64Oracle(x[i], g[i])
					}
					vgeluGradF64(got, x, g)
					if !equalF64Bits(x, xBefore) || !equalF64Bits(g, gBefore) {
						t.Fatal("fallback changed input")
					}
					xAlias, gAlias := slices.Clone(x), slices.Clone(g)
					vgeluGradF64(xAlias, xAlias, g)
					vgeluGradF64(gAlias, x, gAlias)
					for i := range x {
						for _, v := range []float64{got[i], xAlias[i], gAlias[i]} {
							assertGELUF64Value(t, v, want[i])
						}
					}
				})
			}
		}
	}
}

func TestVGELUF64NeonZeroAlloc(t *testing.T) {
	x, g, dst := make([]float64, 2048), make([]float64, 2048), make([]float64, 2048)
	for i := range x {
		x[i], g[i] = float64(i%31)/16-1, 1
	}
	if got := testing.AllocsPerRun(100, func() { vgeluF64(dst, x); vgeluGradF64(dst, x, g) }); got != 0 {
		t.Fatalf("allocs = %g, want 0", got)
	}
}

type geluF64NeonRecorder struct{ calls int }

func (r *geluF64NeonRecorder) Record(backend.Op, []*tensor.Tensor, []*tensor.Tensor, backend.Attrs) {
	r.calls++
}

func TestVGELUF64NeonProductionForward(t *testing.T) {
	be, ok := backend.Get(backend.CPU)
	if !ok {
		t.Fatal("cpu backend unavailable")
	}
	ref := backend.Reference()
	base := tensor.New(tensor.F64, tensor.Shape{11})
	copy(base.Storage().F64(), []float64{99, 0, math.Copysign(0, -1), -0.75, 0.5, math.Nextafter(32, math.Inf(1)), math.NaN(), math.Inf(1), -4, 7, 88})
	in, err := base.Slice(0, 1, 10)
	if err != nil {
		t.Fatal(err)
	}
	before := slices.Clone(base.Storage().F64())
	recorder := &geluF64NeonRecorder{}
	got, err := backend.Execute(backend.NewContext().WithBackend(be).WithRecorder(recorder), backend.OpGELU, []*tensor.Tensor{in}, nil)
	if err != nil {
		t.Fatal(err)
	}
	want, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpGELU, []*tensor.Tensor{in}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if recorder.calls != 1 {
		t.Fatalf("recorder calls=%d, want 1", recorder.calls)
	}
	if got[0].Dtype() != tensor.F64 || !got[0].Shape().Equal(in.Shape()) || !got[0].IsContiguous() || got[0].Offset() != 0 {
		t.Fatalf("bad output metadata")
	}
	if !equalF64Bits(base.Storage().F64(), before) {
		t.Fatal("Execute changed input backing storage")
	}
	for i, v := range got[0].Storage().F64() {
		assertGELUF64Value(t, v, want[0].Storage().F64()[i])
	}

	eligible := tensor.New(tensor.F64, tensor.Shape{2049})
	for i := range eligible.Storage().F64() {
		eligible.Storage().F64()[i] = -32 + 64*float64(i)/2048
	}
	got, err = backend.Execute(backend.NewContext().WithBackend(be), backend.OpGELU, []*tensor.Tensor{eligible}, nil)
	if err != nil {
		t.Fatal(err)
	}
	want, err = backend.Execute(backend.NewContext().WithBackend(ref), backend.OpGELU, []*tensor.Tensor{eligible}, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range got[0].Storage().F64() {
		assertGELUF64Close(t, v, want[0].Storage().F64()[i])
	}
}

func TestVGELUF64NeonPublicReachability(t *testing.T) {
	be, ok := backend.Get(backend.CPU)
	if !ok {
		t.Fatal("cpu backend unavailable")
	}
	for _, n := range []int{3, 4, 200003, 262144} {
		t.Run(fmt.Sprintf("forward/n%d", n), func(t *testing.T) {
			x := make([]float64, n)
			witness := false
			for i := range x {
				x[i] = -8 + 16*float64((i*1137)%10000)/10000
				for step := 1; step < 10000 && !witness; step++ {
					candidate := -8 + 16*float64(step)/10000
					one := []float64{0}
					vgeluF64(one, []float64{candidate})
					if math.Float64bits(one[0]) != math.Float64bits(geluF64Oracle(candidate)) {
						x[i], witness = candidate, true
					}
				}
			}
			if !witness {
				t.Fatal("no forward scalar-reference bit witness found")
			}
			want := make([]float64, n)
			vgeluF64(want, x)
			in := tensor.New(tensor.F64, tensor.Shape{n})
			copy(in.Storage().F64(), x)
			got, err := backend.Execute(backend.NewContext().WithBackend(be), backend.OpGELU, []*tensor.Tensor{in}, nil)
			if err != nil {
				t.Fatal(err)
			}
			for i := range want {
				assertGELUF64Value(t, got[0].Storage().F64()[i], want[i])
			}
		})
		t.Run(fmt.Sprintf("backward/n%d", n), func(t *testing.T) {
			x, g := make([]float64, n), make([]float64, n)
			witness := false
			for i := range x {
				x[i], g[i] = -8+16*float64((i*1777)%10000)/10000, 0.75+float64(i%17)/10
			}
			for step := 1; step < 10000 && !witness; step++ {
				candidate := -8 + 16*float64(step)/10000
				one := []float64{0}
				vgeluGradF64(one, []float64{candidate}, []float64{1.25})
				if math.Float64bits(one[0]) != math.Float64bits(geluGradF64Oracle(candidate, 1.25)) {
					x[0], g[0], witness = candidate, 1.25, true
				}
			}
			if !witness {
				t.Fatal("no backward scalar-reference bit witness found")
			}
			want := make([]float64, n)
			vgeluGradF64(want, x, g)
			xBase, gBase := tensor.New(tensor.F64, tensor.Shape{n + 2}), tensor.New(tensor.F64, tensor.Shape{n + 2})
			copy(xBase.Storage().F64()[1:], x)
			copy(gBase.Storage().F64()[1:], g)
			xView, err := xBase.Slice(0, 1, n+1)
			if err != nil {
				t.Fatal(err)
			}
			gView, err := gBase.Slice(0, 1, n+1)
			if err != nil {
				t.Fatal(err)
			}
			xBefore, gBefore := slices.Clone(xBase.Storage().F64()), slices.Clone(gBase.Storage().F64())
			recorder := &geluF64NeonRecorder{}
			got, err := backend.Execute(backend.NewContext().WithBackend(be).WithRecorder(recorder), backend.OpGELUBackward, []*tensor.Tensor{xView, gView}, nil)
			if err != nil {
				t.Fatal(err)
			}
			if recorder.calls != 1 {
				t.Fatalf("recorder calls=%d, want 1", recorder.calls)
			}
			if got[0].Dtype() != tensor.F64 || !got[0].Shape().Equal(xView.Shape()) || !got[0].IsContiguous() || got[0].Offset() != 0 {
				t.Fatal("bad backward output metadata")
			}
			if got[0].Storage() == xBase.Storage() || got[0].Storage() == gBase.Storage() {
				t.Fatal("backward output aliases input")
			}
			if !equalF64Bits(xBase.Storage().F64(), xBefore) || !equalF64Bits(gBase.Storage().F64(), gBefore) {
				t.Fatal("backward Execute changed input storage")
			}
			for i := range want {
				assertGELUF64Value(t, got[0].Storage().F64()[i], want[i])
			}
		})
	}
}
