package vision

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestSwinAttendUnitsParallelBitIdentical is the §V22 gate for fanning out the window
// attention. It compares the parallel arm against the serial arm of the SAME code — the unit
// body is one closure used by both — so the only difference under test is how units are
// grouped across workers.
//
// Tolerance is ZERO. No accumulation crosses a unit, so chunking cannot legitimately move a
// bit; a relative-tolerance check would pass a genuine slot-indexing error, which is exactly
// the mistake this shape invites.
//
// Both dtypes are covered because the fused score-term path is dtype-specialized, and both a
// shifted and an unshifted stage are exercised by the depths below.
func TestSwinAttendUnitsParallelBitIdentical(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		run := func(threshold int) []float64 {
			old := swinAttendParThreshold
			swinAttendParThreshold = threshold
			defer func() { swinAttendParThreshold = old }()

			m, err := NewSwin(dt, 32, 4, 4, 96, []int{2, 2}, []int{3, 6}, 10, 7,
				WithSwinChannels(3), WithSwinRelativeBias(true))
			if err != nil {
				t.Fatal(err)
			}
			x := tensor.New(dt, tensor.Shape{4, 3, 32, 32})
			switch dt {
			case tensor.F32:
				s := x.Storage().F32()
				for i := range s {
					s[i] = float32(math.Sin(float64(i)*0.013)) * 2
				}
			default:
				s := x.Storage().F64()
				for i := range s {
					s[i] = math.Sin(float64(i)*0.013) * 2
				}
			}
			out, err := m.Forward(backend.NewContext(), x)
			if err != nil {
				t.Fatal(err)
			}
			c := out.Contiguous()
			if c.Dtype() == tensor.F32 {
				f := c.Storage().F32()
				g := make([]float64, len(f))
				for i, v := range f {
					g[i] = float64(v)
				}
				return g
			}
			return c.Storage().F64()
		}
		par := run(8)       // units are 96 and 48 → both stages fan out
		ser := run(1 << 30) // never fans out
		if len(par) != len(ser) {
			t.Fatalf("%v: %d outputs parallel, %d serial", dt, len(par), len(ser))
		}
		for i := range ser {
			if math.Float64bits(par[i]) != math.Float64bits(ser[i]) {
				t.Fatalf("%v: logit %d parallel %v (%016x), serial %v (%016x)",
					dt, i, par[i], math.Float64bits(par[i]), ser[i], math.Float64bits(ser[i]))
			}
		}
	}
}

// TestSwinAttendUnitsRunToRunStable catches a cross-unit dependency that would surface as
// run-to-run variation rather than as a wrong answer on any single run — parallel.Rows does
// not guarantee which worker takes which chunk.
func TestSwinAttendUnitsRunToRunStable(t *testing.T) {
	run := func() []float32 {
		m, err := NewSwin(tensor.F32, 32, 4, 4, 96, []int{2, 2}, []int{3, 6}, 10, 3,
			WithSwinChannels(3), WithSwinRelativeBias(true))
		if err != nil {
			t.Fatal(err)
		}
		x := tensor.New(tensor.F32, tensor.Shape{8, 3, 32, 32})
		s := x.Storage().F32()
		for i := range s {
			s[i] = float32(math.Cos(float64(i) * 0.017))
		}
		out, err := m.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		return out.Contiguous().Storage().F32()
	}
	first := run()
	for r := 1; r < 6; r++ {
		got := run()
		for i := range first {
			if math.Float32bits(got[i]) != math.Float32bits(first[i]) {
				t.Fatalf("run %d logit %d: %08x != %08x — chunk assignment changed a result",
					r, i, math.Float32bits(got[i]), math.Float32bits(first[i]))
			}
		}
	}
}

// TestSwinBlockwiseMatchesDispatch is the correctness ORACLE for the blockwise window
// attention, and it closes a gap that predates the fan-out.
//
// The blockwise path — reusable fill buffers, fused score terms, direct slot placement into
// one output buffer — is an independent reimplementation of the dispatch path's
// slice/transpose/matmul/add/softmax/concat chain. Nothing compared the two. The batched-vs-
// per-image parity test cannot: both arms take the blockwise path, so a wrong slot index or a
// wrong (window, head) decode moves both identically and compares equal. Confirmed by
// mutation: writing each head into its neighbour's slot, and swapping the divisor and modulus
// in the unit decode, BOTH leave the entire Swin suite green without this test.
//
// The oracle is the tape. blockwise requires ctx.Recorder == nil, so attaching a recorder
// selects the dispatch chain while computing the same forward values. The fused arm is
// documented to round exactly where the op chain rounds — scale, then each add — so the two
// must agree bit-for-bit, and a tolerance here would defeat the purpose.
func TestSwinBlockwiseMatchesDispatch(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		mk := func() (*SwinTransformer, *tensor.Tensor) {
			m, err := NewSwin(dt, 32, 4, 4, 96, []int{2, 2}, []int{3, 6}, 10, 5,
				WithSwinChannels(3), WithSwinRelativeBias(true))
			if err != nil {
				t.Fatal(err)
			}
			x := tensor.New(dt, tensor.Shape{2, 3, 32, 32})
			switch dt {
			case tensor.F32:
				s := x.Storage().F32()
				for i := range s {
					s[i] = float32(math.Sin(float64(i)*0.021)) * 1.5
				}
			default:
				s := x.Storage().F64()
				for i := range s {
					s[i] = math.Sin(float64(i)*0.021) * 1.5
				}
			}
			return m, x
		}
		flat := func(o *tensor.Tensor) []float64 {
			c := o.Contiguous()
			if c.Dtype() == tensor.F32 {
				f := c.Storage().F32()
				g := make([]float64, len(f))
				for i, v := range f {
					g[i] = float64(v)
				}
				return g
			}
			return c.Storage().F64()
		}

		mBlock, xBlock := mk()
		outBlock, err := mBlock.Forward(backend.NewContext(), xBlock)
		if err != nil {
			t.Fatal(err)
		}
		mDisp, xDisp := mk()
		tape := autograd.NewTape()
		outDisp, err := mDisp.Forward(tape.Context(), xDisp)
		if err != nil {
			t.Fatal(err)
		}

		bw, dp := flat(outBlock), flat(outDisp)
		if len(bw) != len(dp) {
			t.Fatalf("%v: %d blockwise outputs, %d dispatch", dt, len(bw), len(dp))
		}
		for i := range dp {
			if math.Float64bits(bw[i]) != math.Float64bits(dp[i]) {
				t.Fatalf("%v: logit %d blockwise %v (%016x), dispatch %v (%016x)",
					dt, i, bw[i], math.Float64bits(bw[i]), dp[i], math.Float64bits(dp[i]))
			}
		}
	}
}
