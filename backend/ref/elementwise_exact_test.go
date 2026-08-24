package ref

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
)

// These scalar functions and loops freeze the pre-devirtualization reference
// semantics. The optimized kernels must preserve every output bit, including
// F32's widen/compute/narrow sequence and IEEE-754 signed-zero/NaN behavior.
var exactUnaryOps = []struct {
	name string
	op   backend.Op
	fn   func(float64) float64
}{
	{"stop-gradient", backend.OpStopGradient, func(x float64) float64 { return x }},
	{"neg", backend.OpNeg, func(x float64) float64 { return -x }},
	{"exp", backend.OpExp, math.Exp},
	{"log", backend.OpLog, math.Log},
	{"tanh", backend.OpTanh, math.Tanh},
	{"relu", backend.OpReLU, relu},
	{"gelu", backend.OpGELU, gelu},
	{"sigmoid", backend.OpSigmoid, sigmoid},
	{"silu", backend.OpSiLU, func(x float64) float64 { return x * sigmoid(x) }},
	{"sqrt", backend.OpSqrt, math.Sqrt},
	{"abs", backend.OpAbs, math.Abs},
}

var exactBinaryOps = []struct {
	name string
	op   backend.Op
	fn   func(float64, float64) float64
}{
	{"add", backend.OpAdd, func(a, b float64) float64 { return a + b }},
	{"sub", backend.OpSub, func(a, b float64) float64 { return a - b }},
	{"mul", backend.OpMul, func(a, b float64) float64 { return a * b }},
	{"div", backend.OpDiv, func(a, b float64) float64 { return a / b }},
	{"maximum", backend.OpMaximum, math.Max},
	{"minimum", backend.OpMinimum, math.Min},
}

func TestElementwiseUnaryBitExact(t *testing.T) {
	inputs := []float64{
		math.Float64frombits(0x7ff8000000000042),
		math.Inf(-1), -4, math.Copysign(0, -1), 0, 0.25, 4, math.Inf(1),
	}
	for _, tc := range exactUnaryOps {
		t.Run(tc.name+"/F64", func(t *testing.T) {
			got := execF64(t, tc.op, inputs)
			for i, x := range inputs {
				want := tc.fn(x)
				if math.Float64bits(got[i]) != math.Float64bits(want) {
					t.Fatalf("index %d: got %016x want %016x", i, math.Float64bits(got[i]), math.Float64bits(want))
				}
			}
		})
		t.Run(tc.name+"/F32", func(t *testing.T) {
			got := execF32(t, tc.op, inputs)
			for i, x := range inputs {
				want := float32(tc.fn(float64(float32(x))))
				if math.Float32bits(float32(got[i])) != math.Float32bits(want) {
					t.Fatalf("index %d: got %08x want %08x", i, math.Float32bits(float32(got[i])), math.Float32bits(want))
				}
			}
		})
	}
}

func TestElementwiseBinaryBitExact(t *testing.T) {
	a := []float64{
		math.Float64frombits(0x7ff8000000000042),
		math.Inf(-1), -4, math.Copysign(0, -1), 0, 0.25, 4, math.Inf(1),
	}
	b := []float64{
		1, math.Inf(1), -2, 0, math.Copysign(0, -1), -0.5,
		math.Float64frombits(0x7ff8000000000084), math.Inf(-1),
	}
	for _, tc := range exactBinaryOps {
		t.Run(tc.name+"/F64", func(t *testing.T) {
			got := execF64(t, tc.op, a, b)
			for i := range a {
				want := tc.fn(a[i], b[i])
				if math.Float64bits(got[i]) != math.Float64bits(want) {
					t.Fatalf("index %d: got %016x want %016x", i, math.Float64bits(got[i]), math.Float64bits(want))
				}
			}
		})
		t.Run(tc.name+"/F32", func(t *testing.T) {
			got := execF32(t, tc.op, a, b)
			for i := range a {
				want := float32(tc.fn(float64(float32(a[i])), float64(float32(b[i]))))
				if math.Float32bits(float32(got[i])) != math.Float32bits(want) {
					t.Fatalf("index %d: got %08x want %08x", i, math.Float32bits(float32(got[i])), math.Float32bits(want))
				}
			}
		})
	}
}
