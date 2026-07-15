package cpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// Last-percentile grind benchmarks (§C3 measure-first): realistic model shapes —
// LLM hidden width 2048, vocab-logit width 32000 — for the elementwise/softmax/
// bias kernels. Complements bench_test.go (tiny/medium shapes).

// --- AddBias: out[i,j] = x[i,j] + b[j] ---

func BenchmarkAddBiasF32_128x2048_cpu(b *testing.B) {
	benchOn(b, backend.CPU, backend.OpAddBias,
		bench.RandF32(tensor.Shape{128, 2048}, 1), bench.RandF32(tensor.Shape{2048}, 2))
}
func BenchmarkAddBiasF64_128x2048_cpu(b *testing.B) {
	benchOn(b, backend.CPU, backend.OpAddBias,
		bench.RandF64(tensor.Shape{128, 2048}, 1), bench.RandF64(tensor.Shape{2048}, 2))
}
func BenchmarkAddBiasF32_8x32000_cpu(b *testing.B) {
	benchOn(b, backend.CPU, backend.OpAddBias,
		bench.RandF32(tensor.Shape{8, 32000}, 1), bench.RandF32(tensor.Shape{32000}, 2))
}
func BenchmarkAddBiasF32_1x2048_cpu(b *testing.B) { // decode-step shape (serial path)
	benchOn(b, backend.CPU, backend.OpAddBias,
		bench.RandF32(tensor.Shape{1, 2048}, 1), bench.RandF32(tensor.Shape{2048}, 2))
}

// --- Softmax over last axis ---

func BenchmarkSoftmaxF32_32x2048_cpu(b *testing.B) { // attention-row scale
	benchOn(b, backend.CPU, backend.OpSoftmax, bench.RandF32(tensor.Shape{32, 2048}, 1))
}
func BenchmarkSoftmaxF64_32x2048_cpu(b *testing.B) {
	benchOn(b, backend.CPU, backend.OpSoftmax, bench.RandF64(tensor.Shape{32, 2048}, 1))
}
func BenchmarkSoftmaxF32_1x32000_cpu(b *testing.B) { // LLM logits, single row
	benchOn(b, backend.CPU, backend.OpSoftmax, bench.RandF32(tensor.Shape{1, 32000}, 1))
}
func BenchmarkSoftmaxF32_4x32000_cpu(b *testing.B) { // batched logits
	benchOn(b, backend.CPU, backend.OpSoftmax, bench.RandF32(tensor.Shape{4, 32000}, 1))
}

// --- Unary activations (closure-dispatch candidates) ---

func BenchmarkSigmoidF32_64K_cpu(b *testing.B) {
	benchOn(b, backend.CPU, backend.OpSigmoid, bench.RandF32(tensor.Shape{65536}, 3))
}
func BenchmarkTanhF32_64K_cpu(b *testing.B) {
	benchOn(b, backend.CPU, backend.OpTanh, bench.RandF32(tensor.Shape{65536}, 3))
}
func BenchmarkExpF32_64K_cpu(b *testing.B) {
	benchOn(b, backend.CPU, backend.OpExp, bench.RandF32(tensor.Shape{65536}, 3))
}
func BenchmarkLogF32_64K_cpu(b *testing.B) {
	benchOn(b, backend.CPU, backend.OpLog, bench.RandF32(tensor.Shape{65536}, 4))
}
func BenchmarkSigmoidF32_2048_cpu(b *testing.B) { // small/serial path
	benchOn(b, backend.CPU, backend.OpSigmoid, bench.RandF32(tensor.Shape{2048}, 3))
}
func BenchmarkGELUF32_256x2048_cpu(b *testing.B) { // GPT FFN hidden (256 tok × 4·512)
	benchOn(b, backend.CPU, backend.OpGELU, bench.RandF32(tensor.Shape{256, 2048}, 3))
}
func BenchmarkGELUF64_256x2048_cpu(b *testing.B) {
	benchOn(b, backend.CPU, backend.OpGELU, bench.RandF64(tensor.Shape{256, 2048}, 3))
}
func BenchmarkGELUF32_1x2048_cpu(b *testing.B) { // decode-step shape (serial path)
	benchOn(b, backend.CPU, backend.OpGELU, bench.RandF32(tensor.Shape{1, 2048}, 3))
}

// --- GELU backward (§T664 §V22 A/B): FFN training shape. On the arm64 perf
// build _cpu runs the NEON vgeluGrad kernel; _ref is the scalar-f64 baseline
// (and on every other build _cpu falls back to it — identical numbers). ---

func BenchmarkGELUBackwardF32_256x2048_cpu(b *testing.B) {
	benchOn(b, backend.CPU, backend.OpGELUBackward,
		bench.RandF32(tensor.Shape{256, 2048}, 3), bench.RandF32(tensor.Shape{256, 2048}, 4))
}
func BenchmarkGELUBackwardF32_256x2048_ref(b *testing.B) {
	benchOn(b, backend.Ref, backend.OpGELUBackward,
		bench.RandF32(tensor.Shape{256, 2048}, 3), bench.RandF32(tensor.Shape{256, 2048}, 4))
}

// --- SiLU forward + backward (§T665 §V22 A/B): SwiGLU FFN shapes —
// [256,1365] is the Llama Dim512 hidden (≈ 8·512/3, non-multiple-of-4 width
// exercises the NEON tail), [256,2048] the round training shape. On the arm64
// perf build _cpu runs the NEON vsilu/vsiluGrad kernels; the backward _ref
// lane is the serial scalar baseline the cpu dispatch previously fell back to
// (and still does on every other build — identical numbers there). ---

func BenchmarkSiLUF32_256x1365_cpu(b *testing.B) {
	benchOn(b, backend.CPU, backend.OpSiLU, bench.RandF32(tensor.Shape{256, 1365}, 3))
}
func BenchmarkSiLUF32_256x2048_cpu(b *testing.B) {
	benchOn(b, backend.CPU, backend.OpSiLU, bench.RandF32(tensor.Shape{256, 2048}, 3))
}
func BenchmarkSiLUF32_1x2048_cpu(b *testing.B) { // decode-step shape (serial path)
	benchOn(b, backend.CPU, backend.OpSiLU, bench.RandF32(tensor.Shape{1, 2048}, 3))
}
func BenchmarkSiLUBackwardF32_256x1365_cpu(b *testing.B) {
	benchOn(b, backend.CPU, backend.OpSiLUBackward,
		bench.RandF32(tensor.Shape{256, 1365}, 3), bench.RandF32(tensor.Shape{256, 1365}, 4))
}
func BenchmarkSiLUBackwardF32_256x1365_ref(b *testing.B) {
	benchOn(b, backend.Ref, backend.OpSiLUBackward,
		bench.RandF32(tensor.Shape{256, 1365}, 3), bench.RandF32(tensor.Shape{256, 1365}, 4))
}
func BenchmarkSiLUBackwardF32_256x2048_cpu(b *testing.B) {
	benchOn(b, backend.CPU, backend.OpSiLUBackward,
		bench.RandF32(tensor.Shape{256, 2048}, 3), bench.RandF32(tensor.Shape{256, 2048}, 4))
}
func BenchmarkSiLUBackwardF32_256x2048_ref(b *testing.B) {
	benchOn(b, backend.Ref, backend.OpSiLUBackward,
		bench.RandF32(tensor.Shape{256, 2048}, 3), bench.RandF32(tensor.Shape{256, 2048}, 4))
}

// --- Standalone unary transcendentals (§T666 §V22 A/B): the OpExp / OpTanh /
// OpLog F32 kernels at the training shape [256,2048] (complementing the 64K
// lanes above). On the arm64 perf build _cpu runs the NEON vexp-leaf kernels;
// on every other build these are the scalar f64 math.Exp/Tanh/Log paths.
// Log gets a strictly positive input (log-uniform over ~[0.6, 1.6]) so the
// baseline measures the real math.Log path, not its NaN early-out. ---

func posRandF32(shape tensor.Shape, seed uint64) *tensor.Tensor {
	t := bench.RandF32(shape, seed) // values in [-1,1)
	d := t.Storage().F32()
	for i, v := range d {
		d[i] = float32(math.Exp(float64(v) * 0.5)) // (0.6, 1.65)
	}
	return t
}

func BenchmarkExpF32_256x2048_cpu(b *testing.B) {
	benchOn(b, backend.CPU, backend.OpExp, bench.RandF32(tensor.Shape{256, 2048}, 3))
}
func BenchmarkTanhF32_256x2048_cpu(b *testing.B) {
	benchOn(b, backend.CPU, backend.OpTanh, bench.RandF32(tensor.Shape{256, 2048}, 3))
}
func BenchmarkLogF32_256x2048_cpu(b *testing.B) {
	benchOn(b, backend.CPU, backend.OpLog, posRandF32(tensor.Shape{256, 2048}, 4))
}
func BenchmarkLogPosF32_64K_cpu(b *testing.B) { // positive-input twin of the 64K lane
	benchOn(b, backend.CPU, backend.OpLog, posRandF32(tensor.Shape{65536}, 4))
}

// --- Broadcast binary ops (materialization candidates) ---

func BenchmarkAddBcastRowF32_128x2048_cpu(b *testing.B) { // [M,N] + [N]
	benchOn(b, backend.CPU, backend.OpAdd,
		bench.RandF32(tensor.Shape{128, 2048}, 1), bench.RandF32(tensor.Shape{2048}, 2))
}
func BenchmarkMulBcastScalarF32_128x2048_cpu(b *testing.B) { // [M,N] * [1]
	benchOn(b, backend.CPU, backend.OpMul,
		bench.RandF32(tensor.Shape{128, 2048}, 1), bench.RandF32(tensor.Shape{1}, 2))
}
func BenchmarkAddBcastColF32_128x2048_cpu(b *testing.B) { // [M,N] + [M,1]
	benchOn(b, backend.CPU, backend.OpAdd,
		bench.RandF32(tensor.Shape{128, 2048}, 1), bench.RandF32(tensor.Shape{128, 1}, 2))
}

// --- Same-shape binary at model scale ---

func BenchmarkAddF32_128x2048_cpu(b *testing.B) {
	benchOn(b, backend.CPU, backend.OpAdd,
		bench.RandF32(tensor.Shape{128, 2048}, 1), bench.RandF32(tensor.Shape{128, 2048}, 2))
}
func BenchmarkMulF32_1x2048_cpu(b *testing.B) { // decode-step elementwise (serial)
	benchOn(b, backend.CPU, backend.OpMul,
		bench.RandF32(tensor.Shape{1, 2048}, 1), bench.RandF32(tensor.Shape{1, 2048}, 2))
}
