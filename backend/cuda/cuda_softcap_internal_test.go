//go:build cuda && cgo && linux

package cuda

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

func hostSoftcapSoftmax(x []float32, rows, cols, seqQ, offset int, scale, cap float64) []float32 {
	out := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		lim := (r % seqQ) + offset
		if lim >= cols {
			lim = cols - 1
		}
		m := math.Inf(-1)
		for j := 0; j <= lim; j++ {
			v := cap * math.Tanh(float64(x[r*cols+j])*scale/cap)
			if v > m {
				m = v
			}
		}
		var sum float64
		e := make([]float64, lim+1)
		for j := 0; j <= lim; j++ {
			v := cap * math.Tanh(float64(x[r*cols+j])*scale/cap)
			e[j] = math.Exp(v - m)
			sum += e[j]
		}
		for j := 0; j <= lim; j++ {
			out[r*cols+j] = float32(e[j] / sum)
		}
	}
	return out
}

func TestCUDAAttnSoftmaxCapParity(t *testing.T) {
	if !Available() {
		t.Skip("no gpu")
	}
	const heads, seqQ, cols = 4, 96, 96
	rows := heads * seqQ
	scale, cap := 1.0/math.Sqrt(64), 50.0
	rng := rand.New(rand.NewSource(5))
	x := make([]float32, rows*cols)
	for i := range x {
		x[i] = float32(rng.NormFloat64() * 30)
	}
	want := hostSoftcapSoftmax(x, rows, cols, seqQ, 0, scale, cap)
	d, err := UploadF32(tensor.FromFloat32(tensor.Shape{rows, cols}, x))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Free()
	if err := d.attnSoftmaxCap(float32(scale), 0, seqQ, float32(cap)); err != nil {
		t.Fatal(err)
	}
	got, err := d.ToHost()
	if err != nil {
		t.Fatal(err)
	}
	var maxAbs float64
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			if diff := math.Abs(got.AtF64(r, c) - float64(want[r*cols+c])); diff > maxAbs {
				maxAbs = diff
			}
		}
	}
	t.Logf("softcap-softmax max abs diff vs host ref: %.3g", maxAbs)
	if maxAbs > 1e-5 {
		t.Fatalf("softcap softmax diverges: max abs %.3g > 1e-5", maxAbs)
	}
}

func benchAttnSoftmaxCap(b *testing.B, heads, seq, cols int) {
	if !Available() {
		b.Skip("no gpu")
	}
	rows := heads * seq
	scale, cap := float32(1.0/math.Sqrt(64)), float32(50)
	rng := rand.New(rand.NewSource(5))
	x := make([]float32, rows*cols)
	for i := range x {
		x[i] = float32(rng.NormFloat64() * 30)
	}
	d, err := UploadF32(tensor.FromFloat32(tensor.Shape{rows, cols}, x))
	if err != nil {
		b.Fatal(err)
	}
	defer d.Free()
	_ = d.attnSoftmaxCap(scale, 0, seq, cap)
	GraphSync()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = d.attnSoftmaxCap(scale, 0, seq, cap)
	}
	GraphSync()
	b.StopTimer()
}

func BenchmarkAttnSoftmaxCap_8x1024(b *testing.B)  { benchAttnSoftmaxCap(b, 8, 1024, 1024) }
func BenchmarkAttnSoftmaxCap_16x2048(b *testing.B) { benchAttnSoftmaxCap(b, 16, 2048, 2048) }
