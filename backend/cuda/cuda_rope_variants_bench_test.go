//go:build cuda && cgo && linux

package cuda_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

func benchRoPEPartial(b *testing.B, seq, heads, hd, rotaryDim int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	x := bench.RandF32(tensor.Shape{seq, heads * hd}, 1)
	dx, err := cuda.UploadF32(x)
	if err != nil {
		b.Fatal(err)
	}
	defer dx.Free()
	attrs := backend.RoPEAttrs{Heads: heads, Base: 10000}
	if err := dx.RoPEPartial(attrs, rotaryDim); err != nil {
		b.Fatal(err)
	}
	cuda.GraphSync()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = dx.RoPEPartial(attrs, rotaryDim)
	}
	cuda.GraphSync()
	b.StopTimer()
	b.ReportMetric(2*4*float64(seq)*float64(heads)*float64(rotaryDim)/(b.Elapsed().Seconds()/float64(b.N))/1e9, "GB/s")
}

func benchRoPEBand(b *testing.B, seq, heads, hd int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	stride := heads * hd
	x := bench.RandF32(tensor.Shape{seq, stride}, 1)
	dx, err := cuda.UploadF32(x)
	if err != nil {
		b.Fatal(err)
	}
	defer dx.Free()
	inv, err := cuda.BuildRoPEInv(hd, 10000)
	if err != nil {
		b.Fatal(err)
	}
	defer inv.Free()
	if err := dx.RoPEAtBand(inv, 0, heads, hd, 0, 1, stride); err != nil {
		b.Fatal(err)
	}
	cuda.GraphSync()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = dx.RoPEAtBand(inv, 0, heads, hd, 0, 1, stride)
	}
	cuda.GraphSync()
	b.StopTimer()
	b.ReportMetric(2*4*float64(seq)*float64(heads)*float64(hd)/(b.Elapsed().Seconds()/float64(b.N))/1e9, "GB/s")
}

func BenchmarkRoPEPartial_2048x32x128r64(b *testing.B) { benchRoPEPartial(b, 2048, 32, 128, 64) }
func BenchmarkRoPEBand_2048x32x128(b *testing.B)       { benchRoPEBand(b, 2048, 32, 128) }

func benchRoPEDpos(b *testing.B, seq, heads, hd int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	x := bench.RandF32(tensor.Shape{seq, heads * hd}, 1)
	dx, err := cuda.UploadF32(x)
	if err != nil {
		b.Fatal(err)
	}
	defer dx.Free()
	pos, err := cuda.NewDevicePos()
	if err != nil {
		b.Fatal(err)
	}
	defer pos.Free()
	attrs := backend.RoPEAttrs{Heads: heads, Base: 10000}
	if err := dx.RoPEDpos(attrs, pos); err != nil {
		b.Fatal(err)
	}
	cuda.GraphSync()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = dx.RoPEDpos(attrs, pos)
	}
	cuda.GraphSync()
	b.StopTimer()
	b.ReportMetric(2*4*float64(seq)*float64(heads)*float64(hd)/(b.Elapsed().Seconds()/float64(b.N))/1e9, "GB/s")
}

func BenchmarkRoPEDpos_2048x32x128(b *testing.B) { benchRoPEDpos(b, 2048, 32, 128) }
