//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// binOps binds a built resident weight to its plain (beta=0) and residual-fused (beta=1) GEMMs,
// so the parity/bench harness is quant-agnostic — each K-quant supplies these two closures.
type binOps struct {
	plain func(rec *cuda.Recorder, x, o *cuda.DeviceF32, m int) error
	acc   func(rec *cuda.Recorder, x, dst *cuda.DeviceF32, m int) error
	free  func()
}

func buildQ4K(tb testing.TB, k, n int) binOps {
	blocks, err := gguf.Quantize(transpose2D(bench.RandF32(tensor.Shape{k, n}, 2)), gguf.Q4_K)
	if err != nil {
		tb.Fatal(err)
	}
	w, err := cuda.NewResidentBQ4KFromBlocks(blocks, k, n)
	if err != nil {
		tb.Fatal(err)
	}
	return binOps{
		func(rec *cuda.Recorder, x, o *cuda.DeviceF32, m int) error { return rec.QMatMulResidentQ4K(x, w, o, m) },
		func(rec *cuda.Recorder, x, dst *cuda.DeviceF32, m int) error {
			return rec.QMatMulResidentAccQ4K(x, w, dst, m)
		},
		w.Free,
	}
}

func buildQ3K(tb testing.TB, k, n int) binOps {
	blocks, err := gguf.Quantize(transpose2D(bench.RandF32(tensor.Shape{k, n}, 2)), gguf.Q3_K)
	if err != nil {
		tb.Fatal(err)
	}
	w, err := cuda.NewResidentBQ3KFromBlocks(blocks, k, n)
	if err != nil {
		tb.Fatal(err)
	}
	return binOps{
		func(rec *cuda.Recorder, x, o *cuda.DeviceF32, m int) error { return rec.QMatMulResidentQ3K(x, w, o, m) },
		func(rec *cuda.Recorder, x, dst *cuda.DeviceF32, m int) error {
			return rec.QMatMulResidentAccQ3K(x, w, dst, m)
		},
		w.Free,
	}
}

func buildQ5K(tb testing.TB, k, n int) binOps {
	blocks, err := gguf.Quantize(transpose2D(bench.RandF32(tensor.Shape{k, n}, 2)), gguf.Q5_K)
	if err != nil {
		tb.Fatal(err)
	}
	w, err := cuda.NewResidentBQ5KFromBlocks(blocks, k, n)
	if err != nil {
		tb.Fatal(err)
	}
	return binOps{
		func(rec *cuda.Recorder, x, o *cuda.DeviceF32, m int) error { return rec.QMatMulResidentQ5K(x, w, o, m) },
		func(rec *cuda.Recorder, x, dst *cuda.DeviceF32, m int) error {
			return rec.QMatMulResidentAccQ5K(x, w, dst, m)
		},
		w.Free,
	}
}

func buildQ6K(tb testing.TB, k, n int) binOps {
	blocks, err := gguf.Quantize(transpose2D(bench.RandF32(tensor.Shape{k, n}, 2)), gguf.Q6_K)
	if err != nil {
		tb.Fatal(err)
	}
	w, err := cuda.NewResidentBQ6KFromBlocks(blocks, k, n)
	if err != nil {
		tb.Fatal(err)
	}
	return binOps{
		func(rec *cuda.Recorder, x, o *cuda.DeviceF32, m int) error { return rec.QMatMulResidentQ6K(x, w, o, m) },
		func(rec *cuda.Recorder, x, dst *cuda.DeviceF32, m int) error {
			return rec.QMatMulResidentAccQ6K(x, w, dst, m)
		},
		w.Free,
	}
}

func buildQ2K(tb testing.TB, k, n int) binOps {
	blocks, err := gguf.Quantize(transpose2D(bench.RandF32(tensor.Shape{k, n}, 2)), gguf.Q2_K)
	if err != nil {
		tb.Fatal(err)
	}
	w, err := cuda.NewResidentBQ2KFromBlocks(blocks, k, n)
	if err != nil {
		tb.Fatal(err)
	}
	return binOps{
		func(rec *cuda.Recorder, x, o *cuda.DeviceF32, m int) error { return rec.QMatMulResidentQ2K(x, w, o, m) },
		func(rec *cuda.Recorder, x, dst *cuda.DeviceF32, m int) error {
			return rec.QMatMulResidentAccQ2K(x, w, dst, m)
		},
		w.Free,
	}
}

var kquantBuilders = []struct {
	name  string
	build func(tb testing.TB, k, n int) binOps
}{
	{"Q4K", buildQ4K},
	{"Q3K", buildQ3K},
	{"Q5K", buildQ5K},
	{"Q6K", buildQ6K},
	{"Q2K", buildQ2K},
}

// TestCUDAKQuantAccMTMatchesGEMV: the residual-fused Acc paths now route m>=thresh to the MT (beta=1)
// weight-read-once kernel. dst += x·dequant(w) via MT must equal dst_init + (x·dequant(w) via the plain
// path) bit-for-bit — beta=1 folds the residual add (1.0*dst_init is exact) and the MT is bit-identical
// to the GEMV per row. m=4 hits the MT for all five K-quants (Q4K thr=4, others thr=2).
func TestCUDAKQuantAccMTMatchesGEMV(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const m, k, n = 4, 2048, 5632 // deep up/down-proj shape
	rec, err := cuda.NewRecorder()
	if err != nil {
		t.Fatal(err)
	}
	defer rec.Free()
	for _, bq := range kquantBuilders {
		t.Run(bq.name, func(t *testing.T) {
			ops := bq.build(t, k, n)
			defer ops.free()
			x, _ := cuda.NewDeviceF32(m, k)
			x.UploadF32(bench.RandF32(tensor.Shape{m, k}, 3).Storage().F32())
			defer x.Free()

			plain, _ := cuda.NewDeviceF32(m, n)
			defer plain.Free()
			plain.UploadF32(make([]float32, m*n))
			if err := ops.plain(rec, x, plain, m); err != nil {
				t.Fatalf("plain: %v", err)
			}
			dstInit := bench.RandF32(tensor.Shape{m, n}, 9).Storage().F32()
			acc, _ := cuda.NewDeviceF32(m, n)
			defer acc.Free()
			acc.UploadF32(dstInit)
			if err := ops.acc(rec, x, acc, m); err != nil {
				t.Fatalf("acc: %v", err)
			}
			rec.Wait()

			pv := make([]float32, m*n)
			plain.DownloadF32(pv)
			av := make([]float32, m*n)
			acc.DownloadF32(av)
			for i := range av {
				if want := dstInit[i] + pv[i]; av[i] != want {
					t.Fatalf("%s Acc-via-MT[%d]: got %v want %v (dstInit+plain)", bq.name, i, av[i], want)
				}
			}
		})
	}
}

func benchKQuantAcc(b *testing.B, build func(tb testing.TB, k, n int) binOps, m, k, n int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	rec, err := cuda.NewRecorder()
	if err != nil {
		b.Fatal(err)
	}
	defer rec.Free()
	x, _ := cuda.NewDeviceF32(m, k)
	dst, _ := cuda.NewDeviceF32(m, n)
	ops := build(b, k, n)
	defer func() { x.Free(); dst.Free(); ops.free() }()
	if err := ops.acc(rec, x, dst, m); err != nil {
		b.Skipf("acc: %v", err)
	}
	rec.Wait()
	b.ResetTimer()
	for range b.N {
		_ = ops.acc(rec, x, dst, m)
	}
	rec.Wait()
	b.StopTimer()
	b.ReportMetric(2*float64(m)*float64(k)*float64(n)*float64(b.N)/b.Elapsed().Seconds()/1e12, "TFLOP/s")
}

func BenchmarkQ4KAcc_4x2048x5632(b *testing.B) { benchKQuantAcc(b, buildQ4K, 4, 2048, 5632) }
func BenchmarkQ4KAcc_4x5632x2048(b *testing.B) { benchKQuantAcc(b, buildQ4K, 4, 5632, 2048) }
func BenchmarkQ4KAcc_4x2048x2048(b *testing.B) { benchKQuantAcc(b, buildQ4K, 4, 2048, 2048) }
func BenchmarkQ3KAcc_4x2048x5632(b *testing.B) { benchKQuantAcc(b, buildQ3K, 4, 2048, 5632) }
func BenchmarkQ3KAcc_4x5632x2048(b *testing.B) { benchKQuantAcc(b, buildQ3K, 4, 5632, 2048) }
func BenchmarkQ6KAcc_4x2048x5632(b *testing.B) { benchKQuantAcc(b, buildQ6K, 4, 2048, 5632) }
func BenchmarkQ6KAcc_4x5632x2048(b *testing.B) { benchKQuantAcc(b, buildQ6K, 4, 5632, 2048) }
func BenchmarkQ5KAcc_4x2048x5632(b *testing.B) { benchKQuantAcc(b, buildQ5K, 4, 2048, 5632) }
func BenchmarkQ2KAcc_4x2048x5632(b *testing.B) { benchKQuantAcc(b, buildQ2K, 4, 2048, 5632) }
