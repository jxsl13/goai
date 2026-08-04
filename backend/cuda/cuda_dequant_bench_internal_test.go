//go:build cuda && cgo && linux

package cuda

import "testing"

// benchDequantQ4K isolates the Q4_K→f16 dequant kernel (cu_dequant_q4k_to_f16) — the fixed cost of
// the WMMA prefill path — to check it against the ~200µs roofline (32MB f16 write + 9.4MB Q4_K read
// @360GB/s). If it's several× slower, the strided B[k*N+n] write (warp-per-column) is uncoalesced.
func benchDequantQ4K(b *testing.B, k, n int) {
	if !Available() {
		b.Skip("no gpu")
	}
	raw := make([]byte, (k*n/256)*144)
	rq, err := NewResidentBQ4KFromBlocks(raw, k, n)
	if err != nil {
		b.Fatal(err)
	}
	bf16 := allocU16ForBench(k * n)
	defer func() { rq.Free(); freeF32ForBench(bf16) }()
	dequantQ4KForBench(rq, bf16, k, n)
	GraphSync()
	gb := (float64(k*n*2) + float64((k*n/256)*144)) / 1e9
	b.ResetTimer()
	for range b.N {
		dequantQ4KForBench(rq, bf16, k, n)
	}
	GraphSync()
	b.StopTimer()
	b.ReportMetric(gb/(b.Elapsed().Seconds()/float64(b.N)), "GB/s")
}

func BenchmarkDequantQ4K_4096x4096(b *testing.B) { benchDequantQ4K(b, 4096, 4096) }

func benchDequantQ6K(b *testing.B, k, n int) {
	if !Available() {
		b.Skip("no gpu")
	}
	raw := make([]byte, (k*n/256)*210)
	rq, err := NewResidentBQ6KFromBlocks(raw, k, n)
	if err != nil {
		b.Fatal(err)
	}
	bf16 := allocU16ForBench(k * n)
	defer func() { rq.Free(); freeF32ForBench(bf16) }()
	dequantQ6KForBench(rq, bf16, k, n)
	GraphSync()
	gb := (float64(k*n*2) + float64((k*n/256)*210)) / 1e9
	b.ResetTimer()
	for range b.N {
		dequantQ6KForBench(rq, bf16, k, n)
	}
	GraphSync()
	b.StopTimer()
	b.ReportMetric(gb/(b.Elapsed().Seconds()/float64(b.N)), "GB/s")
}

func BenchmarkDequantQ6K_4096x4096(b *testing.B) { benchDequantQ6K(b, 4096, 4096) }

func benchDequantQ5K(b *testing.B, k, n int) {
	if !Available() {
		b.Skip("no gpu")
	}
	raw := make([]byte, (k*n/256)*176)
	rq, err := NewResidentBQ5KFromBlocks(raw, k, n)
	if err != nil {
		b.Fatal(err)
	}
	bf16 := allocU16ForBench(k * n)
	defer func() { rq.Free(); freeF32ForBench(bf16) }()
	dequantQ5KForBench(rq, bf16, k, n)
	GraphSync()
	gb := (float64(k*n*2) + float64((k*n/256)*176)) / 1e9
	b.ResetTimer()
	for range b.N {
		dequantQ5KForBench(rq, bf16, k, n)
	}
	GraphSync()
	b.StopTimer()
	b.ReportMetric(gb/(b.Elapsed().Seconds()/float64(b.N)), "GB/s")
}

func BenchmarkDequantQ5K_4096x4096(b *testing.B) { benchDequantQ5K(b, 4096, 4096) }
