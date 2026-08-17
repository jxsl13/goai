//go:build arm64

package gguf

import (
	"math"
	"math/rand"
	"os"
	"runtime"
	"runtime/debug"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/jxsl13/goai/tensor"
)

func TestDequantQ6KNeonBitExact(t *testing.T) {
	const blocks = 19
	raw := make([]byte, blocks*q6kBlockSize)
	rng := rand.New(rand.NewSource(17))
	if _, err := rng.Read(raw); err != nil {
		t.Fatal(err)
	}
	// Exercise finite normal, subnormal, signed-zero, and maximum-finite scale
	// encodings. Quantized GGUF weights require finite scales; NaN arithmetic
	// payload propagation is not a model-format semantic.
	halves := [...]uint16{0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x7bff, 0xfbff}
	for block := range blocks {
		h := halves[block%len(halves)]
		raw[block*q6kBlockSize+208] = byte(h)
		raw[block*q6kBlockSize+209] = byte(h >> 8)
	}

	want := make([]float32, blocks*qkK)
	got := make([]float32, blocks*qkK)
	dequantQ6_KIntoScalar(want, raw)
	dequantQ6_KIntoArch(got, raw)
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("element %d: got %08x (%g), want %08x (%g)", i,
				math.Float32bits(got[i]), got[i], math.Float32bits(want[i]), want[i])
		}
	}
}

func TestDequantQ4KNeonBitExact(t *testing.T) {
	const blocks = 19
	raw := make([]byte, blocks*q4kBlockSize)
	rng := rand.New(rand.NewSource(19))
	if _, err := rng.Read(raw); err != nil {
		t.Fatal(err)
	}
	halves := [...]uint16{0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x7bff, 0xfbff}
	for block := range blocks {
		d := halves[block%len(halves)]
		dmin := halves[(block+3)%len(halves)]
		o := block * q4kBlockSize
		raw[o+0], raw[o+1] = byte(d), byte(d>>8)
		raw[o+2], raw[o+3] = byte(dmin), byte(dmin>>8)
	}

	want := make([]float32, blocks*qkK)
	got := make([]float32, blocks*qkK)
	dequantQ4_KIntoScalar(want, raw)
	dequantQ4_KIntoArch(got, raw)
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("element %d: got %08x (%g), want %08x (%g)", i,
				math.Float32bits(got[i]), got[i], math.Float32bits(want[i]), want[i])
		}
	}
}

func BenchmarkDequantQ6KIntoPaths(b *testing.B) {
	raw := make([]byte, (benchN/qkK)*q6kBlockSize)
	rng := rand.New(rand.NewSource(23))
	if _, err := rng.Read(raw); err != nil {
		b.Fatal(err)
	}
	// Keep scale halves finite and representative rather than allowing random
	// NaNs to turn the benchmark into a payload-propagation artifact.
	for block := 0; block < benchN/qkK; block++ {
		raw[block*q6kBlockSize+208] = 0x00
		raw[block*q6kBlockSize+209] = 0x3c // 1.0
	}
	dst := make([]float32, benchN)
	type path struct {
		name string
		fn   func([]float32, []byte)
	}
	paths := []path{{"scalar", dequantQ6_KIntoScalar}, {"neon", dequantQ6_KIntoArch}}
	if os.Getenv("GOAI_GGUF_NEON_FIRST") != "" {
		paths[0], paths[1] = paths[1], paths[0]
	}
	for _, path := range paths {
		b.Run(path.name, func(b *testing.B) {
			b.SetBytes(int64(len(raw)))
			b.ResetTimer()
			for b.Loop() {
				path.fn(dst, raw)
			}
		})
	}
}

func BenchmarkDequantQ4KIntoPaths(b *testing.B) {
	raw := make([]byte, (benchN/qkK)*q4kBlockSize)
	rng := rand.New(rand.NewSource(29))
	if _, err := rng.Read(raw); err != nil {
		b.Fatal(err)
	}
	for block := 0; block < benchN/qkK; block++ {
		o := block * q4kBlockSize
		raw[o+0], raw[o+1] = 0x00, 0x3c
		raw[o+2], raw[o+3] = 0x00, 0x38
	}
	dst := make([]float32, benchN)
	type path struct {
		name string
		fn   func([]float32, []byte)
	}
	paths := []path{{"scalar", dequantQ4_KIntoScalar}, {"neon", dequantQ4_KIntoArch}}
	if os.Getenv("GOAI_GGUF_NEON_FIRST") != "" {
		paths[0], paths[1] = paths[1], paths[0]
	}
	for _, path := range paths {
		b.Run(path.name, func(b *testing.B) {
			b.SetBytes(int64(len(raw)))
			b.ResetTimer()
			for b.Loop() {
				path.fn(dst, raw)
			}
		})
	}
}

type kDequantPath struct {
	name string
	q4   func([]float32, []byte)
	q6   func([]float32, []byte)
}

// decodeTensorWithKPath mirrors decodeTensor only for the two kernels under
// test. All unchanged formats continue through the production decoder, so the
// real-model A/B includes the complete eager materialization workload.
func decodeTensorWithKPath(ti tensorInfo, data []byte, path kDequantPath) (*tensor.Tensor, error) {
	if ti.ggType != tQ4_K && ti.ggType != tQ6_K {
		return decodeTensor(ti, data)
	}
	need, err := byteSize(ti.ggType, ti.shape.Numel())
	if err != nil || ti.offset > uint64(len(data)) || uint64(need) > uint64(len(data))-ti.offset {
		return decodeTensor(ti, data) // preserve the production error contract
	}
	raw := data[ti.offset : ti.offset+uint64(need)]
	out := tensor.New(tensor.F32, ti.shape)
	if ti.ggType == tQ4_K {
		path.q4(out.Storage().F32(), raw)
	} else {
		path.q6(out.Storage().F32(), raw)
	}
	return out, nil
}

// readParsedWithKPath preserves readParsed's bounded work-stealing schedule.
// It exists only to compare scalar and NEON K-quant kernels within one process;
// production dispatch remains direct and pays no function-variable overhead.
func readParsedWithKPath(p *parsed, path kDequantPath) (*File, error) {
	out := &File{Version: p.version, Metadata: p.meta, Tensors: make(map[string]*tensor.Tensor, len(p.infos))}
	decoded := make([]*tensor.Tensor, len(p.infos))
	errs := make([]error, len(p.infos))
	workers := min(runtime.GOMAXPROCS(0), len(p.infos))
	if workers > 1 {
		var next atomic.Int64
		var wg sync.WaitGroup
		for range workers {
			wg.Go(func() {
				for {
					i := int(next.Add(1)) - 1
					if i >= len(p.infos) {
						return
					}
					decoded[i], errs[i] = decodeTensorWithKPath(p.infos[i], p.data, path)
				}
			})
		}
		wg.Wait()
	} else {
		for i := range p.infos {
			decoded[i], errs[i] = decodeTensorWithKPath(p.infos[i], p.data, path)
		}
	}
	for i, ti := range p.infos {
		if errs[i] != nil {
			return nil, errs[i]
		}
		out.Tensors[ti.name] = decoded[i]
	}
	return out, nil
}

func medianDuration(samples []time.Duration) time.Duration {
	ordered := append([]time.Duration(nil), samples...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i] < ordered[j] })
	mid := len(ordered) / 2
	if len(ordered)%2 != 0 {
		return ordered[mid]
	}
	return (ordered[mid-1] + ordered[mid]) / 2
}

// TestReadParsedKPathAB is an opt-in end-to-end evidence harness. Every arm
// gets fresh destination pages; order reverses each round inside one process,
// meeting the interleaving rule for sub-10% model-load claims.
func TestReadParsedKPathAB(t *testing.T) {
	path := os.Getenv("TINYLLAMA_GGUF")
	if path == "" {
		t.Skip("set TINYLLAMA_GGUF to run the real-model K-quant A/B")
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	p, err := parse(f)
	closeErr := f.Close()
	if err != nil {
		t.Fatal(err)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}

	paths := []kDequantPath{
		{name: "scalar", q4: dequantQ4_KIntoScalar, q6: dequantQ6_KIntoScalar},
		{name: "neon", q4: dequantQ4_KIntoArch, q6: dequantQ6_KIntoArch},
	}
	samples := map[string][]time.Duration{"scalar": {}, "neon": {}}
	for round := range 10 {
		order := paths
		if round%2 != 0 {
			order = []kDequantPath{paths[1], paths[0]}
		}
		for _, candidate := range order {
			debug.FreeOSMemory()
			start := time.Now()
			out, err := readParsedWithKPath(p, candidate)
			elapsed := time.Since(start)
			if err != nil {
				t.Fatalf("round %d %s: %v", round+1, candidate.name, err)
			}
			if len(out.Tensors) != len(p.infos) {
				t.Fatalf("round %d %s: decoded %d tensors, want %d", round+1, candidate.name, len(out.Tensors), len(p.infos))
			}
			runtime.KeepAlive(out)
			samples[candidate.name] = append(samples[candidate.name], elapsed)
			t.Logf("round %d %s: %s", round+1, candidate.name, elapsed)
		}
	}
	scalarMedian := medianDuration(samples["scalar"])
	neonMedian := medianDuration(samples["neon"])
	t.Logf("median scalar=%s neon=%s speedup=%.4fx", scalarMedian, neonMedian,
		float64(scalarMedian)/float64(neonMedian))
}
