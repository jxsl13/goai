package gguf

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

func makeIQ4NLRaw(elements int) []byte {
	raw := make([]byte, elements/blockElems*iq4nlBlockSize)
	scales := [...]float32{0.125, -0.25, 0.5, 1, 2}
	for b := 0; b*blockElems < elements; b++ {
		blk := raw[b*iq4nlBlockSize : (b+1)*iq4nlBlockSize]
		//perfscan:ignore PS4001 test setup writes one intentionally varied f16 scale per strided block
		binary.LittleEndian.PutUint16(blk, f32ToF16(scales[b%len(scales)]))
		for i := range 16 {
			lo := byte((b*7 + i*11 + 3) & 0x0f)
			hi := byte((b*13 + i*5 + 9) & 0x0f)
			blk[2+i] = lo | hi<<4
		}
	}
	return raw
}

func iq4Golden(t *testing.T, file string, qt QuantType) {
	t.Helper()
	raw, err := os.ReadFile(file)
	if err != nil {
		t.Skip("golden missing:", err)
	}
	var golden struct {
		Data []byte    `json:"data"`
		Want []float32 `json:"want"`
	}
	if err := json.Unmarshal(raw, &golden); err != nil {
		t.Fatal(err)
	}
	got, err := Dequantize(golden.Data, qt, len(golden.Want))
	if err != nil {
		t.Fatal(err)
	}
	g := got.Storage().F32()
	for i := range golden.Want {
		if g[i] != golden.Want[i] {
			t.Fatalf("[%d]: go %g vs gguf-py %g", i, g[i], golden.Want[i])
		}
	}
}

// §T554 part 4, §V16 tier-1: both IQ4 variants match gguf-py f32-exactly.
func TestIQ4NLDequantMatchesGGUFPy(t *testing.T) {
	iq4Golden(t, "testdata/iq4nl_golden.json", IQ4_NL)
}
func TestIQ4XSDequantMatchesGGUFPy(t *testing.T) {
	iq4Golden(t, "testdata/iq4xs_golden.json", IQ4_XS)
}

func TestDequantIQ4NLIntoMatchesTensorPathExactly(t *testing.T) {
	const n = 4096
	raw := makeIQ4NLRaw(n)
	full, err := dequantIQ4_NL(tensor.Shape{n}, raw)
	if err != nil {
		t.Fatal(err)
	}
	got := make([]float32, n)
	dequantIQ4_NLInto(got, raw)
	want := full.Storage().F32()
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("element %d: into=%v (%#x), tensor=%v (%#x)",
				i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
}

func TestDotIQ4NLRowMatchesMaterializedReferenceExactly(t *testing.T) {
	const k = 4096
	raw := makeIQ4NLRaw(k)
	x := make([]float32, k)
	for i := range x {
		x[i] = float32(math.Sin(float64(i)*0.071) * math.Pow(2, float64(i%9-4)))
	}
	decoded, err := dequantIQ4_NL(tensor.Shape{k}, raw)
	if err != nil {
		t.Fatal(err)
	}
	var want float64
	for i, w := range decoded.Storage().F32() {
		want += float64(x[i]) * float64(w)
	}
	got := dotIQ4NLRow(x, raw, k)
	if math.Float64bits(got) != math.Float64bits(want) {
		t.Fatalf("fused scalar=%v (%#x), materialized=%v (%#x)",
			got, math.Float64bits(got), want, math.Float64bits(want))
	}
}

func TestQMatMulIQ4NLMatchesDequantizedReference(t *testing.T) {
	const n, k = 5, 64
	raw := makeIQ4NLRaw(n * k)
	weights, err := dequantIQ4_NL(tensor.Shape{n, k}, raw)
	if err != nil {
		t.Fatal(err)
	}
	wf := weights.Storage().F32()
	for _, tc := range []struct {
		name string
		m    int
		f64  bool
	}{{"F32_M1", 1, false}, {"F32_M3", 3, false}, {"F64_M1", 1, true}, {"F64_M3", 3, true}} {
		t.Run(tc.name, func(t *testing.T) {
			var x *tensor.Tensor
			if tc.f64 {
				data := make([]float64, tc.m*k)
				for i := range data {
					data[i] = math.Sin(float64(i)*0.19) * math.Pow(2, float64(i%7-3))
				}
				x = tensor.FromFloat64(tensor.Shape{tc.m, k}, data)
			} else {
				data := make([]float32, tc.m*k)
				for i := range data {
					data[i] = float32(math.Sin(float64(i)*0.19) * math.Pow(2, float64(i%7-3)))
				}
				x = tensor.FromFloat32(tensor.Shape{tc.m, k}, data)
			}
			got, err := QMatMul(x, raw, IQ4_NL, n, k)
			if err != nil {
				t.Fatal(err)
			}
			gf := got.Storage().F32()
			for mi := range tc.m {
				for ni := range n {
					var want float64
					for ki := range k {
						want += x.AtF64(mi, ki) * float64(wf[ni*k+ki])
					}
					want32 := float32(want)
					diff := math.Abs(float64(gf[mi*n+ni] - want32))
					if diff > 1e-4*(math.Abs(float64(want32))+1e-9) {
						t.Fatalf("m=%d n=%d: got %v want %v (relative=%g)",
							mi, ni, gf[mi*n+ni], want32, diff/(math.Abs(float64(want32))+1e-9))
					}
				}
			}
		})
	}
}

var iq4QMatMulSink *tensor.Tensor

func TestQMatMulIQ4NLScratchAllocationsDoNotScaleWithOutputRows(t *testing.T) {
	const m, k = 2, 64
	x := tensor.FromFloat32(tensor.Shape{m, k}, make([]float32, m*k))
	allocs := func(n int) float64 {
		raw := makeIQ4NLRaw(n * k)
		return testing.AllocsPerRun(100, func() {
			var err error
			iq4QMatMulSink, err = QMatMul(x, raw, IQ4_NL, n, k)
			if err != nil {
				panic(err)
			}
		})
	}
	one, many := allocs(1), allocs(31)
	if many != one {
		t.Fatalf("QMatMul allocations scale with output rows: N1=%g N31=%g", one, many)
	}
}

func TestQMatMulIQ4NLSelectorScope(t *testing.T) {
	const n, k = 3, 64
	raw := makeIQ4NLRaw(n * k)
	old := dotIQ4NLRowFn
	defer func() { dotIQ4NLRowFn = old }()
	calls := 0
	dotIQ4NLRowFn = func(row []float32, weight []byte, width int) float64 {
		calls++
		return dotIQ4NLRow(row, weight, width)
	}

	f32m1 := tensor.New(tensor.F32, tensor.Shape{1, k})
	if _, err := QMatMul(f32m1, raw, IQ4_NL, n, k); err != nil {
		t.Fatal(err)
	}
	if calls != n {
		t.Fatalf("contiguous F32 M1 selector calls = %d, want %d", calls, n)
	}

	calls = 0
	if _, err := QMatMul(tensor.New(tensor.F32, tensor.Shape{2, k}), raw, IQ4_NL, n, k); err != nil {
		t.Fatal(err)
	}
	if calls != 0 {
		t.Fatalf("F32 M2 dispatched M1 row leaf %d times", calls)
	}
	if _, err := QMatMul(tensor.New(tensor.F64, tensor.Shape{1, k}), raw, IQ4_NL, n, k); err != nil {
		t.Fatal(err)
	}
	if calls != 0 {
		t.Fatalf("F64 M1 dispatched F32 row leaf %d times", calls)
	}
}

// Hostile sizes error; junk decodes without panicking.
func TestIQ4Hostile(t *testing.T) {
	if _, err := Dequantize(make([]byte, 17), IQ4_NL, 32); err == nil {
		t.Fatal("short IQ4_NL must error")
	}
	if _, err := Dequantize(make([]byte, 135), IQ4_XS, 256); err == nil {
		t.Fatal("short IQ4_XS must error")
	}
	junk := make([]byte, 136)
	for i := range junk {
		junk[i] = byte(i*41 + 13)
	}
	if _, err := Dequantize(junk[:18], IQ4_NL, 32); err != nil {
		t.Fatal(err)
	}
	if _, err := Dequantize(junk, IQ4_XS, 256); err != nil {
		t.Fatal(err)
	}
}

// FuzzDequantIQ4XS: any block-size byte soup decodes without panicking.
func FuzzDequantIQ4XS(f *testing.F) {
	f.Add(make([]byte, 136))
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) < 136 {
			return
		}
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("panicked: %v", r)
			}
		}()
		_, _ = Dequantize(data[:136], IQ4_XS, 256)
	})
}
