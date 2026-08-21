package gguf

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

func makeMXFP4Raw(elements int) []byte {
	raw := make([]byte, elements/blockElems*mxfp4BlockSize)
	scales := [...]byte{0, 1, 2, 120, 127, 128, 135}
	for b := 0; b*blockElems < elements; b++ {
		o := raw[b*mxfp4BlockSize : (b+1)*mxfp4BlockSize]
		o[0] = scales[b%len(scales)]
		for i := 1; i < len(o); i++ {
			lo := byte((b*7 + i*3) & 15)
			hi := byte((b*11 + i*5 + 1) & 15)
			o[i] = lo | hi<<4
		}
	}
	return raw
}

func TestE8M0HalfTableExact(t *testing.T) {
	for e := range 256 {
		got, want := math.Float32bits(e8m0HalfTable[e]), math.Float32bits(e8m0ToF32Half(byte(e)))
		if got != want {
			t.Fatalf("e=%d: table bits %#08x, want %#08x", e, got, want)
		}
	}
}

func TestDequantMXFP4IntoMatchesTensorPathExactly(t *testing.T) {
	const elements = 256
	raw := makeMXFP4Raw(elements)
	want, err := dequantMXFP4(tensor.Shape{elements}, raw)
	if err != nil {
		t.Fatal(err)
	}
	got := make([]float32, elements)
	dequantMXFP4Into(got, raw)
	for i, v := range got {
		if math.Float32bits(v) != math.Float32bits(want.Storage().F32()[i]) {
			t.Fatalf("[%d]: into %v (%#08x), tensor %v (%#08x)", i, v, math.Float32bits(v), want.Storage().F32()[i], math.Float32bits(want.Storage().F32()[i]))
		}
	}
}

func TestDotMXFP4RowMatchesMaterializedReferenceExactly(t *testing.T) {
	const k = 4096
	raw := makeMXFP4Raw(k)
	x := make([]float32, k)
	for i := range x {
		x[i] = float32(math.Sin(float64(i)*0.17) * math.Pow(2, float64(i%9-4)))
	}
	w := make([]float32, k)
	dequantMXFP4Into(w, raw)
	var want float64
	for i, v := range w {
		want += float64(x[i]) * float64(v)
	}
	got := dotMXFP4Row(x, raw, k)
	if math.Float64bits(got) != math.Float64bits(want) {
		t.Fatalf("fused %v (%#016x), materialized %v (%#016x)", got, math.Float64bits(got), want, math.Float64bits(want))
	}
}

func TestQMatMulMXFP4MatchesDequantizedReference(t *testing.T) {
	const n, k = 3, 256
	raw := makeMXFP4Raw(n * k)
	weights, err := dequantMXFP4(tensor.Shape{n, k}, raw)
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
			var x32 []float32
			var x64 []float64
			if tc.f64 {
				x64 = make([]float64, tc.m*k)
				for i := range x64 {
					x64[i] = math.Sin(float64(i)*0.19) * math.Pow(2, float64(i%7-3))
				}
				x = tensor.FromFloat64(tensor.Shape{tc.m, k}, x64)
			} else {
				x32 = make([]float32, tc.m*k)
				for i := range x32 {
					x32[i] = float32(math.Sin(float64(i)*0.19) * math.Pow(2, float64(i%7-3)))
				}
				x = tensor.FromFloat32(tensor.Shape{tc.m, k}, x32)
			}
			got, err := QMatMul(x, raw, MXFP4, n, k)
			if err != nil {
				t.Fatal(err)
			}
			for mi := range tc.m {
				for ni := range n {
					var want float64
					if tc.f64 {
						for ki := range k {
							want += x64[mi*k+ki] * float64(wf[ni*k+ki])
						}
					} else {
						for ki := range k {
							want += float64(x32[mi*k+ki]) * float64(wf[ni*k+ki])
						}
					}
					gotv, wantv := got.Storage().F32()[mi*n+ni], float32(want)
					diff := math.Abs(float64(gotv - wantv))
					if diff > 1e-4*(math.Abs(float64(wantv))+1e-9) {
						t.Fatalf("m=%d n=%d: got %v want %v (relative=%g)", mi, ni, gotv, wantv, diff/(math.Abs(float64(wantv))+1e-9))
					}
				}
			}
		})
	}
}

var mxfp4QMatMulSink *tensor.Tensor

func TestQMatMulMXFP4ScratchAllocationsDoNotScaleWithOutputRows(t *testing.T) {
	const m, k = 2, 32
	x := tensor.FromFloat32(tensor.Shape{m, k}, make([]float32, m*k))
	allocs := func(n int) float64 {
		raw := makeMXFP4Raw(n * k)
		return testing.AllocsPerRun(100, func() {
			var err error
			mxfp4QMatMulSink, err = QMatMul(x, raw, MXFP4, n, k)
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

func TestQMatMulMXFP4SelectorScope(t *testing.T) {
	const n, k = 3, 32
	raw := makeMXFP4Raw(n * k)
	old := dotMXFP4RowFn
	defer func() { dotMXFP4RowFn = old }()
	calls := 0
	dotMXFP4RowFn = func(row []float32, weight []byte, width int) float64 {
		calls++
		return dotMXFP4Row(row, weight, width)
	}
	if _, err := QMatMul(tensor.New(tensor.F32, tensor.Shape{1, k}), raw, MXFP4, n, k); err != nil {
		t.Fatal(err)
	}
	if calls != n {
		t.Fatalf("contiguous F32 M1 selector calls = %d, want %d", calls, n)
	}
	calls = 0
	if _, err := QMatMul(tensor.New(tensor.F32, tensor.Shape{2, k}), raw, MXFP4, n, k); err != nil {
		t.Fatal(err)
	}
	if calls != 0 {
		t.Fatalf("F32 M2 dispatched M1 row leaf %d times", calls)
	}
	if _, err := QMatMul(tensor.New(tensor.F64, tensor.Shape{1, k}), raw, MXFP4, n, k); err != nil {
		t.Fatal(err)
	}
	if calls != 0 {
		t.Fatalf("F64 M1 dispatched F32 row leaf %d times", calls)
	}
}

// §T555 §V16 tier-1: BOTH directions match gguf-py exactly — dequant on random
// blocks (f32-exact) and encode on random floats (byte-exact).
func TestMXFP4MatchesGGUFPy(t *testing.T) {
	raw, err := os.ReadFile("testdata/mxfp4_golden.json")
	if err != nil {
		t.Skip("golden missing:", err)
	}
	var golden struct {
		Data []byte    `json:"data"`
		Want []float32 `json:"want"`
		Vals []float32 `json:"vals"`
		Enc  []byte    `json:"enc"`
	}
	if err := json.Unmarshal(raw, &golden); err != nil {
		t.Fatal(err)
	}
	got, err := Dequantize(golden.Data, MXFP4, len(golden.Want))
	if err != nil {
		t.Fatal(err)
	}
	g := got.Storage().F32()
	for i := range golden.Want {
		if g[i] != golden.Want[i] {
			t.Fatalf("dequant [%d]: go %g vs gguf-py %g", i, g[i], golden.Want[i])
		}
	}
	vt := tensor.FromFloat32(tensor.Shape{len(golden.Vals)}, golden.Vals)
	enc, err := Quantize(vt, MXFP4)
	if err != nil {
		t.Fatal(err)
	}
	if len(enc) != len(golden.Enc) {
		t.Fatalf("encode length %d vs %d", len(enc), len(golden.Enc))
	}
	for i := range enc {
		if enc[i] != golden.Enc[i] {
			t.Fatalf("encode [%d]: go %#x vs gguf-py %#x", i, enc[i], golden.Enc[i])
		}
	}
}

// Round trip: encode→decode error bounded by half the E2M1 step at the block
// scale (≤ d·1 for the coarse top of the table; generous bound d·2 guards it).
func TestMXFP4RoundTripBound(t *testing.T) {
	vals := make([]float32, 64)
	for i := range vals {
		vals[i] = float32(i%13) * 0.37 * float32(1-2*(i%2))
	}
	vt := tensor.FromFloat32(tensor.Shape{64}, vals)
	enc, err := Quantize(vt, MXFP4)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := Dequantize(enc, MXFP4, 64)
	if err != nil {
		t.Fatal(err)
	}
	d := dec.Storage().F32()
	for b := range 2 {
		scale := e8m0ToF32Half(enc[b*mxfp4BlockSize])
		for i := range 32 {
			idx := b*32 + i
			if diff := d[idx] - vals[idx]; diff > 2*scale || diff < -2*scale {
				t.Fatalf("[%d]: error %g exceeds 2·scale %g", idx, diff, 2*scale)
			}
		}
	}
}

// Hostile sizes error; junk decodes without panicking.
func TestMXFP4Hostile(t *testing.T) {
	if _, err := Dequantize(make([]byte, 16), MXFP4, 32); err == nil {
		t.Fatal("short data must error")
	}
	junk := make([]byte, 17)
	for i := range junk {
		junk[i] = byte(i*97 + 255)
	}
	if _, err := Dequantize(junk, MXFP4, 32); err != nil {
		t.Fatal(err)
	}
}

// FuzzMXFP4RoundTrip: quantize→dequantize never panics and stays finite for
// finite inputs.
func FuzzMXFP4RoundTrip(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4})
	f.Fuzz(func(t *testing.T, data []byte) {
		vals := make([]float32, 32)
		for i := range vals {
			if i < len(data) {
				vals[i] = float32(int8(data[i])) * 0.25
			}
		}
		vt := tensor.FromFloat32(tensor.Shape{32}, vals)
		enc, err := Quantize(vt, MXFP4)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := Dequantize(enc, MXFP4, 32); err != nil {
			t.Fatal(err)
		}
	})
}
