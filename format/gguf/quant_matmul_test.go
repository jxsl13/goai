package gguf

import (
	"encoding/base64"
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

type qmatGolden struct {
	M, N, K int
	X       []float64 `json:"x"`
	Q8      string    `json:"q8_bytes"`
	Q4      string    `json:"q4_bytes"`
	YQ8     []float64 `json:"y_q8"`
	YQ4     []float64 `json:"y_q4"`
}

// §T39/§V1: quantized-weight matmul (dequant-on-the-fly) matches the gguf-py
// reference (Q8_0 exact to f32, Q4_0 within its coarser quantization).
func TestQMatMulParity(t *testing.T) {
	raw, err := os.ReadFile("testdata/qmatmul.json")
	if err != nil {
		t.Fatalf("read golden (run make golden): %v", err)
	}
	var g qmatGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	x := tensor.FromFloat64(tensor.Shape{g.M, g.K}, g.X)

	check := func(name string, b64 string, qt QuantType, want []float64, rtol float64) {
		wb, err := base64.StdEncoding.DecodeString(b64)
		if err != nil {
			t.Fatal(err)
		}
		y, err := QMatMul(x, wb, qt, g.N, g.K)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		for i := range want {
			got := y.AtF64(tensor.Unravel(i, y.Shape())...)
			if math.Abs(got-want[i]) > rtol*math.Max(1, math.Abs(want[i])) {
				t.Errorf("%s[%d]: got %.9g want %.9g", name, i, got, want[i])
			}
		}
	}
	// Q8_0: our dequant is bit-exact to gguf-py, f64 accum then f32 narrow → tight.
	check("q8", g.Q8, Q8_0, g.YQ8, 1e-5)
	check("q4", g.Q4, Q4_0, g.YQ4, 1e-5)
}

// The quantized weight is far smaller than the full-precision matrix it encodes
// — the memory point of quantized inference.
func TestQuantMemoryFootprint(t *testing.T) {
	raw, _ := os.ReadFile("testdata/qmatmul.json")
	var g qmatGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	q8, _ := base64.StdEncoding.DecodeString(g.Q8)
	q4, _ := base64.StdEncoding.DecodeString(g.Q4)
	f32Bytes := g.N * g.K * 4
	if len(q8) >= f32Bytes {
		t.Errorf("Q8_0 (%d) not smaller than f32 (%d)", len(q8), f32Bytes)
	}
	if len(q4)*2 >= f32Bytes {
		t.Errorf("Q4_0 (%d) not ~4× smaller than f32 (%d)", len(q4), f32Bytes)
	}
	t.Logf("f32 %dB → Q8_0 %dB (%.1f×) → Q4_0 %dB (%.1f×)",
		f32Bytes, len(q8), float64(f32Bytes)/float64(len(q8)), len(q4), float64(f32Bytes)/float64(len(q4)))
}

func TestQMatMulErrors(t *testing.T) {
	x := tensor.New(tensor.F32, tensor.Shape{2, 64})
	if _, err := QMatMul(x, make([]byte, 10), Q8_0, 4, 64); err == nil {
		t.Error("wrong weight byte count must error")
	}
	if _, err := QMatMul(tensor.New(tensor.F32, tensor.Shape{2, 33}), nil, Q8_0, 4, 33); err == nil {
		t.Error("K not multiple of 32 must error")
	}
}
