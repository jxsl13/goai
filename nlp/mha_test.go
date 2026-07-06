package nlp_test

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/ops"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

type tfGolden struct {
	Softmax struct {
		X     []float64 `json:"x"`
		Shape []int     `json:"shape"`
		Y     []float64 `json:"y"`
	} `json:"softmax"`
	Layernorm struct {
		X     []float64 `json:"x"`
		Shape []int     `json:"shape"`
		Gamma []float64 `json:"gamma"`
		Beta  []float64 `json:"beta"`
		Eps   float64   `json:"eps"`
		Y     []float64 `json:"y"`
	} `json:"layernorm"`
	MHA struct {
		Seq    int       `json:"seq"`
		Dmodel int       `json:"dmodel"`
		Heads  int       `json:"heads"`
		X      []float64 `json:"x"`
		Wq     []float64 `json:"wq"`
		Wk     []float64 `json:"wk"`
		Wv     []float64 `json:"wv"`
		Wo     []float64 `json:"wo"`
		Y      []float64 `json:"y"`
	} `json:"mha"`
}

func loadTF(t *testing.T) tfGolden {
	t.Helper()
	raw, err := os.ReadFile("testdata/transformer.json")
	if err != nil {
		t.Fatalf("read golden (run `make golden`): %v", err)
	}
	var g tfGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

func assertClose(t *testing.T, name string, got *tensor.Tensor, want []float64, rtol float64) {
	t.Helper()
	if got.Numel() != len(want) {
		t.Fatalf("%s: numel %d vs %d", name, got.Numel(), len(want))
	}
	for i := range want {
		idx := tensor.Unravel(i, got.Shape())
		g := got.AtF64(idx...)
		if math.Abs(g-want[i]) > rtol*math.Max(1, math.Abs(want[i])) {
			t.Fatalf("%s[%d]: got %.17g want %.17g", name, i, g, want[i])
		}
	}
}

// §V1: softmax matches torch at f64 rtol 1e-12; rows sum to 1; shift-invariant.
func TestSoftmaxParityAndProps(t *testing.T) {
	g := loadTF(t)
	x := tensor.FromFloat64(tensor.Shape(g.Softmax.Shape), g.Softmax.X)
	y, err := ops.Softmax(x)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "softmax", y, g.Softmax.Y, 1e-12)

	// §V6 rows sum to 1
	for i := range 2 {
		var s float64
		for j := range 4 {
			s += y.AtF64(i, j)
		}
		if math.Abs(s-1) > 1e-12 {
			t.Errorf("row %d sums to %v", i, s)
		}
	}
	// §V12 stability: +1000 shift → same output
	shifted := make([]float64, len(g.Softmax.X))
	for i, v := range g.Softmax.X {
		shifted[i] = v + 1000
	}
	ys, err := ops.Softmax(tensor.FromFloat64(tensor.Shape(g.Softmax.Shape), shifted))
	if err != nil {
		t.Fatal(err)
	}
	for i := range g.Softmax.Y {
		idx := tensor.Unravel(i, ys.Shape())
		if math.Abs(ys.AtF64(idx...)-y.AtF64(idx...)) > 1e-12 {
			t.Fatal("softmax not shift-invariant")
		}
	}
}

// §V1: layernorm (op + nn layer) matches torch at f64 rtol 1e-12.
func TestLayerNormParity(t *testing.T) {
	g := loadTF(t)
	x := tensor.FromFloat64(tensor.Shape(g.Layernorm.Shape), g.Layernorm.X)
	gamma := tensor.FromFloat64(tensor.Shape{len(g.Layernorm.Gamma)}, g.Layernorm.Gamma)
	beta := tensor.FromFloat64(tensor.Shape{len(g.Layernorm.Beta)}, g.Layernorm.Beta)

	y, err := ops.LayerNorm(x, gamma, beta, g.Layernorm.Eps)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "layernorm", y, g.Layernorm.Y, 1e-12)

	// layer wrapper with the same params
	l := &nn.LayerNorm{Gamma: gamma, Beta: beta, Eps: g.Layernorm.Eps}
	y2, err := l.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "layernorm-layer", y2, g.Layernorm.Y, 1e-12)
}

// §V1: full MHA forward matches torch at f64 rtol 1e-12.
func TestMHAParity(t *testing.T) {
	g := loadTF(t)
	dm := g.MHA.Dmodel
	sq := tensor.Shape{dm, dm}
	mha, err := nlp.NewMHA(g.MHA.Heads,
		tensor.FromFloat64(sq, g.MHA.Wq),
		tensor.FromFloat64(sq, g.MHA.Wk),
		tensor.FromFloat64(sq, g.MHA.Wv),
		tensor.FromFloat64(sq, g.MHA.Wo),
	)
	if err != nil {
		t.Fatal(err)
	}
	x := tensor.FromFloat64(tensor.Shape{g.MHA.Seq, dm}, g.MHA.X)
	y, err := mha.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "mha", y, g.MHA.Y, 1e-12)
	if got := len(mha.Params()); got != 4 {
		t.Errorf("params %d, want 4", got)
	}
}

func TestMHAErrors(t *testing.T) {
	w := tensor.New(tensor.F64, tensor.Shape{8, 8})
	if _, err := nlp.NewMHA(3, w, w, w, w); err == nil {
		t.Error("dmodel not divisible by heads must error")
	}
	bad := tensor.New(tensor.F64, tensor.Shape{4, 8})
	if _, err := nlp.NewMHA(2, bad, w, w, w); err == nil {
		t.Error("non-square weight must error")
	}
	mha, _ := nlp.NewMHA(2, w, w, w, w)
	if _, err := mha.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{3, 5})); err == nil {
		t.Error("x dmodel mismatch must error")
	}
}
