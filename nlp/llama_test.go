package nlp_test

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/ops"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

type llamaGolden struct {
	RMSNorm struct {
		X     []float64 `json:"x"`
		Shape []int     `json:"shape"`
		Gamma []float64 `json:"gamma"`
		Eps   float64   `json:"eps"`
		Y     []float64 `json:"y"`
	} `json:"rmsnorm"`
	RoPE struct {
		Q    []float64 `json:"q"`
		Seq  int       `json:"seq"`
		HD   int       `json:"hd"`
		Base float64   `json:"base"`
		Y    []float64 `json:"y"`
	} `json:"rope"`
}

func loadLlama(t *testing.T) llamaGolden {
	t.Helper()
	raw, err := os.ReadFile("testdata/llama.json")
	if err != nil {
		t.Fatalf("read golden (run make golden): %v", err)
	}
	var g llamaGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

func closeAll(t *testing.T, name string, got *tensor.Tensor, want []float64) {
	t.Helper()
	for i := range want {
		idx := tensor.Unravel(i, got.Shape())
		if v := got.AtF64(idx...); math.Abs(v-want[i]) > 1e-12*math.Max(1, math.Abs(want[i])) {
			t.Fatalf("%s[%d]: got %.17g want %.17g", name, i, v, want[i])
		}
	}
}

// §V1: RMSNorm and RoPE match torch/HF conventions at f64 rtol 1e-12.
func TestRMSNormParity(t *testing.T) {
	g := loadLlama(t).RMSNorm
	x := tensor.FromFloat64(tensor.Shape(g.Shape), g.X)
	gamma := tensor.FromFloat64(tensor.Shape{len(g.Gamma)}, g.Gamma)
	y, err := ops.RMSNorm(x, gamma, g.Eps)
	if err != nil {
		t.Fatal(err)
	}
	closeAll(t, "rmsnorm", y, g.Y)

	// nn.RMSNorm layer matches too
	l := &nn.RMSNorm{Gamma: gamma, Eps: g.Eps}
	y2, err := l.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	closeAll(t, "rmsnorm-layer", y2, g.Y)
}

func TestRoPEParity(t *testing.T) {
	g := loadLlama(t).RoPE
	q := tensor.FromFloat64(tensor.Shape{g.Seq, g.HD}, g.Q)
	y, err := ops.RoPE(q, g.Base)
	if err != nil {
		t.Fatal(err)
	}
	closeAll(t, "rope", y, g.Y)
}

// §V6: RoPE is an orthogonal rotation → it preserves each position vector's
// L2 norm.
func TestRoPEPreservesNorm(t *testing.T) {
	g := loadLlama(t).RoPE
	q := tensor.FromFloat64(tensor.Shape{g.Seq, g.HD}, g.Q)
	y, _ := ops.RoPE(q, g.Base)
	for p := range g.Seq {
		var nq, ny float64
		for i := range g.HD {
			nq += q.AtF64(p, i) * q.AtF64(p, i)
			ny += y.AtF64(p, i) * y.AtF64(p, i)
		}
		if math.Abs(nq-ny) > 1e-12*math.Max(1, nq) {
			t.Errorf("pos %d: RoPE changed norm %.12g → %.12g", p, math.Sqrt(nq), math.Sqrt(ny))
		}
	}
}
