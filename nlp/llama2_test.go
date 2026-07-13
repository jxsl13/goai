package nlp_test

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/ops"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

type llama2Golden struct {
	SiLU struct {
		X []float64 `json:"x"`
		Y []float64 `json:"y"`
	} `json:"silu"`
	SwiGLU struct {
		X     []float64 `json:"x"`
		Shape []int     `json:"shape"`
		H     int       `json:"h"`
		Wg    []float64 `json:"wg"`
		Wu    []float64 `json:"wu"`
		Wd    []float64 `json:"wd"`
		Y     []float64 `json:"y"`
	} `json:"swiglu"`
	GQA struct {
		Q   []float64 `json:"q"`
		K   []float64 `json:"k"`
		V   []float64 `json:"v"`
		Seq int       `json:"seq"`
		NH  int       `json:"nh"`
		NKV int       `json:"nkv"`
		DK  int       `json:"dk"`
		Y   []float64 `json:"y"`
	} `json:"gqa"`
}

func loadLlama2(t *testing.T) llama2Golden {
	t.Helper()
	raw, err := os.ReadFile("testdata/llama2.json")
	if err != nil {
		t.Fatalf("read golden (run make golden): %v", err)
	}
	var g llama2Golden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

func near(t *testing.T, name string, got *tensor.Tensor, want []float64) {
	t.Helper()
	for i := range want {
		idx := tensor.Unravel(i, got.Shape())
		if v := got.AtF64(idx...); math.Abs(v-want[i]) > 1e-12*math.Max(1, math.Abs(want[i])) {
			t.Fatalf("%s[%d]: got %.17g want %.17g", name, i, v, want[i])
		}
	}
}

// §V1: SiLU matches torch.
func TestSiLUParity(t *testing.T) {
	g := loadLlama2(t).SiLU
	x := tensor.FromFloat64(tensor.Shape{len(g.X)}, g.X)
	y, err := ops.SiLU(x)
	if err != nil {
		t.Fatal(err)
	}
	near(t, "silu", y, g.Y)
}

// §V1: SwiGLU FFN matches torch; §V2: it trains (grads reach all 3 matrices).
func TestSwiGLUParityAndTrain(t *testing.T) {
	g := loadLlama2(t).SwiGLU
	d := g.Shape[1]
	x := tensor.FromFloat64(tensor.Shape(g.Shape), g.X)
	ff := &nn.SwiGLU{
		Wgate: tensor.FromFloat64(tensor.Shape{d, g.H}, g.Wg),
		Wup:   tensor.FromFloat64(tensor.Shape{d, g.H}, g.Wu),
		Wdown: tensor.FromFloat64(tensor.Shape{g.H, d}, g.Wd),
	}
	y, err := ff.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	near(t, "swiglu", y, g.Y)

	// trainable end to end
	tape := autograd.NewTape()
	out, err := ff.Forward(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{out}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	for i, p := range ff.Params() {
		if tape.Grad(p) == nil {
			t.Fatalf("SwiGLU param %d got no gradient", i)
		}
	}
}

// §V1: GQA (4 query heads, 2 kv heads) matches the torch reference.
func TestGQAParity(t *testing.T) {
	g := loadLlama2(t).GQA
	q := tensor.FromFloat64(tensor.Shape{g.Seq, g.NH * g.DK}, g.Q)
	k := tensor.FromFloat64(tensor.Shape{g.Seq, g.NKV * g.DK}, g.K)
	v := tensor.FromFloat64(tensor.Shape{g.Seq, g.NKV * g.DK}, g.V)
	out, err := backend.Execute(backend.NewContext(), backend.OpMHA,
		[]*tensor.Tensor{q, k, v}, backend.AttnAttrs{Heads: g.NH, KVHeads: g.NKV, Causal: false})
	if err != nil {
		t.Fatal(err)
	}
	near(t, "gqa", out[0], g.Y)
}

// MQA (kv_heads=1) is the special case where all query heads share one KV head.
func TestMQAShapes(t *testing.T) {
	q := tensor.New(tensor.F64, tensor.Shape{3, 8}) // 4 heads dk=2
	k := tensor.New(tensor.F64, tensor.Shape{3, 2}) // 1 kv head
	v := tensor.New(tensor.F64, tensor.Shape{3, 2})
	if _, err := backend.Execute(backend.NewContext(), backend.OpMHA,
		[]*tensor.Tensor{q, k, v}, backend.AttnAttrs{Heads: 4, KVHeads: 1}); err != nil {
		t.Fatalf("MQA (kv_heads=1) must work: %v", err)
	}
	// mismatched kv dim errors
	bad := tensor.New(tensor.F64, tensor.Shape{3, 3})
	if _, err := backend.Execute(backend.NewContext(), backend.OpMHA,
		[]*tensor.Tensor{q, bad, bad}, backend.AttnAttrs{Heads: 4, KVHeads: 1}); err == nil {
		t.Error("wrong kv dim must error")
	}
}
