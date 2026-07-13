package ref

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

type cvTensor struct {
	Shape  []int     `json:"shape"`
	Values []float64 `json:"values"`
}
type cvCase struct {
	Out   []float64 `json:"out"`
	Shape []int     `json:"shape"`
}
type cvGolden struct {
	X           cvTensor  `json:"x"`
	W           cvTensor  `json:"w"`
	B           []float64 `json:"b"`
	ConvS1P0    cvCase    `json:"conv_s1p0"`
	ConvS2P1    cvCase    `json:"conv_s2p1"`
	ConvNoBias  cvCase    `json:"conv_nobias"`
	MaxPoolK2   cvCase    `json:"maxpool_k2"`
	AvgPoolK2S1 cvCase    `json:"avgpool_k2s1"`
}

func loadCV(t *testing.T) cvGolden {
	t.Helper()
	raw, err := os.ReadFile("testdata/cv.json")
	if err != nil {
		t.Fatalf("read golden (run `make golden`): %v", err)
	}
	var g cvGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

func assertCV(t *testing.T, name string, got *tensor.Tensor, want cvCase) {
	t.Helper()
	if !got.Shape().Equal(tensor.Shape(want.Shape)) {
		t.Fatalf("%s: shape %v, want %v", name, got.Shape(), want.Shape)
	}
	for i := range want.Out {
		idx := tensor.Unravel(i, got.Shape())
		g := got.AtF64(idx...)
		if math.Abs(g-want.Out[i]) > 1e-12*math.Max(1, math.Abs(want.Out[i])) {
			t.Fatalf("%s[%d]: got %.17g want %.17g", name, i, g, want.Out[i])
		}
	}
}

// §V1: conv2d and pooling match torch F.conv2d / F.max_pool2d / F.avg_pool2d
// at f64 rtol 1e-12 across stride/padding/bias variants.
func TestCVParity(t *testing.T) {
	g := loadCV(t)
	x := tensor.FromFloat64(tensor.Shape(g.X.Shape), g.X.Values)
	w := tensor.FromFloat64(tensor.Shape(g.W.Shape), g.W.Values)
	b := tensor.FromFloat64(tensor.Shape{len(g.B)}, g.B)
	ctx := backend.NewContext()

	run := func(op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) *tensor.Tensor {
		out, err := backend.Execute(ctx, op, ins, attrs)
		if err != nil {
			t.Fatalf("%v: %v", op, err)
		}
		return out[0]
	}

	assertCV(t, "conv_s1p0", run(backend.OpConv2D, backend.ConvAttrs{Stride: 1, Pad: 0}, x, w, b), g.ConvS1P0)
	assertCV(t, "conv_s2p1", run(backend.OpConv2D, backend.ConvAttrs{Stride: 2, Pad: 1}, x, w, b), g.ConvS2P1)
	assertCV(t, "conv_nobias", run(backend.OpConv2D, nil, x, w), g.ConvNoBias)
	assertCV(t, "maxpool_k2", run(backend.OpMaxPool2D, backend.PoolAttrs{Kernel: 2}, x), g.MaxPoolK2)
	assertCV(t, "avgpool_k2s1", run(backend.OpAvgPool2D, backend.PoolAttrs{Kernel: 2, Stride: 1}, x), g.AvgPoolK2S1)
}

func TestCVErrors(t *testing.T) {
	ctx := backend.NewContext()
	x := tensor.New(tensor.F64, tensor.Shape{1, 2, 4, 4})
	wBad := tensor.New(tensor.F64, tensor.Shape{3, 5, 3, 3}) // channel mismatch
	if _, err := backend.Execute(ctx, backend.OpConv2D, []*tensor.Tensor{x, wBad}, nil); err == nil {
		t.Error("channel mismatch must error")
	}
	wBig := tensor.New(tensor.F64, tensor.Shape{1, 2, 9, 9}) // kernel > input
	if _, err := backend.Execute(ctx, backend.OpConv2D, []*tensor.Tensor{x, wBig}, nil); err == nil {
		t.Error("empty output must error")
	}
	if _, err := backend.Execute(ctx, backend.OpMaxPool2D, []*tensor.Tensor{x}, nil); err == nil {
		t.Error("missing kernel attr must error")
	}
}
