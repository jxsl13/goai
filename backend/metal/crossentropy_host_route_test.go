//go:build darwin && cgo

package metal

import (
	"math"
	"runtime"
	"slices"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func closeCrossEntropyHostTensor(t *testing.T, name string, got, want *tensor.Tensor) {
	t.Helper()
	if got == nil || want == nil || !got.Shape().Equal(want.Shape()) {
		t.Fatalf("%s shape mismatch: got=%v want=%v", name, got, want)
	}
	for i := range got.Numel() {
		gv := got.AtF64(tensor.Unravel(i, got.Shape())...)
		wv := want.AtF64(tensor.Unravel(i, want.Shape())...)
		if math.Abs(gv-wv) > 2e-5+2e-5*math.Abs(wv) {
			t.Fatalf("%s[%d]: host=%g metal=%g", name, i, gv, wv)
		}
	}
}

func TestCrossEntropyHostRouteParity(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	ctx := backend.NewContext().WithBackend(Backend{})
	logits := tensor.New(tensor.F32, tensor.Shape{8, 10})
	for i := range logits.Numel() {
		logits.Storage().F32()[i] = float32((i%17)-8) / 7
	}
	targets := tensor.New(tensor.F32, tensor.Shape{8})
	for i := range 8 {
		targets.SetF64(float64((3*i+1)%10), i)
	}
	upstream := tensor.New(tensor.F32, tensor.Shape{})
	upstream.SetF64(0.75)
	inputs := []*tensor.Tensor{logits, targets, upstream}
	before := make([][]float32, len(inputs))
	for i, input := range inputs {
		before[i] = slices.Clone(input.Storage().F32())
	}

	hostLoss, err := crossentropyF32Route(ctx, inputs[:2], nil, true)
	if err != nil {
		t.Fatal(err)
	}
	metalLoss, err := crossentropyF32Route(ctx, inputs[:2], nil, false)
	if err != nil {
		t.Fatal(err)
	}
	hostGrad, err := crossentropyBackwardF32Route(ctx, inputs, nil, true)
	if err != nil {
		t.Fatal(err)
	}
	metalGrad, err := crossentropyBackwardF32Route(ctx, inputs, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	closeCrossEntropyHostTensor(t, "loss", hostLoss[0], metalLoss[0])
	closeCrossEntropyHostTensor(t, "logits-gradient", hostGrad[0], metalGrad[0])
	for i, input := range inputs {
		if !slices.Equal(input.Storage().F32(), before[i]) {
			t.Fatalf("input %d mutated", i)
		}
	}
}

func TestCrossEntropyHostPreferredScope(t *testing.T) {
	wantTarget := runtime.GOARCH == "arm64"
	cases := []struct {
		name           string
		batch, classes int
		attrs          backend.CrossEntropyAttrs
		want           bool
	}{
		{name: "target", batch: 8, classes: 10, want: wantTarget},
		{name: "batch", batch: 7, classes: 10},
		{name: "classes", batch: 8, classes: 11},
		{name: "smoothing", batch: 8, classes: 10, attrs: backend.CrossEntropyAttrs{LabelSmoothing: 0.1}},
		{name: "z-loss", batch: 8, classes: 10, attrs: backend.CrossEntropyAttrs{ZLoss: 1e-4}},
		{name: "sum", batch: 8, classes: 10, attrs: backend.CrossEntropyAttrs{Reduction: backend.ReductionSum}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := crossentropyHostPreferred(tc.batch, tc.classes, tc.attrs); got != tc.want {
				t.Fatalf("crossentropyHostPreferred()=%v want %v", got, tc.want)
			}
		})
	}
}
