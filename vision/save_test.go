package vision_test

import (
	"bytes"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/tensor"
	"github.com/jxsl13/goai/vision"
)

// §T458/§V15: a CNN round-trips through safetensors bit-identically — export the live tensors,
// serialize, parse, rebuild, and the reloaded model produces bit-identical logits.
func TestCNNSafetensorsRoundTrip(t *testing.T) {
	m, err := vision.NewCNN(1, 3, 7, vision.WithDtype(tensor.F64))
	if err != nil {
		t.Fatal(err)
	}
	out := m.Safetensors()
	if len(out) != 6 {
		t.Fatalf("exported %d tensors, want 6", len(out))
	}
	for name, w := range out {
		found := false
		for _, p := range m.Params() {
			if p == w {
				found = true
			}
		}
		if !found {
			t.Fatalf("%q is not one of the model's live tensors", name)
		}
	}

	var buf bytes.Buffer
	if err := safetensors.Save(&buf, out, nil); err != nil {
		t.Fatal(err)
	}
	ts2, _, err := safetensors.Load(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	m2, err := vision.FromSafetensors(ts2)
	if err != nil {
		t.Fatal(err)
	}

	x := tensor.New(tensor.F64, tensor.Shape{2, 1, 8, 8})
	for i := range x.Numel() {
		x.SetF64(float64(i%13)*0.1, tensor.Unravel(i, x.Shape())...)
	}
	ctx := backend.NewContext()
	l1, err := m.Forward(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	l2, err := m2.Forward(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	if !l1.Shape().Equal(l2.Shape()) {
		t.Fatalf("shapes %v vs %v", l1.Shape(), l2.Shape())
	}
	for i := range l1.Numel() {
		idx := tensor.Unravel(i, l1.Shape())
		if a, b := l1.AtF64(idx...), l2.AtF64(idx...); a != b {
			t.Fatalf("logit[%v]: %.17g != %.17g after round-trip", idx, a, b)
		}
	}
	if _, err := vision.FromSafetensors(map[string]*tensor.Tensor{}); err == nil {
		t.Fatal("empty map accepted")
	}
	t.Logf("round-trip: 6 tensors, %d logits bit-identical", l1.Numel())
}
