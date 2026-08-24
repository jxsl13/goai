//go:build darwin && cgo

package llamagpu

import (
	"math"
	"reflect"
	"testing"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

func nonzeroGPTBiasGELUModel(t *testing.T) *nlp.GPT {
	t.Helper()
	cfg := nlp.GPTConfig{Vocab: 19, Ctx: 12, Dim: 16, Heads: 4, Layers: 1, Eps: 1e-5}
	m := gptStorageModel(t, cfg)
	seed := 0
	fill := func(x *tensor.Tensor, scale float32) {
		data := x.Storage().F32()
		for i := range data {
			seed++
			data[i] = float32((seed*7)%23-11) * scale
		}
	}
	gain := func(x *tensor.Tensor) {
		data := x.Storage().F32()
		for i := range data {
			data[i] = 0.9 + float32(i%5)*0.02
		}
	}
	fill(m.TokEmb, 0.02)
	fill(m.PosEmb, 0.02)
	fill(m.Head, 0.02)
	gain(m.LNf.Gamma)
	fill(m.LNf.Beta, 0.01)
	for _, b := range m.Blocks {
		gain(b.LN1.Gamma)
		fill(b.LN1.Beta, 0.01)
		gain(b.LN2.Gamma)
		fill(b.LN2.Beta, 0.01)
		fill(b.Attn.Wq, 0.02)
		fill(b.Attn.Wk, 0.02)
		fill(b.Attn.Wv, 0.02)
		fill(b.Attn.Wo, 0.02)
		fill(b.W1, 0.02)
		fill(b.B1, 0.01)
		fill(b.W2, 0.02)
		fill(b.B2, 0.01)
	}
	return m
}

func newMetalGPTBiasGELUTestDecoder(t *testing.T, m *nlp.GPT, fused, profiled bool) (*GPTDecoder, *metal.RecorderProfile) {
	t.Helper()
	profile := new(metal.RecorderProfile)
	ops := backendOps{
		name:          "metal-test",
		fusedF32QKV:   true,
		fusedBiasGELU: fused,
		newBuffer: func(data []float32) (buffer, error) {
			b, err := metal.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return mBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			if !profiled {
				r, err := metal.NewRecorder()
				if err != nil {
					return nil, err
				}
				return mRec{r: r}, nil
			}
			r, err := metal.NewProfilingRecorder(64)
			if err != nil {
				return nil, err
			}
			return mProfileRec{mRec: mRec{r: r}, profile: profile}, nil
		},
	}
	d, err := newGPTDecoder(m, ops)
	if err != nil {
		t.Fatal(err)
	}
	return d, profile
}

func requireExactF32(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: %d != %d", len(got), len(want))
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("value[%d] fused=%v split=%v", i, got[i], want[i])
		}
	}
}

func TestMetalGPTBiasGELUStepAndStepNMatchSplitControl(t *testing.T) {
	if !metal.Available() {
		t.Skip("Metal unavailable")
	}
	m := nonzeroGPTBiasGELUModel(t)

	fused, _ := newMetalGPTBiasGELUTestDecoder(t, m, true, false)
	defer fused.Release()
	control, _ := newMetalGPTBiasGELUTestDecoder(t, m, false, false)
	defer control.Release()
	got, err := fused.Step(3, 0)
	if err != nil {
		t.Fatal(err)
	}
	want, err := control.Step(3, 0)
	if err != nil {
		t.Fatal(err)
	}
	requireExactF32(t, got, want)

	fusedN, _ := newMetalGPTBiasGELUTestDecoder(t, m, true, false)
	defer fusedN.Release()
	controlN, _ := newMetalGPTBiasGELUTestDecoder(t, m, false, false)
	defer controlN.Release()
	tokens := []int{3, 7, 2, 11}
	got, err = fusedN.StepN(tokens, 0)
	if err != nil {
		t.Fatal(err)
	}
	want, err = controlN.StepN(tokens, 0)
	if err != nil {
		t.Fatal(err)
	}
	requireExactF32(t, got, want)
}

func TestMetalGPTBiasGELUProfileProvesFusedActivation(t *testing.T) {
	if !metal.Available() || !metal.RecorderProfilingAvailable() {
		t.Skip("Metal timestamp profiling unavailable")
	}
	m := nonzeroGPTBiasGELUModel(t)
	labels := func(fused bool) []string {
		d, profile := newMetalGPTBiasGELUTestDecoder(t, m, fused, true)
		defer d.Release()
		if _, err := d.Step(3, 0); err != nil {
			t.Fatal(err)
		}
		out := make([]string, len(profile.Events))
		for i := range profile.Events {
			out[i] = profile.Events[i].Label
		}
		return out
	}
	count := func(labels []string, target string) int {
		n := 0
		for _, label := range labels {
			if label == target {
				n++
			}
		}
		return n
	}
	candidate := labels(true)
	control := labels(false)
	if count(candidate, "bias_gelu") != 1 || count(candidate, "unary.gelu") != 0 || count(candidate, "addbias") != 1 {
		t.Fatalf("candidate profile = %v", candidate)
	}
	if count(control, "bias_gelu") != 0 || count(control, "unary.gelu") != 1 || count(control, "addbias") != 2 {
		t.Fatalf("control profile = %v", control)
	}
	if reflect.DeepEqual(candidate, control) {
		t.Fatalf("toggle arms produced identical profiles: %v", candidate)
	}
}
