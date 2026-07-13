package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §T559 collapse: with every key channel sharing one decay (a_t = α_t·1), KDA IS
// the Gated Delta Rule — compare against the existing GatedDeltaNet at 1e−12.
func TestKDACollapsesToGatedDeltaNet(t *testing.T) {
	const seq, dk, dv = 10, 4, 6
	q, k := mobaProbe(1, seq, dk), mobaProbe(2, seq, dk)
	v := mobaProbe(3, seq, dv)
	alpha := tensor.New(tensor.F64, tensor.Shape{seq, 1})
	beta := tensor.New(tensor.F64, tensor.Shape{seq, 1})
	aFull := tensor.New(tensor.F64, tensor.Shape{seq, dk})
	for i := range seq {
		av := 0.4 + 0.5*math.Abs(math.Sin(float64(i)*1.7))
		alpha.SetF64(av, i, 0)
		beta.SetF64(0.3+0.4*math.Abs(math.Cos(float64(i))), i, 0)
		for c := range dk {
			aFull.SetF64(av, i, c) // uniform channels = the scalar gate
		}
	}
	kda, err := nn.KimiDeltaAttention(q, k, v, aFull, beta)
	if err != nil {
		t.Fatal(err)
	}
	gdn, err := nn.GatedDeltaNet(backend.NewContext(), q, k, v, alpha, beta)
	if err != nil {
		t.Fatal(err)
	}
	for i := range kda.Numel() {
		idx := tensor.Unravel(i, kda.Shape())
		if d := math.Abs(kda.AtF64(idx...) - gdn.AtF64(idx...)); d > 1e-12 {
			t.Fatalf("collapse broken at %v (Δ %g)", idx, d)
		}
	}
}

// §T559 the channelwise point: zeroing ONE key channel's decay erases exactly
// that channel of the memory — a probe key reading only that channel returns 0,
// while an orthogonal channel's memory survives.
func TestKDAChannelwiseForgetting(t *testing.T) {
	const dk, dv = 2, 1
	// t=0: write v=1 with key e0+e1 (memory in both channels).
	// t=1: decay channel 0 to zero, write nothing (β=0).
	// t=2: query e0 → 0 (erased); query e1 → memory survived.
	q := tensor.FromFloat64(tensor.Shape{4, dk}, []float64{0, 0, 0, 0, 1, 0, 0, 1})
	k := tensor.FromFloat64(tensor.Shape{4, dk}, []float64{1, 1, 0, 0, 0, 0, 0, 0})
	v := tensor.FromFloat64(tensor.Shape{4, dv}, []float64{1, 0, 0, 0})
	a := tensor.FromFloat64(tensor.Shape{4, dk}, []float64{
		1, 1, // t0: no decay
		0, 1, // t1: erase channel 0, keep channel 1
		1, 1, 1, 1,
	})
	beta := tensor.FromFloat64(tensor.Shape{4, 1}, []float64{1, 0, 0, 0})
	out, err := nn.KimiDeltaAttention(q, k, v, a, beta)
	if err != nil {
		t.Fatal(err)
	}
	if got := out.AtF64(2, 0); got != 0 {
		t.Fatalf("channel 0 not erased: query e0 read %g, want 0", got)
	}
	// the write used the L2-normalized key (√½,√½), so channel 1 holds √½.
	if got, want := out.AtF64(3, 0), math.Sqrt2/2; math.Abs(got-want) > 1e-15 {
		t.Fatalf("channel 1 damaged: query e1 read %g, want %g", got, want)
	}
}

// KDA: the gated delta rule with per-channel forgetting — Kimi Linear's core.
func ExampleKimiDeltaAttention() {
	q := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{0, 0, 1, 0})
	k := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, 0, 0, 0})
	v := tensor.FromFloat64(tensor.Shape{2, 1}, []float64{2, 0})
	a := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, 1, 1, 1})
	beta := tensor.FromFloat64(tensor.Shape{2, 1}, []float64{1, 0})
	out, _ := nn.KimiDeltaAttention(q, k, v, a, beta)
	fmt.Println(out.AtF64(1, 0)) // t1 reads back the t0 write
	// Output: 2
}
