package autograd

import (
	"github.com/jxsl13/goai/internal/archgold"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// prefCase is one rule: how many operands it takes and with what attributes.
type prefCase struct {
	name  string
	op    backend.Op
	nIn   int
	attrs backend.Attrs
	want  uint64
}

func prefDigest(t *testing.T, c prefCase, n int, dt tensor.Dtype) uint64 {
	t.Helper()
	vjp := vjps[c.op]
	if vjp == nil {
		t.Skipf("%s VJP not registered", c.name)
	}
	in := make([]*tensor.Tensor, c.nIn)
	for j := range c.nIn {
		tn := tensor.New(dt, tensor.Shape{n})
		for i := range n {
			// KTO reads its third operand as a LABEL, so the values must straddle zero rather
			// than being a smooth ramp; sine does both.
			tn.SetF64(math.Sin(float64(i)*0.31+float64(j)*1.3)*0.4, i)
		}
		in[j] = tn
	}
	g := tensor.New(tensor.F64, tensor.Shape{})
	g.Storage().F64()[0] = 1.5
	out, err := vjp(nil, in, nil, c.attrs, g)
	if err != nil {
		t.Fatal(err)
	}
	h := uint64(14695981039346656037)
	for _, o := range out {
		if o == nil {
			continue
		}
		for i := range n {
			u := math.Float64bits(o.AtF64(i))
			for s := 0; s < 64; s += 8 {
				h = (h ^ (u>>s)&0xff) * 1099511628211
			}
		}
	}
	return h
}

func prefCases() []prefCase {
	return []prefCase{
		{"cpo", backend.OpCPO, 2, backend.CPOAttrs{Beta: 0.1, Alpha: 1}, archgold.Pick(6160353642181901900, 8262512422077654180)},
		{"dpo", backend.OpDPO, 4, backend.DPOAttrs{Beta: 0.1}, archgold.Pick(16243699260225276505, 9860811518191949329)},
		{"ipo", backend.OpIPO, 4, backend.IPOAttrs{Beta: 0.1}, archgold.Pick(2411552010633019957, 7190294216285379565)},
		{"kto", backend.OpKTO, 3, backend.KTOAttrs{Beta: 0.1}, archgold.Pick(12382265863399297195, 11653641358527267826)},
		{"simpo", backend.OpSimPO, 2, backend.SimPOAttrs{Beta: 2, Gamma: 0.5}, archgold.Pick(11542092229795165829, 16089967935890736517)},
		{"grpo", backend.OpGRPO, 4, backend.GRPOAttrs{Epsilon: 0.2, Beta: 0.04}, archgold.Pick(6830796145460791928, 8284925590970691940)},
	}
}

// TestPrefVJPsAreBitIdentical freezes every preference-optimization backward. Routing them all
// through one typed walker claims to change no value, and NONE of these rules had a test before
// this, so nothing would have noticed if it did.
func TestPrefVJPsAreBitIdentical(t *testing.T) {
	for _, c := range prefCases() {
		got := prefDigest(t, c, 733, tensor.F64) // 733: not a multiple of anything the walker uses
		if got != c.want {
			t.Fatalf("%s digest %d, want %d", c.name, got, c.want)
		}
	}
}

// TestPrefVJPArmsAgree pins the walker's two arms against each other for every rule. An F32
// operand takes the accessor arm, and its output is stored as float32 — so what must hold is
// that it equals the typed arm's result rounded ONCE, not that the bits are equal.
func TestPrefVJPArmsAgree(t *testing.T) {
	const n = 401
	for _, c := range prefCases() {
		vjp := vjps[c.op]
		if vjp == nil {
			continue
		}
		in32 := make([]*tensor.Tensor, c.nIn)
		in64 := make([]*tensor.Tensor, c.nIn)
		for j := range c.nIn {
			a := tensor.New(tensor.F32, tensor.Shape{n})
			b := tensor.New(tensor.F64, tensor.Shape{n})
			for i := range n {
				a.SetF64(math.Sin(float64(i)*0.31+float64(j)*1.3)*0.4, i)
				b.SetF64(a.AtF64(i), i) // the same, already-rounded values
			}
			in32[j], in64[j] = a, b
		}
		g := tensor.New(tensor.F64, tensor.Shape{})
		g.Storage().F64()[0] = 1.5
		o32, err := vjp(nil, in32, nil, c.attrs, g)
		if err != nil {
			t.Fatal(err)
		}
		o64, err := vjp(nil, in64, nil, c.attrs, g)
		if err != nil {
			t.Fatal(err)
		}
		for k := range o64 {
			if o64[k] == nil || o32[k] == nil {
				continue
			}
			for i := range n {
				want := float32(o64[k].AtF64(i))
				got := float32(o32[k].AtF64(i))
				if math.Float32bits(got) != math.Float32bits(want) {
					t.Fatalf("%s out[%d][%d]: accessor %v, typed rounded %v", c.name, k, i, got, want)
				}
			}
		}
	}
}
