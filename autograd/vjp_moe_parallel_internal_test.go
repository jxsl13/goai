package autograd

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// The token-parallel MoE-combine backward must be BIT-IDENTICAL to a serial reference:
// tokens are independent (per-token disjoint de[i][t,:] and dw[t,:], no cross-token
// reduction), so parallelizing them changes nothing per token. Large tks exercises the
// goroutine split.
func TestMoECombineBackwardParallelBitExact(t *testing.T) {
	vjp := vjps[backend.OpMoECombine]
	if vjp == nil {
		t.Fatal("no OpMoECombine VJP registered")
	}
	const tks, e, d = 4096, 4, 64 // tks*d over the parallel threshold
	rng := rand.New(rand.NewPCG(7, 0x9e3779b9))
	w := tensor.New(tensor.F64, tensor.Shape{tks, e})
	ws := w.Storage().F64()
	for i := range ws {
		ws[i] = math.Abs(rng.NormFloat64()) // nonneg weights; some tokens ~0
	}
	experts := make([]*tensor.Tensor, e)
	for i := range experts {
		experts[i] = tensor.New(tensor.F64, tensor.Shape{tks, d})
		for k, s := 0, experts[i].Storage().F64(); k < len(s); k++ {
			s[k] = rng.NormFloat64()
		}
	}
	g := tensor.New(tensor.F64, tensor.Shape{tks, d})
	for i, s := 0, g.Storage().F64(); i < len(s); i++ {
		s[i] = rng.NormFloat64()
	}

	in := append([]*tensor.Tensor{w}, experts...)
	got, err := vjp(nil, in, nil, nil, g)
	if err != nil {
		t.Fatal(err)
	}
	dwGot := got[0].Storage().F64()
	deGot := make([][]float64, e)
	for i := range experts {
		deGot[i] = got[1+i].Storage().F64()
	}

	// serial reference — exact same formula, one token at a time.
	dwRef := make([]float64, tks*e)
	deRef := make([][]float64, e)
	for i := range deRef {
		deRef[i] = make([]float64, tks*d)
	}
	es, gs := make([][]float64, e), g.Storage().F64()
	for i := range experts {
		es[i] = experts[i].Storage().F64()
	}
	out := make([]float64, d)
	for tk := 0; tk < tks; tk++ {
		wb, eb := tk*e, tk*d
		var denom float64
		for i := 0; i < e; i++ {
			denom += ws[wb+i]
		}
		if denom <= 0 {
			continue
		}
		for j := 0; j < d; j++ {
			var acc float64
			for i := 0; i < e; i++ {
				acc += (ws[wb+i] / denom) * es[i][eb+j]
			}
			out[j] = acc
		}
		for i := 0; i < e; i++ {
			wi := ws[wb+i] / denom
			var dwSum float64
			for j := 0; j < d; j++ {
				gj := gs[eb+j]
				deRef[i][eb+j] = gj * wi
				dwSum += gj * (es[i][eb+j] - out[j])
			}
			dwRef[wb+i] = dwSum / denom
		}
	}

	for k := range dwRef {
		if math.Float64bits(dwGot[k]) != math.Float64bits(dwRef[k]) {
			t.Fatalf("dw[%d]: parallel %v vs serial %v", k, dwGot[k], dwRef[k])
		}
	}
	for i := range experts {
		for k := range deRef[i] {
			if math.Float64bits(deGot[i][k]) != math.Float64bits(deRef[i][k]) {
				t.Fatalf("de[%d][%d]: parallel %v vs serial %v", i, k, deGot[i][k], deRef[i][k])
			}
		}
	}
}
