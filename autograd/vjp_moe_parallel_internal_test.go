package autograd

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestMoECombineBackwardNormalizedWeightsBitExact locks both typed arms to the
// pre-cache formula. The small cells stay serial and the large cells cross the
// token-parallel threshold; neither normalized-weight reuse nor fan-out may move a bit.
func TestMoECombineBackwardNormalizedWeightsBitExact(t *testing.T) {
	vjp := vjps[backend.OpMoECombine]
	if vjp == nil {
		t.Fatal("no OpMoECombine VJP registered")
	}
	for _, tc := range []struct {
		name   string
		dt     tensor.Dtype
		tks, e int
		d      int
	}{
		{name: "f64_serial", dt: tensor.F64, tks: 9, e: 3, d: 17},
		{name: "f64_parallel", dt: tensor.F64, tks: 4096, e: 8, d: 65},
		{name: "f32_serial", dt: tensor.F32, tks: 9, e: 3, d: 17},
		{name: "f32_parallel", dt: tensor.F32, tks: 4096, e: 8, d: 65},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewPCG(7, 0x9e3779b9))
			w := tensor.New(tc.dt, tensor.Shape{tc.tks, tc.e})
			for i := 0; i < tc.tks*tc.e; i++ {
				w.SetF64(math.Abs(rng.NormFloat64()), i/tc.e, i%tc.e)
			}
			for i := 0; i < tc.e; i++ {
				w.SetF64(0, 1, i) // skipped token after a populated token: no stale scratch may escape
			}
			experts := make([]*tensor.Tensor, tc.e)
			for i := range experts {
				experts[i] = tensor.New(tc.dt, tensor.Shape{tc.tks, tc.d})
				for k := 0; k < tc.tks*tc.d; k++ {
					experts[i].SetF64(rng.NormFloat64(), k/tc.d, k%tc.d)
				}
			}
			g := tensor.New(tc.dt, tensor.Shape{tc.tks, tc.d})
			for i := 0; i < tc.tks*tc.d; i++ {
				g.SetF64(rng.NormFloat64(), i/tc.d, i%tc.d)
			}

			in := append([]*tensor.Tensor{w}, experts...)
			got, err := vjp(nil, in, nil, nil, g)
			if err != nil {
				t.Fatal(err)
			}

			// Serial pre-change reference: repeat w_i/denom in the output dot and
			// expert-gradient pass exactly where the old typed implementation did.
			dwRef := make([]float64, tc.tks*tc.e)
			deRef := make([][]float64, tc.e)
			for i := range deRef {
				deRef[i] = make([]float64, tc.tks*tc.d)
			}
			out := make([]float64, tc.d)
			for tk := 0; tk < tc.tks; tk++ {
				var denom float64
				for i := 0; i < tc.e; i++ {
					denom += w.AtF64(tk, i)
				}
				if denom <= 0 {
					continue
				}
				for j := 0; j < tc.d; j++ {
					var acc float64
					for i := 0; i < tc.e; i++ {
						acc += (w.AtF64(tk, i) / denom) * experts[i].AtF64(tk, j)
					}
					out[j] = acc
				}
				for i := 0; i < tc.e; i++ {
					wi := w.AtF64(tk, i) / denom
					var dwSum float64
					for j := 0; j < tc.d; j++ {
						gj := g.AtF64(tk, j)
						deRef[i][tk*tc.d+j] = gj * wi
						dwSum += gj * (experts[i].AtF64(tk, j) - out[j])
					}
					dwRef[tk*tc.e+i] = dwSum / denom
				}
			}

			assertBits := func(name string, got *tensor.Tensor, want []float64) {
				t.Helper()
				switch tc.dt {
				case tensor.F64:
					for i, v := range got.Storage().F64() {
						if math.Float64bits(v) != math.Float64bits(want[i]) {
							t.Fatalf("%s[%d]: got %v, want %v", name, i, v, want[i])
						}
					}
				case tensor.F32:
					for i, v := range got.Storage().F32() {
						if math.Float32bits(v) != math.Float32bits(float32(want[i])) {
							t.Fatalf("%s[%d]: got %v, want %v", name, i, v, float32(want[i]))
						}
					}
				}
			}
			assertBits("dw", got[0], dwRef)
			for i := range experts {
				assertBits("de", got[1+i], deRef[i])
			}
		})
	}
}
