package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// Gated linear attention carries an [dk,dv] state across the sequence, so a one-ulp drift
// compounds across steps exactly as it does in RetNet and SSD. The output dot jammed here adds
// ONE product; the state update beside it adds two and is deliberately left alone (PS3084).
func glaDigest(t *testing.T, seq, dk, dv int) uint64 {
	t.Helper()
	mk := func(rows, cols, seed int, gate bool) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape{rows, cols})
		s := x.Storage().F64()
		for i := range s {
			if gate {
				// Gates live in (0,1): a decay, not an arbitrary coefficient.
				s[i] = 0.5 + 0.45*math.Sin(float64(i*7+seed*11))
				continue
			}
			s[i] = math.Sin(float64(i*7+seed*11)) * 0.4
		}
		return x
	}
	out, err := nn.GatedLinearAttention(backend.NewContext(),
		mk(seq, dk, 1, false), mk(seq, dk, 2, false), mk(seq, dv, 3, false), mk(seq, dk, 4, true))
	if err != nil {
		t.Fatal(err)
	}
	h := uint64(14695981039346656037)
	n := out.Numel()
	for _, val := range out.Storage().F64()[:n] {
		u := math.Float64bits(val)
		for s := 0; s < 64; s += 8 {
			h = (h ^ (u>>s)&0xff) * 1099511628211
		}
	}
	return h
}

func TestGLAIsBitIdentical(t *testing.T) {
	// dk is the jammed dimension and carries the remainders: 13 odd and prime, 22 leaving 6
	// modulo 8, 32 dividing every width so the tail never runs, and 1 skipping the jam.
	cases := []struct {
		seq, dk, dv int
		want        uint64
	}{
		{9, 13, 5, 6023223899936446705},
		{7, 22, 9, 13067282666201662810},
		{6, 32, 8, 17490629942812691155},
		{5, 1, 4, 17003964984244211732},
	}
	for _, c := range cases {
		if got := glaDigest(t, c.seq, c.dk, c.dv); got != c.want {
			t.Errorf("seq=%d dk=%d dv=%d: digest %d", c.seq, c.dk, c.dv, got)
		}
	}
}
