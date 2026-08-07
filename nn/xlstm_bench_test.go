package nn_test

import (
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/nn"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// BenchmarkMLSTMRecurrent times the stabilized mLSTM recurrent scan at head-realistic dims. The step's
// dominant cost is the two O(dv·dk) inner loops — the C_t state update and the (C_t q_t) output dot —
// so this is the gate for the PS3010 latency-break on the serial output-dot reduction.
func BenchmarkMLSTMRecurrent(b *testing.B) {
	for _, d := range []struct{ seq, dk, dv int }{
		{512, 64, 64},
		{512, 128, 128},
		{1024, 128, 128},
	} {
		b.Run(benchMLSTMName(d.seq, d.dk, d.dv), func(b *testing.B) {
			rng := rand.New(rand.NewPCG(7, 11))
			q, k := randMat(rng, d.seq, d.dk), randMat(rng, d.seq, d.dk)
			v := randMat(rng, d.seq, d.dv)
			ipre, fpre := mlstmRandCol(rng, d.seq), mlstmRandCol(rng, d.seq)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				out, err := nn.MLSTMRecurrent(q, k, v, ipre, fpre, false)
				if err != nil {
					b.Fatal(err)
				}
				_ = out
			}
		})
	}
}

func benchMLSTMName(seq, dk, dv int) string {
	return "seq" + itoaB(seq) + "_dk" + itoaB(dk) + "_dv" + itoaB(dv)
}

func itoaB(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
