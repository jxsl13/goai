package ref

import (
	"math"
	"testing"
)

func TestMLAAddRoPEScoresMatchesFrozenLoop(t *testing.T) {
	for _, tc := range []struct {
		seq, dR, jmax int
	}{
		{seq: 5, dR: 4, jmax: 3},
		{seq: 7, dR: 8, jmax: 7},
	} {
		q := make([]float64, tc.dR)
		pattern := [...]float64{1e16, 1, -1e16, 3}
		for e := range q {
			q[e] = pattern[e%len(pattern)]
		}
		k := make([]float64, tc.seq*tc.dR)
		kRT := make([]float64, len(k))
		for j := range tc.seq {
			for e := range tc.dR {
				v := 1 + float64((j+e)%3)*0.25
				k[j*tc.dR+e] = v
				kRT[e*tc.seq+j] = v
			}
		}
		want := make([]float64, tc.seq)
		got := make([]float64, tc.seq)
		for j := range tc.seq {
			want[j] = float64(j+1) * 0.25
			got[j] = want[j]
		}
		for j := range tc.jmax {
			for e := range tc.dR {
				want[j] += q[e] * k[j*tc.dR+e]
			}
		}
		mlaAddRoPEScores(got, q, kRT, 0, tc.seq, tc.dR, tc.jmax)
		for j := range tc.seq {
			if math.Float64bits(got[j]) != math.Float64bits(want[j]) {
				t.Fatalf("seq=%d dR=%d jmax=%d score[%d]=%g, frozen=%g", tc.seq, tc.dR, tc.jmax, j, got[j], want[j])
			}
		}
	}
}
