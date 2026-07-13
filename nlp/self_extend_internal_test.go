package nlp

import "testing"

// §T519 integration audit: the pure position math (SelfExtendRelPos,
// SelfExtendPositions) must be the SPEC of what SelfExtendForward actually
// builds — for every distant causal pair the grouped RoPE positions satisfy
// qpos[i]−kpos[j] == SelfExtendRelPos(i,j,w,G) == SelfExtendPositions[i][j];
// neighbor pairs take the true distance through the ordinary RoPE source.
// Before this test the three symbols were unlinked (the forward built its
// positions inline) — an audit finding, not a behavior change.
func TestSelfExtendPosMatchesRelPosSpec(t *testing.T) {
	const seq, window, group = 40, 8, 4
	qpos, kpos := selfExtendPos(seq, window, group)
	m := SelfExtendPositions(seq, window, group)
	for i := range seq {
		for j := 0; j <= i; j++ {
			want := SelfExtendRelPos(i, j, window, group)
			if m[i][j] != want {
				t.Fatalf("SelfExtendPositions[%d][%d]=%d, RelPos=%d", i, j, m[i][j], want)
			}
			if i-j < window {
				if want != i-j {
					t.Fatalf("neighbor pair (%d,%d): RelPos %d, want true distance %d", i, j, want, i-j)
				}
				continue // forward serves this pair from the true-position source
			}
			if got := qpos[i] - kpos[j]; got != want {
				t.Fatalf("grouped pair (%d,%d): forward rel pos %d, spec %d", i, j, got, want)
			}
		}
	}
}
