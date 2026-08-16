package nn

import (
	"github.com/jxsl13/goai/internal/archgold"
	"math"
	"testing"
)

// TestHQQuantizeIsBitIdentical freezes every bit the quantizer emits. Rewriting its clamp as a
// comparison chain claims to change no value, and the accuracy of a quantizer would absorb a
// real change without complaint — the sibling clamp test pins the rewrite in isolation, and this
// pins it where it is actually used.
func TestHQQuantizeIsBitIdentical(t *testing.T) {
	for _, c := range []struct {
		rows, cols, bits, group int
		want                    uint64
	}{
		{32, 64, 4, 32, archgold.Pick(9348102188691648517, 8727524114837717154)},
		{17, 48, 3, 16, archgold.Pick(8008301601848843183, 16140439446547545984)},
		{8, 128, 8, 64, archgold.Pick(12810509524361636723, 12810509524361636723)},
	} {
		ws := make([]float64, c.rows*c.cols)
		for i := range ws {
			ws[i] = math.Sin(float64(i*11+3)) * 0.9
		}
		codes, scale, zero := HQQuantize(ws, c.bits, c.group)
		h := uint64(14695981039346656037)
		mix := func(u uint64) {
			for s := 0; s < 64; s += 8 {
				h = (h ^ (u>>s)&0xff) * 1099511628211
			}
		}
		for _, v := range codes {
			mix(uint64(v))
		}
		for _, v := range scale {
			mix(math.Float64bits(v))
		}
		for _, v := range zero {
			mix(math.Float64bits(v))
		}
		if h != c.want {
			t.Fatalf("%dx%d b=%d g=%d digest %d, want %d", c.rows, c.cols, c.bits, c.group, h, c.want)
		}
	}
}
