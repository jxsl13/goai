package nlp

import (
	"fmt"
	"math"
	"testing"
)

func BenchmarkFWHT(b *testing.B) {
	for _, n := range []int{128, 512, 4096} {
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			a := make([]float64, n)
			tmpl := make([]float64, n)
			for i := range tmpl {
				tmpl[i] = math.Sin(float64(i))
			}
			b.ResetTimer()
			for range b.N {
				copy(a, tmpl)
				fwht(a)
			}
		})
	}
}
