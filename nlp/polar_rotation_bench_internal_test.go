package nlp

import "testing"

// polarRotation.apply and applyInverse had no benchmark: the TurboQuant benchmarks all exercise the
// Hadamard rotation, so the dense polar path — d*d work per call — was unmeasured, which is why
// perfscan's report on its column walk could not be ranked.
//
// Both directions are benchmarked because they are the pair that makes the column walk interesting:
// apply reads q[i][j] with i outer (row-major) while applyInverse reads it with j outer (column), so
// the gap between them at the same d isolates the layout cost from the arithmetic.
func benchPolar(b *testing.B, d int, inverse bool) {
	p, err := newPolarRotation(d, 7)
	if err != nil {
		b.Fatal(err)
	}
	x := make([]float64, d)
	for i := range x {
		x[i] = float64(i%17) * 0.125
	}
	b.ReportAllocs()
	for b.Loop() {
		var err error
		if inverse {
			_, err = p.applyInverse(x)
		} else {
			_, err = p.apply(x)
		}
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPolarApply(b *testing.B) {
	for _, d := range []int{256, 1024} {
		b.Run(polarName(d), func(b *testing.B) { benchPolar(b, d, false) })
	}
}

func BenchmarkPolarApplyInverse(b *testing.B) {
	for _, d := range []int{256, 1024} {
		b.Run(polarName(d), func(b *testing.B) { benchPolar(b, d, true) })
	}
}

func polarName(n int) string {
	if n == 0 {
		return "d0"
	}
	var buf [8]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return "d" + string(buf[i:])
}
