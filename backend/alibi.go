package backend

import "math"

// ALiBiSlopes returns the per-head attention-bias slopes for ALiBi (Press et al.
// 2021, arXiv:2108.12409). For a power-of-two head count the slopes are a
// geometric sequence with ratio 2^(−8/n): head h (0-indexed) has slope
// 2^(−8·(h+1)/n). For a non-power-of-two n the paper's scheme is used: take the
// slopes of the largest power of two ≤ n, then interleave (every other) slopes of
// the next power of two for the remainder (§R60).
func ALiBiSlopes(n int) []float64 {
	if n <= 0 {
		return nil
	}
	if n&(n-1) == 0 { // power of two
		return powerOfTwoSlopes(n)
	}
	closest := 1
	for closest*2 <= n {
		closest *= 2
	}
	slopes := powerOfTwoSlopes(closest)
	extra := powerOfTwoSlopes(2 * closest)
	for i := 0; i < n-closest; i++ {
		slopes = append(slopes, extra[2*i]) // every other slope
	}
	return slopes
}

func powerOfTwoSlopes(n int) []float64 {
	out := make([]float64, n)
	for h := range n {
		out[h] = math.Exp2(-8.0 * float64(h+1) / float64(n))
	}
	return out
}
