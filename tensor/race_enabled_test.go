//go:build race

package tensor

// raceEnabled reports that this test binary runs under the race detector, whose
// sync.Pool deliberately drops items at random — pool-reuse IDENTITY cannot be
// asserted there (§B51).
const raceEnabled = true
