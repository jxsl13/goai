package main

import "testing"

// TestDetectSincosFusable (PS5008): math.Sin(angle) and math.Cos(angle) on the SAME
// argument in one function → fuse to math.Sincos.
func TestDetectSincosFusable(t *testing.T) {
	src := `package p
import "math"
func fill(pe []float64, n int, f float64) {
	for i := 0; i < n; i++ {
		angle := f * float64(i)
		pe[2*i] = math.Sin(angle)
		pe[2*i+1] = math.Cos(angle)
	}
}`
	if got := countCat(scanSrc(t, src))["sincos-fusable"]; got != 1 {
		t.Fatalf("want 1 sincos-fusable, got %d", got)
	}
}

// Must stay silent when only one of the pair is present, or the two calls take
// DIFFERENT arguments (nothing to fuse).
func TestDetectSincosFusable_Silent(t *testing.T) {
	onlySin := `package p
import "math"
func f(x float64) float64 { return math.Sin(x) + math.Sin(x*2) }`
	if got := countCat(scanSrc(t, onlySin))["sincos-fusable"]; got != 0 {
		t.Fatalf("only-sin must be silent, got %d", got)
	}
	diffArg := `package p
import "math"
func f(x, y float64) float64 { return math.Sin(x) + math.Cos(y) }`
	if got := countCat(scanSrc(t, diffArg))["sincos-fusable"]; got != 0 {
		t.Fatalf("different-arg must be silent, got %d", got)
	}
}
