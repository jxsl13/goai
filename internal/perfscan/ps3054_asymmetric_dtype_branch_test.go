package main

import "testing"

// NOTE ON THE FILE NAME. This file was first called ps3054_asymmetric_dtype_arm_test.go, and Go
// silently excluded it: a name ending in _arm_test.go carries an implicit GOARCH=arm constraint,
// so on arm64 it landed in IgnoredGoFiles. Every fixture below reported PASS and every mutation of
// the check read as green, because none of them ran. Keep architecture and OS words out of the
// last underscore-separated token of a file name.
func asymmetricArmFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "asymmetric-dtype-arm" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3054_OneArmDelegatesTheOtherSpellsItOut is the measured shape: a kernel that picks a
// path from a type-assertion flag, hands the f32 reduction to a helper, and writes the f64 one out
// as a scalar loop.
func TestDetectPS3054_OneArmDelegatesTheOtherSpellsItOut(t *testing.T) {
	src := `package p

func dot4T(a, b []float64) float64 { return 0 }

func kernel(qs, ks []float64, l, kd int, out []float64) {
	_, isF32 := any(qs).([]float64)
	for n := range l {
		qn := qs[n*kd : n*kd+kd]
		km := ks[n*kd : n*kd+kd]
		var a float64
		if isF32 {
			a = dot4T(qn, km)
		} else {
			for i, qv := range qn {
				a += qv * km[i]
			}
		}
		out[n] = a
	}
}`
	fs := asymmetricArmFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// Two things a reader needs that no amount of staring at the code supplies: that the missing
	// benchmark has to come first, and that the delegating arm is not automatically the right one
	// to copy.
	if !containsAll(fs[0].msg, "ADD THE MISSING CELL FIRST", "CHECK WHICH ARM IS BEHIND") {
		t.Fatalf("message omits the benchmark-first instruction or the direction warning:\n%s", fs[0].msg)
	}
}

// TestDetectPS3054_SilentWhenBothArmsAreSpelledOut pins that the finding is about ASYMMETRY. Two
// arms that both write their reduction out differ only in dtype handling, which is what a generic
// kernel looks like and not a half-finished optimization.
func TestDetectPS3054_SilentWhenBothArmsAreSpelledOut(t *testing.T) {
	src := `package p

func kernel(qs, ks []float64, l, kd int, out []float64) {
	_, isF32 := any(qs).([]float64)
	for n := range l {
		qn := qs[n*kd : n*kd+kd]
		km := ks[n*kd : n*kd+kd]
		var a float64
		if isF32 {
			for i, qv := range qn {
				a += qv * km[i] * 2
			}
		} else {
			for i, qv := range qn {
				a += qv * km[i]
			}
		}
		out[n] = a
	}
}`
	if fs := asymmetricArmFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — both arms spell the reduction out:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3054_SilentWhenTheBranchIsOnlyTheRemainder pins the APPLIED form, and it is not a
// hypothetical: the fix for the measured site adds a grouped loop AHEAD of the branch and leaves
// the branch to handle the last few iterations. The asymmetry is still there and no longer costs
// anything, so a check that kept reporting it would be reporting its own fix.
func TestDetectPS3054_SilentWhenTheBranchIsOnlyTheRemainder(t *testing.T) {
	src := `package p

func dot4T(a, b []float64) float64 { return 0 }

func kernel(qs, ks []float64, l, kd int, out []float64) {
	_, isF32 := any(qs).([]float64)
	km := ks[0:kd]
	n := 0
	for ; n+3 < l; n += 4 {
		var a0, a1 float64
		for i, kv := range km {
			a0 += qs[(n+0)*kd+i] * kv
			a1 += qs[(n+1)*kd+i] * kv
		}
		out[n], out[n+1] = a0, a1
	}
	for ; n < l; n++ {
		qn := qs[n*kd : n*kd+kd]
		var a float64
		if isF32 {
			a = dot4T(qn, km)
		} else {
			for i, qv := range qn {
				a += qv * km[i]
			}
		}
		out[n] = a
	}
}`
	if fs := asymmetricArmFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the branch only handles the remainder:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3054_SilentWithoutATypeAssertionFlag pins what the check keys on. An ordinary
// boolean parameter does not mark a dtype split, and branches on one are just branches.
func TestDetectPS3054_SilentWithoutATypeAssertionFlag(t *testing.T) {
	src := `package p

func dot4T(a, b []float64) float64 { return 0 }

func kernel(qs, ks []float64, l, kd int, fast bool, out []float64) {
	for n := range l {
		qn := qs[n*kd : n*kd+kd]
		km := ks[n*kd : n*kd+kd]
		var a float64
		if fast {
			a = dot4T(qn, km)
		} else {
			for i, qv := range qn {
				a += qv * km[i]
			}
		}
		out[n] = a
	}
}`
	if fs := asymmetricArmFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the flag is not a type assertion:\n%s", len(fs), fs[0].msg)
	}
}
