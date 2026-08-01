package main

import (
	"strings"
	"testing"
)

func dtypeArmMsgs(t *testing.T, src string) []string {
	t.Helper()
	var out []string
	for _, f := range scanSrc(t, src) {
		if f.category == "unconverted-dtype-arm" {
			out = append(out, f.msg)
		}
	}
	return out
}

// TestPS1009FiresOnUnconvertedArm is the positive floor. Two shapes, because the message names which
// kind of clause it found and a reader triaging 23 findings needs that to be right.
func TestPS1009FiresOnUnconvertedArm(t *testing.T) {
	for _, c := range []struct{ name, src, want string }{
		{"default-arm", `package p

func rows(t *T, r, c int, buf []float64) {
	switch t.Dtype() {
	case F64:
		src := t.Storage().F64()
		for i := range r {
			buf[i] = src[i]
		}
	default:
		for i := range r {
			for j := range c {
				buf[i*c+j] = t.AtF64(i, j)
			}
		}
	}
}`, "default clause"},
		{"named-case-arm", `package p

func rows(t *T, r, c int, buf []float64) {
	switch t.Dtype() {
	case F64:
		src := t.Storage().F64()
		for i := range r {
			buf[i] = src[i]
		}
	case F32:
		for i := range r {
			for j := range c {
				buf[i*c+j] = t.AtF64(i, j)
			}
		}
	}
}`, "case clause"},
	} {
		t.Run(c.name, func(t *testing.T) {
			msgs := dtypeArmMsgs(t, c.src)
			if len(msgs) != 1 {
				t.Fatalf("%d findings, want 1", len(msgs))
			}
			if !strings.Contains(msgs[0], c.want) {
				t.Fatalf("message does not say %q: %s", c.want, msgs[0])
			}
			// The measured justification is what separates this from a style note.
			if !strings.Contains(msgs[0], "-6.46%") {
				t.Fatalf("message omits the measured win: %s", msgs[0])
			}
		})
	}
}

// TestPS1009Silent covers one clause of the predicate each, as subtests so a broken clause reddens
// exactly its own guard.
func TestPS1009Silent(t *testing.T) {
	quiet := func(name, src string) {
		t.Run(name, func(t *testing.T) {
			if msgs := dtypeArmMsgs(t, src); len(msgs) != 0 {
				t.Fatalf("%s: expected silence, got: %s", name, msgs[0])
			}
		})
	}

	// CLAUSE: a typed sibling must EXIST. With every arm on the accessor there is no partial fast
	// path — the whole function is slow, which is PS1001's domain and its advice, not this one's.
	quiet("no-typed-sibling", `package p

func rows(t *T, r, c int, buf []float64) {
	switch t.Dtype() {
	case F64:
		for i := range r {
			buf[i] = t.AtF64(i, 0)
		}
	default:
		for i := range r {
			buf[i] = t.AtF64(i, 0)
		}
	}
}`)

	// CLAUSE: the switch must be on Dtype() specifically. The tag here is a CALL, so the
	// CallExpr guard passes and only the method NAME rejects it — which is the point: an earlier
	// version of this fixture switched on a plain identifier, so the CallExpr guard rejected it
	// first and the name check was never exercised. Verified by mutation: relaxing the name test
	// now reddens this, and did not before.
	quiet("not-a-dtype-switch", `package p

func rows(t *T, r, c int, buf []float64) {
	switch t.Kind() {
	case 0:
		src := t.Storage().F64()
		for i := range r {
			buf[i] = src[i]
		}
	default:
		for i := range r {
			buf[i] = t.AtF64(i, 0)
		}
	}
}`)

	// CLAUSE: an arm with no accessor at all is already converted; a switch whose arms are all
	// typed is the finished state this check is pointing at.
	quiet("all-arms-converted", `package p

func rows(t *T, r int, buf []float64) {
	switch t.Dtype() {
	case F64:
		src := t.Storage().F64()
		for i := range r {
			buf[i] = src[i]
		}
	case F32:
		s32 := t.Storage().F32()
		for i := range r {
			buf[i] = float64(s32[i])
		}
	}
}`)
	// CLAUSE: an F32 arm plus a default-only tail is the FINISHED state, not an unfinished one.
	// What remains in default is f16/bf16 and the quantized dtypes, which live in u16 or packed
	// storage and need a real conversion rather than the exact widening an f32 arm gets.
	//
	// This floor exists because the check shipped without it and was wrong at scale: all 23 sites
	// it reported already covered both f64 and f32, so once nlp rows2D was fixed the check had zero
	// actionable findings and 23 pieces of permanent noise. The positive floors above are rows2D's
	// PRE-fix shape — typed arms for f64 only — which is what should still fire.
	quiet("f32-covered-default-tail", `package p

func rows(t *T, r int, buf []float64) {
	switch t.Dtype() {
	case F64:
		src := t.Storage().F64()
		for i := range r {
			buf[i] = src[i]
		}
	case F32:
		s32 := t.Storage().F32()
		for i := range r {
			buf[i] = float64(s32[i])
		}
	default:
		for i := range r {
			buf[i] = t.AtF64(i, 0)
		}
	}
}`)

	// CLAUSE: the suppression applies ONLY when the slow arm is `default`. A NAMED case left on the
	// accessor is a deliberate dtype someone chose to list and then did not convert, so it stays
	// reportable even alongside an f32 arm.
	t.Run("named-slow-case-still-fires", func(t *testing.T) {
		src := `package p

func rows(t *T, r int, buf []float64) {
	switch t.Dtype() {
	case F32:
		s32 := t.Storage().F32()
		for i := range r {
			buf[i] = float64(s32[i])
		}
	case F16:
		for i := range r {
			buf[i] = t.AtF64(i, 0)
		}
	}
}`
		if msgs := dtypeArmMsgs(t, src); len(msgs) != 1 {
			t.Fatalf("%d findings, want 1 — a NAMED unconverted case is not the accepted default tail", len(msgs))
		}
	})
}
