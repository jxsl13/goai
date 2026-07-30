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
}
