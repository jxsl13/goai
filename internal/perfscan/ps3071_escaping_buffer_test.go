package main

import "testing"

func escapingBufferFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "local-buffer-escapes-per-call" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3071_LocalArrayHandedToACall is the measured shape: a decode primitive declaring
// its scratch locally and passing a slice of it to a reader.
func TestDetectPS3071_LocalArrayHandedToACall(t *testing.T) {
	src := `package p

import "encoding/binary"

type reader struct{ n int64 }

func (rd *reader) read(p []byte) error { return nil }

func (rd *reader) u32() (uint32, error) {
	var b [4]byte
	if err := rd.read(b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(b[:]), nil
}`
	fs := escapingBufferFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The safety condition is the part a reader most needs before sharing one scratch.
	if !containsAll(fs[0].msg, "Hang the buffer on the receiver",
		"SAFE ONLY IF NOTHING KEEPS THE BUFFER") {
		t.Fatalf("message omits the fix or the safety condition:\n%s", fs[0].msg)
	}
}

// TestDetectPS3071_SilentOnAPlainFunction pins that the fix needs somewhere to put the buffer.
// A free function has no receiver to hang it on, so the finding would have no advice.
func TestDetectPS3071_SilentOnAPlainFunction(t *testing.T) {
	src := `package p

func read(p []byte) error { return nil }

func u32() (uint32, error) {
	var b [4]byte
	if err := read(b[:]); err != nil {
		return 0, err
	}
	return 0, nil
}`
	if fs := escapingBufferFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — no receiver to hang it on:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3071_SilentWhenTheArrayNeverLeaves pins the condition. An array only indexed, not
// sliced into a call, does not escape and costs nothing.
func TestDetectPS3071_SilentWhenTheArrayNeverLeaves(t *testing.T) {
	src := `package p

type reader struct{ n int64 }

func (rd *reader) fill() byte {
	var b [4]byte
	b[0] = 1
	b[1] = 2
	return b[0] + b[1]
}`
	if fs := escapingBufferFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the array never leaves:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3071_SilentOnANonByteArray pins that the shape is a BYTE buffer. A local array of
// another element type handed to a call is an ordinary value, not a decode scratch.
func TestDetectPS3071_SilentOnANonByteArray(t *testing.T) {
	src := `package p

type reader struct{ n int64 }

func (rd *reader) sum(xs []float64) float64 { return 0 }

func (rd *reader) go2() float64 {
	var b [4]float64
	return rd.sum(b[:])
}`
	if fs := escapingBufferFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — not a byte buffer:\n%s", len(fs), fs[0].msg)
	}
}
