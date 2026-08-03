package main

import (
	"strings"
	"testing"
)

const recurrenceMarker = "INTERCHANGE IS NOT AVAILABLE HERE"

func stridedMsg(t *testing.T, src string) string {
	t.Helper()
	for _, f := range scanSrc(t, src) {
		if f.category == "strided-inner-walk" {
			return f.msg
		}
	}
	return ""
}

// TestPS6011ClassifiesSelfReferentialRecurrence is the positive floor: the nest READS the buffer
// strided by the inner variable and WRITES it strided by the outer one, so the strided reads are
// values earlier iterations produced. Interchanging would read x[j] before it exists, which makes
// the check's generic advice not merely unprofitable but wrong — the shipped remedy is contiguous
// scratch plus one scatter.
func TestPS6011ClassifiesSelfReferentialRecurrence(t *testing.T) {
	src := `package p

func back(out, y []float64, lu [][]float64, n, cols int) {
	for c := 0; c < cols; c++ {
		for i := n - 1; i >= 0; i-- {
			s := y[i]
			for j := i + 1; j < n; j++ {
				s -= lu[i][j] * out[j*cols+c]
			}
			out[i*cols+c] = s / lu[i][i]
		}
	}
}`
	msg := stridedMsg(t, src)
	if msg == "" {
		t.Fatal("PS6011 did not fire at all")
	}
	if !strings.Contains(msg, recurrenceMarker) {
		t.Fatalf("self-referential recurrence not classified: %s", msg)
	}
	// The message must carry the measured limit too: at one column the access is already
	// contiguous and the transform is a small LOSS, which is the fact that keeps a reader from
	// applying it blindly.
	if !strings.Contains(msg, "+0.57") {
		t.Fatalf("message omits the single-column regression: %s", msg)
	}
}

// Silence cases, one per clause, as subtests so a broken clause reddens exactly its own guard.
// "Not classified" does not mean silent — PS6011 still fires and still recommends interchange,
// which is correct for these.
func TestPS6011DoesNotClassifyRecurrence(t *testing.T) {
	plain := func(name, src string) {
		t.Run(name, func(t *testing.T) {
			msg := stridedMsg(t, src)
			if msg == "" {
				t.Fatalf("%s: PS6011 should still fire", name)
			}
			if strings.Contains(msg, recurrenceMarker) {
				t.Fatalf("%s: wrongly classified as a recurrence: %s", name, msg)
			}
			if !strings.Contains(msg, "Interchange the loops") {
				t.Fatalf("%s: the generic remedy should still be offered: %s", name, msg)
			}
		})
	}

	// CLAUSE: the strided buffer must be WRITTEN in the nest. A read-only strided traversal is an
	// independent walk, and interchange is exactly the right advice for it.
	plain("read-only-stride", `package p

func mix(dst, src []float64, n, cols int) {
	for c := 0; c < cols; c++ {
		for j := range n {
			dst[c] += src[j*cols+c]
		}
	}
}`)

	// CLAUSE: the write must NOT be strided by the INNER variable. Writing the very slot the
	// inner loop reads is an elementwise update in place, not a recurrence across iterations,
	// and interchange remains the right advice for it.
	plain("strided-by-inner-write", `package p

func mix(src []float64, n, cols int) {
	for c := 0; c < cols; c++ {
		for j := range n {
			src[j*cols+c] = src[j*cols+c] * 2
		}
	}
}`)

	// CLAUSE: an access with NO multiplied variable says nothing about a recurrence. Here the same
	// buffer is written at a contiguous slot and read strided, so the two index shapes differ, but
	// the write does not stride at all — it is a reduction INTO a row, and keeping the intermediate
	// in scratch plus scattering once is not the transform it wants.
	plain("unstrided-write", `package p

func mix(x []float64, n, cols int) {
	for c := 0; c < cols; c++ {
		for j := range n {
			x[c] += x[j*cols+c]
		}
	}
}`)

	// CLAUSE: it must be the SAME buffer. Reading one strided array and writing a different one
	// at the outer stride is an ordinary transform, and the reads are not self-produced.
	plain("different-buffer", `package p

func mix(out, src []float64, lu [][]float64, n, cols int) {
	for c := 0; c < cols; c++ {
		for i := n - 1; i >= 0; i-- {
			var s float64
			for j := i + 1; j < n; j++ {
				s -= lu[i][j] * src[j*cols+c]
			}
			out[i*cols+c] = s
		}
	}
}`)
}
