package nlp

import "testing"

// TestLogSoftmaxRowIntoReslicesToRequest pins the one clause of the reused row that the beam tests
// cannot see: the returned slice must be cut to the logit count, not handed back at full capacity.
//
// Every call in a search asks for the same vocabulary width, so after the first the capacity always
// equals the request and a mutation returning cap() stays green through the whole suite. Calling it
// directly with a shorter row after a longer one is the situation a caller with two vocabularies —
// a draft model and a target model, say — would create.
func TestLogSoftmaxRowIntoReslicesToRequest(t *testing.T) {
	long := logSoftmaxRowInto(nil, []float64{0, 1, 2, 3, 4, 5, 6, 7})
	if len(long) != 8 {
		t.Fatalf("first call: len %d, want 8", len(long))
	}
	short := logSoftmaxRowInto(long, []float64{1, 2, 3})
	if len(short) != 3 {
		t.Fatalf("reused buffer: len %d, want 3 — the row was not cut to the logit count", len(short))
	}
	// And the values must be the log-softmax of the SHORT row, not a stale tail of the long one.
	want := logSoftmaxRowInto(nil, []float64{1, 2, 3})
	for i := range want {
		if short[i] != want[i] {
			t.Fatalf("reused buffer entry %d: %v, fresh %v", i, short[i], want[i])
		}
	}
}
