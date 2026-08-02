package classic

import (
	"math"
	"math/rand"
	"testing"
)

// TestKNNScratchReuseMatchesFreshBuffers pins the per-worker scratch: predicting a batch must
// give exactly what predicting each row on its own gives.
//
// The one-row call is the reference precisely because it cannot reuse anything — a batch of one
// runs a single query against a scratch that was just zero-valued, so any state the batch path
// carries from a previous query shows up as a disagreement. That covers all three buffers: the
// heap's backing array (stale neighbours if it is not truncated), the weight slice (a stale tail
// if it is not resliced to len(nb)), and the class accumulator (votes from the previous row if it
// is not cleared).
//
// The batch is 200 rows so it clears knnParallelChunks' n < 64 serial threshold and actually runs
// several workers, and the weights are inverse-distance so the exact-match branch of the weight
// buffer is exercised rather than the constant-1 one.
func TestKNNScratchReuseMatchesFreshBuffers(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	const n, d = 700, 6
	xs := make([][]float64, n)
	yc := make([]int, n)
	yr := make([]float64, n)
	for i := range xs {
		xs[i] = make([]float64, d)
		for j := range xs[i] {
			xs[i][j] = rng.NormFloat64()
		}
		yc[i] = rng.Intn(4)
		yr[i] = rng.NormFloat64() * 3
	}
	q := xs[:200] // querying training rows also hits the distance-0 exact-match branch

	cl := NewKNNClassifier(WithKNNK(7), WithKNNWeights(KNNDistance))
	if err := cl.Fit(xs, yc); err != nil {
		t.Fatal(err)
	}
	batch, err := cl.Predict(q)
	if err != nil {
		t.Fatal(err)
	}
	proba, err := cl.PredictProba(q)
	if err != nil {
		t.Fatal(err)
	}
	rg := NewKNNRegressor(WithKNNK(7), WithKNNWeights(KNNDistance))
	if err := rg.Fit(xs, yr); err != nil {
		t.Fatal(err)
	}
	rbatch, err := rg.Predict(q)
	if err != nil {
		t.Fatal(err)
	}

	for i, row := range q {
		one := [][]float64{row}
		wantC, err := cl.Predict(one)
		if err != nil {
			t.Fatal(err)
		}
		if batch[i] != wantC[0] {
			t.Fatalf("Predict row %d: batch %d, fresh-scratch %d", i, batch[i], wantC[0])
		}
		wantP, err := cl.PredictProba(one)
		if err != nil {
			t.Fatal(err)
		}
		for c := range wantP[0] {
			if proba[i][c] != wantP[0][c] {
				t.Fatalf("PredictProba row %d class %d: batch %v, fresh-scratch %v",
					i, c, proba[i][c], wantP[0][c])
			}
		}
		wantR, err := rg.Predict(one)
		if err != nil {
			t.Fatal(err)
		}
		if rbatch[i] != wantR[0] && !(math.IsNaN(rbatch[i]) && math.IsNaN(wantR[0])) {
			t.Fatalf("KNNRegressor row %d: batch %v, fresh-scratch %v", i, rbatch[i], wantR[0])
		}
	}
}

// TestKNNProbaRowsAreDistinct pins that PredictProba hands out one slice per row. The vote
// accumulator is now scratch a worker overwrites on its next query, so returning it directly
// would give every row of a chunk the same aliased backing array — and the values would look
// right until the moment a caller kept them.
func TestKNNProbaRowsAreDistinct(t *testing.T) {
	rng := rand.New(rand.NewSource(12))
	const n, d = 300, 4
	xs := make([][]float64, n)
	y := make([]int, n)
	for i := range xs {
		xs[i] = make([]float64, d)
		for j := range xs[i] {
			xs[i][j] = rng.NormFloat64()
		}
		y[i] = rng.Intn(3)
	}
	m := NewKNNClassifier(WithKNNK(3))
	if err := m.Fit(xs, y); err != nil {
		t.Fatal(err)
	}
	p, err := m.PredictProba(xs[:128])
	if err != nil {
		t.Fatal(err)
	}
	seen := make(map[*float64]int, len(p))
	for i := range p {
		if len(p[i]) == 0 {
			t.Fatalf("row %d empty", i)
		}
		if j, dup := seen[&p[i][0]]; dup {
			t.Fatalf("rows %d and %d share one backing array", j, i)
		}
		seen[&p[i][0]] = i
	}
}

// TestKNNWeightsIntoLengthFollowsNeighbours pins the one clause of the scratch contract that the
// end-to-end tests above CANNOT see: the returned weight slice must be resliced to len(nb).
//
// It is invisible through Predict because k is fixed for the life of a model, so every query on a
// given scratch asks for the same number of neighbours and the buffer is always exactly full — a
// mutation that hands back the whole capacity keeps every end-to-end test green. The consumers
// index by neighbour position, so today the extra tail is merely written and never read. This
// test calls the helper directly with a shorter neighbour set after a longer one, which is the
// situation a future caller with a variable k would create.
func TestKNNWeightsIntoLengthFollowsNeighbours(t *testing.T) {
	var s knnScratch
	long := []neighbour{{dist: 1, idx: 0}, {dist: 2, idx: 1}, {dist: 4, idx: 2}, {dist: 8, idx: 3}}
	if got := knnWeightsInto(long, KNNDistance, &s); len(got) != len(long) {
		t.Fatalf("first call: len %d, want %d", len(got), len(long))
	}
	short := long[:2]
	got := knnWeightsInto(short, KNNDistance, &s)
	if len(got) != len(short) {
		t.Fatalf("reused scratch: len %d, want %d — the buffer was not resliced to the neighbour count",
			len(got), len(short))
	}
	for i, n := range short {
		if want := 1 / n.dist; got[i] != want {
			t.Fatalf("weight %d: got %v, want %v", i, got[i], want)
		}
	}
}
