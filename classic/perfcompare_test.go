package classic

import (
	"encoding/csv"
	"fmt"
	"math/rand/v2"
	"os"
	"strconv"
	"testing"
	"time"
)

// TestPerfCompareVsSklearn: §V22 perf-parity probe vs sklearn. Generates a fixed
// synthetic dataset, writes it to a shared CSV (so sklearn times the IDENTICAL
// data), and times GoAI classical fit for the compute-heavy methods. Gated on
// PERF_CSV_DIR. Not a correctness test.
func TestPerfCompareVsSklearn(t *testing.T) {
	dir := os.Getenv("PERF_CSV_DIR")
	if dir == "" {
		t.Skip("set PERF_CSV_DIR to run the sklearn perf comparison")
	}
	const n, d, classes = 4000, 20, 3
	rng := rand.New(rand.NewPCG(42, 42))
	X := make([][]float64, n)
	y := make([]int, n)
	yf := make([]float64, n)
	yb := make([]float64, n)
	ybi := make([]int, n)
	centers := make([][]float64, classes)
	for c := range centers {
		centers[c] = make([]float64, d)
		for j := range centers[c] {
			centers[c][j] = rng.NormFloat64() * 3
		}
	}
	for i := range X {
		c := rng.IntN(classes)
		y[i], yf[i] = c, float64(c)
		if c != 0 {
			yb[i], ybi[i] = 1, 1
		}
		X[i] = make([]float64, d)
		for j := range X[i] {
			X[i][j] = centers[c][j] + rng.NormFloat64()
		}
	}
	perfWriteCSV(t, dir+"/perf_X.csv", X, nil)
	perfWriteCSV(t, dir+"/perf_y.csv", nil, y)

	timeit := func(name string, fit func() error) {
		start := time.Now()
		if err := fit(); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		fmt.Printf("GOAI_FIT\t%s\t%.2f\n", name, float64(time.Since(start).Microseconds())/1000)
	}
	timeit("DecisionTree", func() error { return NewDecisionTreeClassifier(WithMaxDepth(12)).Fit(X, y) })
	timeit("RandomForest100", func() error { return NewRandomForestClassifier(WithNumTrees(100), WithSeed(42)).Fit(X, y) })
	timeit("GradientBoosting100", func() error { return NewGradientBoostingClassifier(WithGBMNEstimators(100)).Fit(X, ybi) })
	timeit("SVC_rbf", func() error { return NewSVC(WithSVMKernel(SVMKernelRBF)).Fit(X, yb) })
	timeit("GaussianNB", func() error { return NewGaussianNB().Fit(X, y) })
	timeit("KNN_fit", func() error { return NewKNNClassifier(WithKNNK(5)).Fit(X, y) })
}

func perfWriteCSV(t *testing.T, path string, X [][]float64, y []int) {
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()
	if X != nil {
		for _, row := range X {
			rec := make([]string, len(row))
			for j, v := range row {
				rec[j] = strconv.FormatFloat(v, 'g', -1, 64)
			}
			_ = w.Write(rec)
		}
	} else {
		for _, v := range y {
			_ = w.Write([]string{strconv.Itoa(v)})
		}
	}
}
