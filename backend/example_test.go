package backend_test

import (
	"fmt"
	"slices"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// --- Level 1: trivial — the zero-effort automatic path -----------------------

// You write no backend-selection code. NewContext uses the best available backend
// automatically: a GPU if you built with its tag (-tags cuda/metal/vulkan) and one
// is present, otherwise the CPU. Here, a plain CPU build reports "cpu".
func Example() {
	ctx := backend.NewContext()
	fmt.Println("running on:", ctx.Backend.Name())
	// Output: running on: cpu
}

// --- Level 2: use-case — override the preference or probe support ------------

// Force your own descending-performance order. Unregistered names (a GPU you did
// not build, or whose device is absent) are safely skipped, so it is fine to list
// every backend you might build with; the reference is always the last resort.
func ExampleSetPreference() {
	defer backend.SetPreference(backend.Preference()...) // restore afterwards
	backend.SetPreference(backend.CUDA, backend.CPU)     // prefer CUDA, fall back to CPU
	fmt.Println(backend.Default().Name())
	// Output: cpu
}

// Available lists exactly the backends that can run on this machine — feature
// detection with no guessing.
func ExampleAvailable() {
	fmt.Println("cpu present:", slices.Contains(backend.Available(), backend.CPU))
	// Output: cpu present: true
}

// Reduction controls whether a batch of per-row losses is averaged, summed,
// or returned intact. Mean and sum therefore produce one scalar; none keeps
// one value per input row so callers can apply their own weighting or mask.
// String uses the same names as PyTorch's reduction argument.
func ExampleReduction() {
	ctx := backend.NewContext()
	logits := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{2, 0, 0, 2})
	targets := tensor.FromFloat64(tensor.Shape{2}, []float64{0, 1})
	reductions := []backend.Reduction{
		backend.ReductionMean,
		backend.ReductionSum,
		backend.ReductionNone,
	}

	for _, reduction := range reductions {
		out, err := backend.Execute(ctx, backend.OpCrossEntropy,
			[]*tensor.Tensor{logits, targets},
			backend.CrossEntropyAttrs{Reduction: reduction})
		if err != nil {
			panic(err)
		}
		fmt.Printf("%s: %v\n", reduction.String(), out[0].Shape())
	}
	// Output:
	// mean: ()
	// sum: ()
	// none: (2)
}

// --- Level 3: embedded — a real matmul on the auto-selected backend ----------

// End to end with zero configuration: build tensors and run a matmul. It executes
// on whatever backend was auto-selected — a GPU if one was built in and detected,
// otherwise the CPU — with identical results either way.
func Example_autoSelectedMatmul() {
	ctx := backend.NewContext() // best available backend, chosen for you
	a := tensor.FromFloat64(tensor.Shape{1, 2}, []float64{1, 2})
	W := tensor.FromFloat64(tensor.Shape{2, 1}, []float64{3, 4})

	y, _ := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{a, W}, nil)
	fmt.Println(y[0].AtF64(0, 0)) // 1·3 + 2·4
	// Output: 11
}
