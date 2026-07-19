package pytorch_test

import (
	"fmt"
	"sort"

	"github.com/jxsl13/goai/format/pytorch"
)

// Example loads a PyTorch state_dict and inspects its parameter map. Nested
// module names remain dot-separated, so the returned keys can be passed
// directly to GoAI's model builders. See the package documentation for the
// restricted-unpickler security contract and supported checkpoint layouts.
func Example() {
	weights, err := pytorch.LoadFile("testdata/golden.pt")
	if err != nil {
		panic(err)
	}

	names := make([]string, 0, len(weights))
	for name := range weights {
		names = append(names, name)
	}
	sort.Strings(names)

	nested := weights["nested.inner.b"]
	fmt.Println(names)
	fmt.Println(nested.Dtype(), nested.Shape(), nested.AtF64(1))
	// Output:
	// [nested.inner.b w16 w32 w64 wbf16 wi64]
	// f32 (2) 8
}
