package npu_test

import (
	"fmt"

	"github.com/jxsl13/goai/backend/npu"
)

// Probe NPU support the same way you would probe a GPU backend. GoAI ships no
// op-level NPU backend (§R45), so this is always false — an explicit "no", never
// a silent gap (§V4).
func ExampleAvailable() {
	if npu.Available() {
		fmt.Println("NPU op backend present")
	} else {
		fmt.Println("no NPU op backend — using CPU/GPU (NPU is a model-level, not op-level, target)")
	}
	// Output: no NPU op backend — using CPU/GPU (NPU is a model-level, not op-level, target)
}
