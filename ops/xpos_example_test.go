package ops_test

import (
	"fmt"

	"github.com/jxsl13/goai/ops"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// XPos applies the RoPE rotation then a per-pair magnitude ζ_i^(±n). For a query
// (downscale=false) at position 0 the scale ζ^0=1, so row 0 is plain RoPE; at
// position 1 the rotated vector is additionally scaled by ζ_0 = 0.4/1.4 ≈ 0.2857.
func ExampleXPos() {
	q := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, 0, 1, 0})
	out, _ := ops.XPos(q, 10000, false)
	fmt.Printf("%.4f %.4f\n", out.AtF64(0, 0), out.AtF64(1, 0))
	// Output: 1.0000 0.1544
}
