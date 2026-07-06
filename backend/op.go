package backend

// Op identifies a primitive operation. Kernels are registered per (Op, Dtype)
// and dispatched through Execute. New ops are appended (monotonic); kernels for
// them are added by the reference and accel backends without changing the
// Backend interface (ADR-0003).
type Op int

const (
	OpInvalid Op = iota

	// elementwise binary
	OpAdd
	OpSub
	OpMul
	OpDiv

	// elementwise unary
	OpNeg
	OpExp
	OpLog
	OpTanh
	OpReLU
	OpGELU
	OpSigmoid
	OpSiLU

	// reductions
	OpSum
	OpMean
	OpMax
	OpMin
	OpArgMax

	// blas-1
	OpDot
	OpAXPY
	OpNrm2

	// blas-3
	OpMatMul

	// nn
	OpAddBias      // x[m,n] + b[n] row-broadcast (general broadcasting: §B18)
	OpCrossEntropy // fused stable log-softmax + NLL, mean over batch (ADR-0007)
	OpSoftmax      // stable max-shift softmax over the LAST axis
	OpLayerNorm    // layer norm over the last axis w/ gamma/beta (torch semantics)

	// cv (NCHW)
	OpConv2D    // cross-correlation, zero-padding, stride; optional bias input
	OpMaxPool2D // window max, stride defaults to kernel
	OpAvgPool2D // window mean, stride defaults to kernel

	// fused attention (heads split/concat/mask internal → one differentiable op)
	OpMHA // multi-head scaled dot-product attention: (Q,K,V)[seq,dmodel]→[seq,dmodel]

	OpEmbed // row gather: (table[n,d], idx[m]) → out[m,d]; grad scatter-adds to table

	// llama-family
	OpRMSNorm // x/√(mean(x²)+eps)·γ over last axis (no mean-sub, no bias)
	OpRoPE    // rotary position embedding (rotate_half), position = row index

	numOps
)

var opName = [...]string{
	OpInvalid:  "invalid",
	OpAdd:      "add",
	OpSub:      "sub",
	OpMul:      "mul",
	OpDiv:      "div",
	OpNeg:      "neg",
	OpExp:      "exp",
	OpLog:      "log",
	OpTanh:     "tanh",
	OpReLU:     "relu",
	OpGELU:     "gelu",
	OpSigmoid:  "sigmoid",
	OpSiLU:     "silu",
	OpSum:      "sum",
	OpMean:     "mean",
	OpMax:      "max",
	OpMin:      "min",
	OpArgMax:   "argmax",
	OpDot:      "dot",
	OpAXPY:     "axpy",
	OpNrm2:     "nrm2",
	OpMatMul:       "matmul",
	OpAddBias:      "addbias",
	OpCrossEntropy: "crossentropy",
	OpSoftmax:      "softmax",
	OpLayerNorm:    "layernorm",
	OpConv2D:       "conv2d",
	OpMaxPool2D:    "maxpool2d",
	OpAvgPool2D:    "avgpool2d",
	OpMHA:          "mha",
	OpEmbed:        "embed",
	OpRMSNorm:      "rmsnorm",
	OpRoPE:         "rope",
}

// String implements fmt.Stringer.
func (o Op) String() string {
	if o < 0 || int(o) >= len(opName) {
		return "op(?)"
	}
	return opName[o]
}

// Attrs carries op-specific parameters (e.g. reduction axis). Values are read
// once per op; typed accessors guard against misuse (ADR-0003).
type Attrs map[string]any

// Int returns the int attr for key, or def if absent/mistyped.
func (a Attrs) Int(key string, def int) int {
	if a != nil {
		if v, ok := a[key].(int); ok {
			return v
		}
	}
	return def
}

// Bool returns the bool attr for key, or def if absent/mistyped.
func (a Attrs) Bool(key string, def bool) bool {
	if a != nil {
		if v, ok := a[key].(bool); ok {
			return v
		}
	}
	return def
}

// Float returns the float64 attr for key, or def if absent/mistyped.
func (a Attrs) Float(key string, def float64) float64 {
	if a != nil {
		if v, ok := a[key].(float64); ok {
			return v
		}
	}
	return def
}

// Ints returns the []int attr for key, or nil if absent/mistyped.
func (a Attrs) Ints(key string) []int {
	if a != nil {
		if v, ok := a[key].([]int); ok {
			return v
		}
	}
	return nil
}
