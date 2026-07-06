package tensor

// Dtype identifies the element type of a Tensor.
//
// f32 and f64 ship first (§C4); f16, bf16 and integer/quantized types are
// appended later. The zero value is Invalid so an unset Dtype is never silently
// treated as a real type.
type Dtype uint8

const (
	Invalid Dtype = iota // zero value: unset / unknown
	F32                  // IEEE-754 binary32, 4 bytes
	F64                  // IEEE-754 binary64, 8 bytes
	F16                  // IEEE-754 binary16, 2 bytes (stored as raw uint16 bits)
	BF16                 // bfloat16 (1/8/7), 2 bytes (stored as raw uint16 bits)
)

// size in bytes per dtype, indexed by Dtype. Invalid → 0.
var dtypeSize = [...]int{
	Invalid: 0,
	F32:     4,
	F64:     8,
	F16:     2,
	BF16:    2,
}

var dtypeName = [...]string{
	Invalid: "invalid",
	F32:     "f32",
	F64:     "f64",
	F16:     "f16",
	BF16:    "bf16",
}

// Size returns the element size in bytes. Invalid returns 0.
func (d Dtype) Size() int {
	if int(d) >= len(dtypeSize) {
		return 0
	}
	return dtypeSize[d]
}

// IsValid reports whether d is a defined, usable dtype.
func (d Dtype) IsValid() bool { return d != Invalid && int(d) < len(dtypeName) }

// IsFloat reports whether d is a floating-point dtype.
func (d Dtype) IsFloat() bool { return d == F32 || d == F64 || d == F16 || d == BF16 }

// IsFloat16 reports whether d is a 16-bit float stored as raw uint16 bits.
func (d Dtype) IsFloat16() bool { return d == F16 || d == BF16 }

// String implements fmt.Stringer.
func (d Dtype) String() string {
	if int(d) >= len(dtypeName) {
		return "dtype(?)"
	}
	return dtypeName[d]
}
