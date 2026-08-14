package tensor

import "fmt"

// Storage is the backing element buffer shared by a Tensor and all zero-copy
// views derived from it (§C5). It is type-erased: a runtime Dtype tag selects
// which typed slice `data` holds (F32→[]float32, F64→[]float64, F16/BF16→
// []uint16 raw bits). See docs/decisions/ADR-0001-type-erased-storage.md.
type Storage struct {
	dtype Dtype
	n     int       // element count
	data  any       // []float32 | []float64 | []uint16 (F16/BF16, tag disambiguates)
	alloc Allocator // owner; used by Release to return the buffer
}

// NewStorage allocates a zeroed heap buffer of n elements of dtype. It panics on
// an invalid dtype or negative n — both are programming errors, like make() with
// a negative length.
func NewStorage(dtype Dtype, n int) *Storage {
	return newStorageWith(Heap(), dtype, n)
}

// newStorageWith allocates n elements of dtype via the given allocator.
func newStorageWith(a Allocator, dtype Dtype, n int) *Storage {
	if n < 0 {
		panic("tensor: negative storage length")
	}
	return &Storage{dtype: dtype, n: n, data: a.Alloc(dtype, n), alloc: a}
}

// Release returns the backing buffer to its allocator for reuse and marks the
// storage empty. It invalidates every Tensor view sharing this storage, so use
// it only for scratch buffers with known lifetimes. Safe to call more than once.
func (s *Storage) Release() {
	if s.alloc != nil && s.data != nil {
		s.alloc.Free(s.data)
	}
	s.data = nil
	s.n = 0
}

// IsReleased reports whether Release has invalidated this storage. It is
// distinct from Len()==0: an allocated tensor with a zero-sized dimension has
// live typed storage of length zero and remains valid.
func (s *Storage) IsReleased() bool { return s == nil || s.data == nil }

// Len returns the number of elements.
func (s *Storage) Len() int { return s.n }

// Dtype returns the element type.
func (s *Storage) Dtype() Dtype { return s.dtype }

// F32 returns the backing slice; it panics if the dtype is not F32.
func (s *Storage) F32() []float32 {
	v, ok := s.data.([]float32)
	if !ok {
		panic(fmt.Sprintf("tensor: F32() on %v storage", s.dtype))
	}
	return v
}

// F64 returns the backing slice; it panics if the dtype is not F64.
func (s *Storage) F64() []float64 {
	v, ok := s.data.([]float64)
	if !ok {
		panic(fmt.Sprintf("tensor: F64() on %v storage", s.dtype))
	}
	return v
}

// U16 returns the raw 16-bit backing slice; it panics unless the dtype is a
// 16-bit float (F16/BF16). The elements are raw bit patterns, not values —
// interpret them via the dtype (§T27).
func (s *Storage) U16() []uint16 {
	v, ok := s.data.([]uint16)
	if !ok {
		panic(fmt.Sprintf("tensor: U16() on %v storage", s.dtype))
	}
	return v
}

// atF64 reads flat element i as float64, widening from the stored dtype. This is
// the uniform reference accessor used by tests and scalar kernels; it trades
// speed for dtype-agnostic correctness (§V9). For F16/BF16 the stored uint16 bits
// are expanded to float32 (exact, lossless) then to float64.
func (s *Storage) atF64(i int) float64 {
	switch d := s.data.(type) {
	case []float32:
		return float64(d[i])
	case []float64:
		return d[i]
	case []uint16:
		if s.dtype == BF16 {
			return float64(bf16ToF32(d[i]))
		}
		return float64(f16ToF32(d[i]))
	default:
		panic("tensor: uninitialized storage")
	}
}

// setF64 writes v into flat element i, narrowing to the stored dtype. For
// F16/BF16 the value is rounded to the 16-bit format (round-to-nearest-even).
func (s *Storage) setF64(i int, v float64) {
	switch d := s.data.(type) {
	case []float32:
		d[i] = float32(v)
	case []float64:
		d[i] = v
	case []uint16:
		if s.dtype == BF16 {
			d[i] = f32ToBF16(float32(v))
		} else {
			d[i] = f32ToF16(float32(v))
		}
	default:
		panic("tensor: uninitialized storage")
	}
}
