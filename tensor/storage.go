package tensor

import "fmt"

// Storage is the backing element buffer shared by a Tensor and all zero-copy
// views derived from it (§C5). It is type-erased: a runtime Dtype tag selects
// which typed slice `data` holds (F32→[]float32, F64→[]float64, F16/BF16→
// []uint16 raw bits). See docs/decisions/ADR-0001-type-erased-storage.md.
type Storage struct {
	dtype Dtype
	n     int // element count
	// Exactly one of these is non-nil, selected by dtype. They replaced a single
	// `data any`, which cost a runtime slice-to-interface box on EVERY allocation —
	// 24 bytes and one object that never recycles, even on a pooled device, where it
	// was one of only three allocations left per tensor (T1033/ADR-01KYX550H2F2J).
	// The struct grows in exchange; that trade was taken deliberately.
	f32   []float32
	f64   []float64
	u16   []uint16  // F16/BF16 raw bits; dtype disambiguates
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
	s := &Storage{dtype: dtype, n: n, alloc: a}
	// The typed path exists to avoid the interface box. Allocators in this repo all
	// implement it; a foreign Allocator keeps the exported any-returning contract
	// unchanged and simply pays the box it always did.
	if ta, ok := a.(typedAllocator); ok {
		switch dtype {
		case F32:
			s.f32 = ta.allocF32(n)
		case F64:
			s.f64 = ta.allocF64(n)
		case F16, BF16:
			s.u16 = ta.allocU16(n)
		default:
			panic(fmt.Sprintf("tensor: cannot alloc dtype %v", dtype))
		}
		return s
	}
	switch v := a.Alloc(dtype, n).(type) {
	case []float32:
		s.f32 = v
	case []float64:
		s.f64 = v
	case []uint16:
		s.u16 = v
	default:
		panic(fmt.Sprintf("tensor: allocator returned %T for dtype %v", v, dtype))
	}
	return s
}

// Release returns the backing buffer to its allocator for reuse and marks the
// storage empty. It invalidates every Tensor view sharing this storage, so use
// it only for scratch buffers with known lifetimes. Safe to call more than once.
func (s *Storage) Release() {
	if s.alloc != nil {
		if ta, ok := s.alloc.(typedAllocator); ok {
			// Typed free for the same reason as typed alloc: handing a slice to the
			// any-taking Free would re-box it and move the allocation rather than
			// remove it.
			switch {
			case s.f32 != nil:
				ta.freeF32(s.f32)
			case s.f64 != nil:
				ta.freeF64(s.f64)
			case s.u16 != nil:
				ta.freeU16(s.u16)
			}
		} else {
			switch {
			case s.f32 != nil:
				s.alloc.Free(s.f32)
			case s.f64 != nil:
				s.alloc.Free(s.f64)
			case s.u16 != nil:
				s.alloc.Free(s.u16)
			}
		}
	}
	s.f32, s.f64, s.u16 = nil, nil, nil
	s.n = 0
}

// Len returns the number of elements.
func (s *Storage) Len() int { return s.n }

// Dtype returns the element type.
func (s *Storage) Dtype() Dtype { return s.dtype }

// F32 returns the backing slice; it panics if the dtype is not F32.
func (s *Storage) F32() []float32 {
	if s.f32 == nil {
		panic(fmt.Sprintf("tensor: F32() on %v storage", s.dtype))
	}
	return s.f32
}

// F64 returns the backing slice; it panics if the dtype is not F64.
func (s *Storage) F64() []float64 {
	if s.f64 == nil {
		panic(fmt.Sprintf("tensor: F64() on %v storage", s.dtype))
	}
	return s.f64
}

// U16 returns the raw 16-bit backing slice; it panics unless the dtype is a
// 16-bit float (F16/BF16). The elements are raw bit patterns, not values —
// interpret them via the dtype (§T27).
func (s *Storage) U16() []uint16 {
	if s.u16 == nil {
		panic(fmt.Sprintf("tensor: U16() on %v storage", s.dtype))
	}
	return s.u16
}

// atF64 reads flat element i as float64, widening from the stored dtype. This is
// the uniform reference accessor used by tests and scalar kernels; it trades
// speed for dtype-agnostic correctness (§V9). For F16/BF16 the stored uint16 bits
// are expanded to float32 (exact, lossless) then to float64.
func (s *Storage) atF64(i int) float64 {
	switch {
	case s.f32 != nil:
		return float64(s.f32[i])
	case s.f64 != nil:
		return s.f64[i]
	case s.u16 != nil:
		if s.dtype == BF16 {
			return float64(bf16ToF32(s.u16[i]))
		}
		return float64(f16ToF32(s.u16[i]))
	default:
		panic("tensor: uninitialized storage")
	}
}

// setF64 writes v into flat element i, narrowing to the stored dtype. For
// F16/BF16 the value is rounded to the 16-bit format (round-to-nearest-even).
func (s *Storage) setF64(i int, v float64) {
	switch {
	case s.f32 != nil:
		s.f32[i] = float32(v)
	case s.f64 != nil:
		s.f64[i] = v
	case s.u16 != nil:
		if s.dtype == BF16 {
			s.u16[i] = f32ToBF16(float32(v))
		} else {
			s.u16[i] = f32ToF16(float32(v))
		}
	default:
		panic("tensor: uninitialized storage")
	}
}
