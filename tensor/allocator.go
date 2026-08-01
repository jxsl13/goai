package tensor

import (
	"fmt"
	"math/bits"
	"sync"
)

// Allocator provides element buffers for Storage. Alloc returns a zeroed typed
// slice (F32→[]float32, F64→[]float64) of length n; Free optionally returns a
// buffer for reuse. Implementations must return zeroed memory so callers never
// observe stale pooled data. See docs/decisions/ADR-0002-allocator-alignment.md.
type Allocator interface {
	Alloc(dtype Dtype, n int) any
	Free(buf any)
	// Alignment reports the requested (advisory) byte alignment. L0 honors only
	// natural element alignment; guaranteed over-alignment is deferred to §T11.
	Alignment() int
}

// --- Heap: make-based, no pooling ---

type heapAllocator struct{}

// Heap returns the default allocator: make-backed, Free is a no-op.
func Heap() Allocator { return heapAllocator{} }

func (heapAllocator) Alloc(dtype Dtype, n int) any {
	if n < 0 {
		panic("tensor: negative alloc length")
	}
	switch dtype {
	case F32:
		return make([]float32, n)
	case F64:
		return make([]float64, n)
	case F16, BF16:
		return make([]uint16, n)
	default:
		panic(fmt.Sprintf("tensor: cannot alloc dtype %v", dtype))
	}
}

// typedAllocator is the box-free twin of Allocator, implemented by every allocator in this
// repository. Alloc returns `any`, so handing back a slice costs a runtime slice-to-interface
// box — 24 bytes and one object per allocation that no pool can recycle. These methods return
// and take the concrete slice instead. The EXPORTED Allocator contract is unchanged, and a
// foreign allocator that does not implement this simply keeps paying the box it always did
// (T1033/ADR-01KYX550H2F2J).
type typedAllocator interface {
	allocF32(n int) []float32
	allocF64(n int) []float64
	allocU16(n int) []uint16
	freeF32([]float32)
	freeF64([]float64)
	freeU16([]uint16)
}

func (heapAllocator) allocF32(n int) []float32 { return make([]float32, negCheck(n)) }
func (heapAllocator) allocF64(n int) []float64 { return make([]float64, negCheck(n)) }
func (heapAllocator) allocU16(n int) []uint16  { return make([]uint16, negCheck(n)) }
func (heapAllocator) freeF32([]float32)        {}
func (heapAllocator) freeF64([]float64)        {}
func (heapAllocator) freeU16([]uint16)         {}

func negCheck(n int) int {
	if n < 0 {
		panic("tensor: negative alloc length")
	}
	return n
}

func (heapAllocator) Free(any)       {}
func (heapAllocator) Alignment() int { return 0 } // natural

// --- Pool: sync.Pool per (dtype, power-of-two size class) ---

// Pool is a reuse-oriented allocator that recycles buffers by power-of-two size
// class. Safe for concurrent use. Alignment is advisory (ADR-0002).
//
// Size classes are indexed directly into fixed per-dtype arrays (§base-perf:
// the previous map[uint64]*sync.Pool guarded by a mutex serialized every
// Alloc/Free; a class can never exceed bits.UintSize, so a flat array needs no
// locking — sync.Pool itself is concurrency-safe).
type Pool struct {
	align int
	f32   [bits.UintSize]sync.Pool // index: size class (1<<class capacity)
	f64   [bits.UintSize]sync.Pool
}

// PoolOption configures a Pool.
type PoolOption func(*Pool)

// WithAlignment records an advisory byte alignment for pooled buffers (see ADR-0002).
//
// In plain terms: a request that buffers begin on an address that is a multiple of `bytes` —
// which SIMD/GPU code paths prefer for aligned loads. Boundary behavior — this is currently a
// NO-OP forward hook: L0 honors only natural element alignment, and guaranteed over-alignment
// is deferred to §T11, so the value is RECORDED (readable via Alignment) but does not yet
// change the layout. SPECIAL VALUE: 0 / unset = natural element alignment only.
//
// Default unset (natural alignment): no research-grounded value applies until the hook is
// active; a typical target once implemented is 64 bytes (a cache line / AVX-512 width).
func WithAlignment(bytes int) PoolOption {
	return func(p *Pool) { p.align = bytes }
}

// NewPool builds a pooling allocator.
func NewPool(opts ...PoolOption) *Pool {
	p := &Pool{}
	for _, o := range opts {
		o(p)
	}
	return p
}

// Alignment returns the pool's byte alignment for allocations.
func (p *Pool) Alignment() int { return p.align }

// sizeClass returns the exponent c such that 1<<c is the smallest power of two
// >= n (with n>=1). n==1 → 0 (cap 1).
func sizeClass(n int) int { return bits.Len(uint(n - 1)) }

// Alloc returns a pooled backing slice for n elements of the given dtype (rounded
// up to a power-of-two size class).
func (p *Pool) Alloc(dtype Dtype, n int) any {
	if n < 0 {
		panic("tensor: negative alloc length")
	}
	if n == 0 {
		return emptyBuf(dtype)
	}
	// F16/BF16 share a []uint16 layout, so Free cannot tell them apart to re-file
	// them by dtype; allocate fresh rather than mis-pool (§T27). Callers needing
	// pooled half buffers get natural GC.
	if dtype == F16 || dtype == BF16 {
		return make([]uint16, n)
	}
	class := sizeClass(n)
	capacity := 1 << class
	switch dtype {
	case F32:
		var s []float32
		if v := p.f32[class].Get(); v == nil {
			s = make([]float32, capacity)
		} else {
			// Pooled entries keep whatever length Free received; only the
			// (power-of-two) capacity is class-invariant, so reslice via cap.
			s = v.([]float32)
		}
		s = s[:n]
		clear(s)
		return s
	case F64:
		var s []float64
		if v := p.f64[class].Get(); v == nil {
			s = make([]float64, capacity)
		} else {
			s = v.([]float64)
		}
		s = s[:n]
		clear(s)
		return s
	default:
		panic(fmt.Sprintf("tensor: cannot alloc dtype %v", dtype))
	}
}

// Free returns buf to its size-class pool. Buffers whose capacity is not a
// power of two (foreign) are dropped rather than mis-filed. The incoming `any`
// box is stored as-is (§base-perf: re-slicing to b[:c] re-boxed the slice
// header, one heap alloc per Free; Alloc reslices by capacity instead).
func (p *Pool) Free(buf any) {
	switch b := buf.(type) {
	case []float32:
		c := cap(b)
		if c == 0 || c&(c-1) != 0 {
			return
		}
		p.f32[bits.Len(uint(c-1))].Put(buf) //nolint:staticcheck // pooling a slice
	case []float64:
		c := cap(b)
		if c == 0 || c&(c-1) != 0 {
			return
		}
		p.f64[bits.Len(uint(c-1))].Put(buf) //nolint:staticcheck
	}
}

func emptyBuf(dtype Dtype) any {
	switch dtype {
	case F32:
		return []float32{}
	case F64:
		return []float64{}
	case F16, BF16:
		return []uint16{}
	default:
		panic(fmt.Sprintf("tensor: cannot alloc dtype %v", dtype))
	}
}

// allocF32/allocF64/allocU16 and their free twins are the box-free path for Pool; the logic
// mirrors Alloc/Free exactly, including the F16/BF16 carve-out where a shared []uint16 layout
// makes Free unable to re-file by dtype.
func (p *Pool) allocF32(n int) []float32 {
	if negCheck(n) == 0 {
		return []float32{}
	}
	class := sizeClass(n)
	var s []float32
	if v := p.f32[class].Get(); v == nil {
		s = make([]float32, 1<<class)
	} else {
		s = v.([]float32)
	}
	s = s[:n]
	clear(s)
	return s
}

func (p *Pool) allocF64(n int) []float64 {
	if negCheck(n) == 0 {
		return []float64{}
	}
	class := sizeClass(n)
	var s []float64
	if v := p.f64[class].Get(); v == nil {
		s = make([]float64, 1<<class)
	} else {
		s = v.([]float64)
	}
	s = s[:n]
	clear(s)
	return s
}

// allocU16 does not pool: F16 and BF16 share a []uint16 layout, so a freed buffer cannot be
// re-filed by dtype. Same carve-out Alloc makes.
func (p *Pool) allocU16(n int) []uint16 { return make([]uint16, negCheck(n)) }

func (p *Pool) freeF32(b []float32) {
	if c := cap(b); c != 0 && c&(c-1) == 0 {
		p.f32[bits.Len(uint(c-1))].Put(b) //nolint:staticcheck // pooling a slice
	}
}

func (p *Pool) freeF64(b []float64) {
	if c := cap(b); c != 0 && c&(c-1) == 0 {
		p.f64[bits.Len(uint(c-1))].Put(b) //nolint:staticcheck
	}
}

func (p *Pool) freeU16([]uint16) {}
