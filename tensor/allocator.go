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

func (heapAllocator) Free(any)      {}
func (heapAllocator) Alignment() int { return 0 } // natural

// --- Pool: sync.Pool per (dtype, power-of-two size class) ---

// Pool is a reuse-oriented allocator that recycles buffers by power-of-two size
// class. Safe for concurrent use. Alignment is advisory (ADR-0002).
type Pool struct {
	align int
	mu    sync.Mutex
	pools map[uint64]*sync.Pool // key: dtype<<32 | class
}

// PoolOption configures a Pool.
type PoolOption func(*Pool)

// WithAlignment records an advisory byte alignment (see ADR-0002). It does not
// currently change the layout — it is a forward hook for §T11.
func WithAlignment(bytes int) PoolOption {
	return func(p *Pool) { p.align = bytes }
}

// NewPool builds a pooling allocator.
func NewPool(opts ...PoolOption) *Pool {
	p := &Pool{pools: make(map[uint64]*sync.Pool)}
	for _, o := range opts {
		o(p)
	}
	return p
}

func (p *Pool) Alignment() int { return p.align }

// sizeClass returns the exponent c such that 1<<c is the smallest power of two
// >= n (with n>=1). n==1 → 0 (cap 1).
func sizeClass(n int) int { return bits.Len(uint(n - 1)) }

func (p *Pool) poolFor(dtype Dtype, class int) *sync.Pool {
	key := uint64(dtype)<<32 | uint64(uint32(class))
	p.mu.Lock()
	sp := p.pools[key]
	if sp == nil {
		sp = &sync.Pool{}
		p.pools[key] = sp
	}
	p.mu.Unlock()
	return sp
}

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
	sp := p.poolFor(dtype, class)
	v := sp.Get()
	switch dtype {
	case F32:
		var s []float32
		if v == nil {
			s = make([]float32, capacity)
		} else {
			s = v.([]float32)
		}
		s = s[:n]
		clear(s)
		return s
	case F64:
		var s []float64
		if v == nil {
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
// power of two (foreign) are dropped rather than mis-filed.
func (p *Pool) Free(buf any) {
	switch b := buf.(type) {
	case []float32:
		c := cap(b)
		if c == 0 || c&(c-1) != 0 {
			return
		}
		p.poolFor(F32, bits.Len(uint(c-1))).Put(b[:c]) //nolint:staticcheck // pooling a slice
	case []float64:
		c := cap(b)
		if c == 0 || c&(c-1) != 0 {
			return
		}
		p.poolFor(F64, bits.Len(uint(c-1))).Put(b[:c]) //nolint:staticcheck
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
