// Package cpu is the optimized Pure-Go CPU backend (layer L1b). It computes the
// same results as backend/ref (the truth, §V9) but fast: contiguous typed-slice
// loops with no per-element allocation, goroutine parallelism above a threshold,
// and SIMD-class primitives from internal/simd (amd64 archsimd override in §T11b,
// ADR-0005). It registers as the preferred Default; ref stays the fallback (§I4).
package cpu

import (
	"runtime"
	"sync"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

type kernelKey struct {
	op    backend.Op
	dtype tensor.Dtype
}

// Backend is the optimized CPU backend. Synchronous, so Synchronize is a no-op.
type Backend struct {
	table map[kernelKey]backend.Kernel
}

var std = &Backend{table: make(map[kernelKey]backend.Kernel)}

func init() { backend.RegisterDefault(std) }

func (b *Backend) add(op backend.Op, dtype tensor.Dtype, k backend.Kernel) {
	b.table[kernelKey{op, dtype}] = k
}

func (b *Backend) Name() string                    { return "cpu" }
func (b *Backend) Device() tensor.Device           { return tensor.CPU() }
func (b *Backend) Synchronize() error              { return nil }
func (b *Backend) Kernel(op backend.Op, dtype tensor.Dtype) (backend.Kernel, bool) {
	k, ok := b.table[kernelKey{op, dtype}]
	return k, ok
}

// parThreshold is the total work (items × work-per-item) below which parallelism
// is not worth the goroutine overhead. Tuned conservatively; refined by
// benchmarks (§V5).
const parThreshold = 1 << 15 // 32768

// parallelWork splits [0,n) into chunks across GOMAXPROCS workers and runs body
// on each, or runs body once inline when the estimated total work
// (n × workPerItem) is small. body must be safe on disjoint sub-ranges.
func parallelWork(n, workPerItem int, body func(lo, hi int)) {
	workers := runtime.GOMAXPROCS(0)
	if workers <= 1 || n*workPerItem < parThreshold {
		body(0, n)
		return
	}
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for lo := 0; lo < n; lo += chunk {
		hi := min(lo+chunk, n)
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			body(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}

// parallel splits [0,n) with unit work per item (elementwise use).
func parallel(n int, body func(lo, hi int)) { parallelWork(n, 1, body) }
