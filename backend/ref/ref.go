// Package ref is the Pure-Go scalar reference backend (layer L1) — the source of
// numeric truth that optimized backends are validated against (§V3, §V9). It is
// synchronous: work completes before Execute returns, so Synchronize is a no-op.
// Kernels are registered per (Op, Dtype) by the op files in this package
// (elementwise, reduce, blas — §T6..§T9).
package ref

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

type kernelKey struct {
	op    backend.Op
	dtype tensor.Dtype
}

// Backend is the reference implementation. Its kernel table is populated by the
// package's op files via add().
type Backend struct {
	table map[kernelKey]backend.Kernel
}

// std is the process-wide reference backend instance. Op files register their
// kernels into it from init(); it self-registers as the reference (§V9, §I4).
var std = &Backend{table: make(map[kernelKey]backend.Kernel)}

func init() { backend.RegisterReference(std) }

// add registers kernel k for (op, dtype). Called from op-file init()s; a
// duplicate is a programming error.
func (b *Backend) add(op backend.Op, dtype tensor.Dtype, k backend.Kernel) {
	key := kernelKey{op, dtype}
	if _, dup := b.table[key]; dup {
		panic("ref: duplicate kernel for " + op.String() + "/" + dtype.String())
	}
	b.table[key] = k
}

// Name implements backend.Backend.
func (b *Backend) Name() backend.Name { return backend.Ref }

// Device returns the host CPU device.
func (b *Backend) Device() tensor.Device { return tensor.CPU() }

// Kernel looks up the kernel for (op, dtype).
func (b *Backend) Kernel(op backend.Op, dtype tensor.Dtype) (backend.Kernel, bool) {
	k, ok := b.table[kernelKey{op, dtype}]
	return k, ok
}

// Synchronize is a no-op: the reference backend is synchronous (§V14).
func (b *Backend) Synchronize() error { return nil }
