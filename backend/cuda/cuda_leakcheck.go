//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

// LiveBufs reports how many caller-owned device buffers (allocated by the cu_upload_*/cu_alloc_*/
// cu_clone_* family behind DeviceF32, ResidentBF16, KV caches, the paged pool, etc.) are currently
// outstanding — i.e. handed to Go and not yet freed via Free()/cu_free_f32. It is the leak-detection
// hook the tests build on: snapshot it, run a workload that should net-zero its device allocations,
// GraphSync, and assert the count is unchanged. A positive delta is a leaked cgo device buffer.
//
// It counts BUFFERS, not bytes, and deliberately so: a byte count via cudaMemGetInfo cannot see
// through the cudaMallocAsync mempool (freed memory is cached, not returned to the driver), so it
// reports "no free" even when every buffer was released. The buffer counter is exact and mempool-
// independent. Internal scratch buffers freed inside a single C call never reach Go and are not
// counted.
func LiveBufs() int64 { return int64(C.cu_live_bufs()) }
