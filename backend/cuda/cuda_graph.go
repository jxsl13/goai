//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// CapturedGraph is an instantiated CUDA graph that replays a previously captured
// device-op sequence with a single launch — the fix for launch-bound decode,
// where the per-token op chain is otherwise ~286 individual host→driver launches.
type CapturedGraph struct{ exec unsafe.Pointer }

// CaptureBegin starts recording device ops issued on the work stream into a graph.
// The calling goroutine MUST be pinned with runtime.LockOSThread from before
// CaptureBegin until CaptureEnd, so thread-local capture records every launch
// (Go goroutines migrate OS threads otherwise). Ops issued between begin and end
// must be capture-safe: async on the stream, no host sync, no plain cudaMalloc.
func CaptureBegin() error {
	if rc := C.cu_capture_begin(); rc != 0 {
		return fmt.Errorf("cuda: capture begin failed (code %d)", int(rc))
	}
	return nil
}

// CaptureEnd stops capture and instantiates the executable graph.
func CaptureEnd() (*CapturedGraph, error) {
	exec := C.cu_capture_end()
	if exec == nil {
		return nil, fmt.Errorf("cuda: capture end / instantiate failed")
	}
	return &CapturedGraph{exec: exec}, nil
}

// Launch replays the captured op sequence on the work stream (asynchronous;
// call GraphSync to wait).
func (g *CapturedGraph) Launch() error {
	if g.exec == nil {
		return fmt.Errorf("cuda: launch on a freed graph")
	}
	if rc := C.cu_graph_launch(g.exec); rc != 0 {
		return fmt.Errorf("cuda: graph launch failed (code %d)", int(rc))
	}
	return nil
}

// GraphSync blocks until all work queued on the stream (including graph launches)
// has completed.
func GraphSync() error {
	if rc := C.cu_graph_sync(); rc != 0 {
		return fmt.Errorf("cuda: graph sync failed (code %d)", int(rc))
	}
	return nil
}

// Free releases the instantiated graph.
func (g *CapturedGraph) Free() {
	if g.exec != nil {
		C.cu_graph_free(g.exec)
		g.exec = nil
	}
}
