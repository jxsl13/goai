package gguf

import (
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/jxsl13/goai/tensor"
)

// Eight concurrent decode-shaped QMatMul calls, the batch-serving shape: each one is
// internally parallel over output rows, so the question is whether the internal fan-out
// multiplies by the number of callers or is bounded by a shared pool.
func TestGoroutinePeakConcurrentDecode(t *testing.T) {
	const n, k, callers = 4096, 4096, 8
	rowBytes, err := byteSize(uint32(Q8_0), k)
	if err != nil {
		t.Fatal(err)
	}
	w := make([]byte, n*rowBytes)
	for i := range w {
		w[i] = byte(i)
	}
	x := tensor.New(tensor.F32, tensor.Shape{1, k})
	xs := x.Storage().F32()
	for i := range xs {
		xs[i] = float32(i%17) * 0.01
	}
	var peak, stop int64
	done := make(chan struct{})
	go func() {
		defer close(done)
		for atomic.LoadInt64(&stop) == 0 {
			if g := int64(runtime.NumGoroutine()); g > atomic.LoadInt64(&peak) {
				atomic.StoreInt64(&peak, g)
			}
			time.Sleep(20 * time.Microsecond)
		}
	}()
	var wg sync.WaitGroup
	for range callers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 40 {
				if _, err := QMatMul(x, w, Q8_0, n, k); err != nil {
					t.Error(err)
					return
				}
			}
		}()
	}
	wg.Wait()
	atomic.StoreInt64(&stop, 1)
	<-done
	t.Logf("PEAK_GOROUTINES=%d callers=%d GOMAXPROCS=%d", atomic.LoadInt64(&peak), callers, runtime.GOMAXPROCS(0))
}

// Throughput under the same concurrent-caller shape the peak probe measures: bounding the
// internal fan-out must not cost aggregate throughput to be worth having.
func BenchmarkQMatMulQ8_0ConcurrentDecode(b *testing.B) {
	const n, k = 4096, 4096
	rowBytes, err := byteSize(uint32(Q8_0), k)
	if err != nil {
		b.Fatal(err)
	}
	w := make([]byte, n*rowBytes)
	for i := range w {
		w[i] = byte(i)
	}
	b.SetBytes(int64(n * rowBytes))
	b.RunParallel(func(pb *testing.PB) {
		x := tensor.New(tensor.F32, tensor.Shape{1, k})
		xs := x.Storage().F32()
		for i := range xs {
			xs[i] = float32(i%17) * 0.01
		}
		for pb.Next() {
			if _, err := QMatMul(x, w, Q8_0, n, k); err != nil {
				b.Error(err)
				return
			}
		}
	})
}
