//go:build darwin && cgo

package metal_test

import (
	"fmt"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
)

// TestMetalQ6KColdProfile measures first-use compilation separately from the
// warm leaf. Run it in fresh processes; the first call compiles the Metal
// library and both pipelines, while the second call reuses them.
func TestMetalQ6KColdProfile(t *testing.T) {
	if os.Getenv("GOAI_Q6K_COLD_PROFILE") == "" {
		t.Skip("set GOAI_Q6K_COLD_PROFILE=1")
	}
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	cooperative, err := strconv.ParseBool(os.Getenv("GOAI_Q6K_COOPERATIVE"))
	if err != nil {
		t.Fatalf("GOAI_Q6K_COOPERATIVE: %v", err)
	}
	previous := metal.SetQ6KCooperative(cooperative)
	defer metal.SetQ6KCooperative(previous)

	const k, n = 5632, 2048
	weight := syntheticQ6K(n, k)
	raw, err := metal.Backend{}.UploadQuant(weight, uint32(gguf.Q6_K), n, k)
	if err != nil {
		t.Fatal(err)
	}
	rw := raw.(*metal.ResidentQWeight)
	defer rw.Close()
	x, err := metal.NewDeviceBufferF32(make([]float32, k))
	if err != nil {
		t.Fatal(err)
	}
	defer x.Release()
	o, err := metal.NewDeviceBufferF32(make([]float32, n))
	if err != nil {
		t.Fatal(err)
	}
	defer o.Release()
	run := func() {
		r, err := metal.NewRecorder()
		if err != nil {
			t.Fatal(err)
		}
		defer r.Free()
		if err := r.QMatMulResident(x, rw, o, 1); err != nil {
			t.Fatal(err)
		}
		if err := r.Finish(); err != nil {
			t.Fatal(err)
		}
	}
	started := time.Now()
	run()
	cold := time.Since(started)
	started = time.Now()
	run()
	warm := time.Since(started)
	fmt.Printf("GOAI_Q6K_COLD cooperative=%t first=%s second=%s\n", cooperative, cold, warm)
}

// BenchmarkMetalQ6KDecodeLeaf measures the device-resident M=1 Q6_K seams in
// a TinyLlama Q4_K_M decode. It uses the same boundary as the Q4_K leaf:
// recorder creation, parameter encoding, command submission, and completion
// are timed; pipeline compilation and weight upload are not.
func BenchmarkMetalQ6KDecodeLeaf(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal device unavailable")
	}
	for _, tc := range []struct {
		name string
		k    int
		n    int
	}{
		{name: "K2048N256", k: 2048, n: 256},
		{name: "K2048N2048", k: 2048, n: 2048},
		{name: "K5632N2048", k: 5632, n: 2048},
		{name: "K2048N32000", k: 2048, n: 32000},
	} {
		b.Run(tc.name, func(b *testing.B) {
			for _, mode := range []struct {
				name        string
				cooperative bool
			}{{name: "cooperative", cooperative: true}, {name: "scalar", cooperative: false}} {
				b.Run(mode.name, func(b *testing.B) {
					previous := metal.SetQ6KCooperative(mode.cooperative)
					defer metal.SetQ6KCooperative(previous)
					weight := syntheticQ6K(tc.n, tc.k)
					raw, err := metal.Backend{}.UploadQuant(weight, uint32(gguf.Q6_K), tc.n, tc.k)
					if err != nil {
						b.Fatal(err)
					}
					rw := raw.(*metal.ResidentQWeight)
					defer rw.Close()
					x, err := metal.NewDeviceBufferF32(make([]float32, tc.k))
					if err != nil {
						b.Fatal(err)
					}
					defer x.Release()
					o, err := metal.NewDeviceBufferF32(make([]float32, tc.n))
					if err != nil {
						b.Fatal(err)
					}
					defer o.Release()

					run := func() {
						r, err := metal.NewRecorder()
						if err != nil {
							b.Fatal(err)
						}
						if err := r.QMatMulResident(x, rw, o, 1); err != nil {
							r.Free()
							b.Fatal(err)
						}
						if err := r.Finish(); err != nil {
							r.Free()
							b.Fatal(err)
						}
						r.Free()
					}
					for range 20 {
						run()
					}
					b.SetBytes(int64(len(weight) + 4*(tc.k+tc.n)))
					b.ReportAllocs()
					b.ResetTimer()
					for range b.N {
						run()
					}
				})
			}
		})
	}
}

func syntheticQ6K(n, k int) []byte {
	const blockBytes = 210
	blocks := n * (k / 256)
	out := make([]byte, blocks*blockBytes)
	for block := range blocks {
		base := block * blockBytes
		for i := 0; i < 208; i++ {
			out[base+i] = byte((block*17 + i*29) & 0xff)
		}
		out[base+208], out[base+209] = 0, 0x3c // f16(1)
	}
	return out
}
