//go:build darwin && cgo

package metal_test

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
)

// TestMetalQ4KColdProfile is a process-level cold/warm probe for T988. It is
// gated because the first call deliberately includes runtime Metal library and
// pipeline compilation; run it in fresh processes for independent samples.
func TestMetalQ4KColdProfile(t *testing.T) {
	if os.Getenv("GOAI_Q4K_COLD_PROFILE") == "" {
		t.Skip("set GOAI_Q4K_COLD_PROFILE=1")
	}
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	cooperative, err := strconv.ParseBool(os.Getenv("GOAI_Q4K_COOPERATIVE"))
	if err != nil {
		t.Fatalf("GOAI_Q4K_COOPERATIVE: %v", err)
	}
	previous := metal.SetQ4KCooperative(cooperative)
	defer metal.SetQ4KCooperative(previous)

	const k, n = 2048, 5632
	weight := syntheticQ4K(n, k)
	raw, err := metal.Backend{}.UploadQuant(weight, uint32(gguf.Q4_K), n, k)
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
	fmt.Printf("GOAI_Q4K_COLD cooperative=%t first=%s second=%s\n", cooperative, cold, warm)
}

// BenchmarkMetalQ4KDecodeLeaf measures the device-resident M=1 Q4_K seam used
// by every quantized decode projection. Recorder creation, parameter encoding,
// command submission, and completion remain inside the timer because a real
// decode step pays them; weight upload and pipeline warm-up do not.
func BenchmarkMetalQ4KDecodeLeaf(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal device unavailable")
	}
	for _, tc := range []struct {
		name string
		k    int
		n    int
	}{
		{name: "K2048N2048", k: 2048, n: 2048},
		{name: "K2048N2560", k: 2048, n: 2560},
		{name: "K2048N5632", k: 2048, n: 5632},
		{name: "K5632N2048", k: 5632, n: 2048},
		{name: "K2048N11008", k: 2048, n: 11008},
		{name: "K2048N16384", k: 2048, n: 16384},
		{name: "K2048N32000", k: 2048, n: 32000},
	} {
		b.Run(tc.name, func(b *testing.B) {
			for _, mode := range []struct {
				name        string
				cooperative bool
			}{{name: "cooperative", cooperative: true}, {name: "scalar", cooperative: false}} {
				b.Run(mode.name, func(b *testing.B) {
					previous := metal.SetQ4KCooperative(mode.cooperative)
					defer metal.SetQ4KCooperative(previous)
					weight := syntheticQ4K(tc.n, tc.k)
					raw, err := metal.Backend{}.UploadQuant(weight, uint32(gguf.Q4_K), tc.n, tc.k)
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
						run() // compile the selected pipeline and reach warm GPU clocks
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

// BenchmarkMetalQ4KGateUpFusion measures the exact TinyLlama SwiGLU projection
// seam in one command buffer. The baseline dispatches two resident
// K2048,N5632 Q4_K matvecs from the same input. The candidate concatenates the
// packed weight rows into one K2048,N11264 matvec and extracts its two bands.
func BenchmarkMetalQ4KGateUpFusion(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal device unavailable")
	}
	const k, n = 2048, 5632
	wGate := syntheticQ4K(n, k)
	wUp := syntheticQ4K(n, k)
	wFused := make([]byte, 0, len(wGate)+len(wUp))
	wFused = append(wFused, wGate...)
	wFused = append(wFused, wUp...)
	upload := func(weight []byte, rows int) *metal.ResidentQWeight {
		raw, err := metal.Backend{}.UploadQuant(weight, uint32(gguf.Q4_K), rows, k)
		if err != nil {
			b.Fatal(err)
		}
		return raw.(*metal.ResidentQWeight)
	}
	gateWeight, upWeight, fusedWeight := upload(wGate, n), upload(wUp, n), upload(wFused, 2*n)
	defer gateWeight.Close()
	defer upWeight.Close()
	defer fusedWeight.Close()
	x, err := metal.NewDeviceBufferF32(make([]float32, k))
	if err != nil {
		b.Fatal(err)
	}
	defer x.Release()
	gate, err := metal.NewDeviceBufferF32(make([]float32, n))
	if err != nil {
		b.Fatal(err)
	}
	defer gate.Release()
	up, err := metal.NewDeviceBufferF32(make([]float32, n))
	if err != nil {
		b.Fatal(err)
	}
	defer up.Release()
	combined, err := metal.NewDeviceBufferF32(make([]float32, 2*n))
	if err != nil {
		b.Fatal(err)
	}
	defer combined.Release()
	previous := metal.SetQ4KCooperative(true)
	defer metal.SetQ4KCooperative(previous)

	for _, mode := range []string{"separate", "fused"} {
		b.Run(mode, func(b *testing.B) {
			run := func() {
				r, err := metal.NewRecorder()
				if err != nil {
					b.Fatal(err)
				}
				if mode == "separate" {
					err = r.QMatMulResident(x, gateWeight, gate, 1)
					if err == nil {
						err = r.QMatMulResident(x, upWeight, up, 1)
					}
				} else {
					err = r.QMatMulResident(x, fusedWeight, combined, 1)
					if err == nil {
						err = r.Copy2D(combined, 0, 2*n, gate, 0, n, 1, n)
					}
					if err == nil {
						err = r.Copy2D(combined, n, 2*n, up, 0, n, 1, n)
					}
				}
				if err == nil {
					err = r.Finish()
				}
				r.Free()
				if err != nil {
					b.Fatal(err)
				}
			}
			for range 20 {
				run()
			}
			b.SetBytes(int64(len(wFused) + 4*(k+2*n)))
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				run()
			}
		})
	}
}

func TestMetalQ4KGateUpFusionMatchesSeparate(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	const k, n = 2048, 5632
	wGate := syntheticQ4K(n, k)
	wUp := syntheticQ4K(n, k)
	wFused := append(append(make([]byte, 0, len(wGate)+len(wUp)), wGate...), wUp...)
	upload := func(weight []byte, rows int) *metal.ResidentQWeight {
		raw, err := metal.Backend{}.UploadQuant(weight, uint32(gguf.Q4_K), rows, k)
		if err != nil {
			t.Fatal(err)
		}
		return raw.(*metal.ResidentQWeight)
	}
	gateWeight, upWeight, fusedWeight := upload(wGate, n), upload(wUp, n), upload(wFused, 2*n)
	defer gateWeight.Close()
	defer upWeight.Close()
	defer fusedWeight.Close()
	xHost := make([]float32, k)
	for i := range xHost {
		xHost[i] = float32(math.Sin(float64(i+1) * 0.013))
	}
	x, err := metal.NewDeviceBufferF32(xHost)
	if err != nil {
		t.Fatal(err)
	}
	defer x.Release()
	newOutput := func(size int) *metal.DeviceBuffer {
		buf, err := metal.NewDeviceBufferF32(make([]float32, size))
		if err != nil {
			t.Fatal(err)
		}
		return buf
	}
	separateGate, separateUp := newOutput(n), newOutput(n)
	fusedGate, fusedUp, combined := newOutput(n), newOutput(n), newOutput(2*n)
	defer separateGate.Release()
	defer separateUp.Release()
	defer fusedGate.Release()
	defer fusedUp.Release()
	defer combined.Release()
	previous := metal.SetQ4KCooperative(true)
	defer metal.SetQ4KCooperative(previous)

	finish := func(record func(*metal.Recorder) error) {
		r, err := metal.NewRecorder()
		if err != nil {
			t.Fatal(err)
		}
		defer r.Free()
		if err := record(r); err != nil {
			t.Fatal(err)
		}
		if err := r.Finish(); err != nil {
			t.Fatal(err)
		}
	}
	finish(func(r *metal.Recorder) error {
		if err := r.QMatMulResident(x, gateWeight, separateGate, 1); err != nil {
			return err
		}
		return r.QMatMulResident(x, upWeight, separateUp, 1)
	})
	finish(func(r *metal.Recorder) error {
		if err := r.QMatMulResident(x, fusedWeight, combined, 1); err != nil {
			return err
		}
		if err := r.Copy2D(combined, 0, 2*n, fusedGate, 0, n, 1, n); err != nil {
			return err
		}
		return r.Copy2D(combined, n, 2*n, fusedUp, 0, n, 1, n)
	})
	for name, pair := range map[string][2]*metal.DeviceBuffer{
		"gate": {separateGate, fusedGate},
		"up":   {separateUp, fusedUp},
	} {
		want, got := make([]float32, n), make([]float32, n)
		if err := pair[0].DownloadF32(want); err != nil {
			t.Fatal(err)
		}
		if err := pair[1].DownloadF32(got); err != nil {
			t.Fatal(err)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("%s[%d] fused=%g separate=%g", name, i, got[i], want[i])
			}
		}
	}
}

func syntheticQ4K(n, k int) []byte {
	const blockBytes = 144
	blocks := n * (k / 256)
	out := make([]byte, blocks*blockBytes)
	for block := range blocks {
		base := block * blockBytes
		out[base], out[base+1] = 0, 0x3c   // f16(1)
		out[base+2], out[base+3] = 0, 0x38 // f16(0.5)
		for i := 4; i < blockBytes; i++ {
			out[base+i] = byte((block*17 + i*29) & 0xff)
		}
	}
	return out
}
