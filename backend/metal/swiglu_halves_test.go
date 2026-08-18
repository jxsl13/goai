//go:build darwin && cgo

package metal

import (
	"math"
	"slices"
	"testing"
)

func TestRecorderSwiGLUHalvesExact(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	for _, rows := range []int{24, 64} {
		t.Run(fmtInt(rows), func(t *testing.T) {
			const hidden = 37
			gateHost := make([]float32, rows*hidden)
			upHost := make([]float32, rows*hidden)
			combinedHost := make([]float32, rows*2*hidden)
			for row := range rows {
				for col := range hidden {
					i := row*hidden + col
					gateHost[i] = float32(math.Sin(float64(i)*0.031)) * 4
					upHost[i] = float32(math.Cos(float64(i)*0.017)) * 3
					combinedHost[row*2*hidden+col] = gateHost[i]
					combinedHost[row*2*hidden+hidden+col] = upHost[i]
				}
			}
			gate, err := NewDeviceBufferF32(gateHost)
			if err != nil {
				t.Fatal(err)
			}
			defer gate.Release()
			up, err := NewDeviceBufferF32(upHost)
			if err != nil {
				t.Fatal(err)
			}
			defer up.Release()
			combined, err := NewDeviceBufferF32(combinedHost)
			if err != nil {
				t.Fatal(err)
			}
			defer combined.Release()
			want, err := NewDeviceBufferF32(make([]float32, rows*hidden))
			if err != nil {
				t.Fatal(err)
			}
			defer want.Release()
			got, err := NewDeviceBufferF32(make([]float32, rows*hidden))
			if err != nil {
				t.Fatal(err)
			}
			defer got.Release()

			r, err := NewRecorder()
			if err != nil {
				t.Fatal(err)
			}
			if err := r.BinaryN(gate, up, want, 6, rows*hidden); err != nil {
				t.Fatal(err)
			}
			if err := r.SwiGLUHalves(combined, got, rows, hidden); err != nil {
				t.Fatal(err)
			}
			if err := r.Finish(); err != nil {
				t.Fatal(err)
			}
			r.Free()

			wantHost := make([]float32, rows*hidden)
			gotHost := make([]float32, rows*hidden)
			if err := want.DownloadF32(wantHost); err != nil {
				t.Fatal(err)
			}
			if err := got.DownloadF32(gotHost); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(gotHost, wantHost) {
				for i := range gotHost {
					if gotHost[i] != wantHost[i] {
						t.Fatalf("element %d = %08x, want %08x", i, math.Float32bits(gotHost[i]), math.Float32bits(wantHost[i]))
					}
				}
			}
		})
	}
}

func fmtInt(v int) string {
	if v == 24 {
		return "rows24"
	}
	return "rows64"
}

func TestRecorderSwiGLUHalvesValidation(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	gu, err := NewDeviceBufferF32(make([]float32, 16))
	if err != nil {
		t.Fatal(err)
	}
	defer gu.Release()
	out, err := NewDeviceBufferF32(make([]float32, 8))
	if err != nil {
		t.Fatal(err)
	}
	defer out.Release()
	r, err := NewRecorder()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Free()
	if err := r.SwiGLUHalves(gu, out, 0, 8); err == nil {
		t.Fatal("zero rows unexpectedly accepted")
	}
	if err := r.SwiGLUHalves(gu, out, 2, 8); err == nil {
		t.Fatal("undersized buffers unexpectedly accepted")
	}
}
