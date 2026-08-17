package safetensors

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// buildFile assembles a minimal single-tensor safetensors byte stream around raw
// element data — the loader's own Save cannot write the widened dtypes (§T577), so
// tests construct the on-disk form directly.
func buildFile(dtype string, shape []int, data []byte) []byte {
	dims := make([]string, len(shape))
	for i, d := range shape {
		dims[i] = strconv.Itoa(d)
	}
	header := fmt.Sprintf(`{"x":{"dtype":%q,"shape":[%s],"data_offsets":[0,%d]}}`,
		dtype, strings.Join(dims, ","), len(data))
	if pad := (8 - (len(header) % 8)) % 8; pad > 0 {
		header += strings.Repeat(" ", pad)
	}
	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.LittleEndian, uint64(len(header)))
	buf.WriteString(header)
	buf.Write(data)
	return buf.Bytes()
}

// torch reference decodings of all 256 byte values (§T577, §V16 tier-1):
// torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn/e5m2).float(),
// torch 2.x. "nan"/"+inf"/"-inf" mark the specials.
const torchE4M3 = `0.0,0.001953125,0.00390625,0.005859375,0.0078125,0.009765625,0.01171875,0.013671875,0.015625,0.017578125,0.01953125,0.021484375,0.0234375,0.025390625,0.02734375,0.029296875,0.03125,0.03515625,0.0390625,0.04296875,0.046875,0.05078125,0.0546875,0.05859375,0.0625,0.0703125,0.078125,0.0859375,0.09375,0.1015625,0.109375,0.1171875,0.125,0.140625,0.15625,0.171875,0.1875,0.203125,0.21875,0.234375,0.25,0.28125,0.3125,0.34375,0.375,0.40625,0.4375,0.46875,0.5,0.5625,0.625,0.6875,0.75,0.8125,0.875,0.9375,1.0,1.125,1.25,1.375,1.5,1.625,1.75,1.875,2.0,2.25,2.5,2.75,3.0,3.25,3.5,3.75,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,18.0,20.0,22.0,24.0,26.0,28.0,30.0,32.0,36.0,40.0,44.0,48.0,52.0,56.0,60.0,64.0,72.0,80.0,88.0,96.0,104.0,112.0,120.0,128.0,144.0,160.0,176.0,192.0,208.0,224.0,240.0,256.0,288.0,320.0,352.0,384.0,416.0,448.0,nan` // positive half; negatives mirror

const torchE5M2 = `0.0,1.52587890625e-05,3.0517578125e-05,4.57763671875e-05,6.103515625e-05,7.62939453125e-05,9.1552734375e-05,0.0001068115234375,0.0001220703125,0.000152587890625,0.00018310546875,0.000213623046875,0.000244140625,0.00030517578125,0.0003662109375,0.00042724609375,0.00048828125,0.0006103515625,0.000732421875,0.0008544921875,0.0009765625,0.001220703125,0.00146484375,0.001708984375,0.001953125,0.00244140625,0.0029296875,0.00341796875,0.00390625,0.0048828125,0.005859375,0.0068359375,0.0078125,0.009765625,0.01171875,0.013671875,0.015625,0.01953125,0.0234375,0.02734375,0.03125,0.0390625,0.046875,0.0546875,0.0625,0.078125,0.09375,0.109375,0.125,0.15625,0.1875,0.21875,0.25,0.3125,0.375,0.4375,0.5,0.625,0.75,0.875,1.0,1.25,1.5,1.75,2.0,2.5,3.0,3.5,4.0,5.0,6.0,7.0,8.0,10.0,12.0,14.0,16.0,20.0,24.0,28.0,32.0,40.0,48.0,56.0,64.0,80.0,96.0,112.0,128.0,160.0,192.0,224.0,256.0,320.0,384.0,448.0,512.0,640.0,768.0,896.0,1024.0,1280.0,1536.0,1792.0,2048.0,2560.0,3072.0,3584.0,4096.0,5120.0,6144.0,7168.0,8192.0,10240.0,12288.0,14336.0,16384.0,20480.0,24576.0,28672.0,32768.0,40960.0,49152.0,57344.0,+inf,nan,nan,nan` // positive half; negatives mirror

func parseTorchTable(t *testing.T, csv string) []float64 {
	t.Helper()
	fields := strings.Split(csv, ",")
	if len(fields) != 128 {
		t.Fatalf("golden table has %d entries, want 128", len(fields))
	}
	out := make([]float64, 128)
	for i, f := range fields {
		switch f {
		case "nan":
			out[i] = math.NaN()
		case "+inf":
			out[i] = math.Inf(1)
		case "-inf":
			out[i] = math.Inf(-1)
		default:
			v, err := strconv.ParseFloat(f, 64)
			if err != nil {
				t.Fatal(err)
			}
			out[i] = v
		}
	}
	return out
}

// §V16 tier-1 (§T577): every one of the 256 possible FP8 bytes decodes exactly to the
// torch reference value — the positive half from the golden table, the negative half by
// the formats' sign-symmetric encoding (torch's tables mirror identically, including -0).
func TestLoadFP8AgainstTorchGoldens(t *testing.T) {
	for _, tc := range []struct {
		dtype string
		table string
	}{
		{"F8_E4M3", torchE4M3},
		{"F8_E5M2", torchE5M2},
	} {
		want := parseTorchTable(t, tc.table)
		data := make([]byte, 256)
		for i := range data {
			data[i] = byte(i)
		}
		tensors, _, err := Load(bytes.NewReader(buildFile(tc.dtype, []int{256}, data)))
		if err != nil {
			t.Fatalf("%s: %v", tc.dtype, err)
		}
		got := tensors["x"].Storage().F32()
		for b := 0; b < 256; b++ {
			w := want[b%128]
			if b >= 128 {
				w = -w // sign bit set: mirror (NaN stays NaN; -0.0 for byte 0x80)
			}
			g := float64(got[b])
			switch {
			case math.IsNaN(w):
				if !math.IsNaN(g) {
					t.Errorf("%s byte 0x%02x: got %v, want NaN", tc.dtype, b, g)
				}
			default:
				if g != w || math.Signbit(g) != math.Signbit(w) {
					t.Errorf("%s byte 0x%02x: got %v, want %v", tc.dtype, b, g, w)
				}
			}
		}
	}
}

// §V16 tier-1 (§T577): integer and BOOL dtypes widen exactly, including the extreme
// values of every width; I64/U64 widen exactly through 2^53.
func TestLoadIntegerWidening(t *testing.T) {
	le := binary.LittleEndian
	u16 := func(vs ...uint16) []byte {
		b := make([]byte, 2*len(vs))
		for i, v := range vs {
			le.PutUint16(b[i*2:], v)
		}
		return b
	}
	u32 := func(vs ...uint32) []byte {
		b := make([]byte, 4*len(vs))
		for i, v := range vs {
			le.PutUint32(b[i*4:], v)
		}
		return b
	}
	u64 := func(vs ...uint64) []byte {
		b := make([]byte, 8*len(vs))
		for i, v := range vs {
			le.PutUint64(b[i*8:], v)
		}
		return b
	}
	cases := []struct {
		dtype string
		data  []byte
		want  []float64
	}{
		{"BOOL", []byte{0, 1, 7}, []float64{0, 1, 1}},
		{"I8", []byte{0x80, 0xFF, 0x7F}, []float64{-128, -1, 127}},
		{"U8", []byte{0, 1, 0xFF}, []float64{0, 1, 255}},
		{"I16", u16(0x8000, 0xFFFF, 0x7FFF), []float64{-32768, -1, 32767}},
		{"U16", u16(0, 0xFFFF), []float64{0, 65535}},
		{"I32", u32(0x80000000, 0xFFFFFFFF, 0x7FFFFFFF), []float64{math.MinInt32, -1, math.MaxInt32}},
		{"U32", u32(0, 0xFFFFFFFF), []float64{0, math.MaxUint32}},
		{"I64", u64(uint64(1<<63), uint64(0xFFFFFFFFFFFFFFFF), 1<<53), []float64{math.MinInt64, -1, 1 << 53}},
		{"U64", u64(0, 1<<53, 12345678901), []float64{0, 1 << 53, 12345678901}},
	}
	for _, tc := range cases {
		tensors, _, err := Load(bytes.NewReader(buildFile(tc.dtype, []int{len(tc.want)}, tc.data)))
		if err != nil {
			t.Fatalf("%s: %v", tc.dtype, err)
		}
		x := tensors["x"]
		var got []float64
		if x.Dtype() == tensor.F64 {
			got = x.Storage().F64()
		} else {
			for _, v := range x.Storage().F32() {
				got = append(got, float64(v))
			}
		}
		for i, w := range tc.want {
			if got[i] != w {
				t.Errorf("%s[%d]: got %v, want %v", tc.dtype, i, got[i], w)
			}
		}
	}
}

// The file-selective path and the full stream path share the same widening decoder. Exercise every
// load-only dtype through both APIs so a future fast path cannot silently diverge by dtype.
func TestLoadTensorWidenedMatchesLoad(t *testing.T) {
	cases := []struct {
		dtype string
		data  []byte
		n     int
	}{
		{"F8_E4M3", []byte{0x00, 0x38, 0xB8}, 3},
		{"F8_E5M2", []byte{0x00, 0x3C, 0xBC}, 3},
		{"BOOL", []byte{0, 1, 7}, 3},
		{"I8", []byte{0x80, 0xFF, 0x7F}, 3},
		{"U8", []byte{0, 1, 0xFF}, 3},
		{"I16", []byte{0x00, 0x80, 0xFF, 0x7F}, 2},
		{"U16", []byte{0, 0, 0xFF, 0xFF}, 2},
		{"I32", []byte{0, 0, 0, 0x80, 0xFF, 0xFF, 0xFF, 0x7F}, 2},
		{"U32", []byte{0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF}, 2},
		{"I64", []byte{0, 0, 0, 0, 0, 0, 0, 0x80, 1, 0, 0, 0, 0, 0, 0, 0}, 2},
		{"U64", []byte{0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0}, 2},
	}
	for _, tc := range cases {
		t.Run(tc.dtype, func(t *testing.T) {
			data := buildFile(tc.dtype, []int{tc.n}, tc.data)
			all, _, err := Load(bytes.NewReader(data))
			if err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(t.TempDir(), "one.safetensors")
			if err := os.WriteFile(path, data, 0o600); err != nil {
				t.Fatal(err)
			}
			one, err := LoadTensor(path, "x")
			if err != nil {
				t.Fatal(err)
			}
			whole := all["x"]
			if one.Dtype() != whole.Dtype() || !one.Shape().Equal(whole.Shape()) {
				t.Fatalf("dtype/shape = %v/%v, want %v/%v", one.Dtype(), one.Shape(), whole.Dtype(), whole.Shape())
			}
			switch one.Dtype() {
			case tensor.F32:
				for i, got := range one.Storage().F32() {
					if math.Float32bits(got) != math.Float32bits(whole.Storage().F32()[i]) {
						t.Fatalf("element %d bits differ: %08x != %08x", i, math.Float32bits(got), math.Float32bits(whole.Storage().F32()[i]))
					}
				}
			case tensor.F64:
				for i, got := range one.Storage().F64() {
					if math.Float64bits(got) != math.Float64bits(whole.Storage().F64()[i]) {
						t.Fatalf("element %d bits differ: %016x != %016x", i, math.Float64bits(got), math.Float64bits(whole.Storage().F64()[i]))
					}
				}
			}
		})
	}
}

func TestLoadTensorRejectsMalformedEntryBeforeAllocation(t *testing.T) {
	truncated := buildFile("F64", []int{2}, make([]byte, 16))
	truncated = truncated[:len(truncated)-1]
	cases := []struct {
		name string
		data []byte
	}{
		{"unknown dtype", buildFile("I128", []int{1}, make([]byte, 16))},
		{"shape byte mismatch", buildFile("F32", []int{2}, make([]byte, 4))},
		{"truncated payload", truncated},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "bad.safetensors")
			if err := os.WriteFile(path, tc.data, 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err := LoadTensor(path, "x"); err == nil {
				t.Fatal("malformed entry loaded without an error")
			}
		})
	}
}

// §V15 hostile headers for the widened dtypes: byte counts must match the SOURCE
// element size (1 byte for FP8, not the widened 4), and unknown dtypes still error.
func TestLoadWidenedHostileHeaders(t *testing.T) {
	// 4 elements × F8 = 4 bytes; offering 16 (as if F32-sized) must fail.
	bad := buildFile("F8_E4M3", []int{4}, make([]byte, 16))
	if _, _, err := Load(bytes.NewReader(bad)); err == nil {
		t.Error("FP8 with F32-sized data section must be rejected")
	}
	// truncated I16 payload: 3 elements × 2 = 6 bytes, only 4 present.
	if _, _, err := Load(bytes.NewReader(buildFile("I16", []int{3}, make([]byte, 4)))); err == nil {
		t.Error("truncated I16 payload must be rejected")
	}
	for _, unknown := range []string{"F8_E8M0", "I128", "COMPLEX64", ""} {
		if _, _, err := Load(bytes.NewReader(buildFile(unknown, []int{1}, make([]byte, 16)))); err == nil {
			t.Errorf("unknown dtype %q must be rejected", unknown)
		}
	}
	// the write path stays narrow: FP8/int dtypes are load-only (§T577).
	if _, err := dtypeName(tensor.F64); err != nil {
		t.Errorf("F64 must stay writable: %v", err)
	}
}
