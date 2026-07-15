package safetensors

import (
	"bytes"
	"encoding/binary"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// Perf guards for the Save encode loop, the Load decode loops and the FP8
// widening path (docs/perf-notes-lowlevel.md).

func benchF32Tensor(n int) *tensor.Tensor {
	t := tensor.New(tensor.F32, tensor.Shape{n})
	s := t.Storage().F32()
	for i := range s {
		s[i] = float32(i)
	}
	return t
}

func BenchmarkSaveF32_1M(b *testing.B) {
	const n = 1 << 20
	ts := map[string]*tensor.Tensor{"w": benchF32Tensor(n)}
	b.SetBytes(4 * n)
	b.ResetTimer()
	for range b.N {
		var buf bytes.Buffer
		buf.Grow(4*n + 256)
		if err := Save(&buf, ts, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLoadF32_1M(b *testing.B) {
	const n = 1 << 20
	var buf bytes.Buffer
	if err := Save(&buf, map[string]*tensor.Tensor{"w": benchF32Tensor(n)}, nil); err != nil {
		b.Fatal(err)
	}
	data := buf.Bytes()
	b.SetBytes(4 * n)
	b.ResetTimer()
	for range b.N {
		if _, _, err := Load(bytes.NewReader(data)); err != nil {
			b.Fatal(err)
		}
	}
}

// fp8File builds a file with one F8_E4M3 tensor of n bytes cycling through all
// 256 encodings.
func fp8File(dtype string, n int) []byte {
	hdr := `{"t":{"dtype":"` + dtype + `","shape":[` + itoa(n) + `],"data_offsets":[0,` + itoa(n) + `]}}`
	for len(hdr)%8 != 0 {
		hdr += " "
	}
	var b bytes.Buffer
	binary.Write(&b, binary.LittleEndian, uint64(len(hdr)))
	b.WriteString(hdr)
	payload := make([]byte, n)
	for i := range payload {
		payload[i] = byte(i)
	}
	b.Write(payload)
	return b.Bytes()
}

func itoa(n int) string {
	return string(bytes.TrimSpace([]byte(fmtInt(n))))
}

func fmtInt(n int) string {
	if n == 0 {
		return "0"
	}
	var d []byte
	for n > 0 {
		d = append([]byte{byte('0' + n%10)}, d...)
		n /= 10
	}
	return string(d)
}

func benchLoadFP8(b *testing.B, dtype string) {
	const n = 1 << 20
	data := fp8File(dtype, n)
	b.SetBytes(n)
	b.ResetTimer()
	for range b.N {
		if _, _, err := Load(bytes.NewReader(data)); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLoadF8E4M3_1M(b *testing.B) { benchLoadFP8(b, "F8_E4M3") }
func BenchmarkLoadF8E5M2_1M(b *testing.B) { benchLoadFP8(b, "F8_E5M2") }
