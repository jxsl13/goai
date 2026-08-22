package gguf

import (
	"bytes"
	"slices"
	"strings"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

func makeIQWireGGUF(ggType uint32, raw []byte) []byte {
	var buf bytes.Buffer
	wr := writer{w: &buf}
	wr.u32(magic)
	wr.u32(3)
	wr.u64(1)
	wr.u64(0)
	wr.str("w")
	wr.u32(1)
	wr.u64(qkK)
	wr.u32(ggType)
	wr.u64(0)
	if pad := (32 - wr.n%32) % 32; pad > 0 {
		wr.pad(int(pad))
	}
	wr.write(raw)
	if wr.err != nil {
		panic(wr.err)
	}
	return buf.Bytes()
}

func TestIQWireTypeIDsMatchPinnedGGML(t *testing.T) {
	if tIQ1_S != 19 || uint32(IQ1_S) != 19 {
		t.Fatalf("IQ1_S wire type = internal %d, public %d; want 19", tIQ1_S, IQ1_S)
	}
	if tIQ2_S != 22 || uint32(IQ2_S) != 22 {
		t.Fatalf("IQ2_S wire type = internal %d, public %d; want 22", tIQ2_S, IQ2_S)
	}
	if got, err := byteSize(19, qkK); err != nil || got != iq1sBlockSize {
		t.Fatalf("wire type 19 byteSize = %d, %v; want %d", got, err, iq1sBlockSize)
	}
	if got, err := byteSize(22, qkK); err != nil || got != iq2sBlockSize {
		t.Fatalf("wire type 22 byteSize = %d, %v; want %d", got, err, iq2sBlockSize)
	}
}

func TestIQWireTypesDispatchEveryReadPath(t *testing.T) {
	for _, tc := range []struct {
		name   string
		ggType uint32
		raw    []byte
		decode func(tensor.Shape, []byte) (*tensor.Tensor, error)
	}{
		{"IQ1_S", 19, makeIQ1SRaw(qkK), dequantIQ1_S},
		{"IQ2_S", 22, makeIQ2SRaw(qkK), dequantIQ2_S},
	} {
		t.Run(tc.name, func(t *testing.T) {
			want, err := tc.decode(tensor.Shape{qkK}, tc.raw)
			if err != nil {
				t.Fatal(err)
			}
			public, err := Dequantize(tc.raw, QuantType(tc.ggType), qkK)
			if err != nil {
				t.Fatal(err)
			}
			eager, err := Read(bytes.NewReader(makeIQWireGGUF(tc.ggType, tc.raw)))
			if err != nil {
				t.Fatal(err)
			}
			rawFile, err := ReadRaw(bytes.NewReader(makeIQWireGGUF(tc.ggType, tc.raw)))
			if err != nil {
				t.Fatal(err)
			}
			qt := rawFile.Tensors["w"]
			if qt.GGType != tc.ggType || !bytes.Equal(qt.Data, tc.raw) {
				t.Fatalf("raw tensor type/data = %d/%v, want %d/true", qt.GGType, bytes.Equal(qt.Data, tc.raw), tc.ggType)
			}
			rawDecoded, err := qt.Dequantize()
			if err != nil {
				t.Fatal(err)
			}
			wantData := want.Storage().F32()
			for name, got := range map[string]*tensor.Tensor{
				"public": public,
				"eager":  eager.Tensors["w"],
				"raw":    rawDecoded,
			} {
				if !slices.Equal(got.Storage().F32(), wantData) {
					t.Fatalf("%s decoder differs from direct %s decode", name, tc.name)
				}
			}
		})
	}
}

func TestIQWireTypeIDsSelectMatchingQMatMulKernels(t *testing.T) {
	old1, old2 := dotIQ1SRowFn, dotIQ2SRowFn
	defer func() {
		dotIQ1SRowFn, dotIQ2SRowFn = old1, old2
	}()
	calls1, calls2 := 0, 0
	dotIQ1SRowFn = func(row []float32, raw []byte, k int) float64 {
		calls1++
		return dotIQ1SRow(row, raw, k)
	}
	dotIQ2SRowFn = func(row []float32, raw []byte, k int) float64 {
		calls2++
		return dotIQ2SRow(row, raw, k)
	}
	x := tensor.New(tensor.F32, tensor.Shape{1, qkK})
	if _, err := QMatMul(x, makeIQ1SRaw(qkK), QuantType(19), 1, qkK); err != nil {
		t.Fatal(err)
	}
	if calls1 != 1 || calls2 != 0 {
		t.Fatalf("wire type 19 kernel calls = IQ1_S %d, IQ2_S %d; want 1,0", calls1, calls2)
	}
	calls1, calls2 = 0, 0
	if _, err := QMatMul(x, makeIQ2SRaw(qkK), QuantType(22), 1, qkK); err != nil {
		t.Fatal(err)
	}
	if calls1 != 0 || calls2 != 1 {
		t.Fatalf("wire type 22 kernel calls = IQ1_S %d, IQ2_S %d; want 0,1", calls1, calls2)
	}
}

func TestGGUFWireType24IsNotMisdecodedAsIQ1S(t *testing.T) {
	const ggType = uint32(24) // GGML_TYPE_I8; support is deliberately not implemented yet.
	assertUnsupported := func(name string, err error) {
		t.Helper()
		if err == nil || !strings.Contains(err.Error(), "unsupported ggml type 24") {
			t.Fatalf("%s error = %v; want unsupported ggml type 24", name, err)
		}
	}
	_, err := byteSize(ggType, qkK)
	assertUnsupported("byteSize", err)
	_, err = Dequantize(make([]byte, iq1sBlockSize), QuantType(ggType), qkK)
	assertUnsupported("Dequantize", err)
	_, err = (QuantTensor{Data: make([]byte, iq1sBlockSize), GGType: ggType, Shape: tensor.Shape{qkK}}).Dequantize()
	assertUnsupported("QuantTensor.Dequantize", err)
	_, err = Read(bytes.NewReader(makeIQWireGGUF(ggType, make([]byte, qkK))))
	assertUnsupported("Read", err)
	_, err = ReadRaw(bytes.NewReader(makeIQWireGGUF(ggType, make([]byte, qkK))))
	assertUnsupported("ReadRaw", err)
	_, err = QMatMul(tensor.New(tensor.F32, tensor.Shape{1, qkK}), make([]byte, iq1sBlockSize), QuantType(ggType), 1, qkK)
	assertUnsupported("QMatMul", err)
}
