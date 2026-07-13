package gguf

import (
	"bytes"
	"encoding/binary"
	"testing"
)

// §V15 (manual-analysis finding): GGUF metadata arrays may nest (element type
// ARRAY). A deeply nested file must error via the depth cap, not blow the stack.
func TestDeepArrayNestingErrorsNotStackOverflow(t *testing.T) {
	var b bytes.Buffer
	b.Write([]byte{0x47, 0x47, 0x55, 0x46})          // GGUF
	binary.Write(&b, binary.LittleEndian, uint32(3)) // version
	binary.Write(&b, binary.LittleEndian, uint64(0)) // 0 tensors
	binary.Write(&b, binary.LittleEndian, uint64(1)) // 1 KV
	binary.Write(&b, binary.LittleEndian, uint64(1)) // key len
	b.WriteString("x")                               // key
	binary.Write(&b, binary.LittleEndian, uint32(9)) // value type = ARRAY

	const depth = 500 // far beyond the cap
	for range depth {
		binary.Write(&b, binary.LittleEndian, uint32(9)) // element type = ARRAY
		binary.Write(&b, binary.LittleEndian, uint64(1)) // length 1
	}
	// terminal element: u8
	binary.Write(&b, binary.LittleEndian, uint32(0)) // element type = U8
	binary.Write(&b, binary.LittleEndian, uint64(1)) // length 1
	b.WriteByte(0x42)

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("deep nesting panicked (stack overflow?): %v", r)
		}
	}()
	if _, err := Read(bytes.NewReader(b.Bytes())); err == nil {
		t.Error("deep array nesting must error via depth cap")
	}
}

// A huge claimed array length with no data must fail fast (grow, not pre-alloc).
func TestHugeArrayLengthNoData(t *testing.T) {
	var b bytes.Buffer
	b.Write([]byte{0x47, 0x47, 0x55, 0x46})
	binary.Write(&b, binary.LittleEndian, uint32(3))
	binary.Write(&b, binary.LittleEndian, uint64(0))
	binary.Write(&b, binary.LittleEndian, uint64(1))
	binary.Write(&b, binary.LittleEndian, uint64(1))
	b.WriteString("x")
	binary.Write(&b, binary.LittleEndian, uint32(9))     // ARRAY
	binary.Write(&b, binary.LittleEndian, uint32(4))     // element type U32
	binary.Write(&b, binary.LittleEndian, uint64(1<<23)) // 8M elements claimed
	// ...but no element bytes follow → must error, not allocate 8M×16B up front
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("panicked: %v", r)
		}
	}()
	if _, err := Read(bytes.NewReader(b.Bytes())); err == nil {
		t.Error("huge array claim with no data must error")
	}
}
