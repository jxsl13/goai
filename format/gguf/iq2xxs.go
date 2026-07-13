package gguf

import (
	"encoding/binary"
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// IQ2_XXS importance-quant READ path (§T554): ggml's 2.06-bit E8-lattice codebook
// quant — the first of the i-quant family (the K-quants' §R99–§R104 sibling built
// on shared grid codebooks instead of per-block affine scales). Ported verbatim
// from gguf-py's IQ2_XXS.dequantize_blocks (the llama.cpp project's own reference
// implementation; §V16 tier-1 = golden vectors generated with that library).
//
// Block layout (66 bytes for 256 elements): f16 d, then 8 pairs of uint32
// (qs0, qs1). Each pair decodes 32 values as 4 groups of 8: the 4 bytes of qs0
// index the 256×8 grid; bits 7g..7g+6 of qs1 index the 128-entry ksigns table
// (one sign bit per element); bits 28..31 of qs1 hold the pair's 4-bit scale s,
// with the effective scale d·(0.5+s)·0.25.

const (
	tIQ2_XXS        = 16 // ggml type id
	iq2xxsBlockSize = 66 // 2 (f16 d) + 8·8 (qs pairs) bytes per 256 elements
)

// iq2xxsKSigns is ggml's ksigns_iq2xs table: entry i has the 7 low bits of i and
// an 8th (parity) bit so every byte has odd... (the table is DATA, copied
// verbatim from the reference — the invariant that matters is byte equality).
var iq2xxsKSigns = [128]byte{
	0x00, 0x81, 0x82, 0x03, 0x84, 0x05, 0x06, 0x87, 0x88, 0x09, 0x0a, 0x8b, 0x0c, 0x8d, 0x8e, 0x0f,
	0x90, 0x11, 0x12, 0x93, 0x14, 0x95, 0x96, 0x17, 0x18, 0x99, 0x9a, 0x1b, 0x9c, 0x1d, 0x1e, 0x9f,
	0xa0, 0x21, 0x22, 0xa3, 0x24, 0xa5, 0xa6, 0x27, 0x28, 0xa9, 0xaa, 0x2b, 0xac, 0x2d, 0x2e, 0xaf,
	0x30, 0xb1, 0xb2, 0x33, 0xb4, 0x35, 0x36, 0xb7, 0xb8, 0x39, 0x3a, 0xbb, 0x3c, 0xbd, 0xbe, 0x3f,
	0xc0, 0x41, 0x42, 0xc3, 0x44, 0xc5, 0xc6, 0x47, 0x48, 0xc9, 0xca, 0x4b, 0xcc, 0x4d, 0x4e, 0xcf,
	0x50, 0xd1, 0xd2, 0x53, 0xd4, 0x55, 0x56, 0xd7, 0xd8, 0x59, 0x5a, 0xdb, 0x5c, 0xdd, 0xde, 0x5f,
	0x60, 0xe1, 0xe2, 0x63, 0xe4, 0x65, 0x66, 0xe7, 0xe8, 0x69, 0x6a, 0xeb, 0x6c, 0xed, 0xee, 0x6f,
	0xf0, 0x71, 0x72, 0xf3, 0x74, 0xf5, 0xf6, 0x77, 0x78, 0xf9, 0xfa, 0x7b, 0xfc, 0x7d, 0x7e, 0xff,
}

// iq2xxsGridHex encodes the 256×8 grid with each element packed in 2 bits
// (0→8, 1→25, 2→43), exactly as gguf-py ships it.
const iq2xxsGridHex = "00000200050008000a00110014002000220028002a0041004400500058006100" +
	"6400800082008a00a20001010401100115014001840198010002020222028202" +
	"010404041004210424044004420448046004810484049004a404000502050805" +
	"200546056905800591050906100640068406a406000805080808140828084108" +
	"440850085208880804094009020a140a01100410101021104010601084109010" +
	"951000110811201150115a118011241245120014081420142514491480141815" +
	"6215001616160118041810184018811800190519a019511a002002200a204420" +
	"6120802082202921482100220222012404241024402456240025412564259026" +
	"082820289428442a014004401040184021402440404048405640604081408440" +
	"9040004120416141804185410142104248425642684200440844204480449944" +
	"124524450046014804481048404845480049584961498249454a904a00500850" +
	"1150195020508050885004514251a4519152905492540a550156545600581158" +
	"195864584059085a046010604060686000615561186260620064056410651265" +
	"84654268008002800a8041808280048118814081118201840484108415844084" +
	"608400854685948509864086608602880489118a0490109024904090a1901691" +
	"8091459200942294449451958198209902a050a085a009a100a218a450a804a9"

// iq2xxsGrid is the decoded [256][8]float32 codebook, built once at init.
var iq2xxsGrid [256][8]float32

func init() {
	if len(iq2xxsGridHex) != 1024 {
		panic("gguf: iq2xxs grid hex length")
	}
	unhex := func(c byte) byte {
		if c > 0x40 {
			c += 9
		}
		return c & 0x0F
	}
	gridMap := [3]float32{0x08, 0x19, 0x2b}
	for e := range 256 {
		for b := range 2 {
			hi := unhex(iq2xxsGridHex[e*4+b*2])
			lo := unhex(iq2xxsGridHex[e*4+b*2+1])
			packed := hi<<4 | lo
			for k := range 4 {
				iq2xxsGrid[e][b*4+k] = gridMap[(packed>>(2*k))&0x3]
			}
		}
	}
}

// dequantIQ2_XXS decodes IQ2_XXS blocks into an F32 tensor of the given shape.
func dequantIQ2_XXS(shape tensor.Shape, data []byte) (*tensor.Tensor, error) {
	n := shape.Numel()
	if n%qkK != 0 {
		return nil, fmt.Errorf("gguf: IQ2_XXS numel %d not multiple of %d", n, qkK)
	}
	nb := n / qkK
	if len(data) != nb*iq2xxsBlockSize {
		return nil, fmt.Errorf("gguf: IQ2_XXS data %dB, want %d", len(data), nb*iq2xxsBlockSize)
	}
	out := tensor.New(tensor.F32, shape)
	dst := out.Storage().F32()
	for b := range nb {
		blk := data[b*iq2xxsBlockSize : (b+1)*iq2xxsBlockSize]
		d := f16ToF32(binary.LittleEndian.Uint16(blk[0:]))
		for pair := range 8 {
			qs0 := binary.LittleEndian.Uint32(blk[2+pair*8:])
			qs1 := binary.LittleEndian.Uint32(blk[2+pair*8+4:])
			db := d * (0.5 + float32(qs1>>28)) * 0.25
			for g := range 4 {
				gridRow := &iq2xxsGrid[byte(qs0>>(8*g))]
				signs := iq2xxsKSigns[(qs1>>(7*g))&0x7F]
				base := b*qkK + pair*32 + g*8
				for k := range 8 {
					v := db * gridRow[k]
					if signs>>k&1 == 1 {
						v = -v
					}
					dst[base+k] = v
				}
			}
		}
	}
	return out, nil
}
