package gguf

import (
	"encoding/binary"
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// IQ2_XS importance-quant READ path (§T554 part 2): the 2.31-bit sibling of
// IQ2_XXS — a 512-entry E8-lattice grid with EXPLICIT per-16-element 4-bit
// scales (XXS packs its scale into the sign word). Ported verbatim from
// gguf-py's IQ2_XS.dequantize_blocks; golden vectors cross-generated with it.
//
// Block layout (74 bytes for 256 elements): f16 d, 32 × uint16 qs (low 9 bits =
// grid index, high 7 bits = ksigns index; each word decodes 8 elements),
// 8 bytes = 16 four-bit scales (one per 16 elements); effective scale
// d·(0.5+sc)·0.25. Shares iq2xxsKSigns.

const (
	tIQ2_XS        = 17 // ggml type id
	iq2xsBlockSize = 74 // 2 + 64 + 8 bytes per 256 elements
)

// iq2xsGridHex encodes the 512×8 grid, 2 bits per element (0→8, 1→25, 2→43),
// exactly as gguf-py ships it (injected programmatically — see gen note in §T554).
const iq2xsGridHex = "00000200050008000a0011001400160019002000220025002800410044004600" +
	"49005000520055005800610064008000820085008800910094009900a0000101" +
	"04010601090110011201150118011a0121012401400142014501480151015401" +
	"6001680181018401900100020202050208021102140220024102440250025502" +
	"80028a0201040404060409041004120415041804210424044004420445044804" +
	"5104540456046004810484049004000502050505080511051405200541054405" +
	"500561058005010604061006260640064206840600080208050808080a081108" +
	"14082008250841084408500858088008a008aa08010904091009400981098909" +
	"000a200a280a960aa00a01100410061009101010121015101810211024104010" +
	"4210451048105110541060106a10811084109010001102110511081111111411" +
	"2011411144115011801194119611011204120612101240126012001402140514" +
	"0814111414142014411444144914501464148014011504151015401500161416" +
	"49160118041810181218401854188618001905196619511aa91a002002200520" +
	"08200a201120142020204120442050208020a020012104211021402148216521" +
	"002222228022a82201240424102429244024002541255225992501261a26a626" +
	"002808280a28202855288828a22868299029082a202a822a882a8a2a01400440" +
	"0640094010401240154018402140244040404240454048404a40514054406040" +
	"6540814084409040004102410541084111411441204141414441504180418541" +
	"a241014204421042124229424042004402440544084411441444194420444144" +
	"4444504480449444014504451045244540459a4500460a464446504601480448" +
	"1048404845485448624800491149444950496949044a00500250055008501150" +
	"145020502850415044505050805001510451105115514051425100524452aa52" +
	"0154045410542154405460548154a154005508558055885521566856a1560058" +
	"14584158505899581a5940594259855a0160046010604060546062608660a960" +
	"006124624a62926200641664106540654565a46501686a682569066a546a626a" +
	"00800280058008801180148020802a8041804480508080808280a880aa800181" +
	"0481068110814081518159810082208280828282a082a8820184048410841284" +
	"158440846084898400854485a58518866a860088088825885a8880888288a888" +
	"0689228a808a888a968aa88a0190049010904090569084900091229164915692" +
	"89920094059444945094589429959095929541965198a6984999159a609a00a0" +
	"02a008a00aa020a02aa0a0a051a159a1a6a100a202a208a22aa280a2a0a240a4" +
	"95a465a698a60aa820a822a828a8a0a8a8a804a984a986a928aa2aaa91aaaaaa"

// iq2xsGrid is the decoded [512][8]float32 codebook, built once at init.
var iq2xsGrid [512][8]float32

func init() {
	if len(iq2xsGridHex) != 2048 {
		panic("gguf: iq2xs grid hex length")
	}
	unhex := func(c byte) byte {
		if c > 0x40 {
			c += 9
		}
		return c & 0x0F
	}
	gridMap := [3]float32{0x08, 0x19, 0x2b}
	for e := range 512 {
		for b := range 2 {
			hi := unhex(iq2xsGridHex[e*4+b*2])
			lo := unhex(iq2xsGridHex[e*4+b*2+1])
			packed := hi<<4 | lo
			for k := range 4 {
				iq2xsGrid[e][b*4+k] = gridMap[(packed>>(2*k))&0x3]
			}
		}
	}
}

// dequantIQ2_XS decodes IQ2_XS blocks into an F32 tensor of the given shape.
func dequantIQ2_XS(shape tensor.Shape, data []byte) (*tensor.Tensor, error) {
	n := shape.Numel()
	if n%qkK != 0 {
		return nil, fmt.Errorf("gguf: IQ2_XS numel %d not multiple of %d", n, qkK)
	}
	nb := n / qkK
	if len(data) != nb*iq2xsBlockSize {
		return nil, fmt.Errorf("gguf: IQ2_XS data %dB, want %d", len(data), nb*iq2xsBlockSize)
	}
	out := tensor.New(tensor.F32, shape)
	dst := out.Storage().F32()
	for b := range nb {
		blk := data[b*iq2xsBlockSize : (b+1)*iq2xsBlockSize]
		d := f16ToF32(binary.LittleEndian.Uint16(blk[0:]))
		scales := blk[66:74]
		for g := range 32 { // one uint16 → 8 elements
			u := binary.LittleEndian.Uint16(blk[2+g*2:])
			s := g / 2 // scale index: one nibble per 16 elements
			sc := scales[s/2] >> (4 * (s % 2)) & 0x0F
			db := d * (0.5 + float32(sc)) * 0.25
			gridRow := &iq2xsGrid[u&511]
			signs := iq2xxsKSigns[u>>9]
			base := b*qkK + g*8
			for k := range 8 {
				v := db * gridRow[k]
				if signs>>k&1 == 1 {
					v = -v
				}
				dst[base+k] = v
			}
		}
	}
	return out, nil
}
