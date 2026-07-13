package gguf

import (
	"encoding/binary"
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// IQ3_XXS importance-quant READ path (§T554 part 3): the 3.06-bit i-quant — a
// 256×4 grid over an 8-value codebook (3-bit elements, two per byte at shifts
// 0/4), signs/scales packed per 32 elements into a uint32 exactly like IQ2_XXS
// (4×7-bit ksigns indices + 4-bit scale, effective d·(0.5+s)·0.5). Ported
// verbatim from gguf-py's IQ3_XXS.dequantize_blocks.
//
// Block layout (98 bytes for 256 elements): f16 d, 64 grid-index bytes (each
// indexes a 4-wide grid row; two bytes = 8 elements), 8 uint32 sign/scale words.

const (
	tIQ3_XXS        = 18 // ggml type id
	iq3xxsBlockSize = 98 // 2 + 64 + 32 bytes per 256 elements
)

// iq3xxsGridHex packs the 256×4 grid, one 3-bit element per nibble (shifts 0/4),
// mapped through iq3xxsGridMap; injected programmatically from gguf-py.
const iq3xxsGridHex = "0000020004001100130017002000220031004200730075000101030110011201" +
	"2101250130013201410154017001000202020402110220022202310233023702" +
	"5102570275020103070310031203250370031304370444045704730475040105" +
	"0705320552053506640610071407160743076107011003101010121021102310" +
	"3010321034104710501000110211111120112211011203121012121221123012" +
	"7212001302132013311346136613011405145014201524154615711505162217" +
	"4017002002201120132020202220262031204220012103210521102112212121" +
	"3021632167217021002202221122172220222222372240225522012310231423" +
	"7023742335245324032527254125742501270327162745270130103012302130" +
	"2330503065307230003102312031313144314631013203321032253252327232" +
	"1133333330344734723400350635223555351436363663363337603704401740" +
	"3540374053405740744120423742404260426642074345430444514464442545" +
	"4345704505471047124730471250415070500051065126515551145232527252" +
	"0253535310542354275472540255315550562457425724604460466064602161" +
	"6161176264623063366344640565526533660367216703700570077010703270" +
	"5270267140711272457252720073157333736073217441740075027524753076"

var iq3xxsGridMap = [8]float32{0x04, 0x0c, 0x14, 0x1c, 0x24, 0x2c, 0x34, 0x3e}

// iq3xxsGrid is the decoded [256][4]float32 codebook, built once at init.
var iq3xxsGrid [256][4]float32

func init() {
	if len(iq3xxsGridHex) != 1024 {
		panic("gguf: iq3xxs grid hex length")
	}
	unhex := func(c byte) byte {
		if c > 0x40 {
			c += 9
		}
		return c & 0x0F
	}
	for e := range 256 {
		for b := range 2 { // two packed bytes per 4-element row
			hi := unhex(iq3xxsGridHex[e*4+b*2])
			lo := unhex(iq3xxsGridHex[e*4+b*2+1])
			packed := hi<<4 | lo
			for k := range 2 { // two 3-bit elements per byte, shifts 0 and 4
				iq3xxsGrid[e][b*2+k] = iq3xxsGridMap[(packed>>(4*k))&0x7]
			}
		}
	}
}

// dequantIQ3_XXS decodes IQ3_XXS blocks into an F32 tensor of the given shape.
func dequantIQ3_XXS(shape tensor.Shape, data []byte) (*tensor.Tensor, error) {
	n := shape.Numel()
	if n%qkK != 0 {
		return nil, fmt.Errorf("gguf: IQ3_XXS numel %d not multiple of %d", n, qkK)
	}
	nb := n / qkK
	if len(data) != nb*iq3xxsBlockSize {
		return nil, fmt.Errorf("gguf: IQ3_XXS data %dB, want %d", len(data), nb*iq3xxsBlockSize)
	}
	out := tensor.New(tensor.F32, shape)
	dst := out.Storage().F32()
	for b := range nb {
		blk := data[b*iq3xxsBlockSize : (b+1)*iq3xxsBlockSize]
		d := f16ToF32(binary.LittleEndian.Uint16(blk[0:]))
		qs := blk[2:66]
		for g := range 8 { // one uint32 → 32 elements (4 sub-groups of 8)
			sg := binary.LittleEndian.Uint32(blk[66+g*4:])
			db := d * (0.5 + float32(sg>>28)) * 0.5
			for j := range 4 { // sub-group: two grid rows of 4 = 8 elements
				r1 := &iq3xxsGrid[qs[g*8+j*2]]
				r2 := &iq3xxsGrid[qs[g*8+j*2+1]]
				signs := iq2xxsKSigns[(sg>>(7*j))&0x7F]
				base := b*qkK + g*32 + j*8
				for k := range 8 {
					var gv float32
					if k < 4 {
						gv = r1[k]
					} else {
						gv = r2[k-4]
					}
					v := db * gv
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
