package gguf

// dotMXFP4Row folds E8M0 scale conversion and signed E2M1 nibble lookup into
// one ascending-weight dot. Each float32 multiply matches the materialized
// decoder before the product widens into the float64 accumulator.
func dotMXFP4Row(row []float32, raw []byte, k int) float64 {
	var acc float64
	for b := 0; b*blockElems < k; b++ {
		blk := raw[b*mxfp4BlockSize : (b+1)*mxfp4BlockSize]
		d := e8m0ToF32Half(blk[0])
		q := blk[1:17]
		base := b * blockElems
		for i, packed := range q {
			weight := d * mxfp4KValues[packed&0x0f]
			acc += float64(row[base+i]) * float64(weight)
		}
		for i, packed := range q {
			weight := d * mxfp4KValues[packed>>4]
			acc += float64(row[base+16+i]) * float64(weight)
		}
	}
	return acc
}
