//go:build arm64 && goexperiment.simd

package cpu

import (
	"math"
	"simd/archsimd"
)

// The erf rational approximations below are derived from Stephen L. Moshier's
// Cephes library, as carried by SciPy's xsf/cephes/ndtr.h at immutable commit
// 4fff9b2cb2b5c31a0cf0b0f609d2699a5eeac53b.
// Copyright 1984, 1987, 1988, 1992 by Stephen L. Moshier.
const (
	geluNeonInvSqrt2 = 0.7071067811865476
	geluNeonT0       = 9.60497373987051638749e0
	geluNeonT1       = 9.00260197203842689217e1
	geluNeonT2       = 2.23200534594684319226e3
	geluNeonT3       = 7.00332514112805075473e3
	geluNeonT4       = 5.55923013010394962768e4
	geluNeonU0       = 3.35617141647503099647e1
	geluNeonU1       = 5.21357949780152679795e2
	geluNeonU2       = 4.59432382970980127987e3
	geluNeonU3       = 2.26290000613890934246e4
	geluNeonU4       = 4.92673942608635921086e4
	geluNeonP0       = 2.46196981473530512524e-10
	geluNeonP1       = 5.64189564831068821977e-1
	geluNeonP2       = 7.46321056442269912687e0
	geluNeonP3       = 4.86371970985681366614e1
	geluNeonP4       = 1.96520832956077098242e2
	geluNeonP5       = 5.26445194995477358631e2
	geluNeonP6       = 9.34528527171957607540e2
	geluNeonP7       = 1.02755188689515710272e3
	geluNeonP8       = 5.57535335369399327526e2
	geluNeonQ0       = 1.32281951154744992508e1
	geluNeonQ1       = 8.67072140885989742329e1
	geluNeonQ2       = 3.54937778887819891062e2
	geluNeonQ3       = 9.75708501743205489753e2
	geluNeonQ4       = 1.82390916687909736289e3
	geluNeonQ5       = 2.24633760818710981792e3
	geluNeonQ6       = 1.65666309194161350182e3
	geluNeonQ7       = 5.57535340817727675546e2
)

var (
	geluNVZero                                       = archsimd.BroadcastFloat64x2(0)
	geluNVHalf                                       = archsimd.BroadcastFloat64x2(0.5)
	geluNVOne                                        = archsimd.BroadcastFloat64x2(1)
	geluNVNegHalf                                    = archsimd.BroadcastFloat64x2(-0.5)
	geluNVInvSqrt2                                   = archsimd.BroadcastFloat64x2(geluNeonInvSqrt2)
	geluNVInvSqrt2Pi                                 = archsimd.BroadcastFloat64x2(cpuInvSqrt2Pi)
	geluNVSix                                        = archsimd.BroadcastFloat64x2(6)
	geluNVExpLo                                      = archsimd.BroadcastFloat64x2(-708)
	geluNVLog2e                                      = archsimd.BroadcastFloat64x2(1.4426950408889634)
	geluNVNLn2Hi                                     = archsimd.BroadcastFloat64x2(-6.93147180369123816490e-01)
	geluNVNLn2Lo                                     = archsimd.BroadcastFloat64x2(-1.90821492927058770002e-10)
	geluNVBias                                       = archsimd.BroadcastInt64x2(1023)
	geluNVAbs                                        = archsimd.BroadcastUint64x2(0x7fffffffffffffff)
	geluNVSign                                       = archsimd.BroadcastUint64x2(0x8000000000000000)
	geluNVE13                                        = geluNVB(1.6059043836821613e-10)
	geluNVE12                                        = geluNVB(2.08767569878681e-09)
	geluNVE11                                        = geluNVB(2.505210838544172e-08)
	geluNVE10                                        = geluNVB(2.755731922398589e-07)
	geluNVE9                                         = geluNVB(2.7557319223985893e-06)
	geluNVE8                                         = geluNVB(2.48015873015873e-05)
	geluNVE7                                         = geluNVB(1.984126984126984e-04)
	geluNVE6                                         = geluNVB(1.388888888888889e-03)
	geluNVE5                                         = geluNVB(8.333333333333333e-03)
	geluNVE4                                         = geluNVB(4.166666666666666e-02)
	geluNVE3                                         = geluNVB(1.6666666666666666e-01)
	geluNVE2                                         = geluNVB(0.5)
	geluNVT0, geluNVT1, geluNVT2, geluNVT3, geluNVT4 = geluNVB(geluNeonT0), geluNVB(geluNeonT1), geluNVB(geluNeonT2), geluNVB(geluNeonT3), geluNVB(geluNeonT4)
	geluNVU0, geluNVU1, geluNVU2, geluNVU3, geluNVU4 = geluNVB(geluNeonU0), geluNVB(geluNeonU1), geluNVB(geluNeonU2), geluNVB(geluNeonU3), geluNVB(geluNeonU4)
	geluNVP0, geluNVP1, geluNVP2, geluNVP3, geluNVP4 = geluNVB(geluNeonP0), geluNVB(geluNeonP1), geluNVB(geluNeonP2), geluNVB(geluNeonP3), geluNVB(geluNeonP4)
	geluNVP5, geluNVP6, geluNVP7, geluNVP8           = geluNVB(geluNeonP5), geluNVB(geluNeonP6), geluNVB(geluNeonP7), geluNVB(geluNeonP8)
	geluNVQ0, geluNVQ1, geluNVQ2, geluNVQ3           = geluNVB(geluNeonQ0), geluNVB(geluNeonQ1), geluNVB(geluNeonQ2), geluNVB(geluNeonQ3)
	geluNVQ4, geluNVQ5, geluNVQ6, geluNVQ7           = geluNVB(geluNeonQ4), geluNVB(geluNeonQ5), geluNVB(geluNeonQ6), geluNVB(geluNeonQ7)
)

func geluNVB(v float64) archsimd.Float64x2 { return archsimd.BroadcastFloat64x2(v) }

func expF64x2GELU(x archsimd.Float64x2) archsimd.Float64x2 {
	x = x.Max(geluNVExpLo)
	kf := x.Mul(geluNVLog2e).Round()
	r := kf.MulAdd(geluNVNLn2Hi, x)
	r = kf.MulAdd(geluNVNLn2Lo, r)
	p := geluNVE13.MulAdd(r, geluNVE12)
	p = p.MulAdd(r, geluNVE11)
	p = p.MulAdd(r, geluNVE10)
	p = p.MulAdd(r, geluNVE9)
	p = p.MulAdd(r, geluNVE8)
	p = p.MulAdd(r, geluNVE7)
	p = p.MulAdd(r, geluNVE6)
	p = p.MulAdd(r, geluNVE5)
	p = p.MulAdd(r, geluNVE4)
	p = p.MulAdd(r, geluNVE3)
	p = p.MulAdd(r, geluNVE2)
	p = p.MulAdd(r, geluNVOne)
	p = p.MulAdd(r, geluNVOne)
	scale := kf.ConvertToInt64().Add(geluNVBias).ShiftAllLeft(52).ToBits().BitsToFloat64()
	return p.Mul(scale)
}

func erfF64x2GELU(y archsimd.Float64x2) archsimd.Float64x2 {
	ay := y.Abs()
	z := y.Mul(y)
	numT := geluNVT0.MulAdd(z, geluNVT1)
	numT = numT.MulAdd(z, geluNVT2)
	numT = numT.MulAdd(z, geluNVT3)
	numT = numT.MulAdd(z, geluNVT4)
	denU := z.Add(geluNVU0)
	denU = denU.MulAdd(z, geluNVU1)
	denU = denU.MulAdd(z, geluNVU2)
	denU = denU.MulAdd(z, geluNVU3)
	denU = denU.MulAdd(z, geluNVU4)
	erfSmall := y.Mul(numT).Div(denU)
	small := ay.Less(geluNVOne)
	lanes := small.ToInt64x2()
	if lanes.GetElem(0) != 0 && lanes.GetElem(1) != 0 {
		return erfSmall
	}
	e := expF64x2GELU(geluNVZero.Sub(z))
	numP := geluNVP0.MulAdd(ay, geluNVP1)
	numP = numP.MulAdd(ay, geluNVP2)
	numP = numP.MulAdd(ay, geluNVP3)
	numP = numP.MulAdd(ay, geluNVP4)
	numP = numP.MulAdd(ay, geluNVP5)
	numP = numP.MulAdd(ay, geluNVP6)
	numP = numP.MulAdd(ay, geluNVP7)
	numP = numP.MulAdd(ay, geluNVP8)
	denQ := ay.Add(geluNVQ0)
	denQ = denQ.MulAdd(ay, geluNVQ1)
	denQ = denQ.MulAdd(ay, geluNVQ2)
	denQ = denQ.MulAdd(ay, geluNVQ3)
	denQ = denQ.MulAdd(ay, geluNVQ4)
	denQ = denQ.MulAdd(ay, geluNVQ5)
	denQ = denQ.MulAdd(ay, geluNVQ6)
	denQ = denQ.MulAdd(ay, geluNVQ7)
	erfc := e.Mul(numP).Div(denQ)
	sign := y.ToBits().And(geluNVSign)
	erfMiddle := geluNVOne.Sub(erfc).ToBits().And(geluNVAbs).Or(sign).BitsToFloat64()
	erfBig := geluNVOne.ToBits().Or(sign).BitsToFloat64()
	erf := erfSmall.IfElse(small, erfMiddle)
	return erfBig.IfElse(ay.GreaterEqual(geluNVSix), erf)
}

func erfF64GELUPoly(y float64) float64 {
	ay, z := math.Abs(y), y*y
	numT := geluNeonT0
	for _, c := range []float64{geluNeonT1, geluNeonT2, geluNeonT3, geluNeonT4} {
		numT = math.FMA(numT, z, c)
	}
	denU := z + geluNeonU0
	for _, c := range []float64{geluNeonU1, geluNeonU2, geluNeonU3, geluNeonU4} {
		denU = math.FMA(denU, z, c)
	}
	if ay < 1 {
		return y * numT / denU
	}
	if ay >= 6 {
		return math.Copysign(1, y)
	}
	numP := geluNeonP0
	for _, c := range []float64{geluNeonP1, geluNeonP2, geluNeonP3, geluNeonP4, geluNeonP5, geluNeonP6, geluNeonP7, geluNeonP8} {
		numP = math.FMA(numP, ay, c)
	}
	denQ := ay + geluNeonQ0
	for _, c := range []float64{geluNeonQ1, geluNeonQ2, geluNeonQ3, geluNeonQ4, geluNeonQ5, geluNeonQ6, geluNeonQ7} {
		denQ = math.FMA(denQ, ay, c)
	}
	return math.Copysign(1-expF64poly(-z)*numP/denQ, y)
}

func geluF64GELUPoly(x float64) float64 { return (x * 0.5) * (1 + erfF64GELUPoly(x*geluNeonInvSqrt2)) }
func geluGradF64GELUPoly(x, g float64) float64 {
	phi := 0.5 * (1 + erfF64GELUPoly(x*geluNeonInvSqrt2))
	pdf := cpuInvSqrt2Pi * expF64poly((-0.5*x)*x)
	xpdf := float64(x * pdf)
	return g * (phi + xpdf)
}

func geluF64Eligible(x []float64) bool {
	for _, v := range x {
		if !isFinite(v) || math.Abs(v) > 32 {
			return false
		}
	}
	return true
}
func geluGradF64Eligible(x, g []float64) bool {
	if !geluF64Eligible(x) {
		return false
	}
	for _, v := range g {
		a := math.Abs(v)
		if !isFinite(v) || a > 8 || (a != 0 && a < 1e-150) {
			return false
		}
	}
	return true
}
func isFinite(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) }

func vgeluF64(dst, src []float64) {
	if !geluF64Eligible(src) {
		for i, x := range src {
			dst[i] = 0.5 * x * (1 + math.Erf(x/math.Sqrt2))
		}
		return
	}
	n2 := len(src) &^ 1
	for i := 0; i < n2; i += 2 {
		x := archsimd.LoadFloat64x2Array((*[2]float64)(src[i:]))
		erf := erfF64x2GELU(x.Mul(geluNVInvSqrt2))
		x.Mul(geluNVHalf).Mul(geluNVOne.Add(erf)).StoreArray((*[2]float64)(dst[i:]))
	}
	for i := n2; i < len(src); i++ {
		dst[i] = geluF64GELUPoly(src[i])
	}
}

func vgeluGradF64(dst, x, g []float64) {
	if !geluGradF64Eligible(x, g) {
		for i := range x {
			dst[i] = geluGradF64(x[i], g[i])
		}
		return
	}
	n2 := len(x) &^ 1
	for i := 0; i < n2; i += 2 {
		xv := archsimd.LoadFloat64x2Array((*[2]float64)(x[i:]))
		gv := archsimd.LoadFloat64x2Array((*[2]float64)(g[i:]))
		phi := geluNVHalf.Mul(geluNVOne.Add(erfF64x2GELU(xv.Mul(geluNVInvSqrt2))))
		pdf := expF64x2GELU(geluNVNegHalf.Mul(xv).Mul(xv)).Mul(geluNVInvSqrt2Pi)
		gv.Mul(phi.Add(xv.Mul(pdf))).StoreArray((*[2]float64)(dst[i:]))
	}
	for i := n2; i < len(x); i++ {
		dst[i] = geluGradF64GELUPoly(x[i], g[i])
	}
}
