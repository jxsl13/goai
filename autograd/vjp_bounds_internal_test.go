package autograd

import (
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// These four functions are intentionally independent copies of the historical
// implementations. They are a same-architecture oracle for changes to the
// production loops, including malformed-input panic timing and callback order.
func historicalBoundsUnaryVJP(f func(x, y, g float64) float64) VJP {
	return func(_ *backend.Context, in, out []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		x, y := in[0], out[0]
		gin := tensor.New(x.Dtype(), x.Shape())
		n := x.Numel()
		xc, yc, gc := x.Contiguous(), y.Contiguous(), g.Contiguous()
		switch x.Dtype() {
		case tensor.F64:
			if yc.Dtype() == tensor.F64 && gc.Dtype() == tensor.F64 {
				xs, ys, gs := xc.Storage().F64(), yc.Storage().F64(), gc.Storage().F64()
				ds := gin.Storage().F64()
				for i := 0; i < n; i++ {
					ds[i] = f(xs[i], ys[i], gs[i])
				}
				return []*tensor.Tensor{gin}, nil
			}
		case tensor.F32:
			if yc.Dtype() == tensor.F32 && gc.Dtype() == tensor.F32 {
				xs, ys, gs := xc.Storage().F32(), yc.Storage().F32(), gc.Storage().F32()
				ds := gin.Storage().F32()
				for i := 0; i < n; i++ {
					ds[i] = float32(f(float64(xs[i]), float64(ys[i]), float64(gs[i])))
				}
				return []*tensor.Tensor{gin}, nil
			}
		}
		for i := 0; i < n; i++ {
			idx := tensor.Unravel(i, x.Shape())
			gin.SetF64(f(x.AtF64(idx...), y.AtF64(idx...), g.AtF64(idx...)), idx...)
		}
		return []*tensor.Tensor{gin}, nil
	}
}

func historicalBoundsReLUVJP(_ *backend.Context, in, out []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
	x := in[0]
	gin := tensor.New(x.Dtype(), x.Shape())
	n := x.Numel()
	xc, gc := x.Contiguous(), g.Contiguous()
	switch x.Dtype() {
	case tensor.F64:
		if gc.Dtype() == tensor.F64 {
			xs, gs := xc.Storage().F64(), gc.Storage().F64()
			ds := gin.Storage().F64()
			for i := 0; i < n; i++ {
				if xs[i] > 0 {
					ds[i] = gs[i]
				}
			}
			return []*tensor.Tensor{gin}, nil
		}
	case tensor.F32:
		if gc.Dtype() == tensor.F32 {
			xs, gs := xc.Storage().F32(), gc.Storage().F32()
			ds := gin.Storage().F32()
			for i := 0; i < n; i++ {
				if xs[i] > 0 {
					ds[i] = gs[i]
				}
			}
			return []*tensor.Tensor{gin}, nil
		}
	}
	return historicalBoundsUnaryVJP(func(xv, _, gv float64) float64 {
		if xv > 0 {
			return gv
		}
		return 0
	})(nil, in, out, attrs, g)
}

func historicalBoundsTanhVJP(_ *backend.Context, in, out []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
	x, y := in[0], out[0]
	gin := tensor.New(x.Dtype(), x.Shape())
	n := x.Numel()
	yc, gc := y.Contiguous(), g.Contiguous()
	if x.Dtype() == tensor.F64 && yc.Dtype() == tensor.F64 && gc.Dtype() == tensor.F64 {
		ys, gs, ds := yc.Storage().F64(), gc.Storage().F64(), gin.Storage().F64()
		for i := 0; i < n; i++ {
			ds[i] = gs[i] * (1 - ys[i]*ys[i])
		}
		return []*tensor.Tensor{gin}, nil
	}
	if x.Dtype() == tensor.F32 && yc.Dtype() == tensor.F32 && gc.Dtype() == tensor.F32 {
		ys, gs, ds := yc.Storage().F32(), gc.Storage().F32(), gin.Storage().F32()
		for i := 0; i < n; i++ {
			yv := float64(ys[i])
			ds[i] = float32(float64(gs[i]) * (1 - yv*yv))
		}
		return []*tensor.Tensor{gin}, nil
	}
	return historicalBoundsUnaryVJP(func(_, yv, gv float64) float64 { return gv * (1 - yv*yv) })(nil, in, out, attrs, g)
}

func historicalBoundsSigmoidVJP(_ *backend.Context, in, out []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
	x, y := in[0], out[0]
	gin := tensor.New(x.Dtype(), x.Shape())
	n := x.Numel()
	yc, gc := y.Contiguous(), g.Contiguous()
	if x.Dtype() == tensor.F64 && yc.Dtype() == tensor.F64 && gc.Dtype() == tensor.F64 {
		ys, gs, ds := yc.Storage().F64(), gc.Storage().F64(), gin.Storage().F64()
		for i := 0; i < n; i++ {
			ds[i] = gs[i] * ys[i] * (1 - ys[i])
		}
		return []*tensor.Tensor{gin}, nil
	}
	if x.Dtype() == tensor.F32 && yc.Dtype() == tensor.F32 && gc.Dtype() == tensor.F32 {
		ys, gs, ds := yc.Storage().F32(), gc.Storage().F32(), gin.Storage().F32()
		for i := 0; i < n; i++ {
			yv := float64(ys[i])
			ds[i] = float32(float64(gs[i]) * yv * (1 - yv))
		}
		return []*tensor.Tensor{gin}, nil
	}
	return historicalBoundsUnaryVJP(func(_, yv, gv float64) float64 { return gv * yv * (1 - yv) })(nil, in, out, attrs, g)
}

type boundsCallbackVisit struct {
	x uint64
	y uint64
	g uint64
}

type boundsTensorImage struct {
	dtype tensor.Dtype
	shape tensor.Shape
	bits  []uint64
}

type boundsOutcome struct {
	outputs []boundsTensorImage
	err     string
	panic   string
	visits  []boundsCallbackVisit
}

func boundsImage(x *tensor.Tensor) boundsTensorImage {
	image := boundsTensorImage{dtype: x.Dtype(), shape: x.Shape().Clone()}
	switch x.Dtype() {
	case tensor.F64:
		for _, v := range x.Storage().F64() {
			image.bits = append(image.bits, math.Float64bits(v))
		}
	case tensor.F32:
		for _, v := range x.Storage().F32() {
			image.bits = append(image.bits, uint64(math.Float32bits(v)))
		}
	case tensor.F16, tensor.BF16:
		for _, v := range x.Storage().U16() {
			image.bits = append(image.bits, uint64(v))
		}
	default:
		panic("boundsImage: unsupported dtype")
	}
	return image
}

func boundsImages(xs []*tensor.Tensor) []boundsTensorImage {
	images := make([]boundsTensorImage, len(xs))
	for i, x := range xs {
		images[i] = boundsImage(x)
	}
	return images
}

func boundsFlipFirst(x *tensor.Tensor) {
	switch x.Dtype() {
	case tensor.F64:
		x.Storage().F64()[0] = math.Float64frombits(math.Float64bits(x.Storage().F64()[0]) ^ 1)
	case tensor.F32:
		x.Storage().F32()[0] = math.Float32frombits(math.Float32bits(x.Storage().F32()[0]) ^ 1)
	case tensor.F16, tensor.BF16:
		x.Storage().U16()[0] ^= 1
	}
}

func boundsRun(t *testing.T, makeRule func(*[]boundsCallbackVisit) VJP, x, y, g *tensor.Tensor) (got boundsOutcome) {
	t.Helper()
	inputs := []*tensor.Tensor{x, y, g}
	before := boundsImages(inputs)
	rule := makeRule(&got.visits)
	defer func() {
		if recovered := recover(); recovered != nil {
			got.panic = fmt.Sprintf("%T:%v", recovered, recovered)
		}
		if after := boundsImages(inputs); !reflect.DeepEqual(after, before) {
			t.Errorf("VJP mutated or aliased an input\nbefore: %#v\nafter:  %#v", before, after)
		}
	}()
	outputs, err := rule(nil, []*tensor.Tensor{x}, []*tensor.Tensor{y}, nil, g)
	if err != nil {
		got.err = fmt.Sprintf("%T:%v", err, err)
	}
	got.outputs = boundsImages(outputs)
	for _, output := range outputs {
		if output.Storage().Len() != 0 {
			boundsFlipFirst(output)
		}
	}
	return got
}

func boundsRule(log *[]boundsCallbackVisit) func(float64, float64, float64) float64 {
	return func(x, y, g float64) float64 {
		*log = append(*log, boundsCallbackVisit{math.Float64bits(x), math.Float64bits(y), math.Float64bits(g)})
		return g*(x+y) + x/(y+3)
	}
}

type boundsOp struct {
	name       string
	production func(*[]boundsCallbackVisit) VJP
	historical func(*[]boundsCallbackVisit) VJP
}

func boundsOps() []boundsOp {
	return []boundsOp{
		{
			name:       "unary",
			production: func(log *[]boundsCallbackVisit) VJP { return unaryVJP(boundsRule(log)) },
			historical: func(log *[]boundsCallbackVisit) VJP { return historicalBoundsUnaryVJP(boundsRule(log)) },
		},
		{
			name:       "relu",
			production: func(*[]boundsCallbackVisit) VJP { return reluVJP },
			historical: func(*[]boundsCallbackVisit) VJP { return historicalBoundsReLUVJP },
		},
		{
			name:       "tanh",
			production: func(*[]boundsCallbackVisit) VJP { return tanhVJP },
			historical: func(*[]boundsCallbackVisit) VJP { return historicalBoundsTanhVJP },
		},
		{
			name:       "sigmoid",
			production: func(*[]boundsCallbackVisit) VJP { return sigmoidVJP },
			historical: func(*[]boundsCallbackVisit) VJP { return historicalBoundsSigmoidVJP },
		},
	}
}

func boundsSpecials(dt tensor.Dtype) []float64 {
	if dt == tensor.F32 {
		return []float64{
			0,
			float64(math.Float32frombits(0x80000000)),
			float64(math.Float32frombits(0x00000001)),
			float64(math.Float32frombits(0x80000001)),
			math.Inf(1),
			math.Inf(-1),
			float64(math.Float32frombits(0x7fc01234)),
			float64(math.Float32frombits(0xffc05678)),
			float64(math.Float32frombits(0x7fa00001)),
		}
	}
	return []float64{
		0,
		math.Copysign(0, -1),
		math.SmallestNonzeroFloat64,
		-math.SmallestNonzeroFloat64,
		math.Inf(1),
		math.Inf(-1),
		math.Float64frombits(0x7ff8000000001234),
		math.Float64frombits(0xfff8000000005678),
		math.Float64frombits(0x7ff4000000000001),
	}
}

func boundsValues(dt tensor.Dtype, n, seed int) []float64 {
	values := make([]float64, n)
	specials := boundsSpecials(dt)
	for i := range values {
		if i < len(specials) {
			values[i] = specials[(i+seed)%len(specials)]
			continue
		}
		values[i] = float64(((i+1)*(seed+11)*37)%257-128) / float64(seed+17)
	}
	return values
}

func boundsFill(x *tensor.Tensor, values []float64) {
	for i, value := range values {
		x.SetF64(value, tensor.Unravel(i, x.Shape())...)
	}
}

func boundsLayout(dt tensor.Dtype, shape tensor.Shape, values []float64, layout string) *tensor.Tensor {
	switch layout {
	case "dense":
		x := tensor.New(dt, shape)
		boundsFill(x, values)
		return x
	case "prefix":
		n := shape.Numel()
		base := tensor.New(dt, tensor.Shape{n + 3})
		boundsFill(base, append(append([]float64{}, values...), 8191.25, -4095.5, 2047.75))
		view, err := base.Slice(0, 0, n)
		if err != nil {
			panic(err)
		}
		return view
	case "offset":
		n := shape.Numel()
		base := tensor.New(dt, tensor.Shape{n + 2})
		all := make([]float64, n+2)
		all[0], all[n+1] = 123.5, -456.25
		copy(all[1:], values)
		boundsFill(base, all)
		view, err := base.Slice(0, 1, n+1)
		if err != nil {
			panic(err)
		}
		return view
	case "transpose":
		base := tensor.New(dt, tensor.Shape{shape[1], shape[0]})
		view, err := base.Transpose(0, 1)
		if err != nil {
			panic(err)
		}
		boundsFill(view, values)
		return view
	default:
		panic("unknown bounds layout: " + layout)
	}
}

func boundsInputs(xdt, ydt, gdt tensor.Dtype, shape tensor.Shape, xl, yl, gl string) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
	n := shape.Numel()
	return boundsLayout(xdt, shape, boundsValues(xdt, n, 0), xl),
		boundsLayout(ydt, shape, boundsValues(ydt, n, 3), yl),
		boundsLayout(gdt, shape, boundsValues(gdt, n, 5), gl)
}

func boundsCheckCase(t *testing.T, op boundsOp, makeInputs func() (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor)) boundsOutcome {
	t.Helper()
	x, y, g := makeInputs()
	want := boundsRun(t, op.historical, x, y, g)
	x, y, g = makeInputs()
	got := boundsRun(t, op.production, x, y, g)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("production differs from historical oracle\nproduction: %#v\nhistorical: %#v", got, want)
	}
	return got
}

func boundsCheckValidCase(t *testing.T, op boundsOp, dtype tensor.Dtype, shape tensor.Shape, makeInputs func() (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor)) {
	t.Helper()
	got := boundsCheckCase(t, op, makeInputs)
	if got.panic != "" || got.err != "" {
		t.Fatalf("valid case failed: panic=%q error=%q", got.panic, got.err)
	}
	if len(got.outputs) != 1 {
		t.Fatalf("valid case returned %d outputs, want 1", len(got.outputs))
	}
	output := got.outputs[0]
	if output.dtype != dtype || !output.shape.Equal(shape) {
		t.Fatalf("output dtype/shape = %v/%v, want %v/%v", output.dtype, output.shape, dtype, shape)
	}
	if len(output.bits) != shape.Numel() {
		t.Fatalf("output backing len = %d, want exact fresh allocation of %d elements", len(output.bits), shape.Numel())
	}
}

func TestUnaryVJPBoundsExact(t *testing.T) {
	sizes := []struct {
		name  string
		shape tensor.Shape
	}{
		{"n0", tensor.Shape{0}},
		{"scalar", tensor.Shape{}},
		{"n1", tensor.Shape{1}},
		{"n3", tensor.Shape{3}},
		{"n7", tensor.Shape{7}},
		{"n8", tensor.Shape{8}},
		{"n9", tensor.Shape{9}},
		{"n31", tensor.Shape{31}},
		{"n32", tensor.Shape{32}},
		{"n33", tensor.Shape{33}},
		{"n2048", tensor.Shape{2048}},
		{"large65537", tensor.Shape{65537}},
	}
	for _, op := range boundsOps() {
		for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
			for _, size := range sizes {
				name := fmt.Sprintf("sizes/%s/%s/%s", op.name, dt, size.name)
				t.Run(name, func(t *testing.T) {
					boundsCheckValidCase(t, op, dt, size.shape, func() (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
						return boundsInputs(dt, dt, dt, size.shape, "dense", "dense", "dense")
					})
				})
			}
		}
	}

	layouts := []struct {
		name       string
		shape      tensor.Shape
		xl, yl, gl string
	}{
		{"prefix-n0", tensor.Shape{0}, "prefix", "prefix", "prefix"},
		{"prefix-x", tensor.Shape{9}, "prefix", "dense", "dense"},
		{"prefix-y", tensor.Shape{9}, "dense", "prefix", "dense"},
		{"prefix-g", tensor.Shape{9}, "dense", "dense", "prefix"},
		{"offset-x", tensor.Shape{9}, "offset", "dense", "dense"},
		{"offset-y", tensor.Shape{9}, "dense", "offset", "dense"},
		{"offset-g", tensor.Shape{9}, "dense", "dense", "offset"},
		{"transpose-x", tensor.Shape{3, 3}, "transpose", "dense", "dense"},
		{"transpose-y", tensor.Shape{3, 3}, "dense", "transpose", "dense"},
		{"transpose-g", tensor.Shape{3, 3}, "dense", "dense", "transpose"},
	}
	for _, op := range boundsOps() {
		for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
			for _, layout := range layouts {
				name := fmt.Sprintf("layouts/%s/%s/%s", op.name, dt, layout.name)
				t.Run(name, func(t *testing.T) {
					boundsCheckValidCase(t, op, dt, layout.shape, func() (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
						return boundsInputs(dt, dt, dt, layout.shape, layout.xl, layout.yl, layout.gl)
					})
				})
			}
		}
	}

	mixed := []struct {
		name          string
		xdt, ydt, gdt tensor.Dtype
	}{
		{"x32-y64-g64", tensor.F32, tensor.F64, tensor.F64},
		{"x64-y32-g32", tensor.F64, tensor.F32, tensor.F32},
		{"f16", tensor.F16, tensor.F16, tensor.F16},
		{"x16-y32-g64", tensor.F16, tensor.F32, tensor.F64},
	}
	for _, op := range boundsOps() {
		for _, mix := range mixed {
			name := fmt.Sprintf("generic/%s/%s", op.name, mix.name)
			t.Run(name, func(t *testing.T) {
				boundsCheckValidCase(t, op, mix.xdt, tensor.Shape{9}, func() (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
					return boundsInputs(mix.xdt, mix.ydt, mix.gdt, tensor.Shape{9}, "dense", "dense", "dense")
				})
			})
		}
	}

	t.Run("finite-normal/tanh/f32", func(t *testing.T) {
		tanh := boundsOps()[2]
		shape := tensor.Shape{3}
		boundsCheckValidCase(t, tanh, tensor.F32, shape, func() (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
			return boundsLayout(tensor.F32, shape, []float64{0.25, -0.5, 1.25}, "dense"),
				boundsLayout(tensor.F32, shape, []float64{0.5, -0.25, 0.75}, "dense"),
				boundsLayout(tensor.F32, shape, []float64{1, 2, -3}, "dense")
		})
	})

	for _, op := range boundsOps() {
		for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
			for _, short := range []struct {
				name       string
				xv, yv, gv []float64
				visits     int
			}{
				{"short-y", []float64{1, 2, 3, 4}, []float64{5, 6}, []float64{7, 8, 9, 10}, 2},
				{"short-g", []float64{1, 2, 3, 4}, []float64{5, 6, 7, 8}, []float64{9, 10}, 2},
			} {
				name := fmt.Sprintf("short/%s/%s/%s", op.name, dt, short.name)
				t.Run(name, func(t *testing.T) {
					got := boundsCheckCase(t, op, func() (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
						return boundsLayout(dt, tensor.Shape{len(short.xv)}, short.xv, "dense"),
							boundsLayout(dt, tensor.Shape{len(short.yv)}, short.yv, "dense"),
							boundsLayout(dt, tensor.Shape{len(short.gv)}, short.gv, "dense")
					})
					wantPanic := op.name != "relu" || short.name == "short-g"
					if (got.panic != "") != wantPanic {
						t.Fatalf("panic = %q, want occurrence %v", got.panic, wantPanic)
					}
					if op.name == "unary" && len(got.visits) != short.visits {
						t.Fatalf("unary callback visits = %d, want %d before panic", len(got.visits), short.visits)
					}
				})
			}
		}
	}

	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		relu := boundsOps()[1]
		for _, malformed := range []struct {
			name string
			xv   []float64
			gv   []float64
			pan  bool
		}{
			{"all-negative-short-g", []float64{-1, -2, -3, -4}, []float64{9}, false},
			{"positive-before-short-g", []float64{1, -2, -3, -4}, []float64{9}, false},
			{"positive-beyond-short-g", []float64{-1, -2, 3, -4}, []float64{9}, true},
			{"positive-before-and-beyond-short-g", []float64{1, -2, 3, -4}, []float64{9}, true},
		} {
			name := fmt.Sprintf("short/relu/%s/%s", dt, malformed.name)
			t.Run(name, func(t *testing.T) {
				got := boundsCheckCase(t, relu, func() (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
					return boundsLayout(dt, tensor.Shape{len(malformed.xv)}, malformed.xv, "dense"),
						boundsLayout(dt, tensor.Shape{len(malformed.xv)}, boundsValues(dt, len(malformed.xv), 3), "dense"),
						boundsLayout(dt, tensor.Shape{len(malformed.gv)}, malformed.gv, "dense")
				})
				if (got.panic != "") != malformed.pan {
					t.Fatalf("panic = %q, want occurrence %v", got.panic, malformed.pan)
				}
			})
		}
	}
}

var boundsBenchmarkSink *tensor.Tensor

func boundsBenchmarkTensor(dt tensor.Dtype, op backend.Op, n, seed int) *tensor.Tensor {
	x := tensor.New(dt, tensor.Shape{n})
	for i := 0; i < n; i++ {
		v := float64(((i+1)*(seed+5)*29)%251-125) / 31
		if op == backend.OpLog {
			v = math.Abs(v) + 0.25
		}
		x.SetF64(v, i)
	}
	return x
}

func boundsBenchmarkDtype(dt tensor.Dtype) string {
	if dt == tensor.F32 {
		return "F32"
	}
	return "F64"
}

func BenchmarkUnaryVJPBounds(b *testing.B) {
	ops := []struct {
		name string
		op   backend.Op
	}{
		{"ReLU", backend.OpReLU},
		{"Tanh", backend.OpTanh},
		{"Sigmoid", backend.OpSigmoid},
		{"Log", backend.OpLog},
	}
	for _, item := range ops {
		for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
			for _, n := range []int{0, 31, 2048, 262144} {
				b.Run(fmt.Sprintf("%s/%s/N%d", item.name, boundsBenchmarkDtype(dt), n), func(b *testing.B) {
					ctx := backend.NewContext()
					x := boundsBenchmarkTensor(dt, item.op, n, 1)
					out, err := backend.Execute(ctx, item.op, []*tensor.Tensor{x}, nil)
					if err != nil {
						b.Fatal(err)
					}
					g := boundsBenchmarkTensor(dt, backend.OpInvalid, n, 7)
					rule := vjps[item.op]
					in := []*tensor.Tensor{x}
					b.ReportAllocs()
					b.SetBytes(int64(n * dt.Size()))
					b.ResetTimer()
					for range b.N {
						grad, err := rule(ctx, in, out, nil, g)
						if err != nil {
							b.Fatal(err)
						}
						boundsBenchmarkSink = grad[0]
					}
				})
			}
		}
	}
}

func BenchmarkUnaryVJPBoundsTape(b *testing.B) {
	ops := []struct {
		name string
		op   backend.Op
	}{
		{"ReLU", backend.OpReLU},
		{"Tanh", backend.OpTanh},
		{"Sigmoid", backend.OpSigmoid},
		{"Log", backend.OpLog},
	}
	for _, item := range ops {
		for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
			for _, n := range []int{2048, 262144} {
				b.Run(fmt.Sprintf("%s/%s/N%d", item.name, boundsBenchmarkDtype(dt), n), func(b *testing.B) {
					tape := NewTape()
					ctx := tape.Context()
					x := boundsBenchmarkTensor(dt, item.op, n, 1)
					out, err := backend.Execute(ctx, item.op, []*tensor.Tensor{x}, nil)
					if err != nil {
						b.Fatal(err)
					}
					g := boundsBenchmarkTensor(dt, backend.OpInvalid, n, 7)
					b.ReportAllocs()
					b.SetBytes(int64(n * dt.Size()))
					b.ResetTimer()
					for range b.N {
						if err := tape.BackwardGrad(out[0], g); err != nil {
							b.Fatal(err)
						}
						boundsBenchmarkSink = tape.Grad(x)
					}
				})
			}
		}
	}
}
