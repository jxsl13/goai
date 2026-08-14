package backend

import (
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

type normalOnlyBackend struct{ name Name }

func (b *normalOnlyBackend) Name() Name            { return b.name }
func (b *normalOnlyBackend) Device() tensor.Device { return tensor.CPU() }
func (b *normalOnlyBackend) Synchronize() error    { return nil }
func (b *normalOnlyBackend) Kernel(op Op, dt tensor.Dtype) (Kernel, bool) {
	if op == OpAdd && dt == tensor.F64 {
		return idKernel, true
	}
	return nil, false
}

type intoTestDevice struct{ kind tensor.DeviceKind }

func (d intoTestDevice) Kind() tensor.DeviceKind   { return d.kind }
func (d intoTestDevice) String() string            { return d.kind.String() }
func (intoTestDevice) Allocator() tensor.Allocator { return tensor.Heap() }

func TestExecuteIntoHappyPathAndFallback(t *testing.T) {
	ctx := NewContext().WithBackend(activeMock)
	x := tensor.FromFloat64(tensor.Shape{3}, []float64{1, -2, 3})
	out := tensor.New(tensor.F64, x.Shape())
	inputs := []*tensor.Tensor{x}
	outputs := []*tensor.Tensor{out}

	if err := ExecuteInto(ctx, OpAdd, inputs, outputs, nil); err != nil {
		t.Fatalf("active into kernel: %v", err)
	}
	if got := out.Storage().F64(); got[0] != 1 || got[1] != -2 || got[2] != 3 {
		t.Fatalf("active into result = %v", got)
	}

	want, err := Execute(ctx, OpSiLU, inputs, nil)
	if err != nil {
		t.Fatalf("allocating CPU fallback: %v", err)
	}
	clear(out.Storage().F64())
	if err := ExecuteInto(ctx, OpSiLU, inputs, outputs, nil); err != nil {
		t.Fatalf("optimized CPU fallback into kernel: %v", err)
	}
	for i, got := range out.Storage().F64() {
		if got != want[0].Storage().F64()[i] {
			t.Fatalf("fallback into result[%d] = %v, want %v", i, got, want[0].Storage().F64()[i])
		}
	}
}

func TestExecuteIntoUnsupportedDoesNotRunAllocatingKernel(t *testing.T) {
	b := &normalOnlyBackend{name: "normal-only-into-test"}
	x := tensor.FromFloat64(tensor.Shape{2}, []float64{1, 2})
	out := tensor.FromFloat64(tensor.Shape{2}, []float64{7, 9})
	err := ExecuteInto(NewContext().WithBackend(b), OpAdd,
		[]*tensor.Tensor{x}, []*tensor.Tensor{out}, nil)
	if !errors.Is(err, ErrIntoUnsupported) {
		t.Fatalf("error = %v, want ErrIntoUnsupported", err)
	}
	if got := out.Storage().F64(); got[0] != 7 || got[1] != 9 {
		t.Fatalf("unsupported path mutated output: %v", got)
	}
}

func TestExecuteIntoRejectsUnsafeOutputsBeforeMutation(t *testing.T) {
	ctx := NewContext().WithBackend(activeMock)
	x3 := tensor.FromFloat64(tensor.Shape{3}, []float64{1, 2, 3})
	valid := func() *tensor.Tensor {
		return tensor.FromFloat64(tensor.Shape{3}, []float64{11, 12, 13})
	}

	base2 := tensor.New(tensor.F64, tensor.Shape{2, 2})
	transposed, err := base2.Transpose(0, 1)
	if err != nil {
		t.Fatal(err)
	}
	base4 := tensor.New(tensor.F64, tensor.Shape{4})
	offset, err := base4.Slice(0, 1, 4)
	if err != nil {
		t.Fatal(err)
	}
	released := valid()
	released.Storage().Release()
	releasedEmpty := tensor.New(tensor.F64, tensor.Shape{0})
	releasedEmpty.Storage().Release()
	otherDevice := tensor.NewOn(intoTestDevice{kind: tensor.KindMetal}, tensor.F64, tensor.Shape{3})

	tests := []struct {
		name    string
		ctx     *Context
		inputs  []*tensor.Tensor
		outputs []*tensor.Tensor
		want    string
	}{
		{name: "recorder", ctx: ctx.WithRecorder(&capRecorder{}), inputs: []*tensor.Tensor{x3}, outputs: []*tensor.Tensor{valid()}, want: "recorder"},
		{name: "nil", ctx: ctx, inputs: []*tensor.Tensor{x3}, outputs: []*tensor.Tensor{nil}, want: "nil"},
		{name: "input alias", ctx: ctx, inputs: []*tensor.Tensor{x3}, outputs: []*tensor.Tensor{x3}, want: "aliases input"},
		{name: "duplicate outputs", ctx: ctx, inputs: []*tensor.Tensor{x3}, outputs: []*tensor.Tensor{valid(), valid()}, want: "want one input and output"},
		{name: "same output twice", ctx: ctx, inputs: []*tensor.Tensor{x3}, outputs: func() []*tensor.Tensor { o := valid(); return []*tensor.Tensor{o, o} }(), want: "alias the same storage"},
		{name: "released", ctx: ctx, inputs: []*tensor.Tensor{x3}, outputs: []*tensor.Tensor{released}, want: "released"},
		{name: "released empty", ctx: ctx, inputs: []*tensor.Tensor{tensor.New(tensor.F64, tensor.Shape{0})}, outputs: []*tensor.Tensor{releasedEmpty}, want: "released"},
		{name: "noncontiguous", ctx: ctx, inputs: []*tensor.Tensor{tensor.New(tensor.F64, tensor.Shape{2, 2})}, outputs: []*tensor.Tensor{transposed}, want: "contiguous base"},
		{name: "offset", ctx: ctx, inputs: []*tensor.Tensor{x3}, outputs: []*tensor.Tensor{offset}, want: "contiguous base"},
		{name: "device", ctx: ctx, inputs: []*tensor.Tensor{x3}, outputs: []*tensor.Tensor{otherDevice}, want: "device"},
		{name: "dtype", ctx: ctx, inputs: []*tensor.Tensor{x3}, outputs: []*tensor.Tensor{tensor.New(tensor.F32, tensor.Shape{3})}, want: "dtype"},
		{name: "shape", ctx: ctx, inputs: []*tensor.Tensor{x3}, outputs: []*tensor.Tensor{tensor.New(tensor.F64, tensor.Shape{1, 3})}, want: "shape"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ExecuteInto(tt.ctx, OpAdd, tt.inputs, tt.outputs, nil)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want substring %q", err, tt.want)
			}
		})
	}
}

func fillInto(value float64) IntoKernel {
	return func(ctx *Context, in, out []*tensor.Tensor, _ Attrs) error {
		if len(in) != 1 || len(out) != 1 {
			return fmt.Errorf("fill into: want one input and output")
		}
		if err := ValidateOutput(ctx, out[0], in[0].Dtype(), in[0].Shape()); err != nil {
			return err
		}
		for i := range out[0].Storage().F64() {
			out[0].Storage().F64()[i] = value
		}
		return nil
	}
}

func TestExecuteIntoHonorsOpRouting(t *testing.T) {
	key := kernelKeyT{OpAdd, tensor.F64}
	home := &mockBackend{
		name:      "into-route-home",
		table:     map[kernelKeyT]Kernel{key: idKernel},
		intoTable: map[kernelKeyT]IntoKernel{key: fillInto(1)},
	}
	target := &mockBackend{
		name:      "into-route-target",
		table:     map[kernelKeyT]Kernel{key: idKernel},
		intoTable: map[kernelKeyT]IntoKernel{key: fillInto(2)},
	}
	Register(home)
	Register(target)

	x := tensor.New(tensor.F64, tensor.Shape{2})
	out := tensor.New(tensor.F64, tensor.Shape{2})
	ctx := NewContext().WithBackend(home).WithOpBackend(OpAdd, target.Name())
	if err := ExecuteInto(ctx, OpAdd, []*tensor.Tensor{x}, []*tensor.Tensor{out}, nil); err != nil {
		t.Fatal(err)
	}
	if got := out.Storage().F64(); got[0] != 2 || got[1] != 2 {
		t.Fatalf("routed result = %v, want target sentinel", got)
	}
}

func TestExecuteIntoConcurrentIndependentOutputs(t *testing.T) {
	ctx := NewContext().WithBackend(activeMock)
	x := tensor.FromFloat64(tensor.Shape{64}, make([]float64, 64))
	for i := range x.Storage().F64() {
		x.Storage().F64()[i] = float64(i)
	}

	const workers = 8
	errCh := make(chan error, workers)
	var wg sync.WaitGroup
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			out := tensor.New(tensor.F64, x.Shape())
			inputs := []*tensor.Tensor{x}
			outputs := []*tensor.Tensor{out}
			for range 100 {
				if err := ExecuteInto(ctx, OpAdd, inputs, outputs, nil); err != nil {
					errCh <- err
					return
				}
			}
			if got := out.Storage().F64()[63]; got != 63 {
				errCh <- fmt.Errorf("last output = %v, want 63", got)
			}
		}()
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Error(err)
	}
}
