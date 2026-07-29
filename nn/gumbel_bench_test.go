package nn_test

import (
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

func BenchmarkSampleGumbelNoise(b *testing.B) {
	shape := tensor.Shape{256, 1024}
	b.ResetTimer()
	for range b.N {
		gsink2 = nn.SampleGumbelNoise(1, shape)
	}
}

var gsink2 *tensor.Tensor
