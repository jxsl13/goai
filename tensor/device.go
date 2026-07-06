package tensor

// DeviceKind identifies a class of compute device. CPU ships now; the accel
// kinds are placeholders wired up by layer L1b behind build tags (§I.L1b).
type DeviceKind uint8

const (
	KindCPU DeviceKind = iota
	KindCUDA
	KindMetal
	KindVulkan
)

func (k DeviceKind) String() string {
	switch k {
	case KindCPU:
		return "cpu"
	case KindCUDA:
		return "cuda"
	case KindMetal:
		return "metal"
	case KindVulkan:
		return "vulkan"
	default:
		return "device(?)"
	}
}

// Device is where a Tensor's memory lives and which Allocator services it. The
// public API is device-agnostic (§I1); accel backends supply their own Device
// implementations without leaking internals to higher layers.
type Device interface {
	Kind() DeviceKind
	String() string
	// Allocator returns the allocator that owns this device's memory.
	Allocator() Allocator
}

// cpuDevice is the host device. Its allocator is configurable so callers can
// opt into pooling for hot paths while the default stays a plain heap.
type cpuDevice struct {
	alloc Allocator
}

func (d cpuDevice) Kind() DeviceKind    { return KindCPU }
func (d cpuDevice) String() string      { return "cpu" }
func (d cpuDevice) Allocator() Allocator { return d.alloc }

// defaultCPU is the process-wide CPU device backed by the heap allocator.
var defaultCPU = cpuDevice{alloc: Heap()}

// CPU returns the default host device (heap-allocated, no pooling).
func CPU() Device { return defaultCPU }

// NewCPUDevice returns a CPU device backed by the given allocator (e.g. a Pool
// for reuse-heavy workloads). A nil allocator falls back to the heap.
func NewCPUDevice(alloc Allocator) Device {
	if alloc == nil {
		alloc = Heap()
	}
	return cpuDevice{alloc: alloc}
}
