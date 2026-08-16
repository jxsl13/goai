//go:build race

package safetensors_test

// raceEnabled reports whether this binary was built with -race. The detector's shadow-memory
// bookkeeping inflates runtime.MemStats.TotalAlloc well past what the code under test allocates,
// so allocation-budget guards measure the instrumentation rather than the program.
const raceEnabled = true
