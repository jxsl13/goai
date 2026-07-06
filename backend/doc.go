// Package backend is layer L1: the compute abstraction.
//
// It defines the Backend and Kernel interfaces plus the registry and runtime
// feature detection. The Pure-Go reference backend (subpackage ref) is the
// source of numeric truth (§V9); accel backends are validated against it, never
// vice versa. The Backend interface defines an execution/sync model so a later
// async GPU backend does not break API stability (§V14).
package backend
