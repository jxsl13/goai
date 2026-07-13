//go:build !darwin || !cgo

package main

// metalStatus: Metal exists only on darwin with cgo compiled in.
func metalStatus() string { return "unavailable (requires darwin + cgo)" }
