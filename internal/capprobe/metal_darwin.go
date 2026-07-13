//go:build darwin && cgo

package main

import "github.com/jxsl13/goai/backend/metal"

// metalStatus asks the Metal backend itself — the same Available() gate the
// tests use, so the probe and the skips can never disagree.
func metalStatus() string {
	if metal.Available() {
		return "available (device present)"
	}
	return "unavailable (no Metal device)"
}
