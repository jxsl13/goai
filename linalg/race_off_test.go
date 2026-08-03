//go:build !race

package linalg_test

// raceEnabled is false in a normal build. See its counterpart in race_on_test.go for why any
// test needs to know.
const raceEnabled = false
