// Command leadership validates, renders, and collects the evidence behind the
// M2-first performance leadership matrix. It deliberately refuses to promote a
// row whose source versions or repeated-sample protocol are incomplete.
package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
)

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "leadership:", err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) error {
	if len(args) == 0 {
		return errors.New("usage: leadership check|render|collect-silu [flags]")
	}
	switch args[0] {
	case "check", "render":
		fs := flag.NewFlagSet(args[0], flag.ContinueOnError)
		fs.SetOutput(stderr)
		manifest := fs.String("manifest", defaultManifest, "leadership matrix JSON")
		if err := fs.Parse(args[1:]); err != nil {
			return err
		}
		m, err := loadMatrix(*manifest)
		if err != nil {
			return err
		}
		report := validateMatrix(m)
		for _, warning := range report.Warnings {
			fmt.Fprintln(stderr, "warning:", warning)
		}
		if len(report.Errors) > 0 {
			for _, problem := range report.Errors {
				fmt.Fprintln(stderr, "error:", problem)
			}
			return fmt.Errorf("matrix has %d validation error(s)", len(report.Errors))
		}
		if args[0] == "render" {
			_, err := io.WriteString(stdout, renderMatrix(m))
			return err
		} else {
			fmt.Fprintf(stdout, "leadership matrix: %d cells valid (%d provisional warning(s))\n", len(m.Cells), len(report.Warnings))
		}
		return nil
	case "collect-silu":
		return collectSiLU(args[1:], stdout, stderr)
	default:
		return fmt.Errorf("unknown command %q", args[0])
	}
}
