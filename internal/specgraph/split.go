package main

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// splitFile cuts a rendered spec file at every `^## ` header. Fragment 0 is
// the preamble (everything before the first header, may be empty); each
// later fragment runs from its header to the byte before the next one, with
// the single separator blank line stripped (renderOutput's join re-adds it).
func splitFile(content string) []string {
	lines := strings.Split(content, "\n")
	var cuts []int
	for i, ln := range lines {
		if strings.HasPrefix(ln, "## ") {
			cuts = append(cuts, i)
		}
	}
	bounds := append([]int{0}, cuts...)
	bounds = append(bounds, len(lines))
	var frags []string
	for i := 0; i+1 < len(bounds); i++ {
		start, end := bounds[i], bounds[i+1]
		if i == 0 && len(cuts) > 0 && start == cuts[0] {
			continue // no preamble
		}
		seg := lines[start:end]
		// strip the trailing separator blank line of every non-last fragment
		for len(seg) > 0 && seg[len(seg)-1] == "" {
			seg = seg[:len(seg)-1]
		}
		frags = append(frags, strings.Join(seg, "\n")+"\n")
	}
	return frags
}

// sectionFileName maps a fragment to its spec/ filename. Known sections get
// the canonical numbered name; unknown ones get a stable slug (keeps split
// usable for future sections without a code change).
func sectionFileName(fragment string, isWorker bool) string {
	head := strings.SplitN(fragment, "\n", 2)[0]
	if !strings.HasPrefix(head, "## ") {
		return "00-preamble.md"
	}
	m := regexp.MustCompile(`^## §?([A-Za-z]+)`).FindStringSubmatch(head)
	key := ""
	if m != nil {
		key = m[1]
	}
	if !isWorker {
		switch key {
		case "G":
			return "10-goals.md"
		case "C":
			return "20-constraints.md"
		case "I":
			return "30-arch-invariants.md"
		case "R":
			return "40-research.md"
		case "V":
			return "50-verification.md"
		case "B":
			return "60-backprop.md"
		case "T":
			return "70-tasks.md"
		}
	} else {
		switch key {
		case "RUN":
			return "10-run.md"
		case "MODELS":
			return "20-models.md"
		case "H":
			return "30-hardware.md"
		case "Iw":
			return "40-invariants.md"
		case "CPU":
			return "50-cpu.md"
		case "GPU":
			return "60-gpu.md"
		case "GAP":
			return "65-gap.md"
		case "PERF":
			return "70-perf.md"
		case "Tw":
			return "80-tasks.md"
		case "GOAL":
			return "90-goal.md"
		case "NEXT":
			return "95-next.md"
		}
	}
	return "85-" + strings.ToLower(regexp.MustCompile(`[^A-Za-z0-9]+`).ReplaceAllString(key, "-")) + ".md"
}

// splitOne builds the fragment set for one rendered file and proves the
// byte round-trip before anything is written.
func splitOne(content, dir string, isWorker bool) (Output, error) {
	o := Output{}
	for _, frag := range splitFile(content) {
		name := sectionFileName(frag, isWorker)
		o.Fragments = append(o.Fragments, Fragment{Path: filepath.ToSlash(filepath.Join(dir, name)), Content: frag})
	}
	// fragments must already be in lexicographic filename order, or render
	// order != source order and the round-trip below fails loudly.
	back, err := renderOutput(Output{RenderPath: "SPEC.md", Fragments: o.Fragments})
	if isWorker {
		back, err = renderOutput(Output{RenderPath: "SPEC-worker-x.md", Fragments: o.Fragments})
	}
	if err != nil {
		return o, err
	}
	if back != content {
		return o, fmt.Errorf("specgraph: split round-trip NOT byte-identical for %s — refusing to write (§V40)", dir)
	}
	return o, nil
}

// cmdSplit performs the one-time migration (or, with force, a re-import of
// hand-edited rendered files during worker phase A): SPEC.md and every
// SPEC-worker-*.md are cut into spec/ fragments, with the byte round-trip
// asserted per file before any write.
func cmdSplit(w *strings.Builder, root string, force bool) int {
	if HasSpecDir(root) && !force {
		fmt.Fprintln(w, "split: spec/ already exists — use -force to re-import from the rendered files")
		return 1
	}
	type job struct {
		src, dir string
		worker   bool
	}
	jobs := []job{{"SPEC.md", specDirName, false}}
	workers, _ := filepath.Glob(filepath.Join(root, "SPEC-worker-*.md"))
	for _, p := range workers {
		name := strings.TrimSuffix(strings.TrimPrefix(filepath.Base(p), "SPEC-worker-"), ".md")
		jobs = append(jobs, job{filepath.Base(p), filepath.Join(specDirName, workerDirName, name), true})
	}
	var outs []Output
	for _, j := range jobs {
		b, err := os.ReadFile(filepath.Join(root, j.src))
		if err != nil {
			fmt.Fprintf(w, "split: %v\n", err)
			return 1
		}
		o, err := splitOne(string(b), j.dir, j.worker)
		if err != nil {
			fmt.Fprintf(w, "split: %v\n", err)
			return 1
		}
		outs = append(outs, o)
	}
	for _, o := range outs {
		for _, f := range o.Fragments {
			full := filepath.Join(root, filepath.FromSlash(f.Path))
			if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
				fmt.Fprintf(w, "split: %v\n", err)
				return 1
			}
			if err := os.WriteFile(full, []byte(f.Content), 0o644); err != nil {
				fmt.Fprintf(w, "split: %v\n", err)
				return 1
			}
			fmt.Fprintf(w, "wrote %s (%d bytes)\n", f.Path, len(f.Content))
		}
	}
	fmt.Fprintln(w, "split: round-trip proven byte-identical for every file")
	return 0
}
