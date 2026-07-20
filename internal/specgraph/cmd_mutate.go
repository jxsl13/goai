package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// applyMutation is the single write path for every entity mutation (§V40):
// load the corpus, mutate IN MEMORY, run the full verify suite on the
// would-be state, and only when clean write the touched fragments AND the
// regenerated rendered views in one pass. A mutation can therefore never
// leave spec/ and SPEC.md out of sync, and a violating mutation writes
// nothing at all (TestMutationVerifyGate).
func applyMutation(w *strings.Builder, root string, dryRun bool, mutate func(c *Corpus) error) int {
	if !HasSpecDir(root) {
		fmt.Fprintln(w, "no spec/ hierarchy — run `specgraph split` first (mutations require the source tree, §V40)")
		return 1
	}
	c, err := LoadCorpus(root)
	if err != nil {
		fmt.Fprintf(w, "%v\n", err)
		return 1
	}
	before := map[string]string{}
	for _, o := range c.Outputs {
		for _, f := range o.Fragments {
			before[f.Path] = f.Content
		}
	}
	if err := mutate(c); err != nil {
		fmt.Fprintf(w, "%v\n", err)
		return 1
	}
	problems, err := verifyCorpus(root, c)
	if err != nil {
		fmt.Fprintf(w, "%v\n", err)
		return 1
	}
	if len(problems) > 0 {
		for _, p := range problems {
			fmt.Fprintf(w, "%s\n", p)
		}
		fmt.Fprintln(w, "mutation rejected — nothing written")
		return 1
	}
	rendered, err := renderAll(c)
	if err != nil {
		fmt.Fprintf(w, "%v\n", err)
		return 1
	}
	if dryRun {
		for _, o := range c.Outputs {
			for _, f := range o.Fragments {
				if before[f.Path] != f.Content {
					fmt.Fprintf(w, "would write %s\n", f.Path)
				}
			}
		}
		fmt.Fprintln(w, "dry-run: verify clean, nothing written")
		return 0
	}
	for _, o := range c.Outputs {
		for _, f := range o.Fragments {
			if before[f.Path] == f.Content {
				continue
			}
			full := filepath.Join(root, filepath.FromSlash(f.Path))
			if err := os.WriteFile(full, []byte(f.Content), 0o644); err != nil {
				fmt.Fprintf(w, "%v\n", err)
				return 1
			}
		}
	}
	if err := writeRendered(root, rendered); err != nil {
		fmt.Fprintf(w, "%v\n", err)
		return 1
	}
	fmt.Fprintln(w, "verify: clean")
	return 0
}

// mutFlags are the shared mutation options parsed by main.go.
type mutFlags struct {
	Cites    string
	Priority string
	Status   string
	ID       string
	Worker   string
	Cause    string
	Fix      string
	Date     string
	Guards   string
	Claim    string
	Source   string
	Conf     string
	Tag      string
	Text     string
	State    string
	Force    bool
	DryRun   bool
}

func cmdNextID(w *strings.Builder, root string, class string) int {
	if class == "" {
		fmt.Fprintln(w, "next-id: class required (T|B|R|V|G|C|I)")
		return 2
	}
	c, err := LoadCorpus(root)
	if err != nil {
		fmt.Fprintf(w, "%v\n", err)
		return 1
	}
	fmt.Fprintf(w, "%s%d\n", class, nextID(c, class))
	return 0
}

func cmdTaskAdd(w *strings.Builder, root, text string, fl mutFlags) int {
	if strings.TrimSpace(text) == "" {
		fmt.Fprintln(w, "task add: text required")
		return 2
	}
	return applyMutation(w, root, fl.DryRun, func(c *Corpus) error {
		if fl.Worker != "" {
			o := c.output("SPEC-worker-" + fl.Worker + ".md")
			if o == nil {
				return fmt.Errorf("unknown worker %q", fl.Worker)
			}
			id := fmt.Sprintf("Tw%d", nextWorkerID(o))
			row, err := buildTwRow(id, orStr(fl.Status, "."), text, fl.Cites)
			if err != nil {
				return err
			}
			f := lastFragment(o, "tasks")
			f.Content = strings.TrimRight(f.Content, "\n") + "\n" + row + "\n"
			fmt.Fprintf(w, "%s allocated\n%s  %s\n", id, f.Path, row)
			return nil
		}
		id := fl.ID
		if id == "" {
			id = fmt.Sprintf("T%d", nextID(c, "T"))
		}
		row, err := buildTaskRow(id, orStr(fl.Status, "."), text, fl.Cites, orStr(fl.Priority, "med"))
		if err != nil {
			return err
		}
		f := c.fragmentByPath(classFile["T"])
		if f == nil {
			return fmt.Errorf("missing %s", classFile["T"])
		}
		if err := insertRow(f, "T", row); err != nil {
			return err
		}
		fmt.Fprintf(w, "%s allocated\n%s  %s\n", id, f.Path, truncate(row, 160))
		return nil
	})
}

func cmdTaskSetStatus(w *strings.Builder, root, id, to string, fl mutFlags) int {
	return applyMutation(w, root, fl.DryRun, func(c *Corpus) error {
		l, ok := findEntry(c, id)
		if !ok {
			return fmt.Errorf("no entry %s", id)
		}
		from, err := setStatus(l, to, fl.Force)
		if err != nil {
			return err
		}
		fmt.Fprintf(w, "%s status %s -> %s (%s)\n", id, from, to, l.Frag.Path)
		return nil
	})
}

func cmdTaskEdit(w *strings.Builder, root, id string, fl mutFlags) int {
	return applyMutation(w, root, fl.DryRun, func(c *Corpus) error {
		l, ok := findEntry(c, id)
		if !ok {
			return fmt.Errorf("no entry %s", id)
		}
		cells := splitRow(l.entryLine())
		if len(cells) < 4 {
			return fmt.Errorf("%s is not a task row", id)
		}
		if fl.Text != "" {
			cells[2] = fl.Text
		}
		if fl.Cites != "" {
			if err := validCites(fl.Cites); err != nil {
				return err
			}
			cells[3] = fl.Cites
		}
		if fl.State != "" && len(cells) >= 5 {
			cells[4] = fl.State
		}
		if fl.Priority != "" && len(cells) >= 6 {
			cells[5] = fl.Priority
		}
		row, err := joinRow(cells)
		if err != nil {
			return err
		}
		replaceLine(l, row)
		fmt.Fprintf(w, "%s edited (%s)\n", id, l.Frag.Path)
		return nil
	})
}

func cmdBugAdd(w *strings.Builder, root string, fl mutFlags) int {
	if fl.Cause == "" || fl.Fix == "" {
		fmt.Fprintln(w, "bug add: -cause and -fix required")
		return 2
	}
	return applyMutation(w, root, fl.DryRun, func(c *Corpus) error {
		id := fmt.Sprintf("B%d", nextID(c, "B"))
		row, err := buildBugRow(id, orStr(fl.Date, time.Now().UTC().Format("2006-01-02")), fl.Cause, fl.Fix, fl.Guards)
		if err != nil {
			return err
		}
		f := c.fragmentByPath(classFile["B"])
		if f == nil {
			return fmt.Errorf("missing %s", classFile["B"])
		}
		if err := insertRow(f, "B", row); err != nil {
			return err
		}
		fmt.Fprintf(w, "%s allocated\n%s  %s\n", id, f.Path, truncate(row, 160))
		return nil
	})
}

func cmdResearchAdd(w *strings.Builder, root string, fl mutFlags) int {
	if fl.Claim == "" || fl.Source == "" {
		fmt.Fprintln(w, "research add: -claim and -source required")
		return 2
	}
	return applyMutation(w, root, fl.DryRun, func(c *Corpus) error {
		id := fmt.Sprintf("R%d", nextID(c, "R"))
		row, err := buildResearchRow(id, fl.Claim, fl.Source, orStr(fl.Conf, "med"))
		if err != nil {
			return err
		}
		f := c.fragmentByPath(classFile["R"])
		if f == nil {
			return fmt.Errorf("missing %s", classFile["R"])
		}
		if err := insertRow(f, "R", row); err != nil {
			return err
		}
		fmt.Fprintf(w, "%s allocated\n%s  %s\n", id, f.Path, truncate(row, 160))
		return nil
	})
}

func cmdVerifAdd(w *strings.Builder, root, text string, fl mutFlags) int {
	if strings.TrimSpace(text) == "" || fl.Tag == "" {
		fmt.Fprintln(w, "verif add: -tag and text required")
		return 2
	}
	return applyMutation(w, root, fl.DryRun, func(c *Corpus) error {
		id := fl.ID
		if id == "" {
			id = fmt.Sprintf("V%d", nextID(c, "V"))
		}
		f := c.fragmentByPath(classFile["V"])
		if f == nil {
			return fmt.Errorf("missing %s", classFile["V"])
		}
		appendDef(f, id+" "+fl.Tag+": "+strings.TrimSpace(text))
		fmt.Fprintf(w, "%s allocated (%s)\n", id, f.Path)
		return nil
	})
}

// cmdDefAdd covers goal/constraint/archinv add.
func cmdDefAdd(w *strings.Builder, root, class, text string, fl mutFlags) int {
	if strings.TrimSpace(text) == "" {
		fmt.Fprintf(w, "%s add: text required\n", strings.ToLower(class))
		return 2
	}
	return applyMutation(w, root, fl.DryRun, func(c *Corpus) error {
		id := fl.ID
		if id == "" {
			id = fmt.Sprintf("%s%d", class, nextID(c, class))
		}
		f := c.fragmentByPath(classFile[class])
		if f == nil {
			return fmt.Errorf("missing %s", classFile[class])
		}
		sep := ": "
		if strings.HasPrefix(id, "I.L") {
			sep = " "
		}
		appendDef(f, id+sep+strings.TrimSpace(text))
		fmt.Fprintf(w, "%s allocated (%s)\n", id, f.Path)
		return nil
	})
}

func cmdDefEdit(w *strings.Builder, root, id, text string, fl mutFlags) int {
	if strings.TrimSpace(text) == "" {
		fmt.Fprintln(w, "edit: new text required")
		return 2
	}
	return applyMutation(w, root, fl.DryRun, func(c *Corpus) error {
		l, ok := findEntry(c, id)
		if !ok {
			return fmt.Errorf("no entry %s", id)
		}
		old := l.entryLine()
		switch {
		case strings.HasPrefix(old, "| "): // table row: edit text cell via task edit
			return fmt.Errorf("%s is a table row — use `task edit`", id)
		case strings.HasPrefix(id, "V"):
			// keep the canonical `V<n> TAG:` prefix; -tag overrides it.
			tag := fl.Tag
			if tag == "" {
				if m := reVDef.FindStringSubmatch(old); m != nil {
					tag = m[2]
				}
			}
			if tag == "" {
				return fmt.Errorf("%s has no TAG — pass -tag", id)
			}
			replaceLine(l, id+" "+tag+": "+strings.TrimSpace(text))
		default:
			sep := ": "
			if strings.HasPrefix(id, "I.L") {
				sep = " "
			}
			replaceLine(l, id+sep+strings.TrimSpace(text))
		}
		fmt.Fprintf(w, "%s edited (%s)\n", id, l.Frag.Path)
		return nil
	})
}

func cmdEntryGet(w *strings.Builder, root, id string) int {
	if !HasSpecDir(root) {
		fmt.Fprintln(w, "no spec/ hierarchy")
		return 1
	}
	c, err := LoadCorpus(root)
	if err != nil {
		fmt.Fprintf(w, "%v\n", err)
		return 1
	}
	l, ok := findEntry(c, id)
	if !ok {
		fmt.Fprintf(w, "no entry %s\n", id)
		return 0
	}
	fmt.Fprintf(w, "%s:%d\n%s\n", l.Frag.Path, l.Line+1, l.entryLine())
	return 0
}

func cmdEntryRm(w *strings.Builder, root, id string, fl mutFlags) int {
	return applyMutation(w, root, fl.DryRun, func(c *Corpus) error {
		l, ok := findEntry(c, id)
		if !ok {
			return fmt.Errorf("no entry %s", id)
		}
		if refs := strongRefsTo(root, c, id); len(refs) > 0 && !fl.Force {
			return fmt.Errorf("%s has strong references from %s — removing would dangle them (§V39); use -force to override", id, strings.Join(refs, ","))
		}
		replaceLine(l, "")
		fmt.Fprintf(w, "%s removed (%s)\n", id, l.Frag.Path)
		return nil
	})
}

// lastFragment returns the output's fragment whose name contains key,
// falling back to the last fragment.
func lastFragment(o *Output, key string) *Fragment {
	for i := range o.Fragments {
		if strings.Contains(o.Fragments[i].Path, key) {
			return &o.Fragments[i]
		}
	}
	return &o.Fragments[len(o.Fragments)-1]
}

func orStr(v, d string) string {
	if v == "" {
		return d
	}
	return v
}
