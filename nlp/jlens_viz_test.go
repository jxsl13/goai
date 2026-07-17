package nlp_test

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/nlp"
)

// jlensVizFixture builds a tiny deterministic GPT + fitted lens + prompt for
// the §T814 structural tests.
func jlensVizFixture(t *testing.T) (*nlp.GPT, *nlp.JLens, []int) {
	t.Helper()
	model := jlensGPT(t, 16, 8, 8, 2, 2)
	lens, err := nlp.FitJLens(model, [][]int{{1, 2, 3, 4}, {5, 6, 7, 8}})
	if err != nil {
		t.Fatal(err)
	}
	return model, lens, []int{3, 1, 4, 1, 5}
}

// jlensVizData extracts the embedded application/json payload of a page.
func jlensVizData(t *testing.T, page string) map[string]any {
	t.Helper()
	m := regexp.MustCompile(`(?s)<script id="jlv-data" type="application/json">(.*?)</script>`).FindStringSubmatch(page)
	if m == nil {
		t.Fatal("page has no embedded jlv-data JSON")
	}
	var data map[string]any
	if err := json.Unmarshal([]byte(m[1]), &data); err != nil {
		t.Fatalf("embedded payload is not valid JSON: %v", err)
	}
	return data
}

// §T814 structure: the page carries the full layer × position grid (row/col
// counts as data attributes, one jlv-cell per grid entry, an "out" bottom
// row), and every token string — including hostile ones full of markup — is
// escaped everywhere it appears.
func TestJLensHTMLStructure(t *testing.T) {
	model, lens, tokens := jlensVizFixture(t)
	page, err := nlp.JLensHTML(model, lens, tokens,
		nlp.WithJLensHTMLTitle(`slice <&> "demo"`),
		nlp.WithJLensHTMLTokenText(func(id int) string { return fmt.Sprintf("<t%d>&", id) }))
	if err != nil {
		t.Fatal(err)
	}
	rows, cols := lens.Layers+1, len(tokens) // GPT lens: every layer fitted
	if want := fmt.Sprintf(`data-rows="%d" data-cols="%d"`, rows, cols); !strings.Contains(page, want) {
		t.Fatalf("page missing grid dimensions %s", want)
	}
	if got := strings.Count(page, `class="jlv-cell`); got != rows*cols {
		t.Fatalf("%d jlv-cell divs, want rows*cols = %d", got, rows*cols)
	}
	if !strings.Contains(page, `>out</div>`) {
		t.Fatal("bottom row is not labeled as the model output row")
	}
	if strings.Count(page, `class="jlv-rank"`) != rows*cols {
		t.Fatal("every cell must carry its vocab-rank superscript")
	}
	// Escaping: the raw token markup must never survive, in HTML or JSON.
	if strings.Contains(page, "<t1>") || strings.Contains(page, "<t3>") {
		t.Fatal("unescaped token text leaked into the page")
	}
	if !strings.Contains(page, "&lt;t3&gt;&amp;") {
		t.Fatal("escaped token text missing from the grid")
	}
	if !strings.Contains(page, `slice &lt;&amp;&gt; &#34;demo&#34;`) {
		t.Fatal("title not escaped")
	}
}

// §T814 data parity: the embedded JSON payload must match JLensSlice output
// field-for-field (layers, per-cell top-1, rank-of-output superscripts, and
// the model's actual output row), and the rank tensors must be consistent
// with the grid (rank of the output token per the tensor == the superscript).
func TestJLensHTMLEmbeddedData(t *testing.T) {
	model, lens, tokens := jlensVizFixture(t)
	page, err := nlp.JLensHTML(model, lens, tokens)
	if err != nil {
		t.Fatal(err)
	}
	data := jlensVizData(t, page)
	sl, err := lens.Slice(model, backend.NewContext(), tokens)
	if err != nil {
		t.Fatal(err)
	}
	ints := func(v any) []int {
		arr := v.([]any)
		out := make([]int, len(arr))
		for i, x := range arr {
			out[i] = int(x.(float64))
		}
		return out
	}
	grid := func(v any) [][]int {
		arr := v.([]any)
		out := make([][]int, len(arr))
		for i, x := range arr {
			out[i] = ints(x)
		}
		return out
	}
	eq := func(name string, got, want []int) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s: length %d != %d", name, len(got), len(want))
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("%s[%d]: %d != %d", name, i, got[i], want[i])
			}
		}
	}
	layers, top, rank := ints(data["layers"]), grid(data["top"]), grid(data["rank"])
	if len(layers) != len(sl.Rows) {
		t.Fatalf("embedded %d rows, JLensSlice has %d", len(layers), len(sl.Rows))
	}
	for i, row := range sl.Rows {
		if layers[i] != row.Layer {
			t.Fatalf("row %d layer %d != %d", i, layers[i], row.Layer)
		}
		eq(fmt.Sprintf("top[%d]", i), top[i], row.Top)
		eq(fmt.Sprintf("rank[%d]", i), rank[i], row.Rank)
	}
	eq("output", ints(data["output"]), sl.Output)

	// Cross-check: the embedded rank tensor of the output token reproduces
	// the superscript grid.
	ranks := data["ranks"].(map[string]any)
	for i := range sl.Rows {
		for p, out := range sl.Output {
			rt := grid(ranks[fmt.Sprint(out)])
			if rt[i][p] != rank[i][p] {
				t.Fatalf("rank tensor of output token %d at [%d,%d] = %d, superscript %d", out, i, p, rt[i][p], rank[i][p])
			}
		}
	}
}

// §T814 zero-dependency gate (§C2): the page must reference NOTHING external
// — no protocol URLs, no protocol-relative refs, no src/href attributes at
// all, no CSS imports or url() loads, no vendored library markers.
func TestJLensHTMLNoExternalRefs(t *testing.T) {
	model, lens, tokens := jlensVizFixture(t)
	page, err := nlp.JLensHTML(model, lens, tokens, nlp.WithJLensHTMLOccupancy(4))
	if err != nil {
		t.Fatal(err)
	}
	for _, bad := range []string{
		"http://", "https://", `"//`, "src=", "href=", "url(", "@import", "<link", "<iframe", "d3.", "cdn",
	} {
		if strings.Contains(page, bad) {
			t.Fatalf("page contains external-resource marker %q", bad)
		}
	}
	// Belt and braces: no "//" anywhere means no protocol-relative URL can
	// hide in any attribute (the inline JS uses block comments only).
	if strings.Contains(page, "//") {
		t.Fatal(`page contains "//" — potential protocol-relative reference`)
	}
}

// §T814 occupancy overlay: with WithJLensHTMLOccupancy the payload carries a
// [rows][cols] occupancy grid with plausible values (0..k) and every cell
// renders a corner badge; without the option there is no occ data or badge.
func TestJLensHTMLOccupancyOverlay(t *testing.T) {
	model, lens, tokens := jlensVizFixture(t)
	const k = 4
	page, err := nlp.JLensHTML(model, lens, tokens, nlp.WithJLensHTMLOccupancy(k))
	if err != nil {
		t.Fatal(err)
	}
	data := jlensVizData(t, page)
	occ, ok := data["occ"].([]any)
	if !ok {
		t.Fatal("occupancy payload missing")
	}
	rows, cols := lens.Layers+1, len(tokens)
	if len(occ) != rows {
		t.Fatalf("occ has %d rows, want %d", len(occ), rows)
	}
	for i, r := range occ {
		vals := r.([]any)
		if len(vals) != cols {
			t.Fatalf("occ row %d has %d cols, want %d", i, len(vals), cols)
		}
		for p, v := range vals {
			if n := int(v.(float64)); n < 0 || n > k {
				t.Fatalf("occ[%d][%d] = %d outside 0..%d", i, p, n, k)
			}
		}
	}
	if got := strings.Count(page, `class="jlv-occ"`); got != rows*cols {
		t.Fatalf("%d occupancy badges, want %d", got, rows*cols)
	}
	plain, err := nlp.JLensHTML(model, lens, tokens)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(plain, `class="jlv-occ"`) || jlensVizData(t, plain)["occ"] != nil {
		t.Fatal("occupancy overlay leaked into a page without the option")
	}
}

// §T814 determinism: two renders of the same inputs are byte-identical (the
// page embeds only integers, and JSON map keys are emitted sorted), so the
// runnable example's summary line is stable.
func TestJLensHTMLDeterministic(t *testing.T) {
	model, lens, tokens := jlensVizFixture(t)
	a, err := nlp.JLensHTML(model, lens, tokens, nlp.WithJLensHTMLOccupancy(4))
	if err != nil {
		t.Fatal(err)
	}
	b, err := nlp.JLensHTML(model, lens, tokens, nlp.WithJLensHTMLOccupancy(4))
	if err != nil {
		t.Fatal(err)
	}
	if a != b {
		t.Fatal("JLensHTML is not deterministic for identical inputs")
	}
}
