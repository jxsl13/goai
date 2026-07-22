package nlp_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// BenchmarkJSONSchemaToGrammar measures compiling a non-trivial nested JSON schema (many
// object properties across nested definitions) into a GBNF grammar — the setup step of
// schema-constrained generation. The final grammar buffer is now pre-sized to the exact
// rule text instead of doubling through reallocations.
func BenchmarkJSONSchemaToGrammar(b *testing.B) {
	// Build a wide, nested object schema so the compiled grammar spans many rules.
	var props []string
	for i := 0; i < 24; i++ {
		props = append(props, fmt.Sprintf(`"field_%d": {"type": %q}`, i, []string{"string", "number", "integer", "boolean"}[i%4]))
	}
	inner := `{"type": "object", "properties": {` + strings.Join(props, ", ") + `}}`
	schema := []byte(`{"type": "object", "properties": {
		"a": ` + inner + `,
		"b": ` + inner + `,
		"c": {"type": "array", "items": ` + inner + `},
		"d": {"type": "string"}, "e": {"type": "number"}, "f": {"type": "boolean"}
	}}`)
	if _, err := nlp.JSONSchemaToGrammar(schema); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := nlp.JSONSchemaToGrammar(schema); err != nil {
			b.Fatal(err)
		}
	}
}
