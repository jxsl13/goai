package nlp_test

import (
	"strings"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// schemaAccepts compiles schema to a grammar guide over a character vocabulary and walks
// text through it, reporting acceptance.
func schemaAccepts(t *testing.T, schema, text string) bool {
	t.Helper()
	gbnf, err := nlp.JSONSchemaToGrammar([]byte(schema))
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	vocab := asciiVocab()
	g, err := nlp.NewGrammarGuide(gbnf, vocab)
	if err != nil {
		t.Fatalf("grammar from schema did not parse: %v\n%s", err, gbnf)
	}
	st := g.Start()
	for _, c := range text {
		id := -1
		for i, v := range vocab {
			if v == string(c) {
				id = i
				break
			}
		}
		st = g.Advance(st, id)
		if st < 0 {
			return false
		}
	}
	return g.Accepting(st)
}

// §V16 tier-1 (§T575): the compiled grammar enforces types, required properties in
// order, optional omission, nested objects/arrays, enums and const — and rejects
// everything off-schema.
func TestJSONSchemaToGrammarConformance(t *testing.T) {
	const schema = `{
	  "type": "object",
	  "properties": {
	    "age":  {"type": "integer"},
	    "name": {"type": "string"},
	    "tags": {"type": "array", "items": {"type": "string"}}
	  },
	  "required": ["age", "name"]
	}`
	yes := []string{
		`{"age":30,"name":"ada"}`,
		`{"age": 0, "name": "x", "tags": ["a", "b"]}`,
		`{"age":-1,"name":"","tags":[]}`,
	}
	no := []string{
		`{"name":"ada","age":30}`,            // property order is fixed by declaration
		`{"age":30}`,                         // required property missing
		`{"age":3.5,"name":"ada"}`,           // integer must not take a fraction
		`{"age":30,"name":"ada",}`,           // trailing comma
		`{"age":30,"name":"ada","extra":1}`,  // additional properties disallowed
		`{"age":30,"name":"ada","tags":[1]}`, // wrong item type
		`"just a string"`,
	}
	for _, s := range yes {
		if !schemaAccepts(t, schema, s) {
			t.Errorf("%s must conform", s)
		}
	}
	for _, s := range no {
		if schemaAccepts(t, schema, s) {
			t.Errorf("%s must be rejected", s)
		}
	}
}

// §V16 tier-1: an all-optional object allows any in-order subset including {} — the
// comma placement construction (first present property carries no comma).
func TestJSONSchemaToGrammarOptionalSubsets(t *testing.T) {
	const schema = `{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"},"c":{"type":"integer"}}}`
	for _, s := range []string{`{}`, `{"a":1}`, `{"b":2}`, `{"c":3}`, `{"a":1,"c":3}`, `{"a":1,"b":2,"c":3}`} {
		if !schemaAccepts(t, schema, s) {
			t.Errorf("%s must conform", s)
		}
	}
	for _, s := range []string{`{"c":3,"a":1}`, `{"a":1,}`, `{,"a":1}`} {
		if schemaAccepts(t, schema, s) {
			t.Errorf("%s must be rejected", s)
		}
	}
}

// §V16 tier-1: enum/const/anyOf/type-list compile to exact alternations.
func TestJSONSchemaToGrammarEnumConstAnyOf(t *testing.T) {
	cases := []struct {
		schema string
		yes    []string
		no     []string
	}{
		{`{"enum": ["red", "green", 7, null]}`,
			[]string{`"red"`, `"green"`, `7`, `null`}, []string{`"blue"`, `8`, `"7"`}},
		{`{"const": {"ok": true}}`,
			[]string{`{"ok":true}`}, []string{`{"ok":false}`, `{}`}},
		{`{"anyOf": [{"type": "integer"}, {"type": "boolean"}]}`,
			[]string{`5`, `true`, `false`, `-12`}, []string{`5.5`, `"x"`, `null`}},
		{`{"type": ["string", "null"]}`,
			[]string{`"hi"`, `null`}, []string{`5`, `true`}},
	}
	for _, tc := range cases {
		for _, s := range tc.yes {
			if !schemaAccepts(t, tc.schema, s) {
				t.Errorf("schema %s: %s must conform", tc.schema, s)
			}
		}
		for _, s := range tc.no {
			if schemaAccepts(t, tc.schema, s) {
				t.Errorf("schema %s: %s must be rejected", tc.schema, s)
			}
		}
	}
}

// §V16 tier-1: deep nesting round-trips — schema-conforming documents with objects in
// arrays in objects are accepted, proving the CFG power end to end.
func TestJSONSchemaToGrammarNested(t *testing.T) {
	const schema = `{
	  "type": "object",
	  "properties": {
	    "matrix": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
	    "meta":   {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]}
	  },
	  "required": ["matrix", "meta"]
	}`
	doc := `{"matrix":[[1,2.5],[3]],"meta":{"id":9}}`
	if !schemaAccepts(t, schema, doc) {
		t.Fatalf("%s must conform", doc)
	}
	if schemaAccepts(t, schema, `{"matrix":[[1,"x"]],"meta":{"id":9}}`) {
		t.Fatal("wrong inner-array type must be rejected")
	}
}

// §V16 tier-1: fail-closed keywords — $ref and patternProperties cannot be ignored
// safely and must error; unknown keywords are ignored (grammar is a superset).
func TestJSONSchemaToGrammarUnsupported(t *testing.T) {
	for _, s := range []string{
		`{"$ref": "#/defs/x"}`,
		`{"type":"object","patternProperties":{"^a": {"type":"string"}}}`,
		`{"schema": false}x`, // invalid JSON
		`{"enum": []}`,
		`{"type": 7}`,
	} {
		if _, err := nlp.JSONSchemaToGrammar([]byte(s)); err == nil {
			t.Errorf("schema %s must be rejected", s)
		}
	}
	// minLength is ignored, not fatal: the grammar stays a superset.
	gbnf, err := nlp.JSONSchemaToGrammar([]byte(`{"type":"string","minLength":3}`))
	if err != nil {
		t.Fatalf("ignorable keyword must not error: %v", err)
	}
	if !strings.Contains(gbnf, "json-string") {
		t.Error("string schema must reference json-string")
	}
	// boolean schema true / absent type: any JSON value.
	if !schemaAccepts(t, `true`, `[1,{"a":null}]`) {
		t.Error("schema true must admit any JSON value")
	}
}

// FuzzJSONSchemaToGrammar: arbitrary bytes must never panic; every successfully compiled
// grammar must itself parse as GBNF (§V16 tier-2 hostile-input discipline).
func FuzzJSONSchemaToGrammar(f *testing.F) {
	f.Add([]byte(`{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}`))
	f.Add([]byte(`{"enum":[1,"two",null]}`))
	f.Add([]byte(`{"anyOf":[{"type":"integer"},{"const":[1,2]}]}`))
	f.Fuzz(func(t *testing.T, schema []byte) {
		gbnf, err := nlp.JSONSchemaToGrammar(schema)
		if err != nil {
			return
		}
		if _, err := nlp.NewGrammarGuide(gbnf, []string{"a", "{", "}"}); err != nil {
			t.Fatalf("compiled grammar does not parse: %v\nschema: %s\ngrammar:\n%s", err, schema, gbnf)
		}
	})
}
