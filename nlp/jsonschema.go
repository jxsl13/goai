package nlp

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// JSONSchemaToGrammar compiles a JSON Schema (draft-07-style subset) into a GBNF grammar
// for GrammarGuide, so generation is FORCED to emit JSON that conforms to the schema —
// the "structured outputs" path of llama.cpp (json_schema_to_grammar), vLLM and SGLang
// (§T575). Use it when a model must return machine-parseable JSON: compile the schema,
// build a GrammarGuide over the vocabulary, and wrap the sampler.
//
// Supported schema subset: "type" (object, array, string, number, integer, boolean, null,
// or a list of these), "properties" + "required" (properties render in declaration order;
// optional ones may be omitted), "items", "enum" and "const" (any JSON values), "anyOf" /
// "oneOf" (alternation), and boolean/absent schemas (any JSON value). Unsupported keywords
// are IGNORED (the grammar is a superset: output may be less constrained, never more) —
// with two exceptions that cannot be ignored safely and return an error: "$ref" and
// "patternProperties". Objects allow no additional properties beyond "properties"
// (matching llama.cpp's default), and arrays don't enforce min/maxItems.
//
// The generated grammar allows minimal whitespace (single spaces/newlines between tokens)
// to keep outputs compact and the automaton small.
func JSONSchemaToGrammar(schema []byte) (string, error) {
	var v any
	if err := json.Unmarshal(schema, &v); err != nil {
		return "", fmt.Errorf("nlp: JSONSchemaToGrammar: %w", err)
	}
	c := &schemaCompiler{rules: map[string]string{}}
	root, err := c.visit(v, "root")
	if err != nil {
		return "", fmt.Errorf("nlp: JSONSchemaToGrammar: %w", err)
	}
	if root != "root" {
		c.rules["root"] = root
	}
	names := make([]string, 0, len(c.rules))
	for n := range c.rules {
		names = append(names, n)
	}
	sort.Strings(names)
	var sb strings.Builder
	fmt.Fprintf(&sb, "root ::= %s\n", c.rules["root"])
	for _, n := range names {
		if n != "root" {
			fmt.Fprintf(&sb, "%s ::= %s\n", n, c.rules[n])
		}
	}
	sb.WriteString(jsonPrimitives)
	return sb.String(), nil
}

// jsonPrimitives are the shared terminal rules every compiled schema builds on: the
// standard JSON string/number/integer/boolean/null grammars plus a compact space rule.
const jsonPrimitives = `json-value ::= json-object | json-array | json-string | json-number | ("true" | "false" | "null")
json-object ::= "{" sp (json-string sp ":" sp json-value (sp "," sp json-string sp ":" sp json-value)*)? sp "}"
json-array ::= "[" sp (json-value (sp "," sp json-value)*)? sp "]"
json-string ::= "\"" ([^"\\\x00-\x1f] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""
json-number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? (("e" | "E") ("-" | "+")? [0-9]+)?
json-integer ::= "-"? ("0" | [1-9] [0-9]*)
json-boolean ::= "true" | "false"
json-null ::= "null"
sp ::= (" " | "\n")?
`

type schemaCompiler struct {
	rules map[string]string // rule name → body (primitives excluded)
	n     int
}

// visit compiles one (sub)schema and returns the grammar EXPRESSION that matches it —
// either a primitive rule reference or the name of a fresh rule added to c.rules. hint
// names fresh rules readably (root, root-prop-x, root-item, ...).
func (c *schemaCompiler) visit(v any, hint string) (string, error) {
	switch s := v.(type) {
	case nil:
		return "json-value", nil
	case bool:
		if !s {
			return "", fmt.Errorf("schema `false` matches nothing — cannot generate")
		}
		return "json-value", nil
	case map[string]any:
		return c.visitObjectSchema(s, hint)
	}
	return "", fmt.Errorf("schema must be an object or boolean, got %T", v)
}

func (c *schemaCompiler) visitObjectSchema(s map[string]any, hint string) (string, error) {
	for _, bad := range []string{"$ref", "patternProperties"} {
		if _, ok := s[bad]; ok {
			return "", fmt.Errorf("unsupported schema keyword %q", bad)
		}
	}
	if v, ok := s["const"]; ok {
		return c.literalRule(v, hint)
	}
	if v, ok := s["enum"]; ok {
		items, ok := v.([]any)
		if !ok || len(items) == 0 {
			return "", fmt.Errorf("enum must be a non-empty array")
		}
		var alts []string
		for _, it := range items {
			lit, err := jsonLiteral(it)
			if err != nil {
				return "", err
			}
			alts = append(alts, lit)
		}
		return c.addRule(hint, strings.Join(alts, " | ")), nil
	}
	for _, key := range []string{"anyOf", "oneOf"} {
		if v, ok := s[key]; ok {
			items, ok := v.([]any)
			if !ok || len(items) == 0 {
				return "", fmt.Errorf("%s must be a non-empty array", key)
			}
			var alts []string
			for i, sub := range items {
				expr, err := c.visit(sub, fmt.Sprintf("%s-alt%d", hint, i))
				if err != nil {
					return "", err
				}
				alts = append(alts, expr)
			}
			return c.addRule(hint, strings.Join(alts, " | ")), nil
		}
	}
	switch t := s["type"].(type) {
	case nil:
		return "json-value", nil
	case string:
		return c.visitType(s, t, hint)
	case []any:
		var alts []string
		for i, tv := range t {
			name, ok := tv.(string)
			if !ok {
				return "", fmt.Errorf("type list entries must be strings")
			}
			expr, err := c.visitType(s, name, fmt.Sprintf("%s-t%d", hint, i))
			if err != nil {
				return "", err
			}
			alts = append(alts, expr)
		}
		return c.addRule(hint, strings.Join(alts, " | ")), nil
	}
	return "", fmt.Errorf(`"type" must be a string or list of strings`)
}

func (c *schemaCompiler) visitType(s map[string]any, typ, hint string) (string, error) {
	switch typ {
	case "string":
		return "json-string", nil
	case "number":
		return "json-number", nil
	case "integer":
		return "json-integer", nil
	case "boolean":
		return "json-boolean", nil
	case "null":
		return "json-null", nil
	case "array":
		item := "json-value"
		if it, ok := s["items"]; ok {
			expr, err := c.visit(it, hint+"-item")
			if err != nil {
				return "", err
			}
			item = expr
		}
		body := fmt.Sprintf(`"[" sp (%s (sp "," sp %s)*)? sp "]"`, item, item)
		return c.addRule(hint, body), nil
	case "object":
		return c.visitObjectType(s, hint)
	}
	return "", fmt.Errorf("unsupported type %q", typ)
}

// visitObjectType renders an object with fixed properties: required properties appear in
// declaration order; optional properties may be omitted but keep their relative order.
// With no required property the first present optional carries no leading comma, so each
// "first" choice becomes an alternate (the llama.cpp construction).
func (c *schemaCompiler) visitObjectType(s map[string]any, hint string) (string, error) {
	props, _ := s["properties"].(map[string]any)
	if len(props) == 0 {
		return c.addRule(hint, `"{" sp "}"`), nil
	}
	required := map[string]bool{}
	if rv, ok := s["required"].([]any); ok {
		for _, r := range rv {
			if name, ok := r.(string); ok {
				required[name] = true
			}
		}
	}
	names := make([]string, 0, len(props))
	for n := range props {
		names = append(names, n)
	}
	sort.Strings(names) // JSON maps are unordered in Go: sorted = stable declaration order
	kv := make(map[string]string, len(names))
	for _, n := range names {
		expr, err := c.visit(props[n], hint+"-prop-"+sanitizeRuleName(n))
		if err != nil {
			return "", err
		}
		key, err := jsonLiteral(n)
		if err != nil {
			return "", err
		}
		kv[n] = fmt.Sprintf(`%s sp ":" sp %s`, key, expr)
	}
	var reqNames, optNames []string
	for _, n := range names {
		if required[n] {
			reqNames = append(reqNames, n)
		} else {
			optNames = append(optNames, n)
		}
	}
	var body strings.Builder
	body.WriteString(`"{" sp `)
	switch {
	case len(reqNames) > 0:
		for i, n := range reqNames {
			if i > 0 {
				body.WriteString(` sp "," sp `)
			}
			body.WriteString(kv[n])
		}
		for _, n := range optNames {
			fmt.Fprintf(&body, ` (sp "," sp %s)?`, kv[n])
		}
	default:
		// all optional: alternate on which property appears first.
		var alts []string
		for i, first := range optNames {
			var alt strings.Builder
			alt.WriteString(kv[first])
			for _, later := range optNames[i+1:] {
				fmt.Fprintf(&alt, ` (sp "," sp %s)?`, kv[later])
			}
			alts = append(alts, alt.String())
		}
		fmt.Fprintf(&body, "(%s)?", strings.Join(alts, " | "))
	}
	body.WriteString(` sp "}"`)
	return c.addRule(hint, body.String()), nil
}

// literalRule compiles a const value to a rule matching exactly its JSON encoding.
func (c *schemaCompiler) literalRule(v any, hint string) (string, error) {
	lit, err := jsonLiteral(v)
	if err != nil {
		return "", err
	}
	return c.addRule(hint, lit), nil
}

// addRule installs body under a collision-free variant of hint and returns the rule name.
func (c *schemaCompiler) addRule(hint, body string) string {
	name := sanitizeRuleName(hint)
	if _, taken := c.rules[name]; taken {
		c.n++
		name = fmt.Sprintf("%s-%d", name, c.n)
	}
	c.rules[name] = body
	return name
}

// jsonLiteral renders any JSON value as a GBNF string-literal sequence matching its
// canonical encoding.
func jsonLiteral(v any) (string, error) {
	enc, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	var sb strings.Builder
	sb.WriteByte('"')
	for _, r := range string(enc) {
		switch r {
		case '"':
			sb.WriteString(`\"`)
		case '\\':
			sb.WriteString(`\\`)
		case '\n':
			sb.WriteString(`\n`)
		case '\r':
			sb.WriteString(`\r`)
		case '\t':
			sb.WriteString(`\t`)
		default:
			sb.WriteRune(r)
		}
	}
	sb.WriteByte('"')
	return sb.String(), nil
}

// sanitizeRuleName maps arbitrary property names onto the GBNF identifier alphabet.
func sanitizeRuleName(s string) string {
	var sb strings.Builder
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_':
			sb.WriteRune(r)
		default:
			sb.WriteByte('x')
		}
	}
	if sb.Len() == 0 {
		return "p"
	}
	out := sb.String()
	if c := out[0]; c >= '0' && c <= '9' {
		out = "p" + out
	}
	return out
}
