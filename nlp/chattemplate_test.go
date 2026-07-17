package nlp_test

import (
	"strings"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// goldenTranscript is the fixture conversation for all golden renders.
var goldenTranscript = []nlp.ChatMessage{
	{Role: "system", Content: "You are helpful."},
	{Role: "user", Content: "Hi!"},
	{Role: "assistant", Content: "Hello!"},
	{Role: "user", Content: "Bye?"},
}

// §V16 tier-1 (§T576): byte-exact parity with the HF transformers apply_chat_template
// reference — goldens produced by rendering the REAL repo templates (Qwen2.5, Meta-
// Llama-3-8B-Instruct, gemma-it, Mistral-7B-Instruct-v0.3, Phi-3-mini-4k-instruct)
// with jinja2 3.1.6, add_generation_prompt=true. gemma/mistral goldens omit the system
// message (their reference templates raise on it).
func TestChatTemplateGoldenRenders(t *testing.T) {
	goldens := map[string]string{
		"chatml": "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi!<|im_end|>\n" +
			"<|im_start|>assistant\nHello!<|im_end|>\n<|im_start|>user\nBye?<|im_end|>\n<|im_start|>assistant\n",
		"llama3": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|>" +
			"<|start_header_id|>user<|end_header_id|>\n\nHi!<|eot_id|>" +
			"<|start_header_id|>assistant<|end_header_id|>\n\nHello!<|eot_id|>" +
			"<|start_header_id|>user<|end_header_id|>\n\nBye?<|eot_id|>" +
			"<|start_header_id|>assistant<|end_header_id|>\n\n",
		"gemma": "<bos><start_of_turn>user\nHi!<end_of_turn>\n<start_of_turn>model\nHello!<end_of_turn>\n" +
			"<start_of_turn>user\nBye?<end_of_turn>\n<start_of_turn>model\n",
		"mistral": "<s>[INST] Hi! [/INST] Hello!</s>[INST] Bye? [/INST]",
		"phi3": "<|system|>\nYou are helpful.<|end|>\n<|user|>\nHi!<|end|>\n<|assistant|>\nHello!<|end|>\n" +
			"<|user|>\nBye?<|end|>\n<|assistant|>\n",
		// Rendered by ibm-granite/granite-3.0-2b-instruct's tokenizer.apply_chat_template
		// (transformers 5.14.1, add_generation_prompt=true) — byte-exact.
		"granite": "<|start_of_role|>system<|end_of_role|>You are helpful.<|end_of_text|>\n" +
			"<|start_of_role|>user<|end_of_role|>Hi!<|end_of_text|>\n" +
			"<|start_of_role|>assistant<|end_of_role|>Hello!<|end_of_text|>\n" +
			"<|start_of_role|>user<|end_of_role|>Bye?<|end_of_text|>\n" +
			"<|start_of_role|>assistant<|end_of_role|>",
		// Rendered by allenai/OLMo-2-1124-7B-Instruct's tokenizer.apply_chat_template
		// (transformers 5.14.1, add_generation_prompt=true) — byte-exact. Note the eos
		// "<|endoftext|>" after the (non-last) assistant turn.
		"olmo2": "<|endoftext|><|system|>\nYou are helpful.\n<|user|>\nHi!\n" +
			"<|assistant|>\nHello!<|endoftext|>\n<|user|>\nBye?\n<|assistant|>\n",
	}
	for family, want := range goldens {
		tpl, err := nlp.NewChatTemplate(family)
		if err != nil {
			t.Fatal(err)
		}
		msgs := goldenTranscript
		if family == "gemma" || family == "mistral" {
			msgs = goldenTranscript[1:] // reference templates have no system role
		}
		got, err := tpl.Render(msgs, nlp.WithGenerationPrompt())
		if err != nil {
			t.Fatalf("%s: %v", family, err)
		}
		if got != want {
			t.Errorf("%s render mismatch:\ngot:  %q\nwant: %q", family, got, want)
		}
	}
}

// §V16 tier-1: phi3 without a generation prompt appends its EOS string, exactly as the
// reference template's else-branch does; chatml appends nothing.
func TestChatTemplateNoGenerationPrompt(t *testing.T) {
	phi, _ := nlp.NewChatTemplate("phi3")
	out, err := phi.Render([]nlp.ChatMessage{{Role: "user", Content: "Hi!"}})
	if err != nil {
		t.Fatal(err)
	}
	if want := "<|user|>\nHi!<|end|>\n<|endoftext|>"; out != want {
		t.Errorf("phi3: got %q, want %q", out, want)
	}
	cm, _ := nlp.NewChatTemplate("chatml")
	out, err = cm.Render([]nlp.ChatMessage{{Role: "user", Content: "Hi!"}})
	if err != nil {
		t.Fatal(err)
	}
	if want := "<|im_start|>user\nHi!<|im_end|>\n"; out != want {
		t.Errorf("chatml: got %q, want %q", out, want)
	}
}

// §V16 tier-1: families without a system role fold a leading system message into the
// first user turn (llama.cpp behavior), and WithoutBOS strips the rendered BOS string.
func TestChatTemplateSystemFoldingAndBOS(t *testing.T) {
	m, _ := nlp.NewChatTemplate("mistral")
	out, err := m.Render(goldenTranscript[:2], nlp.WithGenerationPrompt())
	if err != nil {
		t.Fatal(err)
	}
	if want := "<s>[INST] You are helpful.\n\nHi! [/INST]"; out != want {
		t.Errorf("mistral fold: got %q, want %q", out, want)
	}
	out, err = m.Render(goldenTranscript[:2], nlp.WithoutBOS())
	if err != nil {
		t.Fatal(err)
	}
	if strings.HasPrefix(out, "<s>") {
		t.Error("WithoutBOS must strip the leading <s>")
	}
	g, _ := nlp.NewChatTemplate("gemma")
	out, err = g.Render(goldenTranscript[:2])
	if err != nil {
		t.Fatal(err)
	}
	if want := "<bos><start_of_turn>user\nYou are helpful.\n\nHi!<end_of_turn>\n"; out != want {
		t.Errorf("gemma fold: got %q, want %q", out, want)
	}
}

// §V16 tier-1: validation mirrors the reference templates' raise_exception branches —
// non-alternating turns (gemma/mistral), misplaced system, unknown roles, unknown family.
func TestChatTemplateValidation(t *testing.T) {
	twoUsers := []nlp.ChatMessage{{Role: "user", Content: "a"}, {Role: "user", Content: "b"}}
	for _, family := range []string{"gemma", "mistral"} {
		tpl, _ := nlp.NewChatTemplate(family)
		if _, err := tpl.Render(twoUsers); err == nil {
			t.Errorf("%s must reject non-alternating turns", family)
		}
		if _, err := tpl.Render([]nlp.ChatMessage{{Role: "system", Content: "s"}}); err == nil {
			t.Errorf("%s must reject a system message with no user turn to fold into", family)
		}
	}
	cm, _ := nlp.NewChatTemplate("chatml")
	if _, err := cm.Render([]nlp.ChatMessage{{Role: "user", Content: "a"}, {Role: "system", Content: "s"}}); err == nil {
		t.Error("a system message after position 0 must be rejected")
	}
	if _, err := cm.Render([]nlp.ChatMessage{{Role: "tool", Content: "x"}}); err == nil {
		t.Error("unknown roles must be rejected")
	}
	if _, err := nlp.NewChatTemplate("alpaca"); err == nil {
		t.Error("unknown family must be rejected")
	}
}

// §V16 tier-1: detection maps the REAL Jinja sources (tokenizer.chat_template values)
// onto the right family, and unknown templates error WITH the raw jinja preserved.
func TestDetectChatTemplate(t *testing.T) {
	jinjas := map[string]string{
		"chatml":  `{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}`,
		"llama3":  `{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}`,
		"gemma":   `{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}`,
		"phi3":    `{% elif message['role'] == 'user' %}{{'<|user|>' + '\n' + message['content'] + '<|end|>' + '\n'}}`,
		"mistral": `{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}`,
		"granite": `{%- elif message['role'] == 'user' %}{{- '<|start_of_role|>user<|end_of_role|>' + message['content'] + '<|end_of_text|>\n' }}`,
	}
	for want, jinja := range jinjas {
		tpl, err := nlp.DetectChatTemplate(jinja)
		if err != nil {
			t.Fatalf("%s: %v", want, err)
		}
		if tpl.Family() != want {
			t.Errorf("detected %q, want %q", tpl.Family(), want)
		}
	}
	const alien = `{% for m in messages %}{{ '<<turn>>' + m['content'] }}{% endfor %}`
	_, err := nlp.DetectChatTemplate(alien)
	if err == nil || !strings.Contains(err.Error(), "<<turn>>") {
		t.Errorf("unknown template must error with the raw jinja preserved, got %v", err)
	}
}

// §V16 tier-1: the GGUF metadata path — key present → detection; missing/mistyped → error.
func TestChatTemplateFromGGUF(t *testing.T) {
	tpl, err := nlp.ChatTemplateFromGGUF(map[string]any{
		"tokenizer.chat_template": "{{'<|im_start|>' + role}}",
	})
	if err != nil || tpl.Family() != "chatml" {
		t.Fatalf("got %v, %v; want chatml", tpl, err)
	}
	if _, err := nlp.ChatTemplateFromGGUF(map[string]any{}); err == nil {
		t.Error("missing key must error")
	}
	if _, err := nlp.ChatTemplateFromGGUF(map[string]any{"tokenizer.chat_template": 7}); err == nil {
		t.Error("non-string value must error")
	}
}

// §V16 tier-1: OLMo-2's last-assistant edge case — the reference template's
// loop.last branch drops the trailing newline after a conversation-final
// assistant turn (all other assistant turns close with "<|endoftext|>\n").
// Byte-exact against allenai/OLMo-2-1124-7B-Instruct.
func TestChatTemplateOLMo2LastAssistant(t *testing.T) {
	tpl, err := nlp.NewChatTemplate("olmo2")
	if err != nil {
		t.Fatal(err)
	}
	msgs := []nlp.ChatMessage{
		{Role: "user", Content: "Hi!"},
		{Role: "assistant", Content: "Hello!"},
	}
	got, err := tpl.Render(msgs)
	if err != nil {
		t.Fatal(err)
	}
	want := "<|endoftext|><|user|>\nHi!\n<|assistant|>\nHello!<|endoftext|>"
	if got != want {
		t.Errorf("olmo2 last-assistant mismatch:\ngot:  %q\nwant: %q", got, want)
	}
	// WithoutBOS drops the leading eos-as-bos string.
	got2, _ := tpl.Render(msgs, nlp.WithoutBOS())
	if want2 := "<|user|>\nHi!\n<|assistant|>\nHello!<|endoftext|>"; got2 != want2 {
		t.Errorf("olmo2 WithoutBOS mismatch:\ngot:  %q\nwant: %q", got2, want2)
	}
}
