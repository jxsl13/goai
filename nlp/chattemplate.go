package nlp

import (
	"fmt"
	"sort"
	"strings"
)

// ChatMessage is one turn of a conversation for ChatTemplate rendering: a Role
// ("system", "user" or "assistant") and its Content text.
type ChatMessage struct {
	Role    string // "system", "user" or "assistant"
	Content string // the turn's text
}

// ChatTemplate renders a conversation into the exact prompt string an instruction-tuned
// model was trained on (§T576). Every model family wraps turns in its own control tokens
// — ChatML's <|im_start|>, Llama-3's <|start_header_id|>, Gemma's <start_of_turn>,
// Mistral's [INST], Phi-3's <|user|> — and getting them wrong silently degrades the
// model. GGUF files ship the template as Jinja source under "tokenizer.chat_template";
// DetectChatTemplate maps that source onto one of the native renderers here, which
// reproduce the HF transformers apply_chat_template reference output byte for byte
// (golden-tested against jinja2 renders of the real repo templates).
//
// Supported families (the names NewChatTemplate accepts): "chatml" (Qwen2/2.5 class),
// "llama3" (Meta-Llama-3-Instruct), "gemma" (gemma-it), "mistral"
// (Mistral-7B-Instruct-v0.3 class), "phi3" (Phi-3-instruct), "granite"
// (IBM Granite 3.x instruct), "olmo2" (allenai OLMo-2 Instruct), "deepseek" (DeepSeek-V2/V3 chat), "zephyr" (Zephyr / StableLM-2-Zephyr class).
//
// Family quirks are preserved faithfully: llama3/gemma trim message content; gemma and
// mistral reject non-alternating user/assistant turns and have no system role — a
// leading system message is folded into the first user turn (llama.cpp behavior);
// mistral's trailing "[/INST]" already IS the generation prompt, so WithGenerationPrompt
// adds nothing there; phi3 appends its <|endoftext|> EOS when no generation prompt is
// requested (as the reference template does). Templates that render a BOS string
// (llama3 "<|begin_of_text|>", gemma "<bos>", mistral "<s>") do so by default — pass
// WithoutBOS when the tokenizer inserts the BOS token itself.
type ChatTemplate struct {
	family string
}

// chatRenderCfg collects the ChatRenderOption flags for one Render call.
type chatRenderCfg struct {
	generationPrompt bool
	withoutBOS       bool
}

// ChatRenderOption configures one ChatTemplate.Render call.
type ChatRenderOption func(*chatRenderCfg)

// WithGenerationPrompt appends the family's assistant-turn opener (e.g.
// "<|im_start|>assistant\n") so the model continues as the assistant — the
// add_generation_prompt=true mode used for inference.
func WithGenerationPrompt() ChatRenderOption {
	return func(c *chatRenderCfg) { c.generationPrompt = true }
}

// WithoutBOS suppresses the beginning-of-sequence STRING some templates render (llama3
// "<|begin_of_text|>", gemma "<bos>", mistral "<s>"). Use it when the tokenizer adds the
// BOS token itself — otherwise the sequence would start with two.
func WithoutBOS() ChatRenderOption {
	return func(c *chatRenderCfg) { c.withoutBOS = true }
}

// NewChatTemplate returns the native renderer for a known family name: "chatml",
// "llama3", "gemma", "mistral", "phi3", "granite", "olmo2", "deepseek" or "zephyr".
// olmo2/deepseek/zephyr are explicit-only (no [DetectChatTemplate] entry): their
// distinguishing tokens are shared with other families or live in jinja variables,
// so auto-detection could silently mis-prompt — name the family instead.
func NewChatTemplate(family string) (*ChatTemplate, error) {
	switch family {
	case "chatml", "llama3", "gemma", "mistral", "phi3", "granite", "olmo2", "deepseek", "zephyr":
		return &ChatTemplate{family: family}, nil
	}
	return nil, fmt.Errorf("nlp: NewChatTemplate: unknown family %q (want chatml, llama3, gemma, mistral, phi3, granite, olmo2, deepseek or zephyr)", family)
}

// DetectChatTemplate maps Jinja template source (the GGUF "tokenizer.chat_template"
// value) onto a native renderer by its distinctive control tokens. Unknown templates
// return an error CONTAINING THE RAW JINJA so callers can fall back to their own
// handling rather than silently mis-prompting the model.
//
// # The failure mode this CANNOT catch
//
// Detection is a substring match on marker tokens, and the returned renderer is a
// hand-written Go implementation of that family — the actual Jinja is never
// interpreted. Non-detection is therefore safe (it errors), but MIS-detection is
// not: a fine-tune shipping a customized variant of a known family still matches
// that family, and its customizations are silently ignored. The result is a
// plausible, well-formed prompt that differs from what the model was tuned on, with
// no error.
//
// Interpreting the Jinja for real is the only complete fix and is out of scope for a
// zero-dependency library. So if prompt fidelity matters for your model, check
// [ChatTemplate.Family] against what you expect and compare a rendered sample to the
// template's own output — do not assume a successful detection means an exact match.
func DetectChatTemplate(jinja string) (*ChatTemplate, error) {
	switch {
	case strings.Contains(jinja, "<|im_start|>"):
		return &ChatTemplate{family: "chatml"}, nil
	case strings.Contains(jinja, "<|start_header_id|>"):
		return &ChatTemplate{family: "llama3"}, nil
	case strings.Contains(jinja, "<start_of_turn>"):
		return &ChatTemplate{family: "gemma"}, nil
	case strings.Contains(jinja, "<|user|>") && strings.Contains(jinja, "<|end|>"):
		return &ChatTemplate{family: "phi3"}, nil
	case strings.Contains(jinja, "[INST]"):
		return &ChatTemplate{family: "mistral"}, nil
	case strings.Contains(jinja, "<|start_of_role|>"):
		return &ChatTemplate{family: "granite"}, nil
	}
	return nil, fmt.Errorf("nlp: DetectChatTemplate: unrecognized chat template; raw jinja: %s", jinja)
}

// ChatTemplateFromGGUF reads the "tokenizer.chat_template" key from GGUF metadata (the
// gguf.File.Metadata map) and detects the family. Missing or non-string values error.
func ChatTemplateFromGGUF(metadata map[string]any) (*ChatTemplate, error) {
	v, ok := metadata["tokenizer.chat_template"]
	if !ok {
		return nil, fmt.Errorf("nlp: ChatTemplateFromGGUF: metadata has no tokenizer.chat_template key")
	}
	jinja, ok := v.(string)
	if !ok {
		return nil, fmt.Errorf("nlp: ChatTemplateFromGGUF: tokenizer.chat_template is %T, want string", v)
	}
	return DetectChatTemplate(jinja)
}

// Family returns the template family name ("chatml", "llama3", ...).
func (t *ChatTemplate) Family() string { return t.family }

// SpecialTokens returns the control markers this family's [ChatTemplate.Render] can emit
// — "<|im_start|>", "<|eot_id|>", "[INST]" and so on — in descending length order so
// longest-match wins where markers share a prefix.
//
// Reach for this when you must tokenize rendered output with a tokenizer that carries no
// special-token metadata of its own (a tiktoken rank file, a hand-built vocab); pass it
// and the tokenizer to [RegisterChatSpecials], which resolves each marker against the
// tokenizer's vocabulary. GGUF and tokenizer.json loaders already source their markers
// from file metadata, which is authoritative — prefer that over this list.
//
// This is a HARDCODED per-family list, and that is its limitation: it mirrors the markers
// the renderers in this file write, so a family whose real vocabulary names them
// differently (or adds tool-call markers this package does not render) is not covered.
// It is a fallback for the metadata-free case, not a vocabulary.
//
// Knowing a marker does not make it parsed. Only EncodeSpecial matches these; plain
// Encode still treats them as literal text, which is what keeps untrusted message
// content from forging conversation structure.
func (t *ChatTemplate) SpecialTokens() []string {
	var markers []string
	switch t.family {
	case "chatml":
		markers = []string{"<|im_start|>", "<|im_end|>"}
	case "llama3":
		markers = []string{"<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"}
	case "gemma":
		markers = []string{"<bos>", "<start_of_turn>", "<end_of_turn>"}
	case "mistral":
		markers = []string{"<s>", "</s>", "[INST]", "[/INST]"}
	case "phi3":
		markers = []string{"<|system|>", "<|user|>", "<|assistant|>", "<|end|>", "<|endoftext|>"}
	case "granite":
		markers = []string{"<|start_of_role|>", "<|end_of_role|>", "<|end_of_text|>"}
	case "olmo2":
		markers = []string{"<|system|>", "<|user|>", "<|assistant|>", "<|endoftext|>"}
	case "deepseek":
		markers = []string{"<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"}
	case "zephyr":
		markers = []string{"<|system|>", "<|user|>", "<|assistant|>", "</s>"}
	}
	sort.SliceStable(markers, func(i, j int) bool { return len(markers[i]) > len(markers[j]) })
	return markers
}

// Render formats the conversation exactly as the family's reference template would.
// Roles must be "system", "user" or "assistant"; families without a system role
// (gemma, mistral) fold a LEADING system message into the first user turn. Options:
// WithGenerationPrompt to end with the assistant opener, WithoutBOS to drop a
// template-rendered BOS string.
func (t *ChatTemplate) Render(messages []ChatMessage, opts ...ChatRenderOption) (string, error) {
	var cfg chatRenderCfg
	for _, o := range opts {
		o(&cfg)
	}
	for i, m := range messages {
		switch m.Role {
		case "user", "assistant":
		case "system":
			if i != 0 {
				return "", fmt.Errorf("nlp: ChatTemplate.Render: system message must be first (got position %d)", i)
			}
		default:
			return "", fmt.Errorf("nlp: ChatTemplate.Render: unknown role %q", m.Role)
		}
	}
	switch t.family {
	case "chatml":
		return renderTurnStyle(messages, cfg, "", "<|im_start|>", "\n", "<|im_end|>\n", nil, false), nil
	case "llama3":
		return renderTurnStyle(messages, cfg, "<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>\n\n", "<|eot_id|>", nil, true), nil
	case "phi3":
		out := renderTurnStyle(messages, cfg, "", "<|", "|>\n", "<|end|>\n", nil, false)
		if !cfg.generationPrompt {
			out += "<|endoftext|>" // the reference template emits EOS instead of the opener
		}
		return out, nil
	case "gemma":
		msgs, err := foldSystem(messages, t.family)
		if err != nil {
			return "", err
		}
		for i, m := range msgs { // the reference template raises on non-alternating turns
			if (m.Role == "user") != (i%2 == 0) {
				return "", fmt.Errorf("nlp: ChatTemplate.Render: gemma requires alternating user/assistant turns (position %d is %q)", i, m.Role)
			}
		}
		return renderTurnStyle(msgs, cfg, "<bos>", "<start_of_turn>", "\n", "<end_of_turn>\n",
			map[string]string{"assistant": "model"}, true), nil
	case "mistral":
		return renderMistral(messages, cfg)
	case "granite":
		// IBM Granite 3.x instruct: <|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n
		// per message (system role kept, no folding, no BOS in the template render), and the
		// generation prompt is the assistant role opener. Verified byte-exact against
		// ibm-granite/granite-3.0-2b-instruct's tokenizer.apply_chat_template.
		return renderTurnStyle(messages, cfg, "", "<|start_of_role|>", "<|end_of_role|>", "<|end_of_text|>\n", nil, false), nil
	case "olmo2":
		return renderOLMo2(messages, cfg), nil
	case "deepseek":
		return renderDeepSeek(messages, cfg), nil
	case "zephyr":
		// HuggingFaceH4 Zephyr (and StableLM-2-Zephyr class): uniform
		// <|role|>\n{content}</s>\n turns, generation prompt <|assistant|>\n —
		// phi3's shape with the </s> eos close and none of phi3's no-gen EOS quirk.
		// Verified byte-exact against HuggingFaceH4/zephyr-7b-beta's
		// tokenizer.apply_chat_template.
		return renderTurnStyle(messages, cfg, "", "<|", "|>\n", "</s>\n", nil, false), nil
	}
	return "", fmt.Errorf("nlp: ChatTemplate.Render: unknown family %q", t.family)
}

// renderTurnStyle covers the turn-wrapped families: bos + per message
// open+role+sep+content+close, plus the assistant opener as generation prompt.
// estChatSize is a capacity estimate for a rendered chat prompt: the total message
// content plus a fixed per-turn allowance for role tags and separators. Growing the
// strings.Builder once to this size skips the log(n) doubling reallocations; a rough
// over-estimate is fine (Grow only ensures a lower bound on capacity).
func estChatSize(messages []ChatMessage) int {
	n := 64 // BOS + trailing generation-prompt opener slack
	for _, m := range messages {
		n += len(m.Content) + len(m.Role) + 40
	}
	return n
}

// roleMap renames roles (gemma: assistant→model); trim strips content whitespace.
func renderTurnStyle(messages []ChatMessage, cfg chatRenderCfg, bos, open, sep, closeTok string, roleMap map[string]string, trim bool) string {
	var sb strings.Builder
	sb.Grow(estChatSize(messages))
	if !cfg.withoutBOS {
		sb.WriteString(bos)
	}
	gen := "assistant"
	for _, m := range messages {
		role := m.Role
		if r, ok := roleMap[role]; ok {
			role = r
		}
		content := m.Content
		if trim {
			content = strings.TrimSpace(content)
		}
		sb.WriteString(open)
		sb.WriteString(role)
		sb.WriteString(sep)
		sb.WriteString(content)
		sb.WriteString(closeTok)
	}
	if cfg.generationPrompt {
		if r, ok := roleMap[gen]; ok {
			gen = r
		}
		sb.WriteString(open)
		sb.WriteString(gen)
		sb.WriteString(sep)
	}
	return sb.String()
}

// renderMistral renders the [INST] style: "<s>[INST] u [/INST] a</s>[INST] u2 [/INST]".
// The trailing "[/INST]" after a user turn is itself the generation prompt.
func renderMistral(messages []ChatMessage, cfg chatRenderCfg) (string, error) {
	msgs, err := foldSystem(messages, "mistral")
	if err != nil {
		return "", err
	}
	var sb strings.Builder
	sb.Grow(estChatSize(messages))
	if !cfg.withoutBOS {
		sb.WriteString("<s>")
	}
	for i, m := range msgs {
		wantUser := i%2 == 0
		if (m.Role == "user") != wantUser {
			return "", fmt.Errorf("nlp: ChatTemplate.Render: mistral requires alternating user/assistant turns (position %d is %q)", i, m.Role)
		}
		if m.Role == "user" {
			sb.WriteString("[INST] ")
			sb.WriteString(m.Content)
			sb.WriteString(" [/INST]")
		} else {
			sb.WriteString(" ")
			sb.WriteString(m.Content)
			sb.WriteString("</s>")
		}
	}
	return sb.String(), nil
}

// renderOLMo2 renders the allenai OLMo-2 Instruct template: the bos string
// "<|endoftext|>" (suppressible with WithoutBOS), then per turn "<|role|>\n" +
// content, closed with "\n" for system/user but with the eos "<|endoftext|>"
// for assistant turns — and the LAST assistant turn drops the trailing newline
// (the template's loop.last branch). The generation prompt is "<|assistant|>\n".
// Verified byte-exact against allenai/OLMo-2-1124-7B-Instruct's
// tokenizer.apply_chat_template (transformers 5.14.1).
func renderOLMo2(messages []ChatMessage, cfg chatRenderCfg) string {
	var sb strings.Builder
	sb.Grow(estChatSize(messages))
	if !cfg.withoutBOS {
		sb.WriteString("<|endoftext|>")
	}
	for i, m := range messages {
		switch m.Role {
		case "system":
			sb.WriteString("<|system|>\n" + m.Content + "\n")
		case "user":
			sb.WriteString("<|user|>\n" + m.Content + "\n")
		case "assistant":
			sb.WriteString("<|assistant|>\n" + m.Content + "<|endoftext|>")
			if i != len(messages)-1 { // the last assistant turn omits the trailing newline
				sb.WriteString("\n")
			}
		}
	}
	if cfg.generationPrompt {
		sb.WriteString("<|assistant|>\n")
	}
	return sb.String()
}

// renderDeepSeek renders the DeepSeek-V2/V3 chat template (Vicuna-style): the bos
// string "<｜begin▁of▁sentence｜>" (suppressible with WithoutBOS), a leading system
// message as bare content + "\n\n", each user turn as "User: {content}\n\n" and each
// assistant turn as "Assistant: {content}" closed with the eos "<｜end▁of▁sentence｜>";
// the generation prompt is "Assistant:". The special-token names use the fullwidth
// vertical bar (U+FF5C) and SentencePiece underscore (U+2581), exactly as on disk.
// Verified byte-exact against deepseek-ai/DeepSeek-V2-Lite-Chat's
// tokenizer.apply_chat_template (transformers 5.14.1).
func renderDeepSeek(messages []ChatMessage, cfg chatRenderCfg) string {
	var sb strings.Builder
	sb.Grow(estChatSize(messages))
	if !cfg.withoutBOS {
		sb.WriteString("<｜begin▁of▁sentence｜>")
	}
	for _, m := range messages {
		switch m.Role {
		case "system":
			sb.WriteString(m.Content + "\n\n")
		case "user":
			sb.WriteString("User: " + m.Content + "\n\n")
		case "assistant":
			sb.WriteString("Assistant: " + m.Content + "<｜end▁of▁sentence｜>")
		}
	}
	if cfg.generationPrompt {
		sb.WriteString("Assistant:")
	}
	return sb.String()
}

// foldSystem merges a leading system message into the first user turn
// ("system\n\nuser") for families without a system role, and enforces that a user
// turn follows.
func foldSystem(messages []ChatMessage, family string) ([]ChatMessage, error) {
	if len(messages) == 0 || messages[0].Role != "system" {
		return messages, nil
	}
	if len(messages) < 2 || messages[1].Role != "user" {
		return nil, fmt.Errorf("nlp: ChatTemplate.Render: %s has no system role; a system message must be followed by a user turn to fold into", family)
	}
	out := append([]ChatMessage(nil), messages[1:]...)
	out[0].Content = messages[0].Content + "\n\n" + out[0].Content
	return out, nil
}
