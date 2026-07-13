package nlp_test

import (
	"fmt"

	"github.com/jxsl13/goai/nlp"
)

// Render a conversation for a ChatML-family model (Qwen class) ready for generation:
// the WithGenerationPrompt option ends the prompt with the assistant opener.
func ExampleChatTemplate() {
	tpl, err := nlp.NewChatTemplate("chatml")
	if err != nil {
		panic(err)
	}
	prompt, err := tpl.Render([]nlp.ChatMessage{
		{Role: "system", Content: "Answer briefly."},
		{Role: "user", Content: "2+2?"},
	}, nlp.WithGenerationPrompt())
	if err != nil {
		panic(err)
	}
	fmt.Printf("%q\n", prompt)
	// Output: "<|im_start|>system\nAnswer briefly.<|im_end|>\n<|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n"
}

// Families without a system role (mistral, gemma) fold a leading system message into
// the first user turn; WithoutBOS drops the rendered "<s>" when the tokenizer adds the
// BOS token itself.
func ExampleChatTemplate_Render() {
	tpl, err := nlp.NewChatTemplate("mistral")
	if err != nil {
		panic(err)
	}
	prompt, err := tpl.Render([]nlp.ChatMessage{
		{Role: "system", Content: "Answer briefly."},
		{Role: "user", Content: "2+2?"},
	}, nlp.WithGenerationPrompt(), nlp.WithoutBOS())
	if err != nil {
		panic(err)
	}
	fmt.Printf("%q\n", prompt)
	// Output: "[INST] Answer briefly.\n\n2+2? [/INST]"
}

// DetectChatTemplate maps GGUF Jinja template source onto a native renderer; Family
// names the detected family.
func ExampleDetectChatTemplate() {
	jinja := `{% for m in messages %}{{'<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n'}}{% endfor %}`
	tpl, err := nlp.DetectChatTemplate(jinja)
	if err != nil {
		panic(err)
	}
	fmt.Println(tpl.Family())
	// Output: chatml
}

// ExampleChatTemplateFromGGUF reads the template straight from a GGUF file's metadata
// map (gguf.File.Metadata).
func ExampleChatTemplateFromGGUF() {
	metadata := map[string]any{ // as returned in gguf.File.Metadata
		"tokenizer.chat_template": `{{ '<start_of_turn>' + role + '\n' }}`,
	}
	tpl, err := nlp.ChatTemplateFromGGUF(metadata)
	if err != nil {
		panic(err)
	}
	fmt.Println(tpl.Family())
	// Output: gemma
}

// NewChatTemplate accepts the five family names and rejects everything else.
func ExampleNewChatTemplate() {
	tpl, err := nlp.NewChatTemplate("llama3")
	if err != nil {
		panic(err)
	}
	fmt.Println(tpl.Family())
	_, err = nlp.NewChatTemplate("alpaca")
	fmt.Println(err != nil)
	// Output:
	// llama3
	// true
}
