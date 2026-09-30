// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 alibaba/open-code-review Contributors

package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/jsonschema-go/jsonschema"
	oteltrace "go.opentelemetry.io/otel/trace"
)

func claudeCodeTestRequest() ChatRequest {
	return ChatRequest{Messages: []Message{NewTextMessage("user", "Review this change.")}}
}

func claudeCodeTestTool() ToolDef {
	return ToolDef{Type: "function", Function: FunctionDef{
		Name: "lookup", Description: "Look up a record by its name.",
		Parameters: map[string]any{
			"type": "object", "additionalProperties": false,
			"required":   []string{"name"},
			"properties": map[string]any{"name": map[string]any{"$ref": "#/$defs/name"}},
			"$defs":      map[string]any{"name": map[string]any{"type": "string", "minLength": 1}},
		},
	}}
}

func claudeCodeTestJSON(t *testing.T, value any) []byte {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func claudeCodeTestError(t *testing.T, err error, want string) {
	t.Helper()
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("error = %v, want substring %q", err, want)
	}
}

func TestClaudeCodeRequestSnapshot(t *testing.T) {
	req := ChatRequest{
		Model: "requested-model", SessionID: "not-a-cli-session", MaxTokens: 1024,
		Tools: []ToolDef{claudeCodeTestTool()}, ToolChoice: "auto",
		Messages: []Message{
			NewTextMessage("system", "Follow review rules."),
			{Role: "system", Content: []ContentBlock{{Type: "text", Text: "Preserve locations."}}},
			NewTextMessage("user", "Inspect the record."),
			NewToolCallMessage("Looking it up.", []ToolCall{{ID: "original-call", Type: "function", Function: FunctionCall{Name: "lookup", Arguments: `{"name":"fixture"}`}}}, NativeTurn{Family: "another-provider", Payload: "opaque-state"}, "private-reasoning"),
			NewToolResultMessage("original-call", `{"value":"actual result"}`),
			NewTextMessage("system", "A later instruction stays in order."),
		},
	}
	before := string(claudeCodeTestJSON(t, req))
	input, system, schema, validators, err := buildClaudeCodeRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(system, "Follow review rules.\n\nPreserve locations.\n\n") || !strings.Contains(system, claudeCodeActionPrompt) {
		t.Fatalf("system prompt did not retain leading instructions and action contract: %q", system)
	}
	for _, instruction := range []string{
		"Claude Code's native and MCP tools are disabled here",
		"OCR executes every requested call and returns its result in the next snapshot",
		"request the relevant functions instead of relying on the snapshot alone",
		"Call the CLI's StructuredOutput tool solely to format the response envelope",
		"Never include StructuredOutput in tool_calls",
		"Batch independent context requests in the same tool_calls array",
		"wait for OCR to return those results in the next snapshot rather than guessing them",
	} {
		if !strings.Contains(system, instruction) {
			t.Fatalf("system prompt lost formatting or execution instruction %q", instruction)
		}
	}
	if strings.Contains(system, "Never execute tools") || strings.Contains(system, "Do not execute OCR functions") || strings.Contains(string(input), "StructuredOutput") || strings.Contains(schema, "StructuredOutput") {
		t.Fatal("CLI formatting was forbidden or exposed as an OCR function")
	}
	want := claudeCodeTestJSON(t, struct {
		Messages []Message `json:"messages"`
		Tools    []ToolDef `json:"tools"`
	}{req.Messages[2:], req.Tools})
	if string(input) != string(want) {
		t.Fatalf("snapshot = %s, want %s", input, want)
	}
	for _, omitted := range []string{"opaque-state", "private-reasoning", "not-a-cli-session", "requested-model"} {
		if strings.Contains(string(input), omitted) {
			t.Fatalf("snapshot leaked non-conversation field %q", omitted)
		}
	}
	if string(claudeCodeTestJSON(t, req)) != before {
		t.Fatal("request builder mutated its caller's snapshot")
	}
	if len(validators) != 1 || validators["lookup"] == nil {
		t.Fatalf("validators = %v", validators)
	}
	if strings.Contains(schema, "$ref") || strings.Contains(schema, "$defs") {
		t.Fatal("function-local references must not be relocated into the CLI output schema")
	}
	var outputSchema jsonschema.Schema
	if err := json.Unmarshal([]byte(schema), &outputSchema); err != nil {
		t.Fatal(err)
	}
	resolved, err := outputSchema.Resolve(nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := resolved.Validate(map[string]any{"content": "", "tool_calls": []any{map[string]any{"name": "lookup", "arguments": map[string]any{"name": "fixture"}}}}); err != nil {
		t.Fatalf("generated output schema rejects a valid action: %v", err)
	}
	if err := resolved.Validate(map[string]any{"content": "", "tool_calls": []any{map[string]any{"name": "unlisted", "arguments": map[string]any{}}}}); err == nil {
		t.Fatal("generated output schema accepted an unlisted function")
	}
}

func TestClaudeCodeRequestToolChoice(t *testing.T) {
	for _, choice := range []string{"", "auto", "required", "none"} {
		t.Run(choice, func(t *testing.T) {
			req := claudeCodeTestRequest()
			req.Tools, req.ToolChoice = []ToolDef{claudeCodeTestTool()}, choice
			input, system, schema, validators, err := buildClaudeCodeRequest(req)
			if err != nil {
				t.Fatal(err)
			}
			if choice == "none" {
				if schema != "" || len(validators) != 0 || strings.Contains(string(input), `"tools"`) || !strings.Contains(system, "No tools are available") || strings.Contains(system, "StructuredOutput") || strings.Contains(system, "request the relevant functions") {
					t.Fatal("tool_choice none still exposed tools or structured output")
				}
				return
			}
			if schema == "" || len(validators) != 1 || !strings.Contains(system, claudeCodeActionPrompt) {
				t.Fatal("enabled tools lost their schema, validators, or formatting instruction")
			}
			if strings.Contains(schema, `"minItems":1`) != (choice == "required") {
				t.Fatalf("schema does not implement tool_choice %q: %s", choice, schema)
			}
		})
	}
	t.Run("no available tools", func(t *testing.T) {
		input, system, schema, validators, err := buildClaudeCodeRequest(claudeCodeTestRequest())
		if err != nil {
			t.Fatal(err)
		}
		if schema != "" || len(validators) != 0 || strings.Contains(string(input), `"tools"`) || strings.Contains(system, "StructuredOutput") || !strings.Contains(system, "No tools are available") || strings.Contains(system, "request the relevant functions") {
			t.Fatal("tool-free request exposed tools or a formatting instruction")
		}
	})
	t.Run("nil parameters and omitted function type", func(t *testing.T) {
		req := claudeCodeTestRequest()
		req.Tools = []ToolDef{{Function: FunctionDef{Name: "finish"}}}
		_, _, _, validators, err := buildClaudeCodeRequest(req)
		if err != nil {
			t.Fatal(err)
		}
		if err := validators["finish"].Validate(map[string]any{}); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("disabled invalid tools are not evaluated", func(t *testing.T) {
		req := claudeCodeTestRequest()
		req.ToolChoice, req.Tools = "none", []ToolDef{{Type: "unsupported"}}
		if _, _, _, _, err := buildClaudeCodeRequest(req); err != nil {
			t.Fatal(err)
		}
	})
}

func TestClaudeCodeRequestErrors(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*ChatRequest)
		want   string
	}{
		{"no messages", func(r *ChatRequest) { r.Messages = nil }, "requires conversation messages"},
		{"negative tokens", func(r *ChatRequest) { r.MaxTokens = -1 }, "max_tokens must be non-negative"},
		{"unsupported choice", func(r *ChatRequest) { r.ToolChoice = "lookup" }, "unsupported Claude Code tool_choice"},
		{"required without tools", func(r *ChatRequest) { r.ToolChoice = "required" }, "required needs tools"},
		{"empty name", func(r *ChatRequest) { r.Tools = []ToolDef{{Type: "function"}} }, "invalid Claude Code function"},
		{"unsupported tool type", func(r *ChatRequest) { tool := claudeCodeTestTool(); tool.Type = "computer"; r.Tools = []ToolDef{tool} }, "invalid Claude Code function"},
		{"duplicate names", func(r *ChatRequest) { r.Tools = []ToolDef{claudeCodeTestTool(), claudeCodeTestTool()} }, "duplicate Claude Code function"},
		{"unencodable schema", func(r *ChatRequest) {
			r.Tools = []ToolDef{{Function: FunctionDef{Name: "bad", Parameters: map[string]any{"default": make(chan int)}}}}
		}, "encode Claude Code function"},
		{"invalid schema shape", func(r *ChatRequest) {
			r.Tools = []ToolDef{{Function: FunctionDef{Name: "bad", Parameters: map[string]any{"type": 42}}}}
		}, "decode Claude Code function"},
		{"invalid schema regex", func(r *ChatRequest) {
			r.Tools = []ToolDef{{Function: FunctionDef{Name: "bad", Parameters: map[string]any{"type": "string", "pattern": "["}}}}
		}, "resolve Claude Code function"},
		{"missing local reference", func(r *ChatRequest) {
			r.Tools = []ToolDef{{Function: FunctionDef{Name: "bad", Parameters: map[string]any{"$ref": "#/$defs/missing"}}}}
		}, "resolve Claude Code function"},
		{"remote reference prohibited", func(r *ChatRequest) {
			r.Tools = []ToolDef{{Function: FunctionDef{Name: "bad", Parameters: map[string]any{"$ref": "https://example.invalid/schema.json"}}}}
		}, "resolve Claude Code function"},
		{"file reference prohibited", func(r *ChatRequest) {
			r.Tools = []ToolDef{{Function: FunctionDef{Name: "bad", Parameters: map[string]any{"$ref": "file:///not-an-ocr-schema.json"}}}}
		}, "resolve Claude Code function"},
		{"unencodable content", func(r *ChatRequest) { r.Messages[0].Content = make(chan int) }, "encode Claude Code conversation"},
		{"oversized input", func(r *ChatRequest) { r.Messages[0].Content = strings.Repeat("a", claudeCodeInputLimit) }, "stdin limit"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := claudeCodeTestRequest()
			tt.mutate(&req)
			_, _, _, _, err := buildClaudeCodeRequest(req)
			claudeCodeTestError(t, err, tt.want)
		})
	}
	t.Run("exact input boundary", func(t *testing.T) {
		req := claudeCodeTestRequest()
		req.Messages[0].Content = ""
		base, _, _, _, err := buildClaudeCodeRequest(req)
		if err != nil {
			t.Fatal(err)
		}
		req.Messages[0].Content = strings.Repeat("a", claudeCodeInputLimit-len(base))
		input, _, _, _, err := buildClaudeCodeRequest(req)
		if err != nil || len(input) != claudeCodeInputLimit {
			t.Fatalf("exact input boundary: length %d, error %v", len(input), err)
		}
	})
}

func claudeCodeTestResult() map[string]any {
	return map[string]any{
		"type": "result", "subtype": "success", "is_error": false,
		"result": "The response.", "session_id": "cli-session", "stop_reason": "end_turn",
		"usage": map[string]any{"input_tokens": 10, "output_tokens": 7, "cache_read_input_tokens": 3, "cache_creation_input_tokens": 5},
	}
}

func TestClaudeCodeParseTextAndModel(t *testing.T) {
	for _, text := range []string{"Plain response.", " {\n  \"findings\": []\n}\n", "```json\n{\"findings\":[]}\n```"} {
		t.Run(text, func(t *testing.T) {
			result := claudeCodeTestResult()
			result["result"] = text
			resp, err := parseClaudeCodeResult(claudeCodeTestJSON(t, result), claudeCodeTestRequest(), "requested-alias", nil)
			if err != nil {
				t.Fatal(err)
			}
			if resp.ID != "cli-session" || resp.Model != "requested-alias" || len(resp.Choices) != 1 || resp.Choices[0].Message.Role != "assistant" || resp.Choices[0].FinishReason != "stop" {
				t.Fatalf("incorrect response envelope: %+v", resp)
			}
			if resp.Choices[0].Message.Content == nil || *resp.Choices[0].Message.Content != text || len(resp.ToolCalls()) != 0 {
				t.Fatalf("plain response formatting changed: %+v", resp.Choices[0])
			}
			if resp.Usage == nil || resp.Usage.PromptTokens != 18 || resp.Usage.CompletionTokens != 7 || resp.Usage.TotalTokens != 25 || resp.Usage.CacheReadTokens != 3 || resp.Usage.CacheWriteTokens != 5 {
				t.Fatalf("incorrect token accounting: %+v", resp.Usage)
			}
			if resp.Native().Payload != nil || resp.ReasoningContent() != "" {
				t.Fatal("adapter invented opaque native replay or reasoning state")
			}
		})
	}
	for _, tt := range []struct {
		name   string
		models map[string]any
		want   string
	}{
		{"actual model", map[string]any{"actual-model": map[string]any{}}, "actual-model"},
		{"no models", map[string]any{}, "requested-alias"},
		{"multiple internal models", map[string]any{"first": map[string]any{}, "second": map[string]any{}}, "requested-alias"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			result := claudeCodeTestResult()
			result["modelUsage"] = tt.models
			resp, err := parseClaudeCodeResult(claudeCodeTestJSON(t, result), claudeCodeTestRequest(), "requested-alias", nil)
			if err != nil {
				t.Fatal(err)
			}
			if resp.Model != tt.want {
				t.Fatalf("model = %q, want %q", resp.Model, tt.want)
			}
		})
	}
}

func TestClaudeCodeParseStructuredActions(t *testing.T) {
	req := claudeCodeTestRequest()
	req.ToolChoice, req.Tools = "required", []ToolDef{claudeCodeTestTool()}
	_, _, _, validators, err := buildClaudeCodeRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	result := claudeCodeTestResult()
	result["result"] = "CLI formatting explanation, not assistant content."
	result["structured_output"] = json.RawMessage(`{"content":"","tool_calls":[{"name":"lookup","arguments":{"name":"first"}},{"name":"lookup","arguments":{"name":"second"}}]}`)
	resp, err := parseClaudeCodeResult(claudeCodeTestJSON(t, result), req, "model", validators)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content() != "" || resp.Choices[0].FinishReason != "tool_calls" || len(resp.ToolCalls()) != 2 {
		t.Fatalf("incorrect action response: %+v", resp)
	}
	ids := map[string]bool{}
	for i, call := range resp.ToolCalls() {
		if !strings.HasPrefix(call.ID, "call_") || ids[call.ID] || call.Type != "function" || call.Function.Name != "lookup" {
			t.Fatalf("invalid synthetic call: %+v", call)
		}
		ids[call.ID] = true
		var args map[string]any
		if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
			t.Fatal(err)
		}
		if args["name"] != []string{"first", "second"}[i] {
			t.Fatalf("call order/arguments changed: %+v", args)
		}
	}
	result["structured_output"] = json.RawMessage(`{"content":"Final answer.","tool_calls":[]}`)
	req.ToolChoice = "auto"
	resp, err = parseClaudeCodeResult(claudeCodeTestJSON(t, result), req, "model", validators)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content() != "Final answer." || resp.Choices[0].FinishReason != "stop" {
		t.Fatalf("incorrect final answer: %+v", resp)
	}
}

func TestClaudeCodeParseErrors(t *testing.T) {
	for _, tt := range []struct {
		name   string
		mutate func(map[string]any)
		want   string
	}{
		{"wrong envelope", func(r map[string]any) { r["type"] = "assistant" }, "Claude Code result"},
		{"failed subtype", func(r map[string]any) { r["subtype"] = "error_max_turns"; r["errors"] = []string{"turn limit reached"} }, "turn limit reached"},
		{"error flag", func(r map[string]any) { r["is_error"] = true }, "Claude Code result"},
		{"truncated generation", func(r map[string]any) { r["stop_reason"] = "max_tokens" }, "stopped with max_tokens"},
		{"refusal", func(r map[string]any) { r["stop_reason"] = "refusal" }, "stopped with refusal"},
		{"permission denied", func(r map[string]any) { r["permission_denials"] = []any{map[string]any{"tool_name": "Bash"}} }, "permission denials"},
		{"missing usage", func(r map[string]any) { delete(r, "usage") }, "missing token usage"},
		{"null usage", func(r map[string]any) { r["usage"] = nil }, "missing token usage"},
		{"missing output", func(r map[string]any) { delete(r, "result") }, "empty response"},
		{"empty output", func(r map[string]any) { r["result"] = "" }, "empty response"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			result := claudeCodeTestResult()
			tt.mutate(result)
			resp, err := parseClaudeCodeResult(claudeCodeTestJSON(t, result), claudeCodeTestRequest(), "model", nil)
			claudeCodeTestError(t, err, tt.want)
			if resp != nil {
				t.Fatal("invalid result produced a usable response")
			}
		})
	}
	for _, field := range []string{"input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"} {
		t.Run("negative "+field, func(t *testing.T) {
			result := claudeCodeTestResult()
			result["usage"].(map[string]any)[field] = -1
			_, err := parseClaudeCodeResult(claudeCodeTestJSON(t, result), claudeCodeTestRequest(), "model", nil)
			claudeCodeTestError(t, err, "negative token usage")
		})
	}
	for _, usage := range []map[string]any{
		{"input_tokens": int64(math.MaxInt64), "cache_read_input_tokens": 1},
		{"input_tokens": int64(math.MaxInt64), "cache_creation_input_tokens": 1},
		{"input_tokens": int64(math.MaxInt64), "output_tokens": 1},
	} {
		result := claudeCodeTestResult()
		result["usage"] = usage
		_, err := parseClaudeCodeResult(claudeCodeTestJSON(t, result), claudeCodeTestRequest(), "model", nil)
		claudeCodeTestError(t, err, "overflows int64")
	}
	for _, data := range []string{"", "not JSON", `{"type":"result"`, `{} trailing`, `[]`} {
		t.Run("invalid JSON "+data, func(t *testing.T) {
			_, err := parseClaudeCodeResult([]byte(data), claudeCodeTestRequest(), "model", nil)
			claudeCodeTestError(t, err, "decode Claude Code JSON result")
		})
	}
}

func TestClaudeCodeParseInvalidActions(t *testing.T) {
	req := claudeCodeTestRequest()
	req.Tools = []ToolDef{claudeCodeTestTool()}
	_, _, _, validators, err := buildClaudeCodeRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct{ name, output, want string }{
		{"missing output", "", "decode Claude Code structured_output"},
		{"null output", `null`, "requires content and tool_calls"},
		{"wrong output type", `[]`, "decode Claude Code structured_output"},
		{"missing content", `{"tool_calls":[]}`, "requires content and tool_calls"},
		{"missing calls", `{"content":"done"}`, "requires content and tool_calls"},
		{"null content", `{"content":null,"tool_calls":[]}`, "requires content and tool_calls"},
		{"null calls", `{"content":"done","tool_calls":null}`, "requires content and tool_calls"},
		{"unknown field", `{"content":"done","tool_calls":[],"extra":true}`, "decode Claude Code structured_output"},
		{"unknown call field", `{"content":"","tool_calls":[{"name":"lookup","arguments":{},"id":"pretend"}]}`, "decode Claude Code structured_output"},
		{"unknown function", `{"content":"","tool_calls":[{"name":"unlisted","arguments":{}}]}`, "unknown function"},
		{"missing arguments", `{"content":"","tool_calls":[{"name":"lookup"}]}`, "JSON argument object"},
		{"null arguments", `{"content":"","tool_calls":[{"name":"lookup","arguments":null}]}`, "JSON argument object"},
		{"array arguments", `{"content":"","tool_calls":[{"name":"lookup","arguments":[]}]}`, "JSON argument object"},
		{"string arguments", `{"content":"","tool_calls":[{"name":"lookup","arguments":"{}"}]}`, "JSON argument object"},
		{"missing required parameter", `{"content":"","tool_calls":[{"name":"lookup","arguments":{}}]}`, "invalid arguments"},
		{"wrong parameter type through ref", `{"content":"","tool_calls":[{"name":"lookup","arguments":{"name":5}}]}`, "invalid arguments"},
		{"parameter constraint through ref", `{"content":"","tool_calls":[{"name":"lookup","arguments":{"name":""}}]}`, "invalid arguments"},
		{"additional parameter", `{"content":"","tool_calls":[{"name":"lookup","arguments":{"name":"a","unexpected":true}}]}`, "invalid arguments"},
		{"empty response", `{"content":"","tool_calls":[]}`, "empty response"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			result := claudeCodeTestResult()
			if tt.output != "" {
				result["structured_output"] = json.RawMessage(tt.output)
			}
			resp, err := parseClaudeCodeResult(claudeCodeTestJSON(t, result), req, "model", validators)
			claudeCodeTestError(t, err, tt.want)
			if resp != nil {
				t.Fatal("invalid action was returned to OCR")
			}
		})
	}
	req.ToolChoice = "required"
	result := claudeCodeTestResult()
	result["structured_output"] = json.RawMessage(`{"content":"I am done.","tool_calls":[]}`)
	_, err = parseClaudeCodeResult(claudeCodeTestJSON(t, result), req, "model", validators)
	claudeCodeTestError(t, err, "no calls for tool_choice required")
}

func TestClaudeCodeParseUnknownSchemaTypes(t *testing.T) {
	// Resolve checks schema structure and references, not the full JSON
	// meta-schema. Unknown types may resolve, but argument validation must
	// reject them before any proposed ToolCall can reach OCR execution.
	for _, tt := range []struct {
		name       string
		parameters map[string]any
	}{
		{"root", map[string]any{"type": "not-a-type"}},
		{"nested", map[string]any{
			"type": "object", "required": []string{"name"},
			"properties": map[string]any{"name": map[string]any{"type": "not-a-type"}},
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			req := claudeCodeTestRequest()
			tool := claudeCodeTestTool()
			tool.Function.Parameters = tt.parameters
			req.Tools = []ToolDef{tool}
			_, _, _, validators, err := buildClaudeCodeRequest(req)
			if err != nil {
				t.Fatal(err)
			}
			for _, value := range []any{nil, true, 1, "record", []any{}, map[string]any{}} {
				result := claudeCodeTestResult()
				result["structured_output"] = map[string]any{
					"content": "", "tool_calls": []any{map[string]any{
						"name": "lookup", "arguments": map[string]any{"name": value},
					}},
				}
				resp, err := parseClaudeCodeResult(claudeCodeTestJSON(t, result), req, "model", validators)
				claudeCodeTestError(t, err, "invalid arguments")
				if resp != nil {
					t.Fatal("unknown schema type allowed an action to reach OCR")
				}
			}
		})
	}
}

func TestClaudeCodeEnvironment(t *testing.T) {
	values := map[string]string{
		"ANTHROPIC_AUTH_TOKEN": "fake-test-auth-token", "CLAUDE_CODE_OAUTH_TOKEN": "fake-test-oauth-token",
		"ANTHROPIC_BASE_URL": "https://example.invalid", "CLAUDE_CODE_USE_BEDROCK": "1", "AWS_PROFILE": "fake-test-profile",
		"CLAUDE_CONFIG_DIR": "/fake-test-config", "OCR_CLAUDE_TEST_INHERITED": "a=b=c",
		"CLAUDE_CODE_SIMPLE": "1", "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "99",
	}
	for key, value := range values {
		t.Setenv(key, value)
	}
	for _, maxTokens := range []int{0, 2048} {
		t.Run(strconv.Itoa(maxTokens), func(t *testing.T) {
			got, counts := map[string]string{}, map[string]int{}
			for _, entry := range claudeCodeEnvironment(maxTokens) {
				key, value, _ := strings.Cut(entry, "=")
				got[key] = value
				counts[key]++
			}
			if _, exists := got["CLAUDE_CODE_SIMPLE"]; exists {
				t.Fatal("bare-mode environment was inherited")
			}
			for key, value := range values {
				if key == "CLAUDE_CODE_SIMPLE" {
					continue
				}
				if key == "CLAUDE_CODE_MAX_OUTPUT_TOKENS" && maxTokens > 0 {
					value = strconv.Itoa(maxTokens)
				}
				if got[key] != value || counts[key] != 1 {
					t.Fatalf("environment %s = %q (%d copies), want %q once", key, got[key], counts[key], value)
				}
			}
		})
	}
	if os.Getenv("CLAUDE_CODE_SIMPLE") != "1" || os.Getenv("CLAUDE_CODE_MAX_OUTPUT_TOKENS") != "99" {
		t.Fatal("environment builder mutated parent environment")
	}
}

func TestClaudeCodeErrorRedaction(t *testing.T) {
	for _, suffix := range []string{"AUTH", "TOKEN", "KEY", "SECRET", "CREDENTIAL", "COOKIE"} {
		value := "fake-sensitive-" + suffix
		t.Setenv("OCR_CLAUDE_TEST_"+suffix, value)
		got := redactClaudeCodeError("before " + value + " twice " + value + " after")
		if got != "before [REDACTED] twice [REDACTED] after" {
			t.Fatalf("redaction failed for %s", suffix)
		}
	}
	t.Setenv("OCR_CLAUDE_TEST_ORDINARY", "ordinary-test-value")
	if got := redactClaudeCodeError("ordinary-test-value"); got != "ordinary-test-value" {
		t.Fatal("redacted a non-sensitive value")
	}
	got := redactClaudeCodeError(strings.Repeat("x", claudeCodeStderrLimit+1))
	if got != strings.Repeat("x", claudeCodeStderrLimit-len(" [truncated]"))+" [truncated]" {
		t.Fatal("diagnostic output was not bounded")
	}
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "fake-result-secret")
	result := claudeCodeTestResult()
	result["is_error"] = true
	result["errors"] = []string{"fake-result-secret"}
	result["result"] = "fake-result-secret"
	_, err := parseClaudeCodeResult(claudeCodeTestJSON(t, result), claudeCodeTestRequest(), "model", nil)
	if err == nil || strings.Contains(err.Error(), "fake-result-secret") || !strings.Contains(err.Error(), "[REDACTED]") {
		t.Fatal("structured failure did not redact credentials")
	}
}

func TestClaudeCodeBuffer(t *testing.T) {
	buffer := &claudeCodeBuffer{limit: 8}
	for _, tt := range []struct {
		input, want string
		overflow    bool
	}{
		{"abc", "abc", false}, {"defgh", "abcdefgh", false}, {"i", "abcdefgh", true}, {"more", "abcdefgh", true}, {"", "abcdefgh", true},
	} {
		n, err := buffer.Write([]byte(tt.input))
		if n != len(tt.input) || err != nil || buffer.String() != tt.want || buffer.overflow != tt.overflow {
			t.Fatalf("Write(%q) = (%d, %v), contents %q, overflow %v", tt.input, n, err, buffer.String(), buffer.overflow)
		}
	}
}

func TestClaudeCodeBufferCopy(t *testing.T) {
	buffer := &claudeCodeBuffer{limit: 8}
	// Hide strings.Reader.WriteTo so io.Copy probes the destination's ReaderFrom,
	// just as the os/exec pipe-copy path does. Embedding bytes.Buffer used to
	// promote its unbounded ReadFrom and bypass the adapter's Write limit.
	source := struct{ io.Reader }{strings.NewReader("0123456789abcdef")}
	n, err := io.Copy(buffer, source)
	if err != nil || n != 16 || buffer.String() != "01234567" || !buffer.overflow {
		t.Fatalf("bounded copy = (%d, %v), contents %q, overflow %v", n, err, buffer.String(), buffer.overflow)
	}
}

// The test executable itself is the fake CLI. TestMain intercepts its Claude
// arguments before testing parses flags, so no shell or installed CLI is needed.
const claudeCodeHelperEnv = "_OCR_CLAUDE_CODE_TEST_HELPER"

type claudeCodeHelperConfig struct {
	Mode     string
	Output   string
	Stderr   string
	ExitCode int
	Records  string
}

type claudeCodeInvocation struct {
	Args       []string
	Dir        string
	DirMode    uint32
	PromptFile string
	PromptMode uint32
	System     string
	Input      json.RawMessage
	Env        map[string]string
	Entries    []string
}

func TestMain(m *testing.M) {
	if ready := os.Getenv("_OCR_CLAUDE_CODE_TEST_DESCENDANT"); ready != "" {
		if err := os.WriteFile(ready, []byte(strconv.Itoa(os.Getpid())), 0o600); err != nil {
			os.Exit(2)
		}
		time.Sleep(time.Minute)
		os.Exit(0)
	}
	if config := os.Getenv(claudeCodeHelperEnv); config != "" {
		if err := runClaudeCodeTestHelper(config); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		os.Exit(0)
	}
	os.Exit(m.Run())
}

func runClaudeCodeTestHelper(configPath string) error {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return err
	}
	var cfg claudeCodeHelperConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return err
	}
	input, err := io.ReadAll(os.Stdin)
	if err != nil {
		return err
	}
	dir, err := os.Getwd()
	if err != nil {
		return err
	}
	info, err := os.Stat(dir)
	if err != nil {
		return err
	}
	record := claudeCodeInvocation{Args: os.Args[1:], Dir: dir, DirMode: uint32(info.Mode().Perm()), Input: input, Env: map[string]string{}}
	for i, arg := range record.Args {
		if arg == "--system-prompt-file" && i+1 < len(record.Args) {
			record.PromptFile = record.Args[i+1]
		}
	}
	prompt, err := os.ReadFile(record.PromptFile)
	if err != nil {
		return err
	}
	record.System = string(prompt)
	info, err = os.Stat(record.PromptFile)
	if err != nil {
		return err
	}
	record.PromptMode = uint32(info.Mode().Perm())
	entries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	for _, entry := range entries {
		record.Entries = append(record.Entries, entry.Name())
	}
	// Only these synthetic fixture values are recorded, never the real inherited
	// credential environment, even if a developer has authenticated locally.
	for _, key := range []string{"OCR_CLAUDE_TEST_INHERITED", "CLAUDE_CODE_SIMPLE", "CLAUDE_CODE_MAX_OUTPUT_TOKENS"} {
		if value, exists := os.LookupEnv(key); exists {
			record.Env[key] = value
		}
	}
	data, err = json.Marshal(record)
	if err != nil {
		return err
	}
	recordPath := filepath.Join(cfg.Records, filepath.Base(dir)+".json")
	if err := os.WriteFile(recordPath+".tmp", data, 0o600); err != nil {
		return err
	}
	if err := os.Rename(recordPath+".tmp", recordPath); err != nil {
		return err
	}
	switch cfg.Mode {
	case "hang":
		time.Sleep(time.Minute)
	case "descendant", "orphan":
		self, err := os.Executable()
		if err != nil {
			return err
		}
		child := exec.Command(self)
		child.Env = append(os.Environ(), "_OCR_CLAUDE_CODE_TEST_DESCENDANT="+filepath.Join(cfg.Records, "descendant.pid"))
		child.Stdout, child.Stderr = os.Stdout, os.Stderr
		if err := child.Start(); err != nil {
			return err
		}
		if cfg.Mode == "orphan" {
			// Ensure the child's PID is observable before its parent exits and
			// starts WaitDelay. This avoids depending on child startup speed.
			deadline := time.Now().Add(5 * time.Second)
			for time.Now().Before(deadline) {
				if _, err := os.Stat(filepath.Join(cfg.Records, "descendant.pid")); err == nil {
					return nil
				}
				time.Sleep(10 * time.Millisecond)
			}
			return fmt.Errorf("descendant did not become ready")
		}
		return child.Wait()
	case "overflow":
		_, err := io.CopyN(os.Stdout, strings.NewReader(strings.Repeat("x", claudeCodeOutputLimit+1)), claudeCodeOutputLimit+1)
		return err
	}
	if cfg.Stderr != "" {
		fmt.Fprint(os.Stderr, cfg.Stderr)
	}
	if cfg.Output != "" {
		fmt.Fprint(os.Stdout, cfg.Output)
	} else if cfg.Mode != "empty" {
		var snapshot struct {
			Messages []Message `json:"messages"`
		}
		if err := json.Unmarshal(input, &snapshot); err != nil {
			return err
		}
		text := snapshot.Messages[len(snapshot.Messages)-1].ExtractText()
		result := claudeCodeTestResult()
		result["result"] = text
		for _, arg := range record.Args {
			if arg == "--json-schema" {
				result["structured_output"] = map[string]any{"content": text, "tool_calls": []any{}}
			}
		}
		if cfg.Mode == "tool-roundtrip" && snapshot.Messages[len(snapshot.Messages)-1].Role != "tool" {
			result["structured_output"] = map[string]any{"content": "", "tool_calls": []any{map[string]any{"name": "lookup", "arguments": map[string]any{"name": "requested-record"}}}}
		}
		if err := json.NewEncoder(os.Stdout).Encode(result); err != nil {
			return err
		}
	}
	if cfg.ExitCode != 0 {
		os.Exit(cfg.ExitCode)
	}
	return nil
}

func newClaudeCodeTestCLI(t *testing.T, cfg claudeCodeHelperConfig) (string, string) {
	t.Helper()
	root := t.TempDir()
	cfg.Records = filepath.Join(root, "records")
	if err := os.Mkdir(cfg.Records, 0o700); err != nil {
		t.Fatal(err)
	}
	workspaces := filepath.Join(root, "workspaces")
	if err := os.Mkdir(workspaces, 0o700); err != nil {
		t.Fatal(err)
	}
	config := filepath.Join(root, "fixture.json")
	if err := os.WriteFile(config, claudeCodeTestJSON(t, cfg), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv(claudeCodeHelperEnv, config)
	t.Setenv("GORACE", os.Getenv("GORACE")+" atexit_sleep_ms=0")
	for _, key := range []string{"TMPDIR", "TMP", "TEMP"} {
		t.Setenv(key, workspaces)
	}
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	return self, cfg.Records
}

func claudeCodeTestInvocations(t *testing.T, directory string, count int) []claudeCodeInvocation {
	t.Helper()
	entries, err := os.ReadDir(directory)
	if err != nil {
		t.Fatal(err)
	}
	var records []claudeCodeInvocation
	for _, entry := range entries {
		if filepath.Ext(entry.Name()) != ".json" {
			continue
		}
		data, err := os.ReadFile(filepath.Join(directory, entry.Name()))
		if err != nil {
			t.Fatal(err)
		}
		var record claudeCodeInvocation
		if err := json.Unmarshal(data, &record); err != nil {
			t.Fatal(err)
		}
		records = append(records, record)
	}
	if len(records) != count {
		t.Fatalf("got %d CLI invocations, want %d", len(records), count)
	}
	return records
}

func claudeCodeTestCleaned(t *testing.T, records []claudeCodeInvocation) {
	t.Helper()
	for _, record := range records {
		if _, err := os.Stat(record.Dir); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("temporary workspace was not removed: %v", err)
		}
	}
}

func TestClaudeCodeSubprocessInvocation(t *testing.T) {
	for _, tt := range []struct {
		name, configuredModel, requestedModel, wantModel string
		tools                                            bool
	}{
		{"configured model", "configured-alias", "", "configured-alias", false},
		{"request override", "configured-alias", "request-alias", "request-alias", true},
		{"CLI default", "", "", "", false},
		{"explicit default", "configured-alias", "default", "", false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			command, records := newClaudeCodeTestCLI(t, claudeCodeHelperConfig{Stderr: "harmless CLI diagnostic"})
			t.Setenv("OCR_CLAUDE_TEST_INHERITED", "inherited=fixture")
			t.Setenv("CLAUDE_CODE_SIMPLE", "1")
			t.Setenv("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "99")
			req := claudeCodeTestRequest()
			req.Model, req.MaxTokens = tt.requestedModel, 2048
			req.Messages = append([]Message{NewTextMessage("system", "Private system instruction.")}, req.Messages...)
			if tt.tools {
				req.Tools = []ToolDef{claudeCodeTestTool()}
			}
			client := NewClaudeCodeClient(ClientConfig{ClaudeCommand: command, Model: tt.configuredModel})
			resp, err := client.CompletionsWithCtx(context.Background(), req)
			if err != nil {
				t.Fatal(err)
			}
			if resp.Content() != "Review this change." {
				t.Fatalf("response = %q", resp.Content())
			}
			invocations := claudeCodeTestInvocations(t, records, 1)
			record := invocations[0]
			input, system, schema, _, err := buildClaudeCodeRequest(req)
			if err != nil {
				t.Fatal(err)
			}
			wantArgs := []string{"-p", "--safe-mode", "--tools", "", "--strict-mcp-config", "--disallowedTools", "mcp__*", "--permission-mode", "dontAsk", "--permission-prompts", "none", "--no-session-persistence", "--output-format", "json", "--system-prompt-file", record.PromptFile}
			if tt.wantModel != "" {
				wantArgs = append(wantArgs, "--model", tt.wantModel)
			}
			if tt.tools {
				wantArgs = append(wantArgs, "--json-schema", schema, "--max-turns", "4")
			} else {
				wantArgs = append(wantArgs, "--max-turns", "1")
			}
			if !reflect.DeepEqual(record.Args, wantArgs) {
				t.Fatalf("argv = %#v, want %#v", record.Args, wantArgs)
			}
			if record.System != system || string(record.Input) != string(input) {
				t.Fatal("stdin or system prompt changed in transit")
			}
			if record.PromptFile != filepath.Join(record.Dir, "system.txt") || !reflect.DeepEqual(record.Entries, []string{"system.txt"}) {
				t.Fatalf("unexpected working directory contents: %+v", record.Entries)
			}
			if runtime.GOOS != "windows" && (record.DirMode != 0o700 || record.PromptMode != 0o600) {
				t.Fatalf("insecure workspace/prompt modes: %o/%o", record.DirMode, record.PromptMode)
			}
			if !reflect.DeepEqual(record.Env, map[string]string{"OCR_CLAUDE_TEST_INHERITED": "inherited=fixture", "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "2048"}) {
				t.Fatalf("unexpected child environment: %v", record.Env)
			}
			claudeCodeTestCleaned(t, invocations)
		})
	}
}

func TestClaudeCodeSubprocessToolRoundTrip(t *testing.T) {
	command, records := newClaudeCodeTestCLI(t, claudeCodeHelperConfig{Mode: "tool-roundtrip"})
	client := NewClaudeCodeClient(ClientConfig{ClaudeCommand: command})
	req := claudeCodeTestRequest()
	req.ToolChoice, req.Tools = "required", []ToolDef{claudeCodeTestTool()}
	resp, err := client.CompletionsWithCtx(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.ToolCalls()) != 1 || resp.ToolCalls()[0].Function.Name != "lookup" {
		t.Fatal("CLI did not return a proposed OCR action")
	}
	call := resp.ToolCalls()[0]
	if call.Function.Arguments != `{"name":"requested-record"}` {
		t.Fatalf("tool arguments changed: %s", call.Function.Arguments)
	}
	const actualResult = `{"name":"requested-record","value":"only-known-after-OCR-execution"}`
	req.Messages = append(req.Messages, NewToolCallMessage(resp.VisibleContent(), resp.ToolCalls(), resp.Native(), resp.ReasoningContent()), NewToolResultMessage(call.ID, actualResult))
	req.ToolChoice = "auto"
	resp, err = client.CompletionsWithCtx(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content() != actualResult || len(resp.ToolCalls()) != 0 {
		t.Fatal("follow-up failed to use OCR's actual tool result")
	}
	invocations := claudeCodeTestInvocations(t, records, 2)
	foundResult := false
	for _, record := range invocations {
		var snapshot struct {
			Messages []Message `json:"messages"`
		}
		if err := json.Unmarshal(record.Input, &snapshot); err != nil {
			t.Fatal(err)
		}
		if len(snapshot.Messages) == 3 {
			foundResult = true
			if snapshot.Messages[1].ToolCalls[0].ID != call.ID || snapshot.Messages[2].ToolCallID != call.ID || snapshot.Messages[2].Content != actualResult {
				t.Fatal("tool-call linkage or result was lost in the next snapshot")
			}
		}
	}
	if !foundResult {
		t.Fatal("follow-up snapshot omitted OCR's tool result")
	}
	claudeCodeTestCleaned(t, invocations)
}

func TestClaudeCodeSubprocessHistoryReset(t *testing.T) {
	command, records := newClaudeCodeTestCLI(t, claudeCodeHelperConfig{})
	client := NewClaudeCodeClient(ClientConfig{ClaudeCommand: command})
	first := claudeCodeTestRequest()
	first.SessionID = "same-ocr-session"
	first.Tools = []ToolDef{claudeCodeTestTool()}
	first.Messages = []Message{NewTextMessage("system", "Old rules."), NewTextMessage("user", "Old source that compression removed.")}
	if _, err := client.CompletionsWithCtx(context.Background(), first); err != nil {
		t.Fatal(err)
	}
	second := claudeCodeTestRequest()
	second.SessionID = first.SessionID
	second.Messages = []Message{NewTextMessage("system", "New rules."), NewTextMessage("user", "Compressed current conversation.")}
	if _, err := client.CompletionsWithCtx(context.Background(), second); err != nil {
		t.Fatal(err)
	}
	invocations := claudeCodeTestInvocations(t, records, 2)
	seenNew := false
	for _, record := range invocations {
		if strings.Contains(record.System, "New rules.") {
			seenNew = true
			if strings.Contains(string(record.Input), "Old source") || strings.Contains(record.System, "Old rules") || strings.Contains(string(record.Input), `"tools"`) {
				t.Fatal("reset snapshot retained stale history, rules, or tools")
			}
		}
		for _, arg := range record.Args {
			if arg == "--resume" || arg == "--continue" || arg == "--session-id" {
				t.Fatal("adapter resumed hidden CLI history")
			}
		}
	}
	if !seenNew || invocations[0].Dir == invocations[1].Dir {
		t.Fatal("requests did not use fresh isolated snapshots")
	}
	claudeCodeTestCleaned(t, invocations)
}

func TestClaudeCodeSubprocessFailures(t *testing.T) {
	failure := claudeCodeTestResult()
	failure["is_error"] = true
	failure["errors"] = []string{"authentication failed fake-subprocess-secret"}
	for _, tt := range []struct {
		name    string
		fixture claudeCodeHelperConfig
		want    string
	}{
		{"nonzero structured failure", claudeCodeHelperConfig{Output: string(claudeCodeTestJSON(t, failure)), Stderr: "fake-subprocess-secret", ExitCode: 3}, "authentication failed [REDACTED]"},
		{"nonzero stderr", claudeCodeHelperConfig{Mode: "empty", Stderr: "failed fake-subprocess-secret", ExitCode: 3}, "failed [REDACTED]"},
		{"nonzero despite valid result", claudeCodeHelperConfig{ExitCode: 3}, "Claude Code failed"},
		{"invalid successful stdout", claudeCodeHelperConfig{Output: "not JSON"}, "decode Claude Code JSON result"},
		{"empty successful stdout", claudeCodeHelperConfig{Mode: "empty"}, "decode Claude Code JSON result"},
		{"bounded stdout", claudeCodeHelperConfig{Mode: "overflow"}, "output exceeds"},
		{"bounded stderr", claudeCodeHelperConfig{Mode: "empty", Stderr: strings.Repeat("x", claudeCodeStderrLimit*2), ExitCode: 1}, "Claude Code failed"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			command, records := newClaudeCodeTestCLI(t, tt.fixture)
			t.Setenv("ANTHROPIC_AUTH_TOKEN", "fake-subprocess-secret")
			resp, err := NewClaudeCodeClient(ClientConfig{ClaudeCommand: command}).CompletionsWithCtx(context.Background(), claudeCodeTestRequest())
			claudeCodeTestError(t, err, tt.want)
			if resp != nil || strings.Contains(err.Error(), "fake-subprocess-secret") || len(err.Error()) > claudeCodeStderrLimit+1024 {
				t.Fatal("failure returned a response, leaked a credential, or retained unbounded stderr")
			}
			claudeCodeTestCleaned(t, claudeCodeTestInvocations(t, records, 1))
		})
	}
}

func TestClaudeCodeClientEarlyFailures(t *testing.T) {
	client := NewClaudeCodeClient(ClientConfig{})
	if client.cfg.Timeout != 5*time.Minute || client.cfg.ClaudeCommand != "claude" {
		t.Fatalf("unexpected defaults: %+v", client.cfg)
	}
	for _, tt := range []struct {
		name    string
		cfg     ClientConfig
		request ChatRequest
		want    string
	}{
		{"unsupported config", ClientConfig{APIKey: "must-not-be-logged"}, claudeCodeTestRequest(), "does not support"},
		{"negative timeout", ClientConfig{Timeout: -time.Second}, claudeCodeTestRequest(), "timeout must be non-negative"},
		{"invalid request", ClientConfig{}, ChatRequest{}, "requires conversation messages"},
		{"missing executable", ClientConfig{ClaudeCommand: filepath.Join(t.TempDir(), "missing-claude")}, claudeCodeTestRequest(), "install Claude Code or set claude_command"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := NewClaudeCodeClient(tt.cfg).CompletionsWithCtx(context.Background(), tt.request)
			claudeCodeTestError(t, err, tt.want)
			if resp != nil || strings.Contains(err.Error(), "must-not-be-logged") {
				t.Fatal("early failure exposed a response or credentials")
			}
		})
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := client.CompletionsWithCtx(ctx, claudeCodeTestRequest()); !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-canceled context: %v", err)
	}
	t.Run("working directory creation", func(t *testing.T) {
		command, records := newClaudeCodeTestCLI(t, claudeCodeHelperConfig{})
		missing := filepath.Join(t.TempDir(), "does-not-exist")
		for _, key := range []string{"TMPDIR", "TMP", "TEMP"} {
			t.Setenv(key, missing)
		}
		_, err := NewClaudeCodeClient(ClientConfig{ClaudeCommand: command}).CompletionsWithCtx(context.Background(), claudeCodeTestRequest())
		claudeCodeTestError(t, err, "create Claude Code working directory")
		claudeCodeTestInvocations(t, records, 0)
	})
}

func TestClaudeCodeSubprocessRelativeExecutable(t *testing.T) {
	command, records := newClaudeCodeTestCLI(t, claudeCodeHelperConfig{})
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	relative, err := filepath.Rel(cwd, command)
	if err != nil {
		t.Skip("test executable and working directory are on different volumes")
	}
	if _, err := NewClaudeCodeClient(ClientConfig{ClaudeCommand: relative}).CompletionsWithCtx(context.Background(), claudeCodeTestRequest()); err != nil {
		t.Fatal(err)
	}
	claudeCodeTestCleaned(t, claudeCodeTestInvocations(t, records, 1))
}

func TestClaudeCodeSubprocessCancellation(t *testing.T) {
	for _, timeout := range []bool{false, true} {
		t.Run(fmt.Sprintf("timeout=%t", timeout), func(t *testing.T) {
			command, records := newClaudeCodeTestCLI(t, claudeCodeHelperConfig{Mode: "hang"})
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			cfg := ClientConfig{ClaudeCommand: command, Timeout: 10 * time.Second}
			want := error(context.Canceled)
			if timeout {
				cfg.Timeout = 500 * time.Millisecond
				want = context.DeadlineExceeded
			}
			done := make(chan error, 1)
			go func() {
				_, err := NewClaudeCodeClient(cfg).CompletionsWithCtx(ctx, claudeCodeTestRequest())
				done <- err
			}()
			if !timeout {
				claudeCodeTestWait(t, func() bool { entries, _ := os.ReadDir(records); return len(entries) > 0 })
				cancel()
			}
			select {
			case err := <-done:
				if !errors.Is(err, want) {
					t.Fatalf("completion error = %v, want %v", err, want)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("cancellation did not terminate CLI promptly")
			}
			entries, err := os.ReadDir(filepath.Join(filepath.Dir(records), "workspaces"))
			if err != nil || len(entries) != 0 {
				t.Fatalf("canceled CLI left temporary files: %v, %v", entries, err)
			}
		})
	}
}

func claudeCodeTestWait(t *testing.T, ready func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if ready() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("timed out waiting for helper subprocess")
}

func TestClaudeCodeSubprocessConcurrentIsolation(t *testing.T) {
	command, records := newClaudeCodeTestCLI(t, claudeCodeHelperConfig{})
	client := NewClaudeCodeClient(ClientConfig{ClaudeCommand: command})
	const count = 8
	var wg sync.WaitGroup
	for i := range count {
		wg.Add(1)
		go func() {
			defer wg.Done()
			text := fmt.Sprintf("isolated request %d", i)
			req := claudeCodeTestRequest()
			req.Messages[0].Content = text
			resp, err := client.CompletionsWithCtx(context.Background(), req)
			if err != nil {
				t.Errorf("request %d: %v", i, err)
				return
			}
			if resp.Content() != text {
				t.Errorf("request %d received another request's response", i)
			}
		}()
	}
	wg.Wait()
	invocations := claudeCodeTestInvocations(t, records, count)
	directories := map[string]bool{}
	for _, record := range invocations {
		if directories[record.Dir] {
			t.Fatal("concurrent requests shared a working directory")
		}
		directories[record.Dir] = true
	}
	claudeCodeTestCleaned(t, invocations)
}

func TestClaudeCodeRawCapture(t *testing.T) {
	t.Run("unbound holder", func(t *testing.T) {
		command, _ := newClaudeCodeTestCLI(t, claudeCodeHelperConfig{})
		client := NewClaudeCodeClient(ClientConfig{ClaudeCommand: command, rawHolder: NewRawHolder()})
		if _, err := client.CompletionsWithCtx(context.Background(), claudeCodeTestRequest()); err != nil {
			t.Fatal(err)
		}
	})
	for _, tt := range []struct {
		name, output, stderr string
		exitCode             int
		wantError            bool
	}{
		{"success", string(claudeCodeTestJSON(t, claudeCodeTestResult())), "", 0, false},
		{"invalid output", "invalid fake-raw-secret output", "fake-raw-secret", 1, true},
		{"structured failure", `{"type":"result","subtype":"error_during_execution","is_error":true,"errors":["fake-raw-secret"]}`, "", 1, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			command, _ := newClaudeCodeTestCLI(t, claudeCodeHelperConfig{Output: tt.output, Stderr: tt.stderr, ExitCode: tt.exitCode})
			t.Setenv("ANTHROPIC_AUTH_TOKEN", "fake-raw-secret")
			writer := &recordingRawWriter{}
			holder := NewRawHolder()
			holder.Set(writer)
			client := NewClaudeCodeClient(ClientConfig{ClaudeCommand: command, Model: "raw-model", rawHolder: holder})
			meta := RequestMeta{Model: "raw-model", FilePath: "pkg/review.go", TaskType: "main_task", RequestNo: 4}
			ctx := WithRequestMeta(context.Background(), meta)
			traceID := oteltrace.TraceID{1, 2, 3, 4}
			ctx = oteltrace.ContextWithSpanContext(ctx, oteltrace.NewSpanContext(oteltrace.SpanContextConfig{TraceID: traceID, SpanID: oteltrace.SpanID{1}}))
			req := claudeCodeTestRequest()
			_, err := client.CompletionsWithCtx(ctx, req)
			if (err != nil) != tt.wantError {
				t.Fatalf("unexpected completion success/failure: %v", err)
			}
			record := writer.one(t)
			if record.Transport != "claude-code" || record.RequestID == "" || record.Model != "raw-model" || record.TraceID != traceID.String() || record.FilePath != meta.FilePath || record.TaskType != meta.TaskType || record.RequestNo != meta.RequestNo {
				t.Fatalf("incorrect CLI raw metadata: %+v", record)
			}
			if _, err := time.Parse(time.RFC3339, record.Timestamp); err != nil || record.DurationMs < 0 {
				t.Fatalf("invalid capture timing: %v", err)
			}
			if record.StatusCode != 0 || len(record.RequestHeaders) != 0 || len(record.ResponseHeaders) != 0 || record.SessionID != "" {
				t.Fatal("subprocess capture invented HTTP or session metadata")
			}
			input, system, _, _, err := buildClaudeCodeRequest(req)
			if err != nil {
				t.Fatal(err)
			}
			var request struct {
				System       string          `json:"system"`
				Conversation json.RawMessage `json:"conversation"`
			}
			if err := json.Unmarshal(record.RequestBody, &request); err != nil {
				t.Fatal(err)
			}
			if request.System != system || string(request.Conversation) != string(input) || record.RequestBodyText != "" {
				t.Fatal("raw capture did not record the actual subprocess input")
			}
			if (record.Error != "") != tt.wantError {
				t.Fatal("raw capture lost subprocess error metadata")
			}
			encoded := string(claudeCodeTestJSON(t, record))
			if strings.Contains(encoded, "fake-raw-secret") || strings.Contains(encoded, "ANTHROPIC_AUTH_TOKEN") {
				t.Fatal("raw capture leaked credentials or environment")
			}
			wantOutput := strings.ReplaceAll(tt.output, "fake-raw-secret", "[REDACTED]")
			if json.Valid([]byte(tt.output)) {
				if string(record.ResponseBody) != wantOutput || record.ResponseBodyText != "" {
					t.Fatal("JSON subprocess output was not preserved/redacted correctly")
				}
			} else if record.ResponseBodyText != wantOutput || len(record.ResponseBody) != 0 {
				t.Fatal("plain subprocess output was not preserved/redacted correctly")
			}
		})
	}
}

func TestClaudeCodeLive(t *testing.T) {
	if os.Getenv("OCR_TEST_CLAUDE_CODE") != "1" {
		t.Skip("set OCR_TEST_CLAUDE_CODE=1 to use the installed CLI and current authentication")
	}
	command, err := exec.LookPath("claude")
	if err != nil {
		t.Fatal("OCR_TEST_CLAUDE_CODE=1 requires an installed claude executable")
	}
	client := NewClaudeCodeClient(ClientConfig{ClaudeCommand: command})
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	text := ChatRequest{Messages: []Message{NewTextMessage("user", "Reply with exactly OCR_TEXT_OK and no other text.")}, MaxTokens: 1024}
	resp, err := client.CompletionsWithCtx(ctx, text)
	if err != nil {
		t.Fatal("live Claude Code text request failed; inspect CLI authentication separately")
	}
	if resp.Content() != "OCR_TEXT_OK" || resp.Usage == nil {
		t.Fatal("live text response did not satisfy its contract")
	}

	req := ChatRequest{
		Messages: []Message{NewTextMessage("user", "Call lookup with name sample-731. After OCR returns its result, reply with exactly its value and no other text. Never invent the value.")},
		Tools:    []ToolDef{claudeCodeTestTool()}, ToolChoice: "required", MaxTokens: 2048,
	}
	resp, err = client.CompletionsWithCtx(ctx, req)
	if err != nil {
		t.Fatal("live Claude Code tool request failed")
	}
	if len(resp.ToolCalls()) != 1 || resp.ToolCalls()[0].Function.Name != "lookup" {
		t.Fatal("live response did not request the synthetic OCR tool")
	}
	call := resp.ToolCalls()[0]
	var arguments struct {
		Name string `json:"name"`
	}
	if err := json.Unmarshal([]byte(call.Function.Arguments), &arguments); err != nil || arguments.Name != "sample-731" {
		t.Fatal("live tool call did not request the actual task's data")
	}
	// Generate the value only after the call: it cannot be guessed from the first
	// request, and is supplied exclusively as OCR's real tool result.
	value := fmt.Sprintf("OCR_RESULT_%d", time.Now().UnixNano())
	req.Messages = append(req.Messages, NewToolCallMessage(resp.VisibleContent(), resp.ToolCalls(), resp.Native(), resp.ReasoningContent()), NewToolResultMessage(call.ID, string(claudeCodeTestJSON(t, map[string]string{"name": arguments.Name, "value": value}))))
	req.ToolChoice = "auto"
	resp, err = client.CompletionsWithCtx(ctx, req)
	if err != nil {
		t.Fatal("live Claude Code follow-up request failed")
	}
	if len(resp.ToolCalls()) != 0 || resp.Content() != value {
		t.Fatal("live follow-up did not use the actual OCR tool result")
	}
}
