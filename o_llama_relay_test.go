package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func decodeNDJSON(t *testing.T, body string) []map[string]any {
	t.Helper()

	lines := strings.Split(strings.TrimSpace(body), "\n")
	chunks := make([]map[string]any, 0, len(lines))
	for _, line := range lines {
		if strings.TrimSpace(line) == "" {
			continue
		}

		var chunk map[string]any
		if err := json.Unmarshal([]byte(line), &chunk); err != nil {
			t.Fatalf("failed to decode NDJSON line %q: %v", line, err)
		}
		chunks = append(chunks, chunk)
	}

	return chunks
}

func TestHandleGenerateStream(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/completions" {
			t.Errorf("unexpected path: %s", r.URL.Path)
			http.Error(w, "unexpected path", http.StatusNotFound)
			return
		}

		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("failed to decode upstream request: %v", err)
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		if stream, ok := body["stream"].(bool); !ok || !stream {
			t.Errorf("expected stream=true, got %#v", body["stream"])
		}

		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)

		fmt.Fprint(w, "data: {\"choices\":[{\"text\":\"hel\",\"finish_reason\":null}]}\n\n")
		flusher.Flush()
		fmt.Fprint(w, "data: {\"choices\":[{\"text\":\"lo\",\"finish_reason\":\"stop\"}]}\n\n")
		flusher.Flush()
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer upstream.Close()

	handler := NewOllamaHandler(NewLlamaClient(upstream.URL))

	req := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(`{"model":"demo","prompt":"hi","stream":true}`))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("Content-Type"); !strings.Contains(got, "application/x-ndjson") {
		t.Fatalf("expected NDJSON content type, got %q", got)
	}

	chunks := decodeNDJSON(t, rec.Body.String())
	if len(chunks) != 3 {
		t.Fatalf("expected 3 chunks, got %d: %s", len(chunks), rec.Body.String())
	}

	if response, _ := chunks[0]["response"].(string); response != "hel" {
		t.Fatalf("expected first chunk response hel, got %#v", chunks[0]["response"])
	}
	if done, _ := chunks[0]["done"].(bool); done {
		t.Fatalf("expected first chunk done=false, got %#v", chunks[0]["done"])
	}

	if response, _ := chunks[1]["response"].(string); response != "lo" {
		t.Fatalf("expected second chunk response lo, got %#v", chunks[1]["response"])
	}

	if done, _ := chunks[2]["done"].(bool); !done {
		t.Fatalf("expected final chunk done=true, got %#v", chunks[2]["done"])
	}
	if doneReason, _ := chunks[2]["done_reason"].(string); doneReason != "stop" {
		t.Fatalf("expected final chunk done_reason=stop, got %#v", chunks[2]["done_reason"])
	}
}

func TestHandleChatStream(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("unexpected path: %s", r.URL.Path)
			http.Error(w, "unexpected path", http.StatusNotFound)
			return
		}

		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("failed to decode upstream request: %v", err)
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		if stream, ok := body["stream"].(bool); !ok || !stream {
			t.Errorf("expected stream=true, got %#v", body["stream"])
		}

		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)

		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n")
		flusher.Flush()
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n")
		flusher.Flush()
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\" there\"},\"finish_reason\":null}]}\n\n")
		flusher.Flush()
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		flusher.Flush()
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer upstream.Close()

	handler := NewOllamaHandler(NewLlamaClient(upstream.URL))

	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"model":"demo","messages":[{"role":"user","content":"hi"}],"stream":true}`))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("Content-Type"); !strings.Contains(got, "application/x-ndjson") {
		t.Fatalf("expected NDJSON content type, got %q", got)
	}

	chunks := decodeNDJSON(t, rec.Body.String())
	if len(chunks) != 3 {
		t.Fatalf("expected 3 chunks, got %d: %s", len(chunks), rec.Body.String())
	}

	firstMessage, _ := chunks[0]["message"].(map[string]any)
	if role, _ := firstMessage["role"].(string); role != "assistant" {
		t.Fatalf("expected first chunk role assistant, got %#v", firstMessage["role"])
	}
	if content, _ := firstMessage["content"].(string); content != "hi" {
		t.Fatalf("expected first chunk content hi, got %#v", firstMessage["content"])
	}

	secondMessage, _ := chunks[1]["message"].(map[string]any)
	if content, _ := secondMessage["content"].(string); content != " there" {
		t.Fatalf("expected second chunk content ' there', got %#v", secondMessage["content"])
	}

	if done, _ := chunks[2]["done"].(bool); !done {
		t.Fatalf("expected final chunk done=true, got %#v", chunks[2]["done"])
	}
	if doneReason, _ := chunks[2]["done_reason"].(string); doneReason != "stop" {
		t.Fatalf("expected final chunk done_reason=stop, got %#v", chunks[2]["done_reason"])
	}
}

func TestHandleChatStreamToolCalls(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("unexpected path: %s", r.URL.Path)
			http.Error(w, "unexpected path", http.StatusNotFound)
			return
		}

		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("failed to decode upstream request: %v", err)
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		if stream, ok := body["stream"].(bool); !ok || !stream {
			t.Errorf("expected stream=true, got %#v", body["stream"])
		}

		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)

		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n")
		flusher.Flush()
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_temperature\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n")
		flusher.Flush()
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\":\\\"New\"}}]},\"finish_reason\":null}]}\n\n")
		flusher.Flush()
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\" York\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n")
		flusher.Flush()
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer upstream.Close()

	handler := NewOllamaHandler(NewLlamaClient(upstream.URL))

	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"model":"demo","messages":[{"role":"user","content":"hi"}],"stream":true}`))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}

	chunks := decodeNDJSON(t, rec.Body.String())
	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks, got %d: %s", len(chunks), rec.Body.String())
	}

	message, _ := chunks[0]["message"].(map[string]any)
	toolCalls, _ := message["tool_calls"].([]any)
	if len(toolCalls) != 1 {
		t.Fatalf("expected one streamed tool call, got %#v", message["tool_calls"])
	}

	toolCall, _ := toolCalls[0].(map[string]any)
	function, _ := toolCall["function"].(map[string]any)
	if name, _ := function["name"].(string); name != "get_temperature" {
		t.Fatalf("expected tool call name get_temperature, got %#v", function["name"])
	}

	arguments, _ := function["arguments"].(map[string]any)
	if city, _ := arguments["city"].(string); city != "New York" {
		t.Fatalf("expected tool call city New York, got %#v", arguments["city"])
	}

	if doneReason, _ := chunks[1]["done_reason"].(string); doneReason != "tool_calls" {
		t.Fatalf("expected final chunk done_reason=tool_calls, got %#v", chunks[1]["done_reason"])
	}
}

var unloadCalled = false

func TestHandleGenerate_KeepAliveZero_Unloads(t *testing.T) {
	unloadCalled = false
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/completions":
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"choices":[{"text":"hello","finish_reason":"stop"}]}`)
		case "/models/unload":
			unloadCalled = true
			w.WriteHeader(http.StatusOK)
		default:
			http.Error(w, "not found", http.StatusNotFound)
		}
	}))
	defer upstream.Close()

	handler := NewOllamaHandler(NewLlamaClient(upstream.URL))

	req := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(`{"model":"demo","prompt":"hi","keep_alive":0}`))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}
	if !unloadCalled {
		t.Fatal("expected unload to be called when keep_alive=0")
	}
}

func TestHandleGenerate_KeepAliveDefault_NoUnload(t *testing.T) {
	unloadCalled = false
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/completions":
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"choices":[{"text":"hello","finish_reason":"stop"}]}`)
		case "/models/unload":
			unloadCalled = true
			w.WriteHeader(http.StatusOK)
		default:
			http.Error(w, "not found", http.StatusNotFound)
		}
	}))
	defer upstream.Close()

	handler := NewOllamaHandler(NewLlamaClient(upstream.URL))

	req := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(`{"model":"demo","prompt":"hi"}`))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}
	if unloadCalled {
		t.Fatal("expected unload NOT to be called when keep_alive is omitted")
	}
}

func TestHandleChatToolCalls(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("unexpected path: %s", r.URL.Path)
			http.Error(w, "unexpected path", http.StatusNotFound)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","index":0,"function":{"name":"get_temperature","arguments":"{\"city\":\"New York\"}"}}]},"finish_reason":"tool_calls"}]}`)
	}))
	defer upstream.Close()

	handler := NewOllamaHandler(NewLlamaClient(upstream.URL))

	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"model":"demo","messages":[{"role":"user","content":"hi"}],"stream":false}`))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}

	var response map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	message, _ := response["message"].(map[string]any)
	toolCalls, _ := message["tool_calls"].([]any)
	if len(toolCalls) != 1 {
		t.Fatalf("expected one tool call, got %#v", message["tool_calls"])
	}

	toolCall, _ := toolCalls[0].(map[string]any)
	function, _ := toolCall["function"].(map[string]any)
	arguments, _ := function["arguments"].(map[string]any)
	if city, _ := arguments["city"].(string); city != "New York" {
		t.Fatalf("expected tool call city New York, got %#v", arguments["city"])
	}

	if doneReason, _ := response["done_reason"].(string); doneReason != "tool_calls" {
		t.Fatalf("expected done_reason tool_calls, got %#v", response["done_reason"])
	}
}
