package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	Version     = "0.1.0"
	DefaultPort = 11434
)

var (
	llamaCPPURL  = getEnv("LLAMA_CPP_URL", "http://127.0.0.1:8080")
	verboseLevel int // 0=silent, 1=verbose, 2=very verbose
)

func getEnv(key, defaultVal string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return defaultVal
}

func logVerbose(format string, args ...any) {
	if verboseLevel >= 1 {
		log.Printf(format, args...)
	}
}

func logVeryVerbose(format string, args ...any) {
	if verboseLevel >= 2 {
		log.Printf(format, args...)
	}
}

type ErrorResponse struct {
	Error string `json:"error"`
}

func sendJSON(w http.ResponseWriter, data any, status int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func sendError(w http.ResponseWriter, message string, status int) {
	sendJSON(w, ErrorResponse{Error: message}, status)
}

func doLlamaRequest(path string, data any, method string, stream bool) (*http.Response, error) {
	url := llamaCPPURL + path
	logVerbose("[LLAMA] %s %s", method, url)

	var body []byte
	var err error
	if data != nil {
		body, err = json.Marshal(data)
		if err != nil {
			return nil, err
		}
		logVeryVerbose("[LLAMA] Payload: %s", string(body))
	}

	req, err := http.NewRequest(method, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	logVeryVerbose("[LLAMA] Opening URL, waiting for response...")
	client := &http.Client{}
	if !stream {
		client.Timeout = 60 * time.Second
	}
	resp, err := client.Do(req)
	if err != nil {
		logVerbose("[LLAMA] URLError: %v", err)
		return nil, fmt.Errorf("failed to reach Llama.cpp server: %v", err)
	}
	logVeryVerbose("[LLAMA] Got response with status: %s", resp.Status)

	return resp, nil
}

func readLlamaError(resp *http.Response) error {
	result, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("llama.cpp returned status %d", resp.StatusCode)
	}
	logVeryVerbose("[LLAMA] Error response received: %d bytes", len(result))

	var dataMap map[string]any
	if err := json.Unmarshal(result, &dataMap); err == nil {
		if errorString, ok := dataMap["error"].(string); ok && errorString != "" {
			return fmt.Errorf("llama.cpp returned status %d: %s", resp.StatusCode, errorString)
		}
		if errorMap, ok := dataMap["error"].(map[string]any); ok {
			if message, ok := errorMap["message"].(string); ok && message != "" {
				return fmt.Errorf("llama.cpp returned status %d: %s", resp.StatusCode, message)
			}
		}
		if message, ok := dataMap["message"].(string); ok && message != "" {
			return fmt.Errorf("llama.cpp returned status %d: %s", resp.StatusCode, message)
		}
	}

	if message := strings.TrimSpace(string(result)); message != "" {
		return fmt.Errorf("llama.cpp returned status %d: %s", resp.StatusCode, message)
	}

	return fmt.Errorf("llama.cpp returned status %d", resp.StatusCode)
}

func readLlamaJSON(resp *http.Response) (map[string]any, error) {
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return nil, readLlamaError(resp)
	}

	logVeryVerbose("[LLAMA] Reading response body...")
	result, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	logVeryVerbose("[LLAMA] Response received: %d bytes", len(result))

	var dataMap map[string]any
	if err := json.Unmarshal(result, &dataMap); err != nil {
		return nil, err
	}
	return dataMap, nil
}

func fetchFromLlama(path string, data any, method string) (map[string]any, error) {
	resp, err := doLlamaRequest(path, data, method, false)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	return readLlamaJSON(resp)
}

func writeStreamChunk(encoder *json.Encoder, flusher http.Flusher, data any) error {
	if err := encoder.Encode(data); err != nil {
		return err
	}
	flusher.Flush()
	return nil
}

func consumeSSE(body io.Reader, handle func(string) error) error {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	dataLines := []string{}
	processEvent := func() error {
		if len(dataLines) == 0 {
			return nil
		}
		payload := strings.Join(dataLines, "\n")
		dataLines = dataLines[:0]
		return handle(payload)
	}

	for scanner.Scan() {
		line := strings.TrimRight(scanner.Text(), "\r")
		switch {
		case line == "":
			if err := processEvent(); err != nil {
				return err
			}
		case strings.HasPrefix(line, ":"):
			continue
		case strings.HasPrefix(line, "data:"):
			dataLines = append(dataLines, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}
	}

	if err := scanner.Err(); err != nil {
		return err
	}

	return processEvent()
}

func firstChoice(chunk map[string]any) map[string]any {
	choices, ok := chunk["choices"].([]any)
	if !ok || len(choices) == 0 {
		return nil
	}

	choice, ok := choices[0].(map[string]any)
	if !ok {
		return nil
	}

	return choice
}

func choiceFinishReason(choice map[string]any) string {
	if choice == nil {
		return ""
	}

	finishReason, _ := choice["finish_reason"].(string)
	return finishReason
}

func buildGenerateStreamChunk(model, createdAt string, chunk map[string]any) (map[string]any, string, bool) {
	choice := firstChoice(chunk)
	if choice == nil {
		return nil, "", false
	}

	text, _ := choice["text"].(string)
	finishReason := choiceFinishReason(choice)
	if text == "" {
		return nil, finishReason, false
	}

	return map[string]any{
		"model":      model,
		"created_at": createdAt,
		"response":   text,
		"done":       false,
	}, finishReason, true
}

func buildGenerateFinalChunk(model, createdAt, doneReason string) map[string]any {
	if doneReason == "" {
		doneReason = "stop"
	}

	return map[string]any{
		"model":       model,
		"created_at":  createdAt,
		"response":    "",
		"done":        true,
		"done_reason": doneReason,
	}
}

type chatToolCallAccumulator struct {
	ID                    string
	Index                 int
	Name                  string
	argumentsBuffer       strings.Builder
	arguments             map[string]any
	hasStructuredArgs     bool
	emittedToClientStream bool
}

type chatToolCallState struct {
	calls map[int]*chatToolCallAccumulator
}

func newChatToolCallState() *chatToolCallState {
	return &chatToolCallState{
		calls: make(map[int]*chatToolCallAccumulator),
	}
}

func (s *chatToolCallState) apply(rawToolCalls []any) {
	for i, rawToolCall := range rawToolCalls {
		toolCall, ok := rawToolCall.(map[string]any)
		if !ok {
			continue
		}

		index := i
		switch v := toolCall["index"].(type) {
		case float64:
			index = int(v)
		case int:
			index = v
		}

		accumulator := s.calls[index]
		if accumulator == nil {
			accumulator = &chatToolCallAccumulator{Index: index}
			s.calls[index] = accumulator
		}

		if id, ok := toolCall["id"].(string); ok && id != "" {
			accumulator.ID = id
		}

		function, ok := toolCall["function"].(map[string]any)
		if !ok {
			continue
		}

		if name, ok := function["name"].(string); ok && name != "" {
			if accumulator.Name == "" || strings.HasPrefix(name, accumulator.Name) {
				accumulator.Name = name
			} else {
				accumulator.Name += name
			}
		}

		switch arguments := function["arguments"].(type) {
		case string:
			accumulator.argumentsBuffer.WriteString(arguments)
		case map[string]any:
			accumulator.arguments = arguments
			accumulator.hasStructuredArgs = true
		case nil:
		default:
			rawArguments, err := json.Marshal(arguments)
			if err != nil {
				continue
			}

			var parsedArguments map[string]any
			if err := json.Unmarshal(rawArguments, &parsedArguments); err != nil {
				continue
			}

			accumulator.arguments = parsedArguments
			accumulator.hasStructuredArgs = true
		}
	}
}

func (s *chatToolCallState) readyForClient(allowEmptyArgs bool) []map[string]any {
	indexes := make([]int, 0, len(s.calls))
	for index := range s.calls {
		indexes = append(indexes, index)
	}
	sort.Ints(indexes)

	toolCalls := []map[string]any{}
	for _, index := range indexes {
		accumulator := s.calls[index]
		if accumulator.emittedToClientStream {
			continue
		}

		toolCall, ok := accumulator.toOllamaToolCall(allowEmptyArgs)
		if !ok {
			continue
		}

		accumulator.emittedToClientStream = true
		toolCalls = append(toolCalls, toolCall)
	}

	return toolCalls
}

func (a *chatToolCallAccumulator) toOllamaToolCall(allowEmptyArgs bool) (map[string]any, bool) {
	if a.Name == "" {
		return nil, false
	}

	arguments, ok := a.argumentsMap(allowEmptyArgs)
	if !ok {
		return nil, false
	}

	toolCall := map[string]any{
		"function": map[string]any{
			"index":     a.Index,
			"name":      a.Name,
			"arguments": arguments,
		},
	}
	if a.ID != "" {
		toolCall["id"] = a.ID
	}

	return toolCall, true
}

func (a *chatToolCallAccumulator) argumentsMap(allowEmptyArgs bool) (map[string]any, bool) {
	if a.hasStructuredArgs {
		return a.arguments, true
	}

	rawArguments := strings.TrimSpace(a.argumentsBuffer.String())
	if rawArguments == "" {
		if allowEmptyArgs {
			return map[string]any{}, true
		}
		return nil, false
	}

	var arguments map[string]any
	if err := json.Unmarshal([]byte(rawArguments), &arguments); err != nil {
		return nil, false
	}

	return arguments, true
}

func chatMessageFromChoice(choice map[string]any, toolCallState *chatToolCallState, allowEmptyToolArgs bool) (map[string]any, bool) {
	var source map[string]any
	if delta, ok := choice["delta"].(map[string]any); ok {
		source = delta
	} else if message, ok := choice["message"].(map[string]any); ok {
		source = message
	}
	if source == nil {
		return nil, false
	}

	role, _ := source["role"].(string)
	content, _ := source["content"].(string)
	thinking := ""
	if value, ok := source["thinking"].(string); ok && value != "" {
		thinking = value
	} else if value, ok := source["reasoning"].(string); ok && value != "" {
		thinking = value
	}

	if rawToolCalls, ok := source["tool_calls"].([]any); ok {
		toolCallState.apply(rawToolCalls)
	}
	toolCalls := toolCallState.readyForClient(allowEmptyToolArgs)

	hasPayload := content != "" || thinking != "" || len(toolCalls) > 0
	if !hasPayload {
		return nil, false
	}
	if role == "" {
		role = "assistant"
	}

	message := map[string]any{
		"role": role,
	}
	if content != "" {
		message["content"] = content
	}
	if thinking != "" {
		message["thinking"] = thinking
	}
	if len(toolCalls) > 0 {
		message["tool_calls"] = toolCalls
	}

	return message, true
}

func buildChatStreamChunk(model, createdAt string, chunk map[string]any, toolCallState *chatToolCallState) (map[string]any, string, bool) {
	choice := firstChoice(chunk)
	if choice == nil {
		return nil, "", false
	}

	finishReason := choiceFinishReason(choice)
	message, emit := chatMessageFromChoice(choice, toolCallState, finishReason != "")
	if !emit {
		return nil, finishReason, false
	}

	return map[string]any{
		"model":      model,
		"created_at": createdAt,
		"message":    message,
		"done":       false,
	}, finishReason, true
}

func buildChatFinalChunk(model, createdAt, doneReason string) map[string]any {
	if doneReason == "" {
		doneReason = "stop"
	}

	return map[string]any{
		"model":      model,
		"created_at": createdAt,
		"message": map[string]any{
			"role":    "assistant",
			"content": "",
		},
		"done":        true,
		"done_reason": doneReason,
	}
}

func streamLlamaResponse(
	w http.ResponseWriter,
	path string,
	data any,
	method string,
	model string,
	buildChunk func(string, string, map[string]any) (map[string]any, string, bool),
	buildFinal func(string, string, string) map[string]any,
) error {
	resp, err := doLlamaRequest(path, data, method, true)
	if err != nil {
		sendError(w, err.Error(), 500)
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		err = readLlamaError(resp)
		sendError(w, err.Error(), 500)
		return err
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		err = fmt.Errorf("response writer does not support streaming")
		sendError(w, err.Error(), 500)
		return err
	}

	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Cache-Control", "no-cache")

	createdAt := time.Now().UTC().Format(time.RFC3339)
	encoder := json.NewEncoder(w)
	doneReason := "stop"
	finalSent := false

	err = consumeSSE(resp.Body, func(payload string) error {
		if payload == "[DONE]" {
			if finalSent {
				return nil
			}
			finalSent = true
			return writeStreamChunk(encoder, flusher, buildFinal(model, createdAt, doneReason))
		}

		var chunk map[string]any
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			return err
		}

		responseChunk, finishReason, emit := buildChunk(model, createdAt, chunk)
		if finishReason != "" {
			doneReason = finishReason
		}
		if emit {
			if err := writeStreamChunk(encoder, flusher, responseChunk); err != nil {
				return err
			}
		}
		if finishReason != "" && !finalSent {
			finalSent = true
			return writeStreamChunk(encoder, flusher, buildFinal(model, createdAt, doneReason))
		}

		return nil
	})
	if err != nil {
		return err
	}

	if !finalSent {
		return writeStreamChunk(encoder, flusher, buildFinal(model, createdAt, doneReason))
	}

	return nil
}

func extractModelPath(args []string) string {
	for i, arg := range args {
		if arg == "--model" && i+1 < len(args) {
			return args[i+1]
		}
	}
	return ""
}

func estimateModelSize(filename string) int64 {
	re := regexp.MustCompile(`(\d+\.?\d*)[Bb]`)
	matches := re.FindStringSubmatch(filename)
	if len(matches) < 2 {
		return 0
	}
	params, _ := strconv.ParseFloat(matches[1], 64)
	return int64(params * 1e9 * 0.4)
}

func parseModelDetails(filename string) map[string]string {
	details := map[string]string{
		"format":             "gguf",
		"family":             "unknown",
		"families":           "unknown",
		"parameter_size":     "unknown",
		"quantization_level": "unknown",
	}

	lower := strings.ToLower(filename)

	// Detect family
	switch {
	case strings.Contains(lower, "llama"):
		details["family"] = "llama"
		details["families"] = "[\"llama\"]"
	case strings.Contains(lower, "mistral"):
		details["family"] = "mistral"
		details["families"] = "[\"mistral\"]"
	case strings.Contains(lower, "gemma"):
		details["family"] = "gemma"
		details["families"] = "[\"gemma\"]"
	case strings.Contains(lower, "qwen"):
		details["family"] = "qwen"
		details["families"] = "[\"qwen\"]"
	case strings.Contains(lower, "phi"):
		details["family"] = "phi"
		details["families"] = "[\"phi\"]"
	case strings.Contains(lower, "nemotron"):
		details["family"] = "nemotron"
		details["families"] = "[\"nemotron\"]"
	case strings.Contains(lower, "bert"):
		details["family"] = "bert"
		details["families"] = "[\"bert\"]"
	}

	// Detect quantization
	switch {
	case strings.Contains(lower, "q4_0"):
		details["quantization_level"] = "Q4_0"
	case strings.Contains(lower, "q4_k"):
		if idx := strings.Index(lower, "q4_k"); idx != -1 {
			rest := lower[idx+5:]
			if len(rest) > 0 && rest[0] >= 'a' && rest[0] <= 'z' {
				details["quantization_level"] = strings.ToUpper(string(rest[0]))
			}
		}
	case strings.Contains(lower, "q8_0"):
		details["quantization_level"] = "Q8_0"
	case strings.Contains(lower, "fp16"), strings.Contains(lower, "f16"):
		details["quantization_level"] = "F16"
	case strings.Contains(lower, "mxfp4"):
		details["quantization_level"] = "MXFP4"
	case strings.Contains(lower, "iq3"):
		details["quantization_level"] = "IQ3"
	case strings.Contains(lower, "iq4"):
		details["quantization_level"] = "IQ4"
	}

	// Detect parameter size
	paramRe := regexp.MustCompile(`(\d+\.?\d*)[Bb]`)
	if paramMatches := paramRe.FindStringSubmatch(lower); len(paramMatches) >= 2 {
		if params, err := strconv.ParseFloat(paramMatches[1], 64); err == nil {
			if params < 1 {
				details["parameter_size"] = fmt.Sprintf("%.0fM", params*1000)
			} else {
				details["parameter_size"] = fmt.Sprintf("%.0fB", params)
			}
		}
	}

	return details
}

func llamaToOllamaTags(llamaData map[string]any) map[string]any {
	models := []map[string]any{}

	for _, m := range llamaData["data"].([]any) {
		modelMap := m.(map[string]any)
		modelID, _ := modelMap["id"].(string)
		status, _ := modelMap["status"].(map[string]any)

		var args []string
		if status != nil {
			if a, ok := status["args"].([]any); ok {
				for _, v := range a {
					args = append(args, fmt.Sprintf("%v", v))
				}
			}
		}

		modelPath := extractModelPath(args)
		size := int64(0)

		if modelPath != "" {
			filename := strings.TrimPrefix(modelPath, "/")
			idx := strings.LastIndex(filename, "/")
			if idx != -1 {
				filename = filename[idx+1:]
			}
			size = estimateModelSize(filename)
		}

		details := map[string]string{
			"format":             "gguf",
			"family":             "unknown",
			"families":           "[\"unknown\"]",
			"parameter_size":     "unknown",
			"quantization_level": "unknown",
		}

		if modelPath != "" {
			filename := strings.TrimPrefix(modelPath, "/")
			idx := strings.LastIndex(filename, "/")
			if idx != -1 {
				filename = filename[idx+1:]
			}
			details = parseModelDetails(filename)
		}

		created := modelMap["created"].(float64)
		modifiedAt := time.Unix(int64(created), 0).UTC().Format(time.RFC3339)

		modelEntry := map[string]any{
			"name":        modelID,
			"model":       modelID,
			"modified_at": modifiedAt,
			"size":        size,
			"digest":      "",
			"details":     details,
		}
		models = append(models, modelEntry)
	}

	return map[string]any{"models": models}
}

func llamaToOllamaPS(llamaData map[string]any) map[string]any {
	models := []map[string]any{}

	for _, m := range llamaData["data"].([]any) {
		modelMap := m.(map[string]any)
		status, _ := modelMap["status"].(map[string]any)

		if status == nil || status["value"] != "loaded" {
			continue
		}

		modelID, _ := modelMap["id"].(string)

		var args []string
		if a, ok := status["args"].([]any); ok {
			for _, v := range a {
				args = append(args, fmt.Sprintf("%v", v))
			}
		}

		sleepSeconds := 3600
		for i, arg := range args {
			if arg == "--sleep-idle-seconds" && i+1 < len(args) {
				if s, err := strconv.Atoi(args[i+1]); err == nil {
					sleepSeconds = s
				}
				break
			}
		}

		expiresAt := time.Now().UTC().Add(time.Duration(sleepSeconds) * time.Second)
		expiresAt = expiresAt.Truncate(time.Minute)

		modelEntry := map[string]any{
			"name":   modelID,
			"model":  modelID,
			"size":   0,
			"digest": "",
			"details": map[string]string{
				"format":             "gguf",
				"family":             "unknown",
				"families":           "[\"unknown\"]",
				"parameter_size":     "unknown",
				"quantization_level": "unknown",
			},
			"expires_at":     expiresAt.Format(time.RFC3339),
			"size_vram":      0,
			"context_length": 2048,
		}
		models = append(models, modelEntry)
	}

	return map[string]any{"models": models}
}

type OllamaHandler struct{}

func (h *OllamaHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	logVerbose("[OLLAMA] %s %s", r.Method, r.URL.Path)

	switch r.Method {
	case "GET":
		h.handleGet(w, r)
	case "POST":
		h.handlePost(w, r)
	case "DELETE":
		h.handleDelete(w, r)
	default:
		sendError(w, fmt.Sprintf("Method not allowed: %s", r.Method), 405)
	}
}

func (h *OllamaHandler) handleGet(w http.ResponseWriter, r *http.Request) {
	if strings.HasPrefix(r.URL.Path, "/v1/") {
		h.proxyRequest(w, r)
		return
	}

	switch r.URL.Path {
	case "/", "":
		w.Header().Set("Content-Type", "text/plain")
		w.Write([]byte("Ollama is running"))
	case "/api/tags":
		h.handleTags(w, r)
	case "/api/ps":
		h.handlePS(w, r)
	case "/api/version":
		sendJSON(w, map[string]string{"version": Version}, 200)
	default:
		sendError(w, fmt.Sprintf("Unknown endpoint: %s", r.URL.Path), 404)
	}
}

func (h *OllamaHandler) handlePost(w http.ResponseWriter, r *http.Request) {
	if strings.HasPrefix(r.URL.Path, "/v1/") {
		h.proxyRequest(w, r)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		sendError(w, "Failed to read request body", 400)
		return
	}
	logVeryVerbose("[OLLAMA] POST body: %s", truncateString(string(body), 500))

	var reqBody map[string]any
	if err := json.Unmarshal(body, &reqBody); err != nil {
		reqBody = make(map[string]any)
	}

	switch r.URL.Path {
	case "/api/generate":
		h.handleGenerate(w, reqBody)
	case "/api/chat":
		h.handleChat(w, reqBody)
	case "/api/embed":
		h.handleEmbed(w, reqBody)
	case "/api/show":
		h.handleShow(w, reqBody)
	case "/api/create":
		sendError(w, "Model creation not supported - Llama.cpp uses pre-loaded models", 501)
	case "/api/copy":
		sendError(w, "Model copying not supported - Llama.cpp uses pre-loaded models", 501)
	case "/api/pull":
		sendError(w, "Model pulling not supported - Llama.cpp uses pre-loaded models", 501)
	case "/api/push":
		sendError(w, "Model pushing not supported - Llama.cpp uses pre-loaded models", 501)
	default:
		sendError(w, fmt.Sprintf("Unknown endpoint: %s", r.URL.Path), 404)
	}
}

func (h *OllamaHandler) handleDelete(w http.ResponseWriter, r *http.Request) {
	if strings.HasPrefix(r.URL.Path, "/v1/") {
		h.proxyRequest(w, r)
		return
	}
	if r.URL.Path == "/api/delete" {
		h.handleDeleteModel(w, r)
	} else {
		sendError(w, fmt.Sprintf("Unknown endpoint: %s", r.URL.Path), 404)
	}
}

func (h *OllamaHandler) proxyRequest(w http.ResponseWriter, r *http.Request) {
	url := llamaCPPURL + r.URL.Path
	logVerbose("[PROXY] %s %s", r.Method, url)

	body, err := io.ReadAll(r.Body)
	if err != nil {
		sendError(w, "Failed to read request body", 400)
		return
	}
	if len(body) > 0 {
		logVeryVerbose("[PROXY] Body: %s", truncateString(string(body), 500))
	}

	req, err := http.NewRequest(r.Method, url, bytes.NewReader(body))
	if err != nil {
		sendError(w, "Failed to create proxy request", 500)
		return
	}
	req.Header = r.Header.Clone()

	stream := requestBodyWantsStream(body) || strings.Contains(req.Header.Get("Accept"), "text/event-stream")
	client := &http.Client{}
	if !stream {
		client.Timeout = 60 * time.Second
	}
	resp, err := client.Do(req)
	if err != nil {
		logVeryVerbose("[PROXY] Error: %v", err)
		sendError(w, fmt.Sprintf("Proxy request failed: %v", err), 502)
		return
	}
	defer resp.Body.Close()
	logVeryVerbose("[PROXY] Response status: %s", resp.Status)

	for k, v := range resp.Header {
		w.Header()[k] = v
	}
	w.WriteHeader(resp.StatusCode)

	if flusher, ok := w.(http.Flusher); ok {
		_, err = io.Copy(flushWriter{ResponseWriter: w, Flusher: flusher}, resp.Body)
	} else {
		_, err = io.Copy(w, resp.Body)
	}
	if err != nil {
		logVeryVerbose("[PROXY] Stream copy error: %v", err)
	}
}

func (h *OllamaHandler) handleTags(w http.ResponseWriter, r *http.Request) {
	data, err := fetchFromLlama("/models", nil, "GET")
	if err != nil {
		sendError(w, err.Error(), 500)
		return
	}
	result := llamaToOllamaTags(data)
	sendJSON(w, result, 200)
}

func (h *OllamaHandler) handlePS(w http.ResponseWriter, r *http.Request) {
	data, err := fetchFromLlama("/models", nil, "GET")
	if err != nil {
		sendError(w, err.Error(), 500)
		return
	}
	result := llamaToOllamaPS(data)
	sendJSON(w, result, 200)
}

func wantsStream(reqBody map[string]any) bool {
	stream, ok := reqBody["stream"].(bool)
	return ok && stream
}

func requestBodyWantsStream(body []byte) bool {
	if len(body) == 0 {
		return false
	}

	var reqBody map[string]any
	if err := json.Unmarshal(body, &reqBody); err != nil {
		return false
	}

	return wantsStream(reqBody)
}

func applyCommonOptions(llamaData map[string]any, options map[string]any) {
	if temp, ok := options["temperature"].(float64); ok {
		llamaData["temperature"] = temp
	}
	if topP, ok := options["top_p"].(float64); ok {
		llamaData["top_p"] = topP
	}
	if topK, ok := options["top_k"].(float64); ok {
		llamaData["top_k"] = topK
	}
	if numPredict, ok := options["num_predict"].(float64); ok {
		llamaData["max_tokens"] = int(numPredict)
	} else if maxTokens, ok := options["max_tokens"].(float64); ok {
		llamaData["max_tokens"] = int(maxTokens)
	}
}

func unloadModel(model string) {
	logVerbose("[DEBUG] Unloading model: %s", model)
	if _, err := fetchFromLlama("/models/unload", map[string]string{"model": model}, "POST"); err != nil {
		logVerbose("[DEBUG] Failed to unload model %s: %v", model, err)
	}
}

func (h *OllamaHandler) handleGenerate(w http.ResponseWriter, reqBody map[string]any) {
	model, _ := reqBody["model"].(string)
	prompt, _ := reqBody["prompt"].(string)
	keepAlive := reqBody["keep_alive"]
	stream := wantsStream(reqBody)

	llamaData := map[string]any{
		"prompt": prompt,
		"model":  model,
		"stream": stream,
	}

	if options, ok := reqBody["options"].(map[string]any); ok {
		applyCommonOptions(llamaData, options)
		if stop, ok := options["stop"].([]any); ok {
			stopSlice := make([]string, len(stop))
			for i, s := range stop {
				stopSlice[i] = fmt.Sprintf("%v", s)
			}
			llamaData["stop"] = stopSlice
		}
	}

	if stream {
		if err := streamLlamaResponse(
			w,
			"/v1/completions",
			llamaData,
			"POST",
			model,
			buildGenerateStreamChunk,
			buildGenerateFinalChunk,
		); err != nil {
			logVerbose("[OLLAMA] Generate stream failed: %v", err)
			return
		}
		if shouldUnload(keepAlive) {
			unloadModel(model)
		}
		return
	}

	result, err := fetchFromLlama("/v1/completions", llamaData, "POST")
	if err != nil {
		sendError(w, err.Error(), 500)
		return
	}

	response := ""
	doneReason := "stop"
	if choices, ok := result["choices"].([]any); ok && len(choices) > 0 {
		if choice, ok := choices[0].(map[string]any); ok {
			response, _ = choice["text"].(string)
			if finishReason := choiceFinishReason(choice); finishReason != "" {
				doneReason = finishReason
			}
		}
	}

	sendJSON(w, map[string]any{
		"model":       model,
		"created_at":  time.Now().UTC().Format(time.RFC3339),
		"response":    response,
		"done":        true,
		"done_reason": doneReason,
	}, 200)

	// Handle keep_alive: 0 means unload model
	if shouldUnload(keepAlive) {
		unloadModel(model)
	}
}

func (h *OllamaHandler) handleChat(w http.ResponseWriter, reqBody map[string]any) {
	model, _ := reqBody["model"].(string)
	messages, _ := reqBody["messages"].([]any)
	keepAlive := reqBody["keep_alive"]
	stream := wantsStream(reqBody)

	llamaData := map[string]any{
		"model":    model,
		"messages": messages,
		"stream":   stream,
	}

	if options, ok := reqBody["options"].(map[string]any); ok {
		applyCommonOptions(llamaData, options)
	}

	if stream {
		toolCallState := newChatToolCallState()
		if err := streamLlamaResponse(
			w,
			"/v1/chat/completions",
			llamaData,
			"POST",
			model,
			func(model, createdAt string, chunk map[string]any) (map[string]any, string, bool) {
				return buildChatStreamChunk(model, createdAt, chunk, toolCallState)
			},
			buildChatFinalChunk,
		); err != nil {
			logVerbose("[OLLAMA] Chat stream failed: %v", err)
			return
		}
		if shouldUnload(keepAlive) {
			unloadModel(model)
		}
		return
	}

	result, err := fetchFromLlama("/v1/chat/completions", llamaData, "POST")
	if err != nil {
		sendError(w, err.Error(), 500)
		return
	}

	message := map[string]any{
		"role":    "assistant",
		"content": "",
	}
	doneReason := "stop"
	if choices, ok := result["choices"].([]any); ok && len(choices) > 0 {
		if choice, ok := choices[0].(map[string]any); ok {
			toolCallState := newChatToolCallState()
			if translatedMessage, ok := chatMessageFromChoice(choice, toolCallState, true); ok {
				message = translatedMessage
			}
			if finishReason := choiceFinishReason(choice); finishReason != "" {
				doneReason = finishReason
			} else if _, ok := message["tool_calls"]; ok {
				doneReason = "tool_calls"
			}
		}
	}

	sendJSON(w, map[string]any{
		"model":       model,
		"created_at":  time.Now().UTC().Format(time.RFC3339),
		"message":     message,
		"done":        true,
		"done_reason": doneReason,
	}, 200)

	// Handle keep_alive: 0 means unload model
	if shouldUnload(keepAlive) {
		unloadModel(model)
	}
}

func (h *OllamaHandler) handleEmbed(w http.ResponseWriter, reqBody map[string]any) {
	model, _ := reqBody["model"].(string)
	input, _ := reqBody["input"]

	inputList := []string{}
	switch v := input.(type) {
	case string:
		inputList = append(inputList, v)
	case []any:
		for _, item := range v {
			if s, ok := item.(string); ok {
				inputList = append(inputList, s)
			}
		}
	}

	llamaData := map[string]any{
		"model": model,
		"input": inputList,
	}

	result, err := fetchFromLlama("/v1/embeddings", llamaData, "POST")
	if err != nil {
		sendError(w, err.Error(), 500)
		return
	}

	embeddings := [][]float64{}
	if data, ok := result["data"].([]any); ok {
		for _, item := range data {
			if m, ok := item.(map[string]any); ok {
				if emb, ok := m["embedding"].([]any); ok {
					embFloat := make([]float64, len(emb))
					for i, v := range emb {
						if f, ok := v.(float64); ok {
							embFloat[i] = f
						}
					}
					embeddings = append(embeddings, embFloat)
				}
			}
		}
	}

	sendJSON(w, map[string]any{
		"model":      model,
		"embeddings": embeddings,
	}, 200)
}

func (h *OllamaHandler) handleShow(w http.ResponseWriter, reqBody map[string]any) {
	model, _ := reqBody["model"].(string)

	llamaData := map[string]string{"model": model}
	_, err := fetchFromLlama("/api/show", llamaData, "POST")
	if err != nil {
		sendError(w, err.Error(), 500)
		return
	}

	sendJSON(w, map[string]any{
		"model":       model,
		"modified_at": time.Now().UTC().Format(time.RFC3339),
		"details": map[string]string{
			"format": "gguf",
			"family": "unknown",
		},
	}, 200)
}

func (h *OllamaHandler) handleDeleteModel(w http.ResponseWriter, r *http.Request) {
	sendError(w, "Model deletion not supported - Llama.cpp uses pre-loaded models", 501)
}

func shouldUnload(keepAlive any) bool {
	if keepAlive == nil {
		return false
	}
	switch v := keepAlive.(type) {
	case float64:
		return v == 0
	case string:
		return v == "0" || v == "0s"
	case int:
		return v == 0
	}
	return false
}

type flushWriter struct {
	http.ResponseWriter
	http.Flusher
}

func (w flushWriter) Write(p []byte) (int, error) {
	n, err := w.ResponseWriter.Write(p)
	if err == nil {
		w.Flusher.Flush()
	}
	return n, err
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

func runServer(host string, port int) {
	addr := fmt.Sprintf("%s:%d", host, port)
	log.Printf("Ollama API wrapper running on %s", addr)
	log.Printf("Proxying to Llama.cpp at %s", llamaCPPURL)
	log.Fatal(http.ListenAndServe(addr, &OllamaHandler{}))
}

func parseArgs() (port int, host string) {
	port = DefaultPort
	host = "0.0.0.0"

	for _, arg := range os.Args[1:] {
		switch arg {
		case "-v":
			verboseLevel = 1
		case "-vv":
			verboseLevel = 2
		default:
			if p, err := strconv.Atoi(arg); err == nil {
				port = p
			} else {
				host = arg
			}
		}
	}
	return
}

func main() {
	port, host := parseArgs()

	runServer(host, port)
}
