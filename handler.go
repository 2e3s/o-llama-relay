package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

type ErrorResponse struct {
	Error string `json:"error"`
}

// OllamaHandler translates Ollama API requests into llama.cpp calls.
type OllamaHandler struct {
	Client *LlamaClient
}

func NewOllamaHandler(client *LlamaClient) *OllamaHandler {
	return &OllamaHandler{Client: client}
}

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

func sendJSON(w http.ResponseWriter, data any, status int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func sendError(w http.ResponseWriter, message string, status int) {
	sendJSON(w, ErrorResponse{Error: message}, status)
}

func wantsStream(reqBody map[string]any) bool {
	stream, ok := reqBody["stream"].(bool)
	return ok && stream
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
