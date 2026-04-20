package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"regexp"
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

func logVerbose(format string, args ...interface{}) {
	if verboseLevel >= 1 {
		log.Printf(format, args...)
	}
}

func logVeryVerbose(format string, args ...interface{}) {
	if verboseLevel >= 2 {
		log.Printf(format, args...)
	}
}

type ErrorResponse struct {
	Error string `json:"error"`
}

func sendJSON(w http.ResponseWriter, data interface{}, status int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func sendError(w http.ResponseWriter, message string, status int) {
	sendJSON(w, ErrorResponse{Error: message}, status)
}

func fetchFromLlama(path string, data interface{}, method string) (map[string]interface{}, error) {
	url := llamaCPPURL + path
	logVerbose("[LLAMA] %s %s", method, url)

	var body []byte
	if data != nil {
		body, _ = json.Marshal(data)
		logVeryVerbose("[LLAMA] Payload: %s", string(body))
	}

	req, err := http.NewRequest(method, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	logVeryVerbose("[LLAMA] Opening URL, waiting for response...")
	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		logVerbose("[LLAMA] URLError: %v", err)
		return nil, fmt.Errorf("failed to reach Llama.cpp server: %v", err)
	}
	defer resp.Body.Close()

	logVeryVerbose("[LLAMA] Got response, reading...")
	result, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	logVeryVerbose("[LLAMA] Response received: %d bytes", len(result))

	var dataMap map[string]interface{}
	if err := json.Unmarshal(result, &dataMap); err != nil {
		return nil, err
	}
	return dataMap, nil
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

func llamaToOllamaTags(llamaData map[string]interface{}) map[string]interface{} {
	models := []map[string]interface{}{}

	for _, m := range llamaData["data"].([]interface{}) {
		modelMap := m.(map[string]interface{})
		modelID, _ := modelMap["id"].(string)
		status, _ := modelMap["status"].(map[string]interface{})

		var args []string
		if status != nil {
			if a, ok := status["args"].([]interface{}); ok {
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

		modelEntry := map[string]interface{}{
			"name":        modelID,
			"model":       modelID,
			"modified_at": modifiedAt,
			"size":        size,
			"digest":      "",
			"details":     details,
		}
		models = append(models, modelEntry)
	}

	return map[string]interface{}{"models": models}
}

func llamaToOllamaPS(llamaData map[string]interface{}) map[string]interface{} {
	models := []map[string]interface{}{}

	for _, m := range llamaData["data"].([]interface{}) {
		modelMap := m.(map[string]interface{})
		status, _ := modelMap["status"].(map[string]interface{})

		if status == nil || status["value"] != "loaded" {
			continue
		}

		modelID, _ := modelMap["id"].(string)

		var args []string
		if a, ok := status["args"].([]interface{}); ok {
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

		modelEntry := map[string]interface{}{
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

	return map[string]interface{}{"models": models}
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
	body, err := io.ReadAll(r.Body)
	if err != nil {
		sendError(w, "Failed to read request body", 400)
		return
	}
	logVeryVerbose("[OLLAMA] POST body: %s", truncateString(string(body), 500))

	var reqBody map[string]interface{}
	if err := json.Unmarshal(body, &reqBody); err != nil {
		reqBody = make(map[string]interface{})
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
	if r.URL.Path == "/api/delete" {
		h.handleDeleteModel(w, r)
	} else {
		sendError(w, fmt.Sprintf("Unknown endpoint: %s", r.URL.Path), 404)
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

func (h *OllamaHandler) handleGenerate(w http.ResponseWriter, reqBody map[string]interface{}) {
	model, _ := reqBody["model"].(string)
	prompt, _ := reqBody["prompt"].(string)
	keepAlive := reqBody["keep_alive"]

	llamaData := map[string]interface{}{
		"prompt": prompt,
		"model":  model,
		"stream": false,
	}

	if options, ok := reqBody["options"].(map[string]interface{}); ok {
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
		if stop, ok := options["stop"].([]interface{}); ok {
			stopSlice := make([]string, len(stop))
			for i, s := range stop {
				stopSlice[i] = fmt.Sprintf("%v", s)
			}
			llamaData["stop"] = stopSlice
		}
	}

	result, err := fetchFromLlama("/v1/completions", llamaData, "POST")
	if err != nil {
		sendError(w, err.Error(), 500)
		return
	}

	response := ""
	if choices, ok := result["choices"].([]interface{}); ok && len(choices) > 0 {
		if choice, ok := choices[0].(map[string]interface{}); ok {
			response, _ = choice["text"].(string)
		}
	}

	sendJSON(w, map[string]interface{}{
		"model":       model,
		"created_at":  time.Now().UTC().Format(time.RFC3339),
		"response":    response,
		"done":        true,
		"done_reason": "stop",
	}, 200)

	// Handle keep_alive: 0 means unload model
	if shouldUnload(keepAlive) {
		logVerbose("[DEBUG] Unloading model: %s", model)
		fetchFromLlama("/models/unload", map[string]string{"model": model}, "POST")
	}
}

func (h *OllamaHandler) handleChat(w http.ResponseWriter, reqBody map[string]interface{}) {
	model, _ := reqBody["model"].(string)
	messages, _ := reqBody["messages"].([]interface{})
	keepAlive := reqBody["keep_alive"]

	llamaData := map[string]interface{}{
		"model":    model,
		"messages": messages,
		"stream":   false,
	}

	if options, ok := reqBody["options"].(map[string]interface{}); ok {
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

	result, err := fetchFromLlama("/v1/chat/completions", llamaData, "POST")
	if err != nil {
		sendError(w, err.Error(), 500)
		return
	}

	content := ""
	if choices, ok := result["choices"].([]interface{}); ok && len(choices) > 0 {
		if choice, ok := choices[0].(map[string]interface{}); ok {
			if msg, ok := choice["message"].(map[string]interface{}); ok {
				content, _ = msg["content"].(string)
			}
		}
	}

	sendJSON(w, map[string]interface{}{
		"model":       model,
		"created_at":  time.Now().UTC().Format(time.RFC3339),
		"message":     map[string]string{"role": "assistant", "content": content},
		"done":        true,
		"done_reason": "stop",
	}, 200)

	// Handle keep_alive: 0 means unload model
	if shouldUnload(keepAlive) {
		logVerbose("[DEBUG] Unloading model: %s", model)
		fetchFromLlama("/models/unload", map[string]string{"model": model}, "POST")
	}
}

func (h *OllamaHandler) handleEmbed(w http.ResponseWriter, reqBody map[string]interface{}) {
	model, _ := reqBody["model"].(string)
	input, _ := reqBody["input"]

	inputList := []string{}
	switch v := input.(type) {
	case string:
		inputList = append(inputList, v)
	case []interface{}:
		for _, item := range v {
			if s, ok := item.(string); ok {
				inputList = append(inputList, s)
			}
		}
	}

	llamaData := map[string]interface{}{
		"model": model,
		"input": inputList,
	}

	result, err := fetchFromLlama("/v1/embeddings", llamaData, "POST")
	if err != nil {
		sendError(w, err.Error(), 500)
		return
	}

	embeddings := [][]float64{}
	if data, ok := result["data"].([]interface{}); ok {
		for _, item := range data {
			if m, ok := item.(map[string]interface{}); ok {
				if emb, ok := m["embedding"].([]interface{}); ok {
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

	sendJSON(w, map[string]interface{}{
		"model":      model,
		"embeddings": embeddings,
	}, 200)
}

func (h *OllamaHandler) handleShow(w http.ResponseWriter, reqBody map[string]interface{}) {
	model, _ := reqBody["model"].(string)

	llamaData := map[string]string{"model": model}
	_, err := fetchFromLlama("/api/show", llamaData, "POST")
	if err != nil {
		sendError(w, err.Error(), 500)
		return
	}

	sendJSON(w, map[string]interface{}{
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

func shouldUnload(keepAlive interface{}) bool {
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
