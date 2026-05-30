package main

import (
	"fmt"
	"net/http"
	"strings"
	"time"
)

func (h *OllamaHandler) handleTags(w http.ResponseWriter, r *http.Request) {
	data, err := h.Client.fetch("/models", nil, "GET")
	if err != nil {
		sendError(w, err.Error(), 500)
		return
	}
	result := llamaToOllamaTags(data)
	sendJSON(w, result, 200)
}

func (h *OllamaHandler) handlePS(w http.ResponseWriter, r *http.Request) {
	data, err := h.Client.fetch("/models", nil, "GET")
	if err != nil {
		sendError(w, err.Error(), 500)
		return
	}
	result := llamaToOllamaPS(data)
	sendJSON(w, result, 200)
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

	result, err := h.Client.fetch("/v1/embeddings", llamaData, "POST")
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

	models, err := h.Client.fetch("/v1/models", nil, "GET")
	if err != nil {
		sendError(w, err.Error(), 500)
		return
	}

	// Extract model metadata from /v1/models
	var modelID, paramSize string
	if dataArr, ok := models["data"].([]any); ok {
		for _, item := range dataArr {
			if modelInfo, ok := item.(map[string]any); ok {
				if id, ok := modelInfo["id"].(string); ok && id == model {
					modelID = id
					if meta, ok := modelInfo["meta"].(map[string]any); ok {
						if nParams, ok := meta["n_params"].(float64); ok {
							paramSize = fmt.Sprintf("%.0f", nParams)
						}
					}
					break
				}
			}
		}
	}

	family := inferModelFamily(modelID)

	sendJSON(w, map[string]any{
		"model":       model,
		"modified_at": time.Now().UTC().Format(time.RFC3339),
		"details": map[string]any{
			"parent_model":       "",
			"format":             "gguf",
			"family":             family,
			"families":           []string{family},
			"parameter_size":     paramSize,
			"quantization_level": "unknown",
		},
		"model_info": map[string]any{
			"general.architecture":    family,
			"general.name":            model,
			"general.file_type":       15,
			"general.parameter_count": paramSize,
		},
		"license":    "",
		"modelfile":  "FROM " + model + "\n",
		"parameters": "temperature 1",
		"template":   "{{ .Prompt }}",
	}, 200)
}

func inferModelFamily(modelID string) string {
	lower := strings.ToLower(modelID)
	if strings.Contains(lower, "llama") {
		return "llama"
	} else if strings.Contains(lower, "gemma") {
		return "gemma"
	} else if strings.Contains(lower, "mistral") {
		return "mistral"
	} else if strings.Contains(lower, "qwen") {
		return "qwen"
	} else if strings.Contains(lower, "phi") {
		return "phi"
	} else if strings.Contains(lower, "gpt-oss") || strings.Contains(lower, "gpt_oss") {
		return "gpt-oss"
	}
	return "unknown"
}

func (h *OllamaHandler) handleDeleteModel(w http.ResponseWriter, r *http.Request) {
	sendError(w, "Model deletion not supported - Llama.cpp uses pre-loaded models", 501)
}
