package main

import (
	"net/http"
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

	llamaData := map[string]string{"model": model}
	_, err := h.Client.fetch("/api/show", llamaData, "POST")
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
