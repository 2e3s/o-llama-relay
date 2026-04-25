package main

import (
	"fmt"
	"net/http"
	"time"
)

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
		if err := h.streamLlamaResponse(
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
			h.Client.unloadModel(model)
		}
		return
	}

	result, err := h.Client.fetch("/v1/completions", llamaData, "POST")
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

	if shouldUnload(keepAlive) {
		h.Client.unloadModel(model)
	}
}
