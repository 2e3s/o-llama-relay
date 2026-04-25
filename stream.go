package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

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

func writeStreamChunk(encoder *json.Encoder, flusher http.Flusher, data any) error {
	if err := encoder.Encode(data); err != nil {
		return err
	}
	flusher.Flush()
	return nil
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

func (h *OllamaHandler) streamLlamaResponse(
	w http.ResponseWriter,
	path string,
	data any,
	method string,
	model string,
	buildChunk func(string, string, map[string]any) (map[string]any, string, bool),
	buildFinal func(string, string, string) map[string]any,
) error {
	resp, err := h.Client.doRequest(path, data, method, true)
	if err != nil {
		sendError(w, err.Error(), 500)
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		err = h.Client.readError(resp)
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
