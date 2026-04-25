package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

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

func (h *OllamaHandler) proxyRequest(w http.ResponseWriter, r *http.Request) {
	url := h.Client.BaseURL + r.URL.Path
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
