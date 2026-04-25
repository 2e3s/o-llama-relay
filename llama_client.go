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

// LlamaClient handles all HTTP communication with the llama.cpp server.
type LlamaClient struct {
	BaseURL    string
	HTTPClient *http.Client
}

func NewLlamaClient(baseURL string) *LlamaClient {
	return &LlamaClient{
		BaseURL:    baseURL,
		HTTPClient: &http.Client{},
	}
}

func (c *LlamaClient) doRequest(path string, data any, method string, stream bool) (*http.Response, error) {
	url := c.BaseURL + path
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
	client := c.HTTPClient
	if !stream {
		client = &http.Client{Timeout: 60 * time.Second}
	}
	resp, err := client.Do(req)
	if err != nil {
		logVerbose("[LLAMA] URLError: %v", err)
		return nil, fmt.Errorf("failed to reach Llama.cpp server: %v", err)
	}
	logVeryVerbose("[LLAMA] Got response with status: %s", resp.Status)

	return resp, nil
}

func (c *LlamaClient) readError(resp *http.Response) error {
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

func (c *LlamaClient) readJSON(resp *http.Response) (map[string]any, error) {
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return nil, c.readError(resp)
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

func (c *LlamaClient) fetch(path string, data any, method string) (map[string]any, error) {
	resp, err := c.doRequest(path, data, method, false)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	return c.readJSON(resp)
}

func (c *LlamaClient) unloadModel(model string) {
	logVerbose("[DEBUG] Unloading model: %s", model)
	if _, err := c.fetch("/models/unload", map[string]string{"model": model}, "POST"); err != nil {
		logVerbose("[DEBUG] Failed to unload model %s: %v", model, err)
	}
}
