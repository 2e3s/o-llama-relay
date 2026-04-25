package main

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"
)

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
