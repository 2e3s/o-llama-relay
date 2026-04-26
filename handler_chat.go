package main

import (
	"encoding/json"
	"net/http"
	"sort"
	"strings"
	"time"
)

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
	} else if value, ok := source["reasoning_content"].(string); ok && value != "" {
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
		if err := h.streamLlamaResponse(
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
			h.Client.unloadModel(model)
		}
		return
	}

	result, err := h.Client.fetch("/v1/chat/completions", llamaData, "POST")
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

	if shouldUnload(keepAlive) {
		h.Client.unloadModel(model)
	}
}
