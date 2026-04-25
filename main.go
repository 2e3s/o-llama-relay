package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
)

func runServer(host string, port int, handler *OllamaHandler) {
	addr := fmt.Sprintf("%s:%d", host, port)
	log.Printf("Ollama API wrapper running on %s", addr)
	log.Printf("Proxying to Llama.cpp at %s", handler.Client.BaseURL)
	log.Fatal(http.ListenAndServe(addr, handler))
}

func parseArgs() (port int, host, llamaURL string) {
	port = DefaultPort
	host = DefaultHost
	llamaURL = DefaultLlamaURL

	args := os.Args[1:]

	// Environment variables (lower priority than CLI args)
	if v := os.Getenv("OLLAMA_PORT"); v != "" {
		if p, err := strconv.Atoi(v); err == nil {
			port = p
		}
	}
	if v := os.Getenv("OLLAMA_HOST"); v != "" {
		host = v
	}
	if v := os.Getenv("LLAMA_CPP_URL"); v != "" {
		llamaURL = v
	}

	for i := 0; i < len(args); i++ {
		key, val, hasVal := strings.Cut(args[i], "=")

		switch key {
		case "--help", "-h":
			fmt.Printf(`Usage: %s [OPTIONS]

Options:
  -v, --verbose        Enable verbose output
  -vv, --debug         Enable very verbose output
  -h, --help           Show this help message and exit
  --port=N             Port to listen on (default: %d)
  --host=H             Host to bind to (default: %s)
  --llama-cpp-url=URL  Upstream llama.cpp URL (default: %s)

Environment variables:
  OLLAMA_PORT       Port to listen on (overridden by --port)
  OLLAMA_HOST       Host to bind to (overridden by --host)
  LLAMA_CPP_URL     Upstream llama.cpp URL (overridden by --llama-cpp-url)
`, os.Args[0], DefaultPort, DefaultHost, DefaultLlamaURL)
			os.Exit(0)
		case "-v", "--verbose":
			verboseLevel = 1
		case "-vv", "--debug":
			verboseLevel = 2
		case "--port":
			if !hasVal && i+1 < len(args) {
				i++
				val = args[i]
			}
			if p, err := strconv.Atoi(val); err == nil {
				port = p
			}
		case "--host":
			if !hasVal && i+1 < len(args) {
				i++
				host = args[i]
			}
			if hasVal {
				host = val
			}
		case "--llama-cpp-url":
			if !hasVal && i+1 < len(args) {
				i++
				llamaURL = args[i]
			}
			if hasVal {
				llamaURL = val
			}
		default:
			log.Fatalf("Unknown argument: %s", args[i])
		}
	}
	return
}

func main() {
	port, host, llamaURL := parseArgs()

	client := NewLlamaClient(llamaURL)
	handler := NewOllamaHandler(client)

	runServer(host, port, handler)
}
