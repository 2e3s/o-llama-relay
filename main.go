package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
)

func runServer(host string, port int, handler *OllamaHandler) {
	addr := fmt.Sprintf("%s:%d", host, port)
	log.Printf("Ollama API wrapper running on %s", addr)
	log.Printf("Proxying to Llama.cpp at %s", handler.Client.BaseURL)
	log.Fatal(http.ListenAndServe(addr, handler))
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

	client := NewLlamaClient(getEnv("LLAMA_CPP_URL", "http://127.0.0.1:8080"))
	handler := NewOllamaHandler(client)

	runServer(host, port, handler)
}
