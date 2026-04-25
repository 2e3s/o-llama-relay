Some application nowadays have a proper support of Ollama, but they support llama.cpp only as OpenAI-compatible API despite a functional overlap with Ollama. This limits its capabilities for its local LLM management.
This is a compatibility layer which mimics as Ollama instance and translates [Ollama](https://docs.ollama.com/api/introduction) REST endpoints into [llama.cpp](https://github.com/ggml-org/llama.cpp) server calls.

## How it works

```
Ollama client -> o_llama_wrapper (:11434) -> llama.cpp server (:8080)
```

- **`/api/*` endpoints** — Translated from Ollama format to llama.cpp's OpenAI-compatible format.
- **`/v1/*` endpoints** — Forwarded to llama.cpp without transformation.
- **Streaming** — `stream: true` is supported for translated `/api/generate` and `/api/chat` requests.
- **unloading** - Supported by `keep_alive: 0`.

## Build

```sh
go build
```

No external dependencies.

## Usage

```sh
./o_llama_wrapper [port] [host] [-v] [-vv]
```

| Argument | Description | Default |
|----------|-------------|---------|
| `port` | Listen port | `11434` |
| `host` | Listen address | `0.0.0.0` |
| `-v` | Verbose logging (show request routing) | off |
| `-vv` | Very verbose logging (show bodies/payloads) | off |

### Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLAMA_CPP_URL` | llama.cpp server base URL | `http://127.0.0.1:8080` |
