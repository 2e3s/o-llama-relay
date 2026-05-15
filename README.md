# Llama.cpp as Ollama

Some application nowadays have a proper support of Ollama, but they support llama.cpp only as OpenAI-compatible API despite its functional overlap with Ollama. This limits its capabilities for its local LLM management, like detecting or unloading currently loaded models.

This application mimics llama.cpp as Ollama, serving as a compatibility layer which mimics as Ollama instance and translates [Ollama](https://docs.ollama.com/api/introduction) REST endpoints into [llama.cpp](https://github.com/ggml-org/llama.cpp) server calls.


## How it works

```
Ollama client -> o_llama_relay (:11434) -> llama.cpp server (:8080)
```

- **`/api/*` endpoints** — Translated from Ollama format to llama.cpp's OpenAI-compatible format.
- **`/v1/*` endpoints** — Forwarded to llama.cpp without transformation.
- **Streaming** — `stream: true` is supported for translated `/api/generate` and `/api/chat` requests.
- **unloading** - Supported by `keep_alive: 0`.

## Build

```sh
go build
./o_llama_relay --port=1234
```

### Docker Compose

```sh
git clone https://github.com/2e3s/o-llama-relay && cd o-llama-relay
LLAMA_CPP_URL=http://host.docker.internal:8080 docker compose up --build
```
Or keep the variables in `.env` file:
```
LLAMA_CPP_URL=http://host.docker.internal:8080
OLLAMA_PORT=12345
```

| Environment variable | Description |
|---|---|
| `OLLAMA_PORT` | External port to expose |
| `LLAMA_CPP_URL` | llama.cpp server URL |

## Usage

```sh
./o_llama_relay [options] [port] [host]
```

Every parameter can be set via a CLI argument **or** an environment variable.
CLI arguments take priority over environment variables.

| CLI argument | Environment variable | Description | Default |
|---|---|---|---|
| `--port <n>` or positional number | `OLLAMA_PORT` | Listen port | `11434` |
| `--host <addr>` or positional string | `OLLAMA_HOST` | Listen address | `0.0.0.0` |
| `--llama-cpp-url <url>` | `LLAMA_CPP_URL` | llama.cpp server base URL | `http://127.0.0.1:8080` |
| `-v` | — | Verbose logging (show request routing) | off |
| `-vv` | — | Very verbose logging (show bodies/payloads) | off |

Positional arguments are supported for backward compatibility: a bare number
is treated as the port, a bare string as the host address.
