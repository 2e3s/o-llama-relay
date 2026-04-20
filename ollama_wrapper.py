#!/usr/bin/env python3
"""
Ollama API wrapper for Llama.cpp server.
Translates Ollama API endpoints to Llama.cpp API format.
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import urlopen, Request
from urllib.error import URLError

LLAMA_CPP_URL = os.environ.get("LLAMA_CPP_URL", "http://127.0.0.1:8080")
VERSION = "0.1.0"


def fetch_from_llama(path: str, data: dict | None = None, method: str = "GET") -> dict:
    """Fetch data from Llama.cpp server."""
    url = f"{LLAMA_CPP_URL}{path}"

    # Log request
    print(f"[LLAMA] {method} {url}")
    if data:
        print(f"[LLAMA] Payload: {json.dumps(data)}")
    sys.stdout.flush()

    headers = {"Content-Type": "application/json"}

    body = None
    if data is not None:
        body = json.dumps(data).encode()

    req = Request(url, data=body, headers=headers, method=method)

    print(f"[LLAMA] Opening URL, waiting for response...")
    sys.stdout.flush()

    try:
        with urlopen(req, timeout=60) as response:
            print(f"[LLAMA] Got response, reading...")
            sys.stdout.flush()
            result = json.loads(response.read().decode())
            print(f"[LLAMA] Response received: {len(json.dumps(result))} bytes")
            sys.stdout.flush()
            return result
    except URLError as e:
        print(f"[LLAMA] URLError: {e}")
        sys.stdout.flush()
        raise Exception(f"Failed to reach Llama.cpp server: {e}")
    except Exception as e:
        print(f"[LLAMA] Exception: {e}")
        sys.stdout.flush()
        raise


def llama_to_ollama_tags(llama_data: dict) -> dict:
    """Convert Llama.cpp /models format to Ollama /api/tags format."""
    models = []

    for m in llama_data.get("data", []):
        model_id = m.get("id", "")
        status = m.get("status", {})
        args = status.get("args", [])
        preset = status.get("preset", "")

        # Extract model path from args
        model_path = None
        for i, arg in enumerate(args):
            if arg == "--model" and i + 1 < len(args):
                model_path = args[i + 1]
                break

        # Try to extract size from model path (heuristic)
        size = 0
        if model_path:
            try:
                # Very rough estimate based on filename patterns
                filename = os.path.basename(model_path)
                if "Q4" in filename:
                    # Estimate 0.4 bytes per parameter
                    match = re.search(r'(\d+\.?\d*)[Bb]', filename)
                    if match:
                        params = float(match.group(1))
                        size = int(params * 1e9 * 0.4)
            except:
                pass

        # Extract model family and quantization from filename
        details = {
            "format": "gguf",
            "family": "unknown",
            "families": ["unknown"],
            "parameter_size": "unknown",
            "quantization_level": "unknown"
        }

        if model_path:
            filename = os.path.basename(model_path).lower()

            # Detect family
            if "llama" in filename:
                details["family"] = "llama"
                details["families"] = ["llama"]
            elif "mistral" in filename:
                details["family"] = "mistral"
                details["families"] = ["mistral"]
            elif "gemma" in filename:
                details["family"] = "gemma"
                details["families"] = ["gemma"]
            elif "qwen" in filename:
                details["family"] = "qwen"
                details["families"] = ["qwen"]
            elif "phi" in filename:
                details["family"] = "phi"
                details["families"] = ["phi"]
            elif "nemotron" in filename:
                details["family"] = "nemotron"
                details["families"] = ["nemotron"]
            elif "bert" in filename:
                details["family"] = "bert"
                details["families"] = ["bert"]

            # Detect quantization
            if "q4_0" in filename:
                details["quantization_level"] = "Q4_0"
            elif "q4_k_s" in filename or "q4_k_m" in filename or "q4_k_n" in filename:
                details["quantization_level"] = filename.split("q4_k_")[1].split("-")[0].upper()
            elif "q8_0" in filename:
                details["quantization_level"] = "Q8_0"
            elif "fp16" in filename or "f16" in filename:
                details["quantization_level"] = "F16"
            elif "mxfp4" in filename:
                details["quantization_level"] = "MXFP4"
            elif "iq3" in filename:
                details["quantization_level"] = "IQ3"
            elif "iq4" in filename:
                details["quantization_level"] = "IQ4"

            # Detect parameter size
            param_match = re.search(r'(\d+\.?\d*)[Bb]', filename)
            if param_match:
                params = float(param_match.group(1))
                if params < 1:
                    details["parameter_size"] = f"{params * 1000}M"
                else:
                    details["parameter_size"] = f"{params}B"

        model_entry = {
            "name": model_id,
            "model": model_id,
            "modified_at": datetime.fromtimestamp(m.get("created", 0), tz=timezone.utc).isoformat(),
            "size": size,
            "digest": "",  # Llama.cpp doesn't provide digest
            "details": details
        }
        models.append(model_entry)

    return {"models": models}


def llama_to_ollama_ps(llama_data: dict) -> dict:
    """Convert Llama.cpp /models format to Ollama /api/ps format (loaded models only)."""
    models = []

    for m in llama_data.get("data", []):
        status = m.get("status", {})
        if status.get("value") != "loaded":
            continue

        model_id = m.get("id", "")
        args = status.get("args", [])

        # Extract port from args (for loaded models)
        port = 0
        for i, arg in enumerate(args):
            if arg == "--port" and i + 1 < len(args):
                try:
                    port = int(args[i + 1])
                except:
                    pass
                break

        # Build expires_at based on sleep-idle-seconds
        sleep_seconds = 3600  # default
        for i, arg in enumerate(args):
            if arg == "--sleep-idle-seconds" and i + 1 < len(args):
                try:
                    sleep_seconds = int(args[i + 1])
                except:
                    pass
                break

        expires_at = datetime.now(timezone.utc)
        expires_at = expires_at.replace(second=0, microsecond=0)
        from datetime import timedelta
        expires_at = expires_at + timedelta(seconds=sleep_seconds)

        model_entry = {
            "name": model_id,
            "model": model_id,
            "size": 0,
            "digest": "",
            "details": {
                "format": "gguf",
                "family": "unknown",
                "families": ["unknown"],
                "parameter_size": "unknown",
                "quantization_level": "unknown"
            },
            "expires_at": expires_at.isoformat(),
            "size_vram": 0,
            "context_length": 2048
        }
        models.append(model_entry)

    return {"models": models}


class OllamaHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress default logging

    def send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def send_error_json(self, message: str, status: int = 500):
        self.send_json({"error": message}, status)

    def do_GET(self):
        print(f"[OLLAMA] GET {self.path}")
        if self.path == "/api/tags":
            self.handle_tags()
        elif self.path == "/api/ps":
            self.handle_ps()
        elif self.path == "/api/version":
            self.handle_version()
        else:
            self.send_error_json(f"Unknown endpoint: {self.path}", 404)

    def do_POST(self):
        # Read body ONCE and store it
        content_length = int(self.headers.get("Content-Length", 0))
        body_raw = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        self.request_body = json.loads(body_raw)

        display_body = body_raw if len(body_raw) <= 500 else body_raw[:500] + "..."
        print(f"[OLLAMA] POST {self.path} Body: {display_body}")
        sys.stdout.flush()

        print(f"[DEBUG] Starting request handling for {self.path}")
        sys.stdout.flush()

        if self.path == "/api/generate":
            print(f"[DEBUG] Calling handle_generate")
            sys.stdout.flush()
            self.handle_generate()
        elif self.path == "/api/chat":
            print(f"[DEBUG] Calling handle_chat")
            sys.stdout.flush()
            self.handle_chat()
        elif self.path == "/api/embed":
            self.handle_embed()
        elif self.path == "/api/create":
            self.handle_create()
        elif self.path == "/api/copy":
            self.handle_copy()
        elif self.path == "/api/pull":
            self.handle_pull()
        elif self.path == "/api/push":
            self.handle_push()
        elif self.path == "/api/show":
            self.handle_show()
        else:
            self.send_error_json(f"Unknown endpoint: {self.path}", 404)

    def do_DELETE(self):
        print(f"[OLLAMA] DELETE {self.path}")
        if self.path == "/api/delete":
            self.handle_delete()
        else:
            self.send_error_json(f"Unknown endpoint: {self.path}", 404)

    def handle_tags(self):
        try:
            data = fetch_from_llama("/models")
            result = llama_to_ollama_tags(data)
            self.send_json(result)
        except Exception as e:
            self.send_error_json(str(e))

    def handle_ps(self):
        try:
            data = fetch_from_llama("/models")
            result = llama_to_ollama_ps(data)
            self.send_json(result)
        except Exception as e:
            self.send_error_json(str(e))

    def handle_version(self):
        self.send_json({"version": VERSION})

    def handle_generate(self):
        try:
            req_body = self.request_body
            print(
                f"[DEBUG] handle_generate: model={req_body.get('model')}, prompt={req_body.get('prompt', '')[:50]}...")
            sys.stdout.flush()
            model = req_body.get("model", "")
            prompt = req_body.get("prompt", "")
            stream = req_body.get("stream", True)

            print(f"[DEBUG] handle_generate: model={model}, prompt={prompt[:50] if prompt else ''}...")
            sys.stdout.flush()

            # Map to Llama.cpp completions endpoint
            llama_data = {
                "prompt": prompt,
                "model": model,
                "stream": stream
            }

            # Map options if present
            options = req_body.get("options", {})
            if options:
                if "temperature" in options:
                    llama_data["temperature"] = options["temperature"]
                if "top_p" in options:
                    llama_data["top_p"] = options["top_p"]
                if "top_k" in options:
                    llama_data["top_k"] = options["top_k"]
                if "num_predict" in options or "max_tokens" in options:
                    llama_data["max_tokens"] = options.get("num_predict") or options.get("max_tokens")
                if "stop" in options:
                    llama_data["stop"] = options["stop"]

            result = fetch_from_llama("/v1/completions", llama_data, "POST")

            # Convert to Ollama format
            if stream:
                # For streaming, we'd need to handle chunked responses
                # This is a simplified version
                self.send_json({"error": "Streaming not fully implemented"}, 501)
            else:
                response = result.get("choices", [{}])[0].get("text", "")
                self.send_json({
                    "model": model,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "response": response,
                    "done": True,
                    "done_reason": "stop"
                })
        except Exception as e:
            self.send_error_json(str(e))

    def handle_chat(self):
        try:
            req_body = self.request_body
            model = req_body.get("model", "")
            messages = req_body.get("messages", [])
            stream = req_body.get("stream", True)

            print(f"[DEBUG] handle_chat: model={model}, messages={len(messages)}, stream={stream}")
            sys.stdout.flush()

            # Map to Llama.cpp chat completions endpoint
            llama_data = {
                "model": model,
                "messages": messages,
                "stream": False  # Simplified: always non-stream for now
            }

            print(f"[DEBUG] handle_chat: calling fetch_from_llama with llama_data")
            sys.stdout.flush()

            # Map options if present
            options = req_body.get("options", {})
            if options:
                if "temperature" in options:
                    llama_data["temperature"] = options["temperature"]
                if "top_p" in options:
                    llama_data["top_p"] = options["top_p"]
                if "top_k" in options:
                    llama_data["top_k"] = options["top_k"]
                if "num_predict" in options or "max_tokens" in options:
                    llama_data["max_tokens"] = options.get("num_predict") or options.get("max_tokens")

            result = fetch_from_llama("/v1/chat/completions", llama_data, "POST")

            # Convert to Ollama format
            if result.get("choices"):
                choice = result["choices"][0]
                content = choice.get("message", {}).get("content", "")
                self.send_json({
                    "model": model,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "done": True,
                    "done_reason": "stop"
                })
            else:
                self.send_error_json("No response from model")
        except Exception as e:
            self.send_error_json(str(e))

    def handle_embed(self):
        try:
            req_body = self.request_body
            model = req_body.get("model", "")
            input_text = req_body.get("input", "")

            # Convert input to list if string
            if isinstance(input_text, str):
                input_text = [input_text]

            # Map to Llama.cpp embeddings endpoint
            llama_data = {
                "model": model,
                "input": input_text
            }

            result = fetch_from_llama("/v1/embeddings", llama_data, "POST")

            # Convert to Ollama format
            embeddings = []
            if result.get("data"):
                for item in result["data"]:
                    embeddings.append(item.get("embedding", []))

            self.send_json({
                "model": model,
                "embeddings": embeddings
            })
        except Exception as e:
            self.send_error_json(str(e))

    def handle_show(self):
        try:
            req_body = self.request_body
            model = req_body.get("model", "")
            verbose = req_body.get("verbose", False)

            # Use Llama.cpp /api/show endpoint
            llama_data = {"model": model}
            result = fetch_from_llama("/api/show", llama_data, "POST")

            # Convert to Ollama format
            self.send_json({
                "model": model,
                "modified_at": datetime.now(timezone.utc).isoformat(),
                "details": {
                    "format": "gguf",
                    "family": "unknown"
                }
            })
        except Exception as e:
            self.send_error_json(str(e))

    def handle_create(self):
        # Ollama create is not directly supported by Llama.cpp
        self.send_error_json("Model creation not supported - Llama.cpp uses pre-loaded models", 501)

    def handle_copy(self):
        # Ollama copy is not directly supported by Llama.cpp
        self.send_error_json("Model copying not supported - Llama.cpp uses pre-loaded models", 501)

    def handle_delete(self):
        # Ollama delete is not directly supported by Llama.cpp
        self.send_error_json("Model deletion not supported - Llama.cpp uses pre-loaded models", 501)

    def handle_pull(self):
        # Ollama pull is not directly supported by Llama.cpp
        self.send_error_json("Model pulling not supported - Llama.cpp uses pre-loaded models", 501)

    def handle_push(self):
        # Ollama push is not directly supported by Llama.cpp
        self.send_error_json("Model pushing not supported - Llama.cpp uses pre-loaded models", 501)


def run_server(host: str = "0.0.0.0", port: int = 11434):
    server = HTTPServer((host, port), OllamaHandler)
    print(f"Ollama API wrapper running on {host}:{port}")
    print(f"Proxying to Llama.cpp at {LLAMA_CPP_URL}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    default_host = "0.0.0.0"
    default_port = 11434

    if len(sys.argv) > 1:
        pdefault_portort = int(sys.argv[1])
    if len(sys.argv) > 2:
        default_host = sys.argv[2]

    run_server(default_host, default_port)
