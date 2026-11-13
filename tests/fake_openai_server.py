"""Servidor HTTP mínimo compatible con la API de OpenAI (para tests).

Simula a Ollama en http://localhost:11434/v1: responde /v1/chat/completions
con una tool call de 'ExtraccionTecnica' y guarda cada pedido recibido.
`respuestas` es la lista de argumentos (dict) a devolver en cada llamada.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class FakeOpenAIServer:
    def __init__(self, respuestas: list[dict[str, Any]]) -> None:
        self.respuestas = respuestas
        self.requests: list[dict[str, Any]] = []
        servidor = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                i = len(servidor.requests)
                servidor.requests.append(body)
                args = servidor.respuestas[min(i, len(servidor.respuestas) - 1)]
                data = json.dumps({
                    "id": f"chatcmpl-{i}", "object": "chat.completion", "created": 0,
                    "model": body["model"],
                    "choices": [{
                        "index": 0, "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant", "content": None,
                            "tool_calls": [{
                                "id": f"call_{i}", "type": "function",
                                "function": {"name": "ExtraccionTecnica",
                                             "arguments": json.dumps(args)},
                            }],
                        },
                    }],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._httpd.server_address[1]}/v1"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> FakeOpenAIServer:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
