"""Small dependency-free HTTP serving layer for LLAISYS.

This module intentionally provides APIs rather than a web UI, authentication,
or persistence. It exposes OpenAI-shaped completion endpoints plus explicit
in-memory session and request management.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import unquote, urlparse

from .serving import (
    ChatMessage,
    ChatService,
    FinishReason,
    GenerationRequest,
    RequestStatus,
    RoundRobinScheduler,
    TokenEvent,
)


class ServingMetrics:
    """Thread-safe counters and latency aggregates for the HTTP layer."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.requests_total = 0
        self.requests_active = 0
        self.requests_completed = 0
        self.requests_failed = 0
        self.requests_cancelled = 0
        self.prompt_tokens_total = 0
        self.generated_tokens_total = 0
        self._first_token_at: Dict[str, float] = {}
        self._ttft_sum = 0.0
        self._duration_sum = 0.0
        self._terminal_requests: set[str] = set()

    def submitted(self, request: GenerationRequest) -> None:
        with self._lock:
            self.requests_total += 1
            self.requests_active += 1
            self.prompt_tokens_total += len(request.input_tokens)

    def observed(self, request: GenerationRequest, event: TokenEvent) -> None:
        with self._lock:
            if event.token_id is not None:
                self.generated_tokens_total += 1
                if request.request_id not in self._first_token_at:
                    now = time.time()
                    self._first_token_at[request.request_id] = now
                    self._ttft_sum += now - request.created_at
            if not event.finished or request.request_id in self._terminal_requests:
                return
            self._terminal_requests.add(request.request_id)
            self.requests_active = max(0, self.requests_active - 1)
            self._duration_sum += time.time() - request.created_at
            if request.status is RequestStatus.FINISHED:
                self.requests_completed += 1
            elif request.status is RequestStatus.FAILED:
                self.requests_failed += 1
            else:
                self.requests_cancelled += 1

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            terminal = (
                self.requests_completed
                + self.requests_failed
                + self.requests_cancelled
            )
            return {
                "requests_total": self.requests_total,
                "requests_active": self.requests_active,
                "requests_completed": self.requests_completed,
                "requests_failed": self.requests_failed,
                "requests_cancelled": self.requests_cancelled,
                "prompt_tokens_total": self.prompt_tokens_total,
                "generated_tokens_total": self.generated_tokens_total,
                "mean_ttft_seconds": (
                    self._ttft_sum / len(self._first_token_at)
                    if self._first_token_at
                    else 0.0
                ),
                "mean_request_duration_seconds": (
                    self._duration_sum / terminal if terminal else 0.0
                ),
            }

    def prometheus(self) -> str:
        values = self.snapshot()
        lines = []
        for name, value in values.items():
            metric_type = (
                "gauge"
                if name.startswith("mean_") or name == "requests_active"
                else "counter"
            )
            metric_name = f"llaisys_{name}"
            lines.extend(
                [f"# TYPE {metric_name} {metric_type}", f"{metric_name} {value}"]
            )
        return "\n".join(lines) + "\n"


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class OpenAIAPIServer:
    """OpenAI-shaped HTTP server backed by ``RoundRobinScheduler``."""

    def __init__(
        self,
        chat: ChatService,
        model_id: str = "DeepSeek-R1-Distill-Qwen-1.5B",
        host: str = "127.0.0.1",
        port: int = 8000,
    ) -> None:
        self.chat = chat
        self.scheduler: RoundRobinScheduler = chat.scheduler
        self.model_id = model_id
        self.metrics = ServingMetrics()
        self._httpd = _Server((host, port), self._handler_type())
        self._thread: Optional[threading.Thread] = None

    @property
    def address(self) -> Tuple[str, int]:
        host, port = self._httpd.server_address[:2]
        return str(host), int(port)

    def start(self) -> None:
        self.scheduler.start()
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="llaisys-http-server",
            daemon=True,
        )
        self._thread.start()

    def serve_forever(self) -> None:
        self.scheduler.start()
        self._httpd.serve_forever()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join()
        self.scheduler.stop()

    def _handler_type(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, _format, *_args):
                return

            def do_GET(self):
                owner._handle_get(self)

            def do_POST(self):
                owner._handle_post(self)

            def do_DELETE(self):
                owner._handle_delete(self)

        return Handler

    def _handle_get(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            path = urlparse(handler.path).path
            if path == "/health":
                self._json(handler, HTTPStatus.OK, {"status": "ok"})
                return
            if path == "/v1/models":
                self._json(
                    handler,
                    HTTPStatus.OK,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": self.model_id,
                                "object": "model",
                                "owned_by": "llaisys",
                            }
                        ],
                    },
                )
                return
            if path == "/metrics":
                self._text(
                    handler,
                    HTTPStatus.OK,
                    self.metrics.prometheus(),
                    "text/plain; version=0.0.4; charset=utf-8",
                )
                return
            if path == "/sessions":
                sessions = [
                    self.chat.export_session(session.session_id)
                    for session in self.scheduler.sessions.list()
                ]
                self._json(handler, HTTPStatus.OK, {"data": sessions})
                return
            if path.startswith("/sessions/"):
                session_id = unquote(path.removeprefix("/sessions/"))
                self._json(
                    handler,
                    HTTPStatus.OK,
                    self.chat.export_session(session_id),
                )
                return
            if path.startswith("/requests/"):
                request_id = unquote(path.removeprefix("/requests/"))
                request = self.scheduler.request_pool.get(request_id)
                self._json(handler, HTTPStatus.OK, self._request_json(request))
                return
            self._error(handler, HTTPStatus.NOT_FOUND, "Unknown endpoint")
        except Exception as error:
            self._exception(handler, error)

    def _handle_post(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            path = urlparse(handler.path).path
            body = self._read_json(handler)
            if path == "/v1/chat/completions":
                self._chat_completion(handler, body)
                return
            if path == "/v1/completions":
                self._text_completion(handler, body)
                return
            if path == "/sessions":
                session_id = self._optional_string(body, "session_id")
                user_id = self._optional_string(body, "user_id")
                system_prompt = self._optional_string(body, "system_prompt")
                metadata = body.get("metadata")
                if metadata is not None and not isinstance(metadata, Mapping):
                    raise ValueError("metadata must be an object")
                session = self.chat.create_session(
                    session_id=session_id,
                    user_id=user_id,
                    system_prompt=system_prompt,
                    metadata=metadata,
                )
                self._json(
                    handler,
                    HTTPStatus.CREATED,
                    self.chat.export_session(session.session_id),
                )
                return
            if path.startswith("/sessions/") and path.endswith("/messages"):
                session_id = unquote(
                    path.removeprefix("/sessions/").removesuffix("/messages")
                ).rstrip("/")
                content = body.get("content")
                if not isinstance(content, str) or not content:
                    raise ValueError("content must be a non-empty string")
                request = self.chat.submit_message(
                    session_id, content, **self._generation_args(body)
                )
                self.metrics.submitted(request)
                self._observe_in_background(request)
                self._json(
                    handler,
                    HTTPStatus.ACCEPTED,
                    self._request_json(request),
                )
                return
            if path.startswith("/requests/") and path.endswith("/cancel"):
                request_id = unquote(
                    path.removeprefix("/requests/").removesuffix("/cancel")
                ).rstrip("/")
                cancelled = self.scheduler.cancel(request_id)
                self._json(handler, HTTPStatus.OK, {"cancelled": cancelled})
                return
            self._error(handler, HTTPStatus.NOT_FOUND, "Unknown endpoint")
        except Exception as error:
            self._exception(handler, error)

    def _handle_delete(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            path = urlparse(handler.path).path
            if path.startswith("/sessions/"):
                session_id = unquote(path.removeprefix("/sessions/"))
                self.scheduler.sessions.delete(session_id)
                self._json(handler, HTTPStatus.OK, {"deleted": True})
                return
            self._error(handler, HTTPStatus.NOT_FOUND, "Unknown endpoint")
        except Exception as error:
            self._exception(handler, error)

    def _chat_completion(
        self, handler: BaseHTTPRequestHandler, body: Mapping[str, object]
    ) -> None:
        self._validate_model(body)
        raw_messages = body.get("messages")
        if not isinstance(raw_messages, Sequence) or isinstance(
            raw_messages, (str, bytes)
        ):
            raise ValueError("messages must be a non-empty sequence")
        messages = self._messages(raw_messages)
        if not messages or messages[-1].role != "user":
            raise ValueError("The final chat message must have role=user")
        generation_args = self._generation_args(body)
        stream = self._streaming(body)

        requested_session = self._optional_string(body, "session_id")
        if requested_session == "":
            raise ValueError("session_id must not be empty")
        ephemeral = requested_session is None
        session_id = str(requested_session or f"openai-{uuid.uuid4().hex}")
        if ephemeral:
            self.scheduler.sessions.create(
                session_id=session_id,
                initial_messages=messages[:-1],
            )
        else:
            try:
                self.scheduler.sessions.get(session_id)
            except KeyError:
                self.scheduler.sessions.create(
                    session_id=session_id,
                    initial_messages=messages[:-1],
                )

        request: Optional[GenerationRequest] = None
        try:
            request = self.chat.submit_message(
                session_id,
                messages[-1].content,
                **generation_args,
            )
            self.metrics.submitted(request)
            if stream:
                self._stream_chat(handler, request)
            else:
                self._complete_chat(handler, request)
        finally:
            if ephemeral:
                session = self.scheduler.sessions.get(session_id)
                if not session.busy:
                    self.scheduler.sessions.delete(session_id)

    def _text_completion(
        self, handler: BaseHTTPRequestHandler, body: Mapping[str, object]
    ) -> None:
        self._validate_model(body)
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        generation_args = self._generation_args(body)
        stream = self._streaming(body)
        session_id = f"completion-{uuid.uuid4().hex}"
        self.scheduler.sessions.create(session_id=session_id)
        request: Optional[GenerationRequest] = None
        try:
            input_tokens = [
                int(token) for token in self.chat.tokenizer.encode(prompt)
            ]
            request = self.scheduler.submit(
                session_id, input_tokens, **generation_args
            )
            self.metrics.submitted(request)
            if stream:
                self._stream_text(handler, request)
            else:
                events = list(self.scheduler.events(request.request_id, timeout=None))
                for event in events:
                    self.metrics.observed(request, event)
                text = request.streamed_text
                self._json(
                    handler,
                    HTTPStatus.OK,
                    {
                        "id": request.request_id,
                        "object": "text_completion",
                        "model": self.model_id,
                        "choices": [
                            {
                                "index": 0,
                                "text": text,
                                "finish_reason": self._openai_reason(
                                    request.finish_reason
                                ),
                            }
                        ],
                        "usage": self._usage(request),
                    },
                )
        finally:
            session = self.scheduler.sessions.get(session_id)
            if not session.busy:
                self.scheduler.sessions.delete(session_id)

    def _complete_chat(
        self, handler: BaseHTTPRequestHandler, request: GenerationRequest
    ) -> None:
        events = list(self.chat.events(request.request_id, timeout=None))
        for event in events:
            self.metrics.observed(request, event)
        if request.status is RequestStatus.FAILED:
            raise RuntimeError(request.error or "Model inference failed")
        content = request.streamed_text
        self._json(
            handler,
            HTTPStatus.OK,
            {
                "id": request.request_id,
                "object": "chat.completion",
                "created": int(request.created_at),
                "model": self.model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": self._openai_reason(request.finish_reason),
                    }
                ],
                "usage": self._usage(request),
            },
        )

    def _stream_chat(
        self, handler: BaseHTTPRequestHandler, request: GenerationRequest
    ) -> None:
        self._begin_sse(handler)
        try:
            for event in self.chat.events(request.request_id, timeout=None):
                self.metrics.observed(request, event)
                chunk = {
                    "id": request.request_id,
                    "object": "chat.completion.chunk",
                    "created": int(request.created_at),
                    "model": self.model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": ({"content": event.text} if event.text else {}),
                            "finish_reason": (
                                self._openai_reason(event.finish_reason)
                                if event.finished
                                else None
                            ),
                        }
                    ],
                }
                self._sse(handler, chunk)
            self._sse_done(handler)
        except (BrokenPipeError, ConnectionResetError):
            self.scheduler.cancel(request.request_id)
            for event in self.chat.events(request.request_id, timeout=None):
                self.metrics.observed(request, event)

    def _stream_text(
        self, handler: BaseHTTPRequestHandler, request: GenerationRequest
    ) -> None:
        self._begin_sse(handler)
        try:
            for event in self.scheduler.events(request.request_id, timeout=None):
                self.metrics.observed(request, event)
                self._sse(
                    handler,
                    {
                        "id": request.request_id,
                        "object": "text_completion",
                        "model": self.model_id,
                        "choices": [
                            {
                                "index": 0,
                                "text": event.text or "",
                                "finish_reason": (
                                    self._openai_reason(event.finish_reason)
                                    if event.finished
                                    else None
                                ),
                            }
                        ],
                    },
                )
            self._sse_done(handler)
        except (BrokenPipeError, ConnectionResetError):
            self.scheduler.cancel(request.request_id)
            for event in self.scheduler.events(request.request_id, timeout=None):
                self.metrics.observed(request, event)

    def _observe_in_background(self, request: GenerationRequest) -> None:
        """Finalizes an asynchronous chat turn and records its metrics."""

        def observe() -> None:
            for event in self.chat.events(request.request_id, timeout=None):
                self.metrics.observed(request, event)

        threading.Thread(
            target=observe,
            name=f"llaisys-observer-{request.request_id}",
            daemon=True,
        ).start()

    @staticmethod
    def _messages(raw_messages: Sequence[object]) -> List[ChatMessage]:
        messages = []
        for raw in raw_messages:
            if not isinstance(raw, Mapping):
                raise ValueError("Each message must be an object")
            role = raw.get("role")
            content = raw.get("content")
            if not isinstance(role, str):
                raise ValueError("message role must be a string")
            if not isinstance(content, str):
                raise ValueError("message content must be a string")
            messages.append(ChatMessage(role=role, content=content))
        return messages

    @staticmethod
    def _generation_args(body: Mapping[str, object]) -> Dict[str, object]:
        raw_stop_tokens = body.get("stop_token_ids", ())
        if raw_stop_tokens is None:
            raw_stop_tokens = ()
        if not isinstance(raw_stop_tokens, Sequence) or isinstance(
            raw_stop_tokens, (str, bytes)
        ):
            raise ValueError("stop_token_ids must be a sequence of integers")
        stop_tokens = []
        for token in raw_stop_tokens:
            if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                raise ValueError(
                    "stop_token_ids must contain non-negative integers"
                )
            stop_tokens.append(token)

        raw_stop = body.get("stop", ())
        if raw_stop is None:
            stop_strings: Tuple[str, ...] = ()
        elif isinstance(raw_stop, str):
            stop_strings = (raw_stop,)
        elif isinstance(raw_stop, Sequence) and not isinstance(raw_stop, bytes):
            if any(not isinstance(stop, str) for stop in raw_stop):
                raise ValueError("stop must contain only strings")
            stop_strings = tuple(raw_stop)
        else:
            raise ValueError("stop must be a string or a sequence of strings")
        if any(not stop for stop in stop_strings):
            raise ValueError("stop strings must not be empty")

        max_tokens_key = (
            "max_tokens" if "max_tokens" in body else "max_completion_tokens"
        )
        max_tokens = OpenAIAPIServer._integer(
            body, max_tokens_key, 128, minimum=1
        )
        priority = OpenAIAPIServer._integer(body, "priority", 0)
        top_k = OpenAIAPIServer._integer(body, "top_k", 1, minimum=0)
        seed = OpenAIAPIServer._integer(body, "seed", 0, minimum=0)
        if seed >= 2**64:
            raise ValueError("seed must fit in an unsigned 64-bit integer")
        timeout = body.get("timeout_seconds")
        if timeout is not None:
            timeout = OpenAIAPIServer._number(
                body, "timeout_seconds", 0.0, minimum=0.0, strict_minimum=True
            )
        truncate_prompt = body.get("truncate_prompt", False)
        if not isinstance(truncate_prompt, bool):
            raise ValueError("truncate_prompt must be a boolean")
        return {
            "max_new_tokens": max_tokens,
            "priority": priority,
            "stop_token_ids": tuple(stop_tokens),
            "stop_strings": stop_strings,
            "truncate_prompt": truncate_prompt,
            "timeout_seconds": timeout,
            "top_k": top_k,
            "top_p": OpenAIAPIServer._number(
                body, "top_p", 0.8, minimum=0.0, maximum=1.0,
                strict_minimum=True
            ),
            "temperature": OpenAIAPIServer._number(
                body, "temperature", 0.8, minimum=0.0, strict_minimum=True
            ),
            "repetition_penalty": OpenAIAPIServer._number(
                body,
                "repetition_penalty",
                1.0,
                minimum=0.0,
                strict_minimum=True,
            ),
            "seed": seed,
        }

    @staticmethod
    def _integer(
        body: Mapping[str, object],
        name: str,
        default: int,
        minimum: Optional[int] = None,
    ) -> int:
        value = body.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if minimum is not None and value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
        return value

    @staticmethod
    def _number(
        body: Mapping[str, object],
        name: str,
        default: float,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        strict_minimum: bool = False,
    ) -> float:
        value = body.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        if minimum is not None and (
            result <= minimum if strict_minimum else result < minimum
        ):
            comparison = "greater than" if strict_minimum else "at least"
            raise ValueError(f"{name} must be {comparison} {minimum}")
        if maximum is not None and result > maximum:
            raise ValueError(f"{name} must be at most {maximum}")
        return result

    @staticmethod
    def _optional_string(
        body: Mapping[str, object], name: str
    ) -> Optional[str]:
        value = body.get(name)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{name} must be a string")
        return value

    def _validate_model(self, body: Mapping[str, object]) -> None:
        model = body.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a non-empty string")
        if model != self.model_id:
            raise ValueError(f"Unknown model: {model}")

    @staticmethod
    def _streaming(body: Mapping[str, object]) -> bool:
        stream = body.get("stream", False)
        if not isinstance(stream, bool):
            raise ValueError("stream must be a boolean")
        return stream

    @staticmethod
    def _usage(request: GenerationRequest) -> Dict[str, int]:
        prompt_tokens = request.context_length or len(request.input_tokens)
        completion_tokens = len(request.generated_tokens)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    @staticmethod
    def _openai_reason(reason: Optional[FinishReason]) -> Optional[str]:
        if reason in (FinishReason.EOS, FinishReason.STOP):
            return "stop"
        return reason.value if reason is not None else None

    @staticmethod
    def _request_json(request: GenerationRequest) -> Dict[str, object]:
        return {
            "request_id": request.request_id,
            "session_id": request.session_id,
            "status": request.status.value,
            "finish_reason": (
                request.finish_reason.value if request.finish_reason else None
            ),
            "generated_tokens": list(request.generated_tokens),
            "error": request.error,
            "created_at": request.created_at,
            "started_at": request.started_at,
            "finished_at": request.finished_at,
        }

    @staticmethod
    def _read_json(handler: BaseHTTPRequestHandler) -> Mapping[str, object]:
        length = int(handler.headers.get("Content-Length", "0"))
        raw = handler.rfile.read(length) if length else b"{}"
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError("JSON body must be an object")
        return data

    @staticmethod
    def _json(
        handler: BaseHTTPRequestHandler,
        status: HTTPStatus,
        data: Mapping[str, object],
    ) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)

    @staticmethod
    def _text(
        handler: BaseHTTPRequestHandler,
        status: HTTPStatus,
        text: str,
        content_type: str,
    ) -> None:
        payload = text.encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)

    @staticmethod
    def _begin_sse(handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.close_connection = True

    @staticmethod
    def _sse(handler: BaseHTTPRequestHandler, data: Mapping[str, object]) -> None:
        payload = f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode(
            "utf-8"
        )
        handler.wfile.write(payload)
        handler.wfile.flush()

    @staticmethod
    def _sse_done(handler: BaseHTTPRequestHandler) -> None:
        handler.wfile.write(b"data: [DONE]\n\n")
        handler.wfile.flush()

    def _exception(
        self, handler: BaseHTTPRequestHandler, error: Exception
    ) -> None:
        if isinstance(error, (KeyError, ValueError, TypeError)):
            status = HTTPStatus.BAD_REQUEST
        else:
            status = HTTPStatus.INTERNAL_SERVER_ERROR
        self._error(handler, status, str(error))

    @staticmethod
    def _error(
        handler: BaseHTTPRequestHandler, status: HTTPStatus, message: str
    ) -> None:
        OpenAIAPIServer._json(
            handler,
            status,
            {"error": {"message": message, "type": status.phrase}},
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Runs the HTTP service with a local Hugging Face model directory."""

    parser = argparse.ArgumentParser(description="Serve a Qwen2 model with LLAISYS")
    parser.add_argument("--model", required=True, help="local model directory")
    parser.add_argument("--device", choices=("cpu", "nvidia"), default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-id", default=None)
    args = parser.parse_args(argv)

    # Imported lazily so importing llaisys.server does not require transformers.
    from transformers import AutoTokenizer

    from .libllaisys import DeviceType
    from .models import Qwen2

    device = DeviceType.NVIDIA if args.device == "nvidia" else DeviceType.CPU
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    model = Qwen2(args.model, device)
    scheduler = RoundRobinScheduler(model)
    chat = ChatService(scheduler, tokenizer)
    server = OpenAIAPIServer(
        chat,
        model_id=args.model_id or args.model,
        host=args.host,
        port=args.port,
    )
    server.start()
    host, port = server.address
    print(f"LLAISYS is serving http://{host}:{port}", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return 0
    finally:
        server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
