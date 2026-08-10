"""Small dependency-free HTTP serving layer for LLAISYS.

This module intentionally provides APIs rather than a web UI, authentication,
or persistence. It exposes OpenAI-shaped completion endpoints plus explicit
in-memory session and request management.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import threading
import time
import uuid
import weakref
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Deque, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import parse_qs, unquote, urlparse

from .serving import (
    ChatMessage,
    ChatService,
    FinishReason,
    GenerationRequest,
    OrcaScheduler,
    RequestPool,
    RequestStatus,
    RoundRobinScheduler,
    ServiceOverloadedError,
    ServiceUnavailableError,
    TokenEvent,
)


LOGGER = logging.getLogger("llaisys.requests")


class ServingMetrics:
    """Thread-safe counters and latency aggregates for the HTTP layer."""

    def __init__(
        self,
        resource_provider: Optional[Callable[[], Mapping[str, object]]] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._started_at = time.time()
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
        self._generation_seconds_sum = 0.0
        self._completed_generated_tokens = 0
        self._terminal_requests: set[str] = set()
        self._ttft_samples: Deque[float] = deque(maxlen=10000)
        self._duration_samples: Deque[float] = deque(maxlen=10000)
        self._token_rate_samples: Deque[float] = deque(maxlen=10000)
        self._queue_samples: Deque[float] = deque(maxlen=10000)
        self._prefill_samples: Deque[float] = deque(maxlen=10000)
        self._decode_samples: Deque[float] = deque(maxlen=10000)
        self._resource_provider = resource_provider

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
                    ttft = now - request.created_at
                    self._ttft_sum += ttft
                    self._ttft_samples.append(ttft)
            if not event.finished or request.request_id in self._terminal_requests:
                return
            self._terminal_requests.add(request.request_id)
            self.requests_active = max(0, self.requests_active - 1)
            now = time.time()
            duration = now - request.created_at
            self._duration_sum += duration
            self._duration_samples.append(duration)
            first_token_at = self._first_token_at.get(request.request_id)
            if first_token_at is not None:
                generation_seconds = max(now - first_token_at, 1e-9)
                generated_tokens = len(request.generated_tokens)
                self._generation_seconds_sum += generation_seconds
                self._completed_generated_tokens += generated_tokens
                self._token_rate_samples.append(
                    generated_tokens / generation_seconds
                )
            if request.status is RequestStatus.FINISHED:
                self.requests_completed += 1
            elif request.status is RequestStatus.FAILED:
                self.requests_failed += 1
            else:
                self.requests_cancelled += 1
            queue_seconds = (
                request.started_at - request.created_at
                if request.started_at is not None
                else None
            )
            prefill_seconds = (
                request.prefill_finished_at - request.prefill_started_at
                if request.prefill_finished_at is not None
                and request.prefill_started_at is not None
                else None
            )
            decode_seconds = (
                request.finished_at - request.decode_started_at
                if request.finished_at is not None
                and request.decode_started_at is not None
                else None
            )
            for value, samples in (
                (queue_seconds, self._queue_samples),
                (prefill_seconds, self._prefill_samples),
                (decode_seconds, self._decode_samples),
            ):
                if value is not None:
                    samples.append(max(0.0, value))
            LOGGER.info(
                json.dumps(
                    {
                        "event": "request_finished",
                        "request_id": request.request_id,
                        "session_id": request.session_id,
                        "status": request.status.value,
                        "finish_reason": (
                            request.finish_reason.value
                            if request.finish_reason is not None
                            else None
                        ),
                        "timeout_phase": request.timeout_phase,
                        "prompt_tokens": (
                            request.context_length or len(request.input_tokens)
                        ),
                        "output_tokens": len(request.generated_tokens),
                        "queue_ms": self._milliseconds(queue_seconds),
                        "prefill_ms": self._milliseconds(prefill_seconds),
                        "decode_ms": self._milliseconds(decode_seconds),
                        "ttft_ms": self._milliseconds(
                            request.first_token_at - request.created_at
                            if request.first_token_at is not None
                            else None
                        ),
                        "total_ms": self._milliseconds(
                            request.finished_at - request.created_at
                            if request.finished_at is not None
                            else None
                        ),
                        "error": request.error,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            terminal = (
                self.requests_completed
                + self.requests_failed
                + self.requests_cancelled
            )
            elapsed = max(time.time() - self._started_at, 1e-9)
            snapshot = {
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
                "requests_per_second": terminal / elapsed,
                "generated_tokens_per_second": (
                    self._completed_generated_tokens
                    / self._generation_seconds_sum
                    if self._generation_seconds_sum
                    else 0.0
                ),
            }
            for name, samples in (
                ("ttft_seconds", self._ttft_samples),
                ("request_duration_seconds", self._duration_samples),
                ("request_token_rate", self._token_rate_samples),
                ("queue_seconds", self._queue_samples),
                ("prefill_seconds", self._prefill_samples),
                ("decode_seconds", self._decode_samples),
            ):
                for label, quantile in (("p50", 0.50), ("p95", 0.95), ("p99", 0.99)):
                    snapshot[f"{name}_{label}"] = self._percentile(
                        samples, quantile
                    )
            if self._resource_provider is not None:
                snapshot.update(self._resource_provider())
            return snapshot

    @staticmethod
    def _milliseconds(value: Optional[float]) -> Optional[float]:
        return None if value is None else value * 1000.0

    @staticmethod
    def _percentile(samples: Sequence[float], quantile: float) -> float:
        if not samples:
            return 0.0
        ordered = sorted(samples)
        position = (len(ordered) - 1) * quantile
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return float(ordered[lower])
        weight = position - lower
        return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)

    def prometheus(self) -> str:
        values = self.snapshot()
        lines = []
        counters = {
            "requests_total",
            "requests_completed",
            "requests_failed",
            "requests_cancelled",
            "prompt_tokens_total",
            "generated_tokens_total",
        }
        for name, value in values.items():
            metric_type = (
                "counter"
                if name in counters or name.endswith("_total")
                else "gauge"
            )
            metric_name = f"llaisys_{name}"
            lines.extend(
                [f"# TYPE {metric_name} {metric_type}", f"{metric_name} {value}"]
            )
        return "\n".join(lines) + "\n"


class _Server(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True
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
        self.metrics = ServingMetrics(
            resource_provider=self.scheduler.resource_snapshot
        )
        self._httpd = _Server((host, port), self._handler_type())
        self._thread: Optional[threading.Thread] = None
        self._lifecycle_lock = threading.RLock()
        self._accepting_requests = False

    @property
    def address(self) -> Tuple[str, int]:
        host, port = self._httpd.server_address[:2]
        return str(host), int(port)

    def start(self) -> None:
        self.scheduler.start()
        with self._lifecycle_lock:
            self._accepting_requests = True
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
        with self._lifecycle_lock:
            self._accepting_requests = True
        self._httpd.serve_forever()

    def begin_draining(self) -> None:
        with self._lifecycle_lock:
            self._accepting_requests = False

    def stop(
        self, graceful: bool = True, timeout_seconds: float = 30.0
    ) -> bool:
        self.begin_draining()
        self._httpd.shutdown()
        drained = (
            self.scheduler.wait_until_idle(timeout_seconds) if graceful else False
        )
        if not drained:
            self.scheduler.cancel_all()
            self.scheduler.wait_until_idle(timeout=1.0)
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join()
        self.scheduler.stop()
        return drained

    @property
    def accepting_requests(self) -> bool:
        with self._lifecycle_lock:
            return self._accepting_requests

    def _ensure_accepting(self) -> None:
        if not self.accepting_requests:
            raise ServiceUnavailableError(
                "Server is draining and no longer accepts new requests"
            )

    def _handler_type(self):
        owner_ref = weakref.ref(self)

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, _format, *_args):
                return

            def do_GET(self):
                owner = owner_ref()
                if owner is not None:
                    owner._handle_get(self)

            def do_POST(self):
                owner = owner_ref()
                if owner is not None:
                    owner._handle_post(self)

            def do_DELETE(self):
                owner = owner_ref()
                if owner is not None:
                    owner._handle_delete(self)

        return Handler

    def _handle_get(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            parsed = urlparse(handler.path)
            path = parsed.path
            if path == "/health":
                self._json(
                    handler,
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "accepting_requests": self.accepting_requests,
                    },
                )
                return
            if path == "/ready":
                if not self.accepting_requests:
                    raise ServiceUnavailableError("Server is draining")
                self._json(handler, HTTPStatus.OK, {"status": "ready"})
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
            if path == "/requests":
                query = parse_qs(parsed.query)
                status_filter = query.get("status", [None])[0]
                session_filter = query.get("session_id", [None])[0]
                if status_filter is not None:
                    try:
                        status = RequestStatus(status_filter)
                    except ValueError as error:
                        raise ValueError(
                            f"Unknown request status: {status_filter}"
                        ) from error
                else:
                    status = None
                raw_limit = query.get("limit", ["100"])[0]
                try:
                    limit = int(raw_limit)
                except ValueError as error:
                    raise ValueError("limit must be an integer") from error
                if not 1 <= limit <= 1000:
                    raise ValueError("limit must be between 1 and 1000")
                requests = sorted(
                    self.scheduler.request_pool.requests(),
                    key=lambda request: request.created_at,
                    reverse=True,
                )
                if status is not None:
                    requests = [
                        request for request in requests if request.status is status
                    ]
                if session_filter is not None:
                    requests = [
                        request
                        for request in requests
                        if request.session_id == session_filter
                    ]
                data = [self._request_json(request) for request in requests[:limit]]
                self._json(
                    handler,
                    HTTPStatus.OK,
                    {"object": "list", "count": len(data), "data": data},
                )
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
            is_cancel = path.startswith("/requests/") and path.endswith("/cancel")
            if not is_cancel:
                self._ensure_accepting()
            if path == "/v1/chat/completions":
                self._chat_completion(handler, body)
                return
            if path == "/v1/completions":
                self._text_completion(handler, body)
                return
            if path == "/v1/tokenize":
                self._tokenize(handler, body)
                return
            if path == "/v1/detokenize":
                self._detokenize(handler, body)
                return
            if path == "/sessions/import":
                raw_session = body.get("session", body)
                if not isinstance(raw_session, Mapping):
                    raise ValueError("session must be an object")
                new_session_id = self._optional_string(body, "new_session_id")
                if new_session_id == "":
                    raise ValueError("new_session_id must not be empty")
                session = self.chat.import_session(
                    raw_session, session_id=new_session_id
                )
                self._json(
                    handler,
                    HTTPStatus.CREATED,
                    self.chat.export_session(session.session_id),
                )
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
                                "logprobs": self._request_logprobs(request),
                            }
                        ],
                        "usage": self._usage(request),
                    },
                )
        finally:
            session = self.scheduler.sessions.get(session_id)
            if not session.busy:
                self.scheduler.sessions.delete(session_id)

    def _tokenize(
        self, handler: BaseHTTPRequestHandler, body: Mapping[str, object]
    ) -> None:
        self._validate_model(body)
        if "messages" in body:
            raw_messages = body["messages"]
            if not isinstance(raw_messages, Sequence) or isinstance(
                raw_messages, (str, bytes)
            ):
                raise ValueError("messages must be a sequence")
            messages = self._messages(raw_messages)
            add_generation_prompt = body.get("add_generation_prompt", True)
            if not isinstance(add_generation_prompt, bool):
                raise ValueError("add_generation_prompt must be a boolean")
            rendered = self.chat.tokenizer.apply_chat_template(
                [message.as_dict() for message in messages],
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
            )
            if not isinstance(rendered, str):
                raise TypeError("Chat tokenizer must return text")
            text = rendered
        else:
            text = body.get("text", body.get("prompt"))
            if not isinstance(text, str):
                raise ValueError("text must be a string")
        tokens = [int(token) for token in self.chat.tokenizer.encode(text)]
        self._json(
            handler,
            HTTPStatus.OK,
            {"object": "tokens", "tokens": tokens, "count": len(tokens)},
        )

    def _detokenize(
        self, handler: BaseHTTPRequestHandler, body: Mapping[str, object]
    ) -> None:
        self._validate_model(body)
        raw_tokens = body.get("tokens")
        if not isinstance(raw_tokens, Sequence) or isinstance(
            raw_tokens, (str, bytes)
        ):
            raise ValueError("tokens must be a sequence of integers")
        tokens = []
        for token in raw_tokens:
            if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                raise ValueError("tokens must contain non-negative integers")
            tokens.append(token)
        skip_special_tokens = body.get("skip_special_tokens", True)
        if not isinstance(skip_special_tokens, bool):
            raise ValueError("skip_special_tokens must be a boolean")
        text = self.chat.tokenizer.decode(
            tokens, skip_special_tokens=skip_special_tokens
        )
        self._json(
            handler,
            HTTPStatus.OK,
            {"object": "text", "text": text, "count": len(tokens)},
        )

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
                        "logprobs": self._request_logprobs(request),
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
                            "logprobs": self._event_logprobs(event),
                        }
                    ],
                }
                self._sse(handler, chunk)
            self._sse_done(handler)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self._cancel_disconnected_request(request, chat=True)

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
                                "logprobs": self._event_logprobs(event),
                            }
                        ],
                    },
                )
            self._sse_done(handler)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self._cancel_disconnected_request(request, chat=False)

    def _cancel_disconnected_request(
        self, request: GenerationRequest, chat: bool
    ) -> None:
        """Cancels work only when it is still live, then drains its terminal event."""

        if not self.scheduler.cancel(request.request_id):
            return
        events = (
            self.chat.events(request.request_id, timeout=None)
            if chat
            else self.scheduler.events(request.request_id, timeout=None)
        )
        for event in events:
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
        min_tokens = OpenAIAPIServer._integer(
            body, "min_tokens", 0, minimum=0
        )
        if min_tokens > max_tokens:
            raise ValueError("min_tokens must not exceed max_tokens")
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
        phase_timeouts = {}
        for name in (
            "queue_timeout_seconds",
            "prefill_timeout_seconds",
            "decode_timeout_seconds",
        ):
            value = body.get(name)
            if value is not None:
                value = OpenAIAPIServer._number(
                    body, name, 0.0, minimum=0.0, strict_minimum=True
                )
            phase_timeouts[name] = value
        truncate_prompt = body.get("truncate_prompt", False)
        if not isinstance(truncate_prompt, bool):
            raise ValueError("truncate_prompt must be a boolean")
        ignore_eos = body.get("ignore_eos", False)
        if not isinstance(ignore_eos, bool):
            raise ValueError("ignore_eos must be a boolean")
        raw_logprobs = body.get("logprobs", False)
        raw_top_logprobs = body.get("top_logprobs", 0)
        if isinstance(raw_logprobs, bool):
            if isinstance(raw_top_logprobs, bool) or not isinstance(
                raw_top_logprobs, int
            ):
                raise ValueError("top_logprobs must be an integer")
            if not 0 <= raw_top_logprobs <= 20:
                raise ValueError("top_logprobs must be between 0 and 20")
            if raw_top_logprobs and not raw_logprobs:
                raise ValueError("top_logprobs requires logprobs=true")
            logprobs = max(1, raw_top_logprobs) if raw_logprobs else 0
        elif isinstance(raw_logprobs, int):
            if not 0 <= raw_logprobs <= 20:
                raise ValueError("logprobs must be between 0 and 20")
            if raw_top_logprobs:
                raise ValueError(
                    "top_logprobs is only valid when logprobs is a boolean"
                )
            logprobs = raw_logprobs
        else:
            raise ValueError("logprobs must be a boolean or integer")
        return {
            "max_new_tokens": max_tokens,
            "min_new_tokens": min_tokens,
            "priority": priority,
            "stop_token_ids": tuple(stop_tokens),
            "stop_strings": stop_strings,
            "truncate_prompt": truncate_prompt,
            "ignore_eos": ignore_eos,
            "logprobs": logprobs,
            "timeout_seconds": timeout,
            **phase_timeouts,
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

    def _format_token_logprobs(
        self, token_logprobs: Mapping[str, object]
    ) -> Dict[str, object]:
        token_id = int(token_logprobs["token_id"])
        token = self.chat.tokenizer.decode(
            [token_id], skip_special_tokens=False
        )
        top = []
        for candidate in token_logprobs.get("top_logprobs", []):
            candidate_id = int(candidate["token_id"])
            candidate_text = self.chat.tokenizer.decode(
                [candidate_id], skip_special_tokens=False
            )
            top.append(
                {
                    "token": candidate_text,
                    "token_id": candidate_id,
                    "logprob": float(candidate["logprob"]),
                    "bytes": list(candidate_text.encode("utf-8")),
                }
            )
        return {
            "token": token,
            "token_id": token_id,
            "logprob": float(token_logprobs["logprob"]),
            "bytes": list(token.encode("utf-8")),
            "top_logprobs": top,
        }

    def _event_logprobs(
        self, event: TokenEvent
    ) -> Optional[Dict[str, object]]:
        if event.logprobs is None:
            return None
        return {"content": [self._format_token_logprobs(event.logprobs)]}

    def _request_logprobs(
        self, request: GenerationRequest
    ) -> Optional[Dict[str, object]]:
        if not request.generated_logprobs:
            return None
        return {
            "content": [
                self._format_token_logprobs(token_logprobs)
                for token_logprobs in request.generated_logprobs
            ]
        }

    @classmethod
    def _request_json(cls, request: GenerationRequest) -> Dict[str, object]:
        now = time.time()
        queue_seconds = (
            request.started_at - request.created_at
            if request.started_at is not None
            else None
        )
        ttft_seconds = (
            request.first_token_at - request.created_at
            if request.first_token_at is not None
            else None
        )
        total_seconds = (
            request.finished_at - request.created_at
            if request.finished_at is not None
            else None
        )
        generation_seconds = (
            request.finished_at - request.first_token_at
            if request.finished_at is not None
            and request.first_token_at is not None
            else None
        )
        return {
            "request_id": request.request_id,
            "session_id": request.session_id,
            "status": request.status.value,
            "phase": request.phase.value,
            "finish_reason": (
                request.finish_reason.value if request.finish_reason else None
            ),
            "timeout_phase": request.timeout_phase,
            "generated_tokens": list(request.generated_tokens),
            "logprobs": list(request.generated_logprobs),
            "output_text": request.streamed_text,
            "usage": cls._usage(request),
            "timing": {
                "queue_seconds": queue_seconds,
                "prefill_seconds": (
                    request.prefill_finished_at - request.prefill_started_at
                    if request.prefill_finished_at is not None
                    and request.prefill_started_at is not None
                    else None
                ),
                "decode_seconds": (
                    request.finished_at - request.decode_started_at
                    if request.finished_at is not None
                    and request.decode_started_at is not None
                    else None
                ),
                "ttft_seconds": ttft_seconds,
                "generation_seconds": generation_seconds,
                "total_seconds": total_seconds,
                "elapsed_seconds": now - request.created_at,
            },
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
        if isinstance(error, ServiceOverloadedError):
            status = HTTPStatus.TOO_MANY_REQUESTS
        elif isinstance(error, ServiceUnavailableError):
            status = HTTPStatus.SERVICE_UNAVAILABLE
        elif isinstance(error, KeyError):
            status = HTTPStatus.NOT_FOUND
        elif isinstance(error, (ValueError, TypeError)):
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
    parser.add_argument("--max-queue-size", type=int, default=256)
    parser.add_argument("--max-active-requests", type=int, default=32)
    parser.add_argument(
        "--scheduler", choices=("orca", "round-robin"), default="orca"
    )
    parser.add_argument("--max-prefill-per-iteration", type=int, default=1)
    parser.add_argument(
        "--max-kv-cache-mib",
        type=int,
        default=0,
        help="KV cache reservation budget in MiB; 0 disables the limit",
    )
    parser.add_argument(
        "--kv-cache-high-watermark", type=float, default=0.9
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args(argv)
    if args.max_kv_cache_mib < 0:
        parser.error("--max-kv-cache-mib must be non-negative")
    if not 0.0 < args.kv_cache_high_watermark <= 1.0:
        parser.error("--kv-cache-high-watermark must be in (0, 1]")
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Imported lazily so importing llaisys.server does not require transformers.
    from transformers import AutoTokenizer

    from .libllaisys import DeviceType
    from .models import Qwen2

    device = DeviceType.NVIDIA if args.device == "nvidia" else DeviceType.CPU
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    model = Qwen2(args.model, device)
    request_pool = RequestPool(max_pending_requests=args.max_queue_size)
    if args.scheduler == "orca":
        scheduler = OrcaScheduler(
            model,
            request_pool=request_pool,
            max_active_requests=args.max_active_requests,
            max_prefill_per_iteration=args.max_prefill_per_iteration,
            max_kv_cache_bytes=(
                args.max_kv_cache_mib * 1024 * 1024
                if args.max_kv_cache_mib > 0
                else None
            ),
            kv_cache_high_watermark=args.kv_cache_high_watermark,
        )
    else:
        scheduler = RoundRobinScheduler(
            model,
            request_pool=request_pool,
            max_active_requests=args.max_active_requests,
        )
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
