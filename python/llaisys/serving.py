"""Lightweight request, session, and scheduling primitives for LLAISYS serving.

The current Qwen2 backend owns a single KV cache, so one model instance cannot
interleave decode steps from different sequences.  This module therefore uses a
thread-safe request pool and a single-worker scheduler.  It provides correct
multi-session semantics today and leaves continuous batching to a future
multi-sequence KV-cache backend.
"""

from __future__ import annotations

import heapq
import itertools
import math
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Deque, Dict, Iterator, List, Mapping, Optional, Protocol, Sequence, Tuple


class GenerationModel(Protocol):
    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: Optional[int] = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
        repetition_penalty: float = 1.0,
        seed: int = 0,
    ) -> Sequence[int]: ...


class SequenceBatchModel(GenerationModel, Protocol):
    supports_sequence_batching: bool

    def create_sequence(self, sequence_id: str, capacity: int) -> None: ...

    def destroy_sequence(self, sequence_id: str) -> None: ...

    def prefill_sequence(
        self,
        sequence_id: str,
        input_tokens: Sequence[int],
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
        repetition_penalty: float = 1.0,
        seed: int = 0,
    ) -> int: ...

    def decode_batch(
        self,
        sequence_ids: Sequence[str],
        token_ids: Sequence[int],
        sampling_configs: Sequence[Mapping[str, object]],
    ) -> Sequence[int]: ...


class ChatTokenizer(Protocol):
    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, str]],
        add_generation_prompt: bool = True,
        tokenize: bool = False,
    ) -> object: ...

    def encode(self, text: str) -> Sequence[int]: ...

    def decode(
        self, token_ids: Sequence[int], skip_special_tokens: bool = True
    ) -> str: ...


class RequestStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RequestPhase(str, Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"


class FinishReason(str, Enum):
    EOS = "eos"
    STOP = "stop"
    LENGTH = "length"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    ERROR = "error"


class ServiceOverloadedError(RuntimeError):
    """Raised when the configured waiting queue has reached capacity."""


class ServiceUnavailableError(RuntimeError):
    """Raised when the server is draining or no longer accepting work."""


class KVCacheBudget:
    """Thread-safe reservation accounting for per-sequence KV cache memory."""

    def __init__(
        self, max_bytes: Optional[int] = None, high_watermark: float = 0.9
    ) -> None:
        if max_bytes is not None and (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes <= 0
        ):
            raise ValueError("max_bytes must be a positive integer")
        if not math.isfinite(high_watermark) or not 0.0 < high_watermark <= 1.0:
            raise ValueError("high_watermark must be in (0, 1]")
        self.max_bytes = max_bytes
        self.high_watermark = float(high_watermark)
        self._reservations: Dict[str, int] = {}
        self._lock = threading.RLock()
        self.rejections_total = 0

    @property
    def limit_bytes(self) -> Optional[int]:
        if self.max_bytes is None:
            return None
        return int(self.max_bytes * self.high_watermark)

    def reserve(self, request_id: str, size_bytes: int) -> None:
        if size_bytes < 0:
            raise ValueError("KV cache reservation must be non-negative")
        with self._lock:
            if request_id in self._reservations:
                raise ValueError(f"KV cache is already reserved: {request_id}")
            limit = self.limit_bytes
            used = sum(self._reservations.values())
            if limit is not None and used + size_bytes > limit:
                self.rejections_total += 1
                raise ServiceOverloadedError(
                    "KV cache budget exceeded: "
                    f"requested={size_bytes} available={max(0, limit - used)}"
                )
            self._reservations[request_id] = size_bytes

    def release(self, request_id: str) -> int:
        with self._lock:
            return self._reservations.pop(request_id, 0)

    def snapshot(self) -> Dict[str, int | float]:
        with self._lock:
            reserved = sum(self._reservations.values())
            limit = self.limit_bytes
            return {
                "kv_cache_capacity_bytes": self.max_bytes or 0,
                "kv_cache_limit_bytes": limit or 0,
                "kv_cache_reserved_bytes": reserved,
                "kv_cache_available_bytes": (
                    max(0, limit - reserved) if limit is not None else 0
                ),
                "kv_cache_active_reservations": len(self._reservations),
                "kv_cache_high_watermark": self.high_watermark,
                "kv_cache_rejections_total": self.rejections_total,
            }


@dataclass
class GenerationRequest:
    request_id: str
    session_id: str
    input_tokens: Tuple[int, ...]
    max_new_tokens: int
    min_new_tokens: int = 0
    priority: int = 0
    top_k: int = 1
    top_p: float = 0.8
    temperature: float = 0.8
    repetition_penalty: float = 1.0
    seed: int = 0
    ignore_eos: bool = False
    stop_token_ids: Tuple[int, ...] = ()
    stop_strings: Tuple[str, ...] = ()
    truncate_prompt: bool = False
    timeout_seconds: Optional[float] = None
    queue_timeout_seconds: Optional[float] = None
    prefill_timeout_seconds: Optional[float] = None
    decode_timeout_seconds: Optional[float] = None
    status: RequestStatus = RequestStatus.WAITING
    phase: RequestPhase = RequestPhase.WAITING
    finish_reason: Optional[FinishReason] = None
    timeout_phase: Optional[str] = None
    cancel_requested: bool = False
    context_length: Optional[int] = None
    context_tokens: Optional[Tuple[int, ...]] = None
    generated_token_ids: List[int] = field(default_factory=list)
    streamed_text: str = ""
    output_tokens: Optional[Tuple[int, ...]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    prefill_started_at: Optional[float] = None
    prefill_finished_at: Optional[float] = None
    decode_started_at: Optional[float] = None
    first_token_at: Optional[float] = None
    finished_at: Optional[float] = None

    @property
    def generated_tokens(self) -> Tuple[int, ...]:
        if self.generated_token_ids:
            return tuple(self.generated_token_ids)
        if self.output_tokens is not None and self.context_length is not None:
            return self.output_tokens[self.context_length :]
        return ()

    @property
    def deadline(self) -> Optional[float]:
        if self.timeout_seconds is None:
            return None
        return self.created_at + self.timeout_seconds

    @property
    def queue_deadline(self) -> Optional[float]:
        if self.queue_timeout_seconds is None:
            return None
        return self.created_at + self.queue_timeout_seconds

    @property
    def prefill_deadline(self) -> Optional[float]:
        if (
            self.prefill_timeout_seconds is None
            or self.prefill_started_at is None
        ):
            return None
        return self.prefill_started_at + self.prefill_timeout_seconds

    @property
    def decode_deadline(self) -> Optional[float]:
        if self.decode_timeout_seconds is None or self.decode_started_at is None:
            return None
        return self.decode_started_at + self.decode_timeout_seconds


@dataclass(frozen=True)
class TokenEvent:
    request_id: str
    session_id: str
    token_id: Optional[int] = None
    text: Optional[str] = None
    finished: bool = False
    finish_reason: Optional[FinishReason] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported chat role: {self.role}")

    def as_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class Session:
    session_id: str
    user_id: Optional[str] = None
    token_history: List[int] = field(default_factory=list)
    messages: List[ChatMessage] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)
    busy: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class SessionManager:
    """Stores independent token histories for users or conversations."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.RLock()

    def create(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        initial_tokens: Sequence[int] = (),
        initial_messages: Sequence[ChatMessage] = (),
        metadata: Optional[Mapping[str, object]] = None,
    ) -> Session:
        session_id = session_id or uuid.uuid4().hex
        with self._lock:
            if session_id in self._sessions:
                raise ValueError(f"Session already exists: {session_id}")
            session = Session(
                session_id=session_id,
                user_id=user_id,
                token_history=[int(token) for token in initial_tokens],
                messages=list(initial_messages),
                metadata=dict(metadata or {}),
            )
            self._sessions[session_id] = session
            return self._snapshot(session)

    def get(self, session_id: str) -> Session:
        with self._lock:
            return self._snapshot(self._require(session_id))

    def list(self, user_id: Optional[str] = None) -> List[Session]:
        with self._lock:
            sessions = self._sessions.values()
            if user_id is not None:
                sessions = (
                    session for session in sessions if session.user_id == user_id
                )
            return [self._snapshot(session) for session in sessions]

    def clear(self, session_id: str) -> None:
        with self._lock:
            session = self._require(session_id)
            if session.busy:
                raise RuntimeError(f"Session has a pending request: {session_id}")
            session.token_history.clear()
            session.messages.clear()
            session.updated_at = time.time()

    def set_token_history(
        self, session_id: str, token_history: Sequence[int]
    ) -> None:
        with self._lock:
            session = self._require(session_id)
            if session.busy:
                raise RuntimeError(f"Session has a pending request: {session_id}")
            session.token_history = [int(token) for token in token_history]
            session.updated_at = time.time()

    def append_message(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            session = self._require(session_id)
            if session.busy:
                raise RuntimeError(f"Session has a pending request: {session_id}")
            session.messages.append(ChatMessage(role=role, content=str(content)))
            session.updated_at = time.time()

    def replace_messages(
        self, session_id: str, messages: Sequence[ChatMessage]
    ) -> None:
        with self._lock:
            session = self._require(session_id)
            if session.busy:
                raise RuntimeError(f"Session has a pending request: {session_id}")
            session.messages = list(messages)
            session.updated_at = time.time()

    def delete(self, session_id: str) -> None:
        with self._lock:
            session = self._require(session_id)
            if session.busy:
                raise RuntimeError(f"Session has a pending request: {session_id}")
            del self._sessions[session_id]

    def acquire(self, session_id: str) -> None:
        with self._lock:
            session = self._require(session_id)
            if session.busy:
                raise RuntimeError(
                    f"Session already has a pending request: {session_id}"
                )
            session.busy = True
            session.updated_at = time.time()

    def release(self, session_id: str) -> None:
        with self._lock:
            session = self._require(session_id)
            session.busy = False
            session.updated_at = time.time()

    def build_context(
        self, session_id: str, input_tokens: Sequence[int]
    ) -> List[int]:
        with self._lock:
            session = self._require(session_id)
            return session.token_history + [int(token) for token in input_tokens]

    def commit(self, session_id: str, output_tokens: Sequence[int]) -> None:
        with self._lock:
            session = self._require(session_id)
            session.token_history = [int(token) for token in output_tokens]
            session.updated_at = time.time()

    def _require(self, session_id: str) -> Session:
        try:
            return self._sessions[session_id]
        except KeyError as error:
            raise KeyError(f"Unknown session: {session_id}") from error

    @staticmethod
    def _snapshot(session: Session) -> Session:
        return replace(
            session,
            token_history=list(session.token_history),
            messages=list(session.messages),
            metadata=dict(session.metadata),
        )


class RequestPool:
    """Thread-safe priority FIFO queue.

    Higher priority values run first. Requests with equal priority preserve
    submission order.
    """

    def __init__(self, max_pending_requests: Optional[int] = None) -> None:
        if max_pending_requests is not None and max_pending_requests <= 0:
            raise ValueError("max_pending_requests must be positive")
        self._heap: List[Tuple[int, int, str]] = []
        self._requests: Dict[str, GenerationRequest] = {}
        self._sequence = itertools.count()
        self._lock = threading.RLock()
        self.max_pending_requests = max_pending_requests

    def submit(self, request: GenerationRequest) -> None:
        with self._lock:
            if request.request_id in self._requests:
                raise ValueError(f"Request already exists: {request.request_id}")
            if request.status is not RequestStatus.WAITING:
                raise ValueError("Only waiting requests may enter the request pool")
            if (
                self.max_pending_requests is not None
                and self.pending_count() >= self.max_pending_requests
            ):
                raise ServiceOverloadedError(
                    "Request queue is full; retry after an active request finishes"
                )
            self._requests[request.request_id] = request
            heapq.heappush(
                self._heap,
                (-request.priority, next(self._sequence), request.request_id),
            )

    def pop_next(self) -> Optional[GenerationRequest]:
        with self._lock:
            while self._heap:
                _, _, request_id = heapq.heappop(self._heap)
                request = self._requests[request_id]
                if request.status is not RequestStatus.WAITING:
                    continue
                request.status = RequestStatus.RUNNING
                request.started_at = time.time()
                return request
            return None

    def get(self, request_id: str) -> GenerationRequest:
        with self._lock:
            try:
                return self._requests[request_id]
            except KeyError as error:
                raise KeyError(f"Unknown request: {request_id}") from error

    def cancel(self, request_id: str) -> bool:
        with self._lock:
            request = self.get(request_id)
            if request.status is not RequestStatus.WAITING:
                return False
            request.status = RequestStatus.CANCELLED
            request.finished_at = time.time()
            return True

    def pending_count(self) -> int:
        with self._lock:
            return sum(
                request.status is RequestStatus.WAITING
                for request in self._requests.values()
            )

    def requests(self) -> List[GenerationRequest]:
        with self._lock:
            return list(self._requests.values())

    def unfinished_count(self) -> int:
        with self._lock:
            return sum(
                request.status in (RequestStatus.WAITING, RequestStatus.RUNNING)
                for request in self._requests.values()
            )


class RequestScheduler:
    """Schedules requests on one Qwen2 model instance.

    Requests from different sessions share model weights but run sequentially.
    Each request rebuilds its session context during prefill because the current
    backend exposes only one KV cache. The scheduler is safe to submit to from
    multiple threads, while ``run_once`` is serialized by a worker lock.
    """

    supports_continuous_batching = False

    def __init__(
        self,
        model: GenerationModel,
        sessions: Optional[SessionManager] = None,
        request_pool: Optional[RequestPool] = None,
        max_sequence_length: Optional[int] = None,
    ) -> None:
        self.model = model
        self.sessions = sessions or SessionManager()
        self.request_pool = request_pool or RequestPool()
        self._worker_lock = threading.Lock()
        model_limit = getattr(model, "max_sequence_length", None)
        self.max_sequence_length = (
            int(max_sequence_length)
            if max_sequence_length is not None
            else (int(model_limit) if model_limit is not None else None)
        )
        if self.max_sequence_length is not None and self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")

    def submit(
        self,
        session_id: str,
        input_tokens: Sequence[int],
        max_new_tokens: int = 128,
        min_new_tokens: int = 0,
        priority: int = 0,
        request_id: Optional[str] = None,
        stop_token_ids: Sequence[int] = (),
        stop_strings: Sequence[str] = (),
        truncate_prompt: bool = False,
        timeout_seconds: Optional[float] = None,
        queue_timeout_seconds: Optional[float] = None,
        prefill_timeout_seconds: Optional[float] = None,
        decode_timeout_seconds: Optional[float] = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
        repetition_penalty: float = 1.0,
        seed: int = 0,
        ignore_eos: bool = False,
    ) -> GenerationRequest:
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise ValueError("max_new_tokens must be an integer")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if isinstance(min_new_tokens, bool) or not isinstance(min_new_tokens, int):
            raise ValueError("min_new_tokens must be an integer")
        if not 0 <= min_new_tokens <= max_new_tokens:
            raise ValueError("min_new_tokens must be between 0 and max_new_tokens")
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise ValueError("priority must be an integer")
        for timeout_name, timeout_value in (
            ("timeout_seconds", timeout_seconds),
            ("queue_timeout_seconds", queue_timeout_seconds),
            ("prefill_timeout_seconds", prefill_timeout_seconds),
            ("decode_timeout_seconds", decode_timeout_seconds),
        ):
            if timeout_value is None:
                continue
            if isinstance(timeout_value, bool) or not isinstance(
                timeout_value, (int, float)
            ):
                raise ValueError(f"{timeout_name} must be a number")
            if not math.isfinite(timeout_value) or timeout_value <= 0:
                raise ValueError(
                    f"{timeout_name} must be positive and finite"
                )
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError("top_k must be an integer")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if (
            isinstance(top_p, bool)
            or not isinstance(top_p, (int, float))
            or not math.isfinite(top_p)
            or not 0.0 < top_p <= 1.0
        ):
            raise ValueError("top_p must be in (0, 1]")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature <= 0.0
        ):
            raise ValueError("temperature must be positive and finite")
        if (
            isinstance(repetition_penalty, bool)
            or not isinstance(repetition_penalty, (int, float))
            or not math.isfinite(repetition_penalty)
            or repetition_penalty <= 0.0
        ):
            raise ValueError("repetition_penalty must be positive and finite")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if not 0 <= seed < 2**64:
            raise ValueError("seed must fit in an unsigned 64-bit integer")
        if not isinstance(ignore_eos, bool):
            raise ValueError("ignore_eos must be a boolean")
        if any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in input_tokens
        ):
            raise ValueError("input_tokens must contain non-negative integers")
        if any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in stop_token_ids
        ):
            raise ValueError(
                "stop_token_ids must contain non-negative integers"
            )
        if isinstance(stop_strings, (str, bytes)) or any(
            not isinstance(stop, str) for stop in stop_strings
        ):
            raise ValueError("stop_strings must be a sequence of strings")
        tokens = tuple(input_tokens)
        stop_tokens = tuple(stop_token_ids)
        normalized_stops = tuple(dict.fromkeys(stop_strings))
        if any(not stop for stop in normalized_stops):
            raise ValueError("stop strings must not be empty")
        if not isinstance(truncate_prompt, bool):
            raise ValueError("truncate_prompt must be a boolean")
        session = self.sessions.get(session_id)
        self._validate_context_length(
            len(session.token_history) + len(tokens),
            max_new_tokens,
            truncate_prompt,
        )
        self.sessions.acquire(session_id)
        request = GenerationRequest(
            request_id=request_id or uuid.uuid4().hex,
            session_id=session_id,
            input_tokens=tokens,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            priority=priority,
            top_k=top_k,
            top_p=float(top_p),
            temperature=float(temperature),
            repetition_penalty=float(repetition_penalty),
            seed=seed,
            ignore_eos=ignore_eos,
            stop_token_ids=stop_tokens,
            stop_strings=normalized_stops,
            truncate_prompt=truncate_prompt,
            timeout_seconds=timeout_seconds,
            queue_timeout_seconds=queue_timeout_seconds,
            prefill_timeout_seconds=prefill_timeout_seconds,
            decode_timeout_seconds=decode_timeout_seconds,
        )
        try:
            self.request_pool.submit(request)
        except Exception:
            self.sessions.release(session_id)
            raise
        return request

    def _validate_context_length(
        self, context_length: int, max_new_tokens: int, truncate_prompt: bool
    ) -> None:
        if self.max_sequence_length is None:
            return
        available = self.max_sequence_length - max_new_tokens
        if available <= 0:
            raise ValueError(
                "max_new_tokens must be smaller than max_sequence_length "
                f"({self.max_sequence_length})"
            )
        if context_length > available and not truncate_prompt:
            raise ValueError(
                f"Prompt has {context_length} tokens, but at most {available} "
                "tokens fit with the requested output; set truncate_prompt=true "
                "to left-truncate the prompt"
            )

    def _build_context(self, request: GenerationRequest) -> Tuple[int, ...]:
        context = tuple(
            self.sessions.build_context(request.session_id, request.input_tokens)
        )
        if not context:
            raise ValueError("A generation request needs at least one token")
        if self.max_sequence_length is not None:
            available = self.max_sequence_length - request.max_new_tokens
            if len(context) > available:
                if not request.truncate_prompt:
                    self._validate_context_length(
                        len(context), request.max_new_tokens, False
                    )
                context = context[-available:]
        return context

    def cancel(self, request_id: str) -> bool:
        request = self.request_pool.get(request_id)
        cancelled = self.request_pool.cancel(request_id)
        if cancelled:
            self.sessions.release(request.session_id)
        return cancelled

    def run_once(self, raise_on_error: bool = False) -> Optional[GenerationRequest]:
        with self._worker_lock:
            request = self.request_pool.pop_next()
            if request is None:
                return None

            try:
                context = self._build_context(request)
                request.context_length = len(context)
                output = tuple(
                    int(token)
                    for token in self.model.generate(
                        context,
                        max_new_tokens=request.max_new_tokens,
                        top_k=request.top_k,
                        top_p=request.top_p,
                        temperature=request.temperature,
                        repetition_penalty=request.repetition_penalty,
                        seed=request.seed,
                    )
                )
                if output[: len(context)] != tuple(context):
                    raise RuntimeError(
                        "The model must return the input context followed by generated tokens"
                    )
                request.output_tokens = output
                if len(output) > len(context):
                    request.first_token_at = time.time()
                request.status = RequestStatus.FINISHED
                self.sessions.commit(request.session_id, output)
            except Exception as error:
                request.status = RequestStatus.FAILED
                request.error = str(error)
                if raise_on_error:
                    raise
            finally:
                request.finished_at = time.time()
                self.sessions.release(request.session_id)
            return request

    def run_until_idle(self, raise_on_error: bool = False) -> List[GenerationRequest]:
        completed: List[GenerationRequest] = []
        while True:
            request = self.run_once(raise_on_error=raise_on_error)
            if request is None:
                return completed
            completed.append(request)


class RoundRobinScheduler(RequestScheduler):
    """Function-first token scheduler with streaming and cancellation.

    The current backend has one KV cache, so each token step recomputes the
    selected request from its complete context. This is intentionally slow but
    provides correct round-robin semantics without duplicating model weights.
    A future multi-sequence backend can replace ``_generate_one`` while keeping
    the request, session, and event APIs unchanged.
    """

    supports_continuous_batching = False

    def __init__(
        self,
        model: GenerationModel,
        sessions: Optional[SessionManager] = None,
        request_pool: Optional[RequestPool] = None,
        token_decoder: Optional[Callable[[int], str]] = None,
        sequence_decoder: Optional[Callable[[Sequence[int]], str]] = None,
        commit_partial_on_abort: bool = True,
        max_sequence_length: Optional[int] = None,
        max_active_requests: Optional[int] = None,
    ) -> None:
        if max_active_requests is not None and max_active_requests <= 0:
            raise ValueError("max_active_requests must be positive")
        super().__init__(
            model,
            sessions=sessions,
            request_pool=request_pool,
            max_sequence_length=max_sequence_length,
        )
        self._active: Deque[str] = deque()
        self._event_queues: Dict[str, "queue.Queue[TokenEvent]"] = {}
        self._events_lock = threading.RLock()
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._token_decoder = token_decoder
        self._sequence_decoder = sequence_decoder
        self._commit_partial_on_abort = commit_partial_on_abort
        self._eos_token_id = getattr(model, "eos_token_id", None)
        self.max_active_requests = max_active_requests
        self.backend_errors_total = 0
        self.backend_oom_total = 0

    def submit(
        self,
        session_id: str,
        input_tokens: Sequence[int],
        max_new_tokens: int = 128,
        min_new_tokens: int = 0,
        priority: int = 0,
        request_id: Optional[str] = None,
        stop_token_ids: Sequence[int] = (),
        stop_strings: Sequence[str] = (),
        truncate_prompt: bool = False,
        timeout_seconds: Optional[float] = None,
        queue_timeout_seconds: Optional[float] = None,
        prefill_timeout_seconds: Optional[float] = None,
        decode_timeout_seconds: Optional[float] = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
        repetition_penalty: float = 1.0,
        seed: int = 0,
        ignore_eos: bool = False,
    ) -> GenerationRequest:
        request_id = request_id or uuid.uuid4().hex
        with self._events_lock:
            if request_id in self._event_queues:
                raise ValueError(f"Request already exists: {request_id}")
            self._event_queues[request_id] = queue.Queue()
        try:
            request = super().submit(
                session_id=session_id,
                input_tokens=input_tokens,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                priority=priority,
                request_id=request_id,
                stop_token_ids=stop_token_ids,
                stop_strings=stop_strings,
                truncate_prompt=truncate_prompt,
                timeout_seconds=timeout_seconds,
                queue_timeout_seconds=queue_timeout_seconds,
                prefill_timeout_seconds=prefill_timeout_seconds,
                decode_timeout_seconds=decode_timeout_seconds,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                seed=seed,
                ignore_eos=ignore_eos,
            )
            self._reserve_request_resources(request)
        except Exception:
            try:
                request = self.request_pool.get(request_id)
            except KeyError:
                request = None
            if request is not None and request.status is RequestStatus.WAITING:
                self.request_pool.cancel(request_id)
                self.sessions.release(request.session_id)
            with self._events_lock:
                del self._event_queues[request_id]
            raise
        self._wake_event.set()
        return request

    def cancel(self, request_id: str) -> bool:
        request = self.request_pool.get(request_id)
        if request.status is RequestStatus.WAITING:
            if not self.request_pool.cancel(request_id):
                return False
            request.finish_reason = FinishReason.CANCELLED
            self.sessions.release(request.session_id)
            self._release_request_resources(request)
            self._publish(self._terminal_event(request))
            return True
        if request.status is RequestStatus.RUNNING:
            request.cancel_requested = True
            self._wake_event.set()
            return True
        return False

    def step(self) -> Optional[TokenEvent]:
        """Advances one active request by at most one generated token."""

        with self._worker_lock:
            expired = self._expire_waiting_requests()
            if expired:
                return expired[0]
            self._admit_waiting()
            while self._active:
                request = self.request_pool.get(self._active.popleft())
                if request.status is not RequestStatus.RUNNING:
                    continue

                if request.cancel_requested:
                    return self._finalize_abort(request, FinishReason.CANCELLED)
                if timeout_phase := self._expired_timeout_phase(request):
                    return self._finalize_timeout(request, timeout_phase)

                try:
                    if request.context_tokens is None:
                        context = self._build_context(request)
                        request.context_tokens = context
                        request.context_length = len(context)

                    is_prefill = not request.generated_token_ids
                    if is_prefill:
                        request.phase = RequestPhase.PREFILL
                        request.prefill_started_at = (
                            request.prefill_started_at or time.time()
                        )
                    else:
                        request.phase = RequestPhase.DECODE
                        request.decode_started_at = (
                            request.decode_started_at or time.time()
                        )

                    token_id = self._generate_one(request)
                    if request.cancel_requested:
                        return self._finalize_abort(
                            request, FinishReason.CANCELLED
                        )
                    if is_prefill:
                        request.prefill_finished_at = time.time()
                    if timeout_phase := self._expired_timeout_phase(request):
                        return self._finalize_timeout(request, timeout_phase)

                    request.generated_token_ids.append(token_id)
                    if request.first_token_at is None:
                        request.first_token_at = time.time()
                    reason = self._finish_reason(request, token_id)
                    text, stopped_by_string = self._decode_increment(
                        request, force_flush=reason is not None
                    )
                    if stopped_by_string:
                        reason = FinishReason.STOP
                    if reason is not None:
                        return self._finalize_success(
                            request, reason, token_id=token_id, text=text
                        )

                    self._active.append(request.request_id)
                    event = TokenEvent(
                        request_id=request.request_id,
                        session_id=request.session_id,
                        token_id=token_id,
                        text=text,
                    )
                    self._publish(event)
                    return event
                except Exception as error:
                    return self._finalize_error(request, error)
            return None

    def _decode_increment(
        self, request: GenerationRequest, force_flush: bool = False
    ) -> Tuple[Optional[str], bool]:
        if self._sequence_decoder is not None:
            decoded = self._sequence_decoder(request.generated_token_ids)
        elif self._token_decoder is not None:
            decoded = "".join(
                self._token_decoder(token)
                for token in request.generated_token_ids
            )
        else:
            return None, False

        stop_positions = [
            position
            for stop in request.stop_strings
            if (position := decoded.find(stop)) >= 0
        ]
        stopped = bool(stop_positions)
        if stopped:
            visible = decoded[: min(stop_positions)]
        else:
            visible = decoded
            if request.stop_strings and not force_flush:
                held = max(
                    (
                        size
                        for stop in request.stop_strings
                        for size in range(1, min(len(stop), len(decoded)) + 1)
                        if decoded.endswith(stop[:size])
                    ),
                    default=0,
                )
                if held:
                    visible = decoded[:-held]

        if visible.startswith(request.streamed_text):
            delta = visible[len(request.streamed_text) :]
        else:
            delta = visible
        request.streamed_text = visible
        return delta or None, stopped

    def run_until_idle_stream(self) -> Iterator[TokenEvent]:
        """Runs synchronously and yields token or terminal events."""

        while True:
            event = self.step()
            if event is None:
                return
            yield event

    def events(
        self, request_id: str, timeout: Optional[float] = None
    ) -> Iterator[TokenEvent]:
        """Consumes events for one request, normally while a worker is running."""

        with self._events_lock:
            try:
                event_queue = self._event_queues[request_id]
            except KeyError as error:
                raise KeyError(f"Unknown request: {request_id}") from error
        while True:
            try:
                event = event_queue.get(timeout=timeout)
            except queue.Empty as error:
                raise TimeoutError(
                    f"Timed out waiting for request events: {request_id}"
                ) from error
            yield event
            if event.finished:
                return

    def start(self) -> None:
        """Starts the optional background worker."""

        if self._worker is not None and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="llaisys-round-robin-scheduler",
            daemon=True,
        )
        self._worker.start()

    def stop(self, wait: bool = True) -> None:
        """Stops the worker; queued requests remain available for a restart."""

        self._stop_event.set()
        self._wake_event.set()
        if wait and self._worker is not None:
            self._worker.join()

    def cancel_all(self) -> None:
        for request in self.request_pool.requests():
            if request.status in (RequestStatus.WAITING, RequestStatus.RUNNING):
                self.cancel(request.request_id)
        while self.step() is not None:
            pass

    def wait_until_idle(self, timeout: Optional[float] = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.request_pool.unfinished_count():
            if deadline is not None and time.monotonic() >= deadline:
                return False
            self._wake_event.set()
            self._stop_event.wait(0.01)
        return True

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            event = self.step()
            if event is None:
                self._wake_event.wait(timeout=0.05)
                self._wake_event.clear()

    def _admit_waiting(self) -> None:
        while True:
            if (
                self.max_active_requests is not None
                and len(self._active) >= self.max_active_requests
            ):
                return
            request = self.request_pool.pop_next()
            if request is None:
                return
            request.phase = RequestPhase.PREFILL
            self._active.append(request.request_id)

    def _reserve_request_resources(self, request: GenerationRequest) -> None:
        del request

    def _release_request_resources(self, request: GenerationRequest) -> None:
        del request

    def resource_snapshot(self) -> Dict[str, int | float]:
        return {
            "request_queue_pending": self.request_pool.pending_count(),
            "request_active_sequences": len(self._active),
            "backend_errors_total": self.backend_errors_total,
            "backend_oom_total": self.backend_oom_total,
        }

    def _expire_waiting_requests(self) -> List[TokenEvent]:
        events = []
        now = time.time()
        for request in self.request_pool.requests():
            if request.status is not RequestStatus.WAITING:
                continue
            total_expired = request.deadline is not None and now >= request.deadline
            queue_expired = (
                request.queue_deadline is not None
                and now >= request.queue_deadline
            )
            if not total_expired and not queue_expired:
                continue
            if self.request_pool.cancel(request.request_id):
                request.finish_reason = FinishReason.TIMEOUT
                request.timeout_phase = "total" if total_expired else "queue"
                request.phase = RequestPhase.FINISHED
                request.error = f"{request.timeout_phase} timeout exceeded"
                self.sessions.release(request.session_id)
                self._release_request_resources(request)
                event = self._terminal_event(request)
                self._publish(event)
                events.append(event)
                break
        return events

    @staticmethod
    def _expired_timeout_phase(
        request: GenerationRequest, now: Optional[float] = None
    ) -> Optional[str]:
        now = time.time() if now is None else now
        if request.deadline is not None and now >= request.deadline:
            return "total"
        if (
            request.phase is RequestPhase.PREFILL
            and request.prefill_deadline is not None
            and now >= request.prefill_deadline
        ):
            return "prefill"
        if (
            request.phase is RequestPhase.DECODE
            and request.decode_deadline is not None
            and now >= request.decode_deadline
        ):
            return "decode"
        return None

    def _finalize_timeout(
        self, request: GenerationRequest, timeout_phase: str
    ) -> TokenEvent:
        request.timeout_phase = timeout_phase
        request.error = f"{timeout_phase} timeout exceeded"
        return self._finalize_abort(request, FinishReason.TIMEOUT)

    def _generate_one(self, request: GenerationRequest) -> int:
        inputs = request.context_tokens + tuple(request.generated_token_ids)
        output = tuple(
            int(token)
            for token in self.model.generate(
                inputs,
                max_new_tokens=1,
                top_k=request.top_k,
                top_p=request.top_p,
                temperature=request.temperature,
                repetition_penalty=request.repetition_penalty,
                seed=(request.seed + len(request.generated_token_ids)) % 2**64,
            )
        )
        if output[: len(inputs)] != inputs or len(output) != len(inputs) + 1:
            raise RuntimeError(
                "A token step must return its input followed by exactly one token"
            )
        return output[-1]

    def _finish_reason(
        self, request: GenerationRequest, token_id: int
    ) -> Optional[FinishReason]:
        minimum_reached = (
            len(request.generated_token_ids) >= request.min_new_tokens
        )
        if minimum_reached:
            if (
                not request.ignore_eos
                and self._eos_token_id is not None
                and token_id == self._eos_token_id
            ):
                return FinishReason.EOS
            if token_id in request.stop_token_ids:
                return FinishReason.STOP
        if len(request.generated_token_ids) >= request.max_new_tokens:
            return FinishReason.LENGTH
        return None

    def _finalize_success(
        self,
        request: GenerationRequest,
        reason: FinishReason,
        token_id: Optional[int] = None,
        text: Optional[str] = None,
    ) -> TokenEvent:
        request.status = RequestStatus.FINISHED
        request.phase = RequestPhase.FINISHED
        request.finish_reason = reason
        request.output_tokens = request.context_tokens + tuple(
            request.generated_token_ids
        )
        self.sessions.commit(request.session_id, request.output_tokens)
        self.sessions.release(request.session_id)
        self._release_request_resources(request)
        request.finished_at = time.time()
        event = TokenEvent(
            request_id=request.request_id,
            session_id=request.session_id,
            token_id=token_id,
            text=text,
            finished=True,
            finish_reason=reason,
        )
        self._publish(event)
        return event

    def _finalize_abort(
        self, request: GenerationRequest, reason: FinishReason
    ) -> TokenEvent:
        text, _ = self._decode_increment(request, force_flush=True)
        request.status = RequestStatus.CANCELLED
        request.phase = RequestPhase.FINISHED
        request.finish_reason = reason
        if request.context_tokens is not None:
            request.output_tokens = request.context_tokens + tuple(
                request.generated_token_ids
            )
            if self._commit_partial_on_abort:
                self.sessions.commit(request.session_id, request.output_tokens)
        self.sessions.release(request.session_id)
        self._release_request_resources(request)
        request.finished_at = time.time()
        event = self._terminal_event(request, text=text)
        self._publish(event)
        return event

    def _finalize_error(
        self, request: GenerationRequest, error: Exception
    ) -> TokenEvent:
        text, _ = self._decode_increment(request, force_flush=True)
        request.status = RequestStatus.FAILED
        request.phase = RequestPhase.FINISHED
        request.finish_reason = FinishReason.ERROR
        request.error = str(error)
        self.backend_errors_total += 1
        if "out of memory" in request.error.lower():
            self.backend_oom_total += 1
        self.sessions.release(request.session_id)
        self._release_request_resources(request)
        request.finished_at = time.time()
        event = self._terminal_event(request, text=text)
        self._publish(event)
        return event

    @staticmethod
    def _terminal_event(
        request: GenerationRequest, text: Optional[str] = None
    ) -> TokenEvent:
        return TokenEvent(
            request_id=request.request_id,
            session_id=request.session_id,
            text=text,
            finished=True,
            finish_reason=request.finish_reason,
            error=request.error,
        )

    def _publish(self, event: TokenEvent) -> None:
        with self._events_lock:
            self._event_queues[event.request_id].put(event)


class OrcaScheduler(RoundRobinScheduler):
    """Iteration-level scheduler with persistent per-sequence KV caches.

    Prefill is admitted selectively while all already-prefilled sequences are
    decoded together in one model call. Finished sequences leave immediately,
    and waiting sequences can join the next iteration.
    """

    supports_continuous_batching = True

    def __init__(
        self,
        model: SequenceBatchModel,
        sessions: Optional[SessionManager] = None,
        request_pool: Optional[RequestPool] = None,
        token_decoder: Optional[Callable[[int], str]] = None,
        sequence_decoder: Optional[Callable[[Sequence[int]], str]] = None,
        commit_partial_on_abort: bool = True,
        max_sequence_length: Optional[int] = None,
        max_active_requests: Optional[int] = 32,
        max_prefill_per_iteration: int = 1,
        max_kv_cache_bytes: Optional[int] = None,
        kv_cache_high_watermark: float = 0.9,
    ) -> None:
        if not getattr(model, "supports_sequence_batching", False):
            raise ValueError("OrcaScheduler requires a sequence-batched model")
        if max_prefill_per_iteration <= 0:
            raise ValueError("max_prefill_per_iteration must be positive")
        super().__init__(
            model,
            sessions=sessions,
            request_pool=request_pool,
            token_decoder=token_decoder,
            sequence_decoder=sequence_decoder,
            commit_partial_on_abort=commit_partial_on_abort,
            max_sequence_length=max_sequence_length,
            max_active_requests=max_active_requests,
        )
        self.model: SequenceBatchModel = model
        self.max_prefill_per_iteration = max_prefill_per_iteration
        self.decode_batch_sizes: Deque[int] = deque(maxlen=10000)
        self.prefill_batch_sizes: Deque[int] = deque(maxlen=10000)
        self.scheduler_iterations_total = 0
        self.kv_cache_budget = KVCacheBudget(
            max_bytes=max_kv_cache_bytes,
            high_watermark=kv_cache_high_watermark,
        )

    def _estimate_request_kv_bytes(self, request: GenerationRequest) -> int:
        estimate = getattr(self.model, "estimate_kv_cache_bytes", None)
        if estimate is None:
            return 0
        session = self.sessions.get(request.session_id)
        prompt_tokens = len(session.token_history) + len(request.input_tokens)
        if self.max_sequence_length is not None:
            prompt_tokens = min(
                prompt_tokens,
                self.max_sequence_length - request.max_new_tokens,
            )
        return int(estimate(prompt_tokens + request.max_new_tokens))

    def _reserve_request_resources(self, request: GenerationRequest) -> None:
        self.kv_cache_budget.reserve(
            request.request_id, self._estimate_request_kv_bytes(request)
        )

    def _release_request_resources(self, request: GenerationRequest) -> None:
        self.kv_cache_budget.release(request.request_id)

    def resource_snapshot(self) -> Dict[str, int | float]:
        snapshot = super().resource_snapshot()
        snapshot.update(self.kv_cache_budget.snapshot())
        snapshot.update(
            {
                "orca_scheduler_iterations_total": self.scheduler_iterations_total,
                "orca_prefill_batch_size_current": (
                    self.prefill_batch_sizes[-1]
                    if self.prefill_batch_sizes
                    else 0
                ),
                "orca_prefill_batch_size_max": max(
                    self.prefill_batch_sizes, default=0
                ),
                "orca_prefill_batch_size_mean": (
                    sum(self.prefill_batch_sizes) / len(self.prefill_batch_sizes)
                    if self.prefill_batch_sizes
                    else 0.0
                ),
                "orca_decode_batch_size_current": (
                    self.decode_batch_sizes[-1]
                    if self.decode_batch_sizes
                    else 0
                ),
                "orca_decode_batch_size_max": max(
                    self.decode_batch_sizes, default=0
                ),
                "orca_decode_batch_size_mean": (
                    sum(self.decode_batch_sizes) / len(self.decode_batch_sizes)
                    if self.decode_batch_sizes
                    else 0.0
                ),
            }
        )
        model_snapshot = getattr(self.model, "kv_cache_snapshot", None)
        if model_snapshot is not None:
            snapshot.update(model_snapshot())
        return snapshot

    @staticmethod
    def _sampling_args(request: GenerationRequest) -> Dict[str, object]:
        return {
            "top_k": request.top_k,
            "top_p": request.top_p,
            "temperature": request.temperature,
            "repetition_penalty": request.repetition_penalty,
            "seed": request.seed,
        }

    def _accept_generated_token(
        self, request: GenerationRequest, token_id: int
    ) -> TokenEvent:
        request.generated_token_ids.append(int(token_id))
        if request.first_token_at is None:
            request.first_token_at = time.time()
        reason = self._finish_reason(request, int(token_id))
        text, stopped_by_string = self._decode_increment(
            request, force_flush=reason is not None
        )
        if stopped_by_string:
            reason = FinishReason.STOP
        if reason is not None:
            return self._finalize_success(
                request, reason, token_id=int(token_id), text=text
            )
        event = TokenEvent(
            request_id=request.request_id,
            session_id=request.session_id,
            token_id=int(token_id),
            text=text,
        )
        self._publish(event)
        return event

    def step_batch(self) -> List[TokenEvent]:
        """Runs one Orca iteration and returns every event from that iteration."""

        with self._worker_lock:
            events: List[TokenEvent] = self._expire_waiting_requests()
            self._admit_waiting()
            active_ids = list(self._active)
            self._active.clear()
            if not active_ids:
                return events
            self.scheduler_iterations_total += 1

            survivors: List[str] = []
            decode_requests: List[GenerationRequest] = []
            prefills = 0

            for request_id in active_ids:
                request = self.request_pool.get(request_id)
                if request.status is not RequestStatus.RUNNING:
                    continue
                if request.cancel_requested:
                    events.append(
                        self._finalize_abort(request, FinishReason.CANCELLED)
                    )
                    continue
                if timeout_phase := self._expired_timeout_phase(request):
                    events.append(self._finalize_timeout(request, timeout_phase))
                    continue
                if request.context_tokens is not None:
                    decode_requests.append(request)
                    continue
                if prefills >= self.max_prefill_per_iteration:
                    survivors.append(request.request_id)
                    continue

                prefills += 1
                try:
                    request.phase = RequestPhase.PREFILL
                    request.prefill_started_at = (
                        request.prefill_started_at or time.time()
                    )
                    context = self._build_context(request)
                    request.context_tokens = context
                    request.context_length = len(context)
                    self.model.create_sequence(
                        request.request_id,
                        len(context) + request.max_new_tokens,
                    )
                    token_id = self.model.prefill_sequence(
                        request.request_id,
                        context,
                        **self._sampling_args(request),
                    )
                    request.prefill_finished_at = time.time()
                    if request.cancel_requested:
                        event = self._finalize_abort(
                            request, FinishReason.CANCELLED
                        )
                    elif timeout_phase := self._expired_timeout_phase(request):
                        event = self._finalize_timeout(request, timeout_phase)
                    else:
                        event = self._accept_generated_token(request, token_id)
                except Exception as error:
                    event = self._finalize_error(request, error)
                events.append(event)
                if request.status is RequestStatus.RUNNING:
                    survivors.append(request.request_id)

            if prefills:
                self.prefill_batch_sizes.append(prefills)

            if decode_requests:
                ready_decode_requests = []
                for request in decode_requests:
                    request.phase = RequestPhase.DECODE
                    request.decode_started_at = (
                        request.decode_started_at or time.time()
                    )
                    if timeout_phase := self._expired_timeout_phase(request):
                        events.append(
                            self._finalize_timeout(request, timeout_phase)
                        )
                    else:
                        ready_decode_requests.append(request)
                decode_requests = ready_decode_requests

            if decode_requests:
                self.decode_batch_sizes.append(len(decode_requests))
                try:
                    output_ids = self.model.decode_batch(
                        [request.request_id for request in decode_requests],
                        [request.generated_token_ids[-1] for request in decode_requests],
                        [self._sampling_args(request) for request in decode_requests],
                    )
                    if len(output_ids) != len(decode_requests):
                        raise RuntimeError("Batched decode returned the wrong size")
                    for request, token_id in zip(decode_requests, output_ids):
                        if request.cancel_requested:
                            event = self._finalize_abort(
                                request, FinishReason.CANCELLED
                            )
                        elif timeout_phase := self._expired_timeout_phase(request):
                            event = self._finalize_timeout(
                                request, timeout_phase
                            )
                        else:
                            event = self._accept_generated_token(
                                request, int(token_id)
                            )
                        events.append(event)
                        if request.status is RequestStatus.RUNNING:
                            survivors.append(request.request_id)
                except Exception as error:
                    for request in decode_requests:
                        if request.status is RequestStatus.RUNNING:
                            events.append(self._finalize_error(request, error))

            self._active.extend(survivors)
            return events

    def step(self) -> Optional[TokenEvent]:
        events = self.step_batch()
        return events[0] if events else None

    def run_until_idle_stream(self) -> Iterator[TokenEvent]:
        while True:
            events = self.step_batch()
            if not events:
                return
            yield from events

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            if not self.step_batch():
                self._wake_event.wait(timeout=0.05)
                self._wake_event.clear()

    def _destroy_model_sequence(self, request: GenerationRequest) -> None:
        self.model.destroy_sequence(request.request_id)

    def _finalize_success(
        self,
        request: GenerationRequest,
        reason: FinishReason,
        token_id: Optional[int] = None,
        text: Optional[str] = None,
    ) -> TokenEvent:
        try:
            return super()._finalize_success(
                request, reason, token_id=token_id, text=text
            )
        finally:
            self._destroy_model_sequence(request)

    def _finalize_abort(
        self, request: GenerationRequest, reason: FinishReason
    ) -> TokenEvent:
        try:
            return super()._finalize_abort(request, reason)
        finally:
            self._destroy_model_sequence(request)

    def _finalize_error(
        self, request: GenerationRequest, error: Exception
    ) -> TokenEvent:
        try:
            return super()._finalize_error(request, error)
        finally:
            self._destroy_model_sequence(request)


class ChatService:
    """Structured chat history on top of ``RoundRobinScheduler``.

    Messages are the source of truth. Before each turn the complete conversation
    is rendered with the tokenizer's chat template and re-encoded. This is slower
    than reusing a per-session KV cache, but it keeps roles and conversations
    correct while the backend still owns only one cache.
    """

    def __init__(
        self, scheduler: RoundRobinScheduler, tokenizer: ChatTokenizer
    ) -> None:
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self._finalized_requests: set[str] = set()
        self._finalize_lock = threading.RLock()
        if self.scheduler._token_decoder is None:
            self.scheduler._token_decoder = lambda token_id: self.tokenizer.decode(
                [token_id], skip_special_tokens=False
            )
        if self.scheduler._sequence_decoder is None:
            self.scheduler._sequence_decoder = lambda token_ids: self.tokenizer.decode(
                token_ids, skip_special_tokens=True
            )

    def create_session(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> Session:
        messages = (
            [ChatMessage("system", system_prompt)]
            if system_prompt is not None
            else []
        )
        return self.scheduler.sessions.create(
            session_id=session_id,
            user_id=user_id,
            initial_messages=messages,
            metadata=metadata,
        )

    def submit_message(
        self,
        session_id: str,
        content: str,
        max_new_tokens: int = 128,
        min_new_tokens: int = 0,
        priority: int = 0,
        request_id: Optional[str] = None,
        stop_token_ids: Sequence[int] = (),
        stop_strings: Sequence[str] = (),
        truncate_prompt: bool = False,
        timeout_seconds: Optional[float] = None,
        queue_timeout_seconds: Optional[float] = None,
        prefill_timeout_seconds: Optional[float] = None,
        decode_timeout_seconds: Optional[float] = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
        repetition_penalty: float = 1.0,
        seed: int = 0,
        ignore_eos: bool = False,
    ) -> GenerationRequest:
        previous = self.scheduler.sessions.get(session_id)
        messages = previous.messages + [ChatMessage("user", str(content))]
        prompt = self.tokenizer.apply_chat_template(
            [message.as_dict() for message in messages],
            add_generation_prompt=True,
            tokenize=False,
        )
        if not isinstance(prompt, str):
            raise TypeError("Chat tokenizer must return text when tokenize=False")
        input_tokens = [int(token) for token in self.tokenizer.encode(prompt)]

        self.scheduler.sessions.replace_messages(session_id, messages)
        self.scheduler.sessions.set_token_history(session_id, ())
        try:
            return self.scheduler.submit(
                session_id=session_id,
                input_tokens=input_tokens,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                priority=priority,
                request_id=request_id,
                stop_token_ids=stop_token_ids,
                stop_strings=stop_strings,
                truncate_prompt=truncate_prompt,
                timeout_seconds=timeout_seconds,
                queue_timeout_seconds=queue_timeout_seconds,
                prefill_timeout_seconds=prefill_timeout_seconds,
                decode_timeout_seconds=decode_timeout_seconds,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                seed=seed,
                ignore_eos=ignore_eos,
            )
        except Exception:
            self.scheduler.sessions.replace_messages(
                session_id, previous.messages
            )
            self.scheduler.sessions.set_token_history(
                session_id, previous.token_history
            )
            raise

    def events(
        self, request_id: str, timeout: Optional[float] = None
    ) -> Iterator[TokenEvent]:
        for event in self.scheduler.events(request_id, timeout=timeout):
            if event.finished:
                self.finalize_message(request_id)
            yield event

    def run_until_idle_stream(self) -> Iterator[TokenEvent]:
        for event in self.scheduler.run_until_idle_stream():
            if event.finished:
                self.finalize_message(event.request_id)
            yield event

    def finalize_message(self, request_id: str) -> Optional[ChatMessage]:
        with self._finalize_lock:
            if request_id in self._finalized_requests:
                return None
            request = self.scheduler.request_pool.get(request_id)
            if request.status in (RequestStatus.WAITING, RequestStatus.RUNNING):
                raise RuntimeError(f"Request has not finished: {request_id}")
            self._finalized_requests.add(request_id)
            if request.status is RequestStatus.FAILED or not request.generated_tokens:
                return None
            content = request.streamed_text
            message = ChatMessage("assistant", content)
            self.scheduler.sessions.append_message(
                request.session_id, message.role, message.content
            )
            return message

    def export_session(self, session_id: str) -> Dict[str, object]:
        session = self.scheduler.sessions.get(session_id)
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "messages": [message.as_dict() for message in session.messages],
            "metadata": dict(session.metadata),
            "created_at": session.created_at,
            "updated_at": session.updated_at,
        }

    def import_session(
        self,
        data: Mapping[str, object],
        session_id: Optional[str] = None,
    ) -> Session:
        raw_messages = data.get("messages", [])
        if not isinstance(raw_messages, Sequence):
            raise TypeError("messages must be a sequence")
        messages: List[ChatMessage] = []
        for item in raw_messages:
            if not isinstance(item, Mapping):
                raise TypeError("Each message must be a mapping")
            role = item.get("role")
            content = item.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise TypeError("message role and content must be strings")
            messages.append(ChatMessage(role=role, content=content))
        metadata = data.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        imported_session_id = session_id or data.get("session_id")
        imported_user_id = data.get("user_id")
        if imported_session_id is not None and not isinstance(
            imported_session_id, str
        ):
            raise TypeError("session_id must be a string")
        if imported_user_id is not None and not isinstance(imported_user_id, str):
            raise TypeError("user_id must be a string")
        return self.scheduler.sessions.create(
            session_id=imported_session_id,
            user_id=imported_user_id,
            initial_messages=messages,
            metadata=metadata,
        )
