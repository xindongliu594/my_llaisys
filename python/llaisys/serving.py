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


class FinishReason(str, Enum):
    EOS = "eos"
    STOP = "stop"
    LENGTH = "length"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class GenerationRequest:
    request_id: str
    session_id: str
    input_tokens: Tuple[int, ...]
    max_new_tokens: int
    priority: int = 0
    stop_token_ids: Tuple[int, ...] = ()
    timeout_seconds: Optional[float] = None
    status: RequestStatus = RequestStatus.WAITING
    finish_reason: Optional[FinishReason] = None
    cancel_requested: bool = False
    context_length: Optional[int] = None
    context_tokens: Optional[Tuple[int, ...]] = None
    generated_token_ids: List[int] = field(default_factory=list)
    output_tokens: Optional[Tuple[int, ...]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
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

    def __init__(self) -> None:
        self._heap: List[Tuple[int, int, str]] = []
        self._requests: Dict[str, GenerationRequest] = {}
        self._sequence = itertools.count()
        self._lock = threading.RLock()

    def submit(self, request: GenerationRequest) -> None:
        with self._lock:
            if request.request_id in self._requests:
                raise ValueError(f"Request already exists: {request.request_id}")
            if request.status is not RequestStatus.WAITING:
                raise ValueError("Only waiting requests may enter the request pool")
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
    ) -> None:
        self.model = model
        self.sessions = sessions or SessionManager()
        self.request_pool = request_pool or RequestPool()
        self._worker_lock = threading.Lock()

    def submit(
        self,
        session_id: str,
        input_tokens: Sequence[int],
        max_new_tokens: int = 128,
        priority: int = 0,
        request_id: Optional[str] = None,
        stop_token_ids: Sequence[int] = (),
        timeout_seconds: Optional[float] = None,
    ) -> GenerationRequest:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        tokens = tuple(int(token) for token in input_tokens)
        priority = int(priority)
        stop_tokens = tuple(int(token) for token in stop_token_ids)
        self.sessions.acquire(session_id)
        request = GenerationRequest(
            request_id=request_id or uuid.uuid4().hex,
            session_id=session_id,
            input_tokens=tokens,
            max_new_tokens=max_new_tokens,
            priority=priority,
            stop_token_ids=stop_tokens,
            timeout_seconds=timeout_seconds,
        )
        try:
            self.request_pool.submit(request)
        except Exception:
            self.sessions.release(session_id)
            raise
        return request

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
                context = self.sessions.build_context(
                    request.session_id, request.input_tokens
                )
                if not context:
                    raise ValueError("A generation request needs at least one token")
                request.context_length = len(context)
                output = tuple(
                    int(token)
                    for token in self.model.generate(
                        context,
                        max_new_tokens=request.max_new_tokens,
                        top_k=1,
                    )
                )
                if output[: len(context)] != tuple(context):
                    raise RuntimeError(
                        "The model must return the input context followed by generated tokens"
                    )
                request.output_tokens = output
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
        commit_partial_on_abort: bool = True,
    ) -> None:
        super().__init__(model, sessions=sessions, request_pool=request_pool)
        self._active: Deque[str] = deque()
        self._event_queues: Dict[str, "queue.Queue[TokenEvent]"] = {}
        self._events_lock = threading.RLock()
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._token_decoder = token_decoder
        self._commit_partial_on_abort = commit_partial_on_abort
        self._eos_token_id = getattr(model, "eos_token_id", None)

    def submit(
        self,
        session_id: str,
        input_tokens: Sequence[int],
        max_new_tokens: int = 128,
        priority: int = 0,
        request_id: Optional[str] = None,
        stop_token_ids: Sequence[int] = (),
        timeout_seconds: Optional[float] = None,
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
                priority=priority,
                request_id=request_id,
                stop_token_ids=stop_token_ids,
                timeout_seconds=timeout_seconds,
            )
        except Exception:
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
            self._admit_waiting()
            while self._active:
                request = self.request_pool.get(self._active.popleft())
                if request.status is not RequestStatus.RUNNING:
                    continue

                if request.cancel_requested:
                    return self._finalize_abort(request, FinishReason.CANCELLED)
                if request.deadline is not None and time.time() >= request.deadline:
                    return self._finalize_abort(request, FinishReason.TIMEOUT)

                try:
                    if request.context_tokens is None:
                        context = tuple(
                            self.sessions.build_context(
                                request.session_id, request.input_tokens
                            )
                        )
                        if not context:
                            raise ValueError(
                                "A generation request needs at least one token"
                            )
                        request.context_tokens = context
                        request.context_length = len(context)

                    token_id = self._generate_one(request)
                    if request.cancel_requested:
                        return self._finalize_abort(
                            request, FinishReason.CANCELLED
                        )
                    if request.deadline is not None and time.time() >= request.deadline:
                        return self._finalize_abort(request, FinishReason.TIMEOUT)

                    request.generated_token_ids.append(token_id)
                    text = (
                        self._token_decoder(token_id)
                        if self._token_decoder is not None
                        else None
                    )
                    reason = self._finish_reason(request, token_id)
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

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            event = self.step()
            if event is None:
                self._wake_event.wait(timeout=0.05)
                self._wake_event.clear()

    def _admit_waiting(self) -> None:
        while True:
            request = self.request_pool.pop_next()
            if request is None:
                return
            self._active.append(request.request_id)

    def _generate_one(self, request: GenerationRequest) -> int:
        inputs = request.context_tokens + tuple(request.generated_token_ids)
        output = tuple(
            int(token)
            for token in self.model.generate(inputs, max_new_tokens=1, top_k=1)
        )
        if output[: len(inputs)] != inputs or len(output) != len(inputs) + 1:
            raise RuntimeError(
                "A token step must return its input followed by exactly one token"
            )
        return output[-1]

    def _finish_reason(
        self, request: GenerationRequest, token_id: int
    ) -> Optional[FinishReason]:
        if self._eos_token_id is not None and token_id == self._eos_token_id:
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
        request.finish_reason = reason
        request.output_tokens = request.context_tokens + tuple(
            request.generated_token_ids
        )
        self.sessions.commit(request.session_id, request.output_tokens)
        self.sessions.release(request.session_id)
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
        request.status = RequestStatus.CANCELLED
        request.finish_reason = reason
        if request.context_tokens is not None:
            request.output_tokens = request.context_tokens + tuple(
                request.generated_token_ids
            )
            if self._commit_partial_on_abort:
                self.sessions.commit(request.session_id, request.output_tokens)
        self.sessions.release(request.session_id)
        request.finished_at = time.time()
        event = self._terminal_event(request)
        self._publish(event)
        return event

    def _finalize_error(
        self, request: GenerationRequest, error: Exception
    ) -> TokenEvent:
        request.status = RequestStatus.FAILED
        request.finish_reason = FinishReason.ERROR
        request.error = str(error)
        self.sessions.release(request.session_id)
        request.finished_at = time.time()
        event = self._terminal_event(request)
        self._publish(event)
        return event

    @staticmethod
    def _terminal_event(request: GenerationRequest) -> TokenEvent:
        return TokenEvent(
            request_id=request.request_id,
            session_id=request.session_id,
            finished=True,
            finish_reason=request.finish_reason,
            error=request.error,
        )

    def _publish(self, event: TokenEvent) -> None:
        with self._events_lock:
            self._event_queues[event.request_id].put(event)


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
        priority: int = 0,
        request_id: Optional[str] = None,
        stop_token_ids: Sequence[int] = (),
        timeout_seconds: Optional[float] = None,
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
                priority=priority,
                request_id=request_id,
                stop_token_ids=stop_token_ids,
                timeout_seconds=timeout_seconds,
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
            content = self.tokenizer.decode(
                request.generated_tokens, skip_special_tokens=True
            )
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
            messages.append(
                ChatMessage(role=str(item["role"]), content=str(item["content"]))
            )
        metadata = data.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        imported_session_id = session_id or data.get("session_id")
        imported_user_id = data.get("user_id")
        return self.scheduler.sessions.create(
            session_id=(
                str(imported_session_id)
                if imported_session_id is not None
                else None
            ),
            user_id=(str(imported_user_id) if imported_user_id is not None else None),
            initial_messages=messages,
            metadata=metadata,
        )
