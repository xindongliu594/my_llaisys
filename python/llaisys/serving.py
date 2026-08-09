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
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Tuple


class GenerationModel(Protocol):
    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: Optional[int] = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ) -> Sequence[int]: ...


class RequestStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class GenerationRequest:
    request_id: str
    session_id: str
    input_tokens: Tuple[int, ...]
    max_new_tokens: int
    priority: int = 0
    status: RequestStatus = RequestStatus.WAITING
    context_length: Optional[int] = None
    output_tokens: Optional[Tuple[int, ...]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    @property
    def generated_tokens(self) -> Optional[Tuple[int, ...]]:
        if self.output_tokens is None or self.context_length is None:
            return None
        return self.output_tokens[self.context_length :]


@dataclass
class Session:
    session_id: str
    user_id: Optional[str] = None
    token_history: List[int] = field(default_factory=list)
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
    ) -> GenerationRequest:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        tokens = tuple(int(token) for token in input_tokens)
        priority = int(priority)
        self.sessions.acquire(session_id)
        request = GenerationRequest(
            request_id=request_id or uuid.uuid4().hex,
            session_id=session_id,
            input_tokens=tokens,
            max_new_tokens=max_new_tokens,
            priority=priority,
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
