# A plain in-memory, thread-keyed session store, playing the same role
# Phase 7's CLI local `messages` variable plays, just addressable by
# thread_id instead of scoped to one process's loop. No persistence
# across restarts, on purpose, matching Phase 7's InMemorySaver — this is
# a dev/demo API, not a production service.
import threading
import uuid

_sessions: dict[str, list] = {}
_pending: set[str] = set()
_lock = threading.Lock()


def get_or_create(thread_id: str | None) -> tuple[str, list]:
    with _lock:
        if thread_id and thread_id in _sessions:
            return thread_id, list(_sessions[thread_id])
        new_id = thread_id or str(uuid.uuid4())
        _sessions[new_id] = []
        return new_id, []


def save_messages(thread_id: str, messages: list) -> None:
    with _lock:
        _sessions[thread_id] = messages


def thread_exists(thread_id: str) -> bool:
    with _lock:
        return thread_id in _sessions


def mark_pending(thread_id: str) -> None:
    with _lock:
        _pending.add(thread_id)


def clear_pending(thread_id: str) -> None:
    with _lock:
        _pending.discard(thread_id)


def is_pending(thread_id: str) -> bool:
    with _lock:
        return thread_id in _pending
