"""
Request context — thread-local store for the current request_id.

Usage
-----
At the start of each ask_with_trace() call (which runs inside asyncio.to_thread),
call set_request_id(generate_request_id()).  Every subsequent audit log write on
the same thread automatically picks up the id via get_request_id().

IMPORTANT: set_request_id() must be called from INSIDE the worker thread
(i.e. after asyncio.to_thread() has entered the new thread).  Thread-locals
do not inherit across the asyncio event-loop / thread-pool boundary.
"""

import threading
import uuid
from typing import Optional

_ctx = threading.local()


def generate_request_id() -> str:
    """Return a new UUID4 string suitable for use as a request_id."""
    return str(uuid.uuid4())


def set_request_id(rid: str) -> None:
    """Store *rid* in the current thread's local context."""
    _ctx.request_id = rid


def get_request_id() -> Optional[str]:
    """Return the request_id stored for this thread, or None if not set."""
    return getattr(_ctx, "request_id", None)


def clear_request_id() -> None:
    """Remove the request_id from the current thread's context."""
    if hasattr(_ctx, "request_id"):
        del _ctx.request_id
