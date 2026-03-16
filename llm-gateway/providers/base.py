from abc import ABC, abstractmethod
from typing import Any, AsyncIterator


class BaseProvider(ABC):

    @abstractmethod
    async def chat(self, request_body: dict) -> Any:
        """Non-streaming chat completion. Returns the upstream JSON response."""

    @abstractmethod
    async def stream_chat(self, request_body: dict) -> AsyncIterator[bytes]:
        """Streaming chat completion. Yields raw SSE bytes from upstream."""

    @abstractmethod
    async def embeddings(self, request_body: dict) -> Any:
        """Embeddings request. Returns the upstream JSON response."""
