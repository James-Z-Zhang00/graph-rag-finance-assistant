"""
OpenAI-compatible provider.

Forwards requests transparently to the upstream base URL (OpenAI, One-API,
or any compatible endpoint). Swaps in the upstream API key so clients only
ever need to know the gateway key.
"""

import logging
from typing import Any, AsyncIterator

import httpx

from config.settings import UPSTREAM_BASE_URL, UPSTREAM_API_KEY, REQUEST_TIMEOUT
from providers.base import BaseProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {UPSTREAM_API_KEY}",
            "Content-Type": "application/json",
        }

    async def chat(self, request_body: dict) -> Any:
        url = f"{UPSTREAM_BASE_URL}/chat/completions"
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(url, json=request_body, headers=self._headers())
            response.raise_for_status()
            return response.json()

    async def stream_chat(self, request_body: dict) -> AsyncIterator[bytes]:
        url = f"{UPSTREAM_BASE_URL}/chat/completions"
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            async with client.stream("POST", url, json=request_body, headers=self._headers()) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk

    async def embeddings(self, request_body: dict) -> Any:
        url = f"{UPSTREAM_BASE_URL}/embeddings"
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(url, json=request_body, headers=self._headers())
            response.raise_for_status()
            return response.json()
