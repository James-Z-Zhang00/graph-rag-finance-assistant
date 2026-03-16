"""
Pydantic models mirroring the OpenAI API request schemas.
extra="allow" ensures any fields not listed here are still forwarded upstream.
"""

from __future__ import annotations
from typing import Any, List, Optional, Union
from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: Union[str, List[Any], None] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Any]] = None
    tool_call_id: Optional[str] = None

    model_config = {"extra": "allow"}


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    tools: Optional[List[Any]] = None
    tool_choice: Optional[Any] = None

    model_config = {"extra": "allow"}


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str], List[int], List[List[int]]]
    encoding_format: Optional[str] = None
    dimensions: Optional[int] = None
    user: Optional[str] = None

    model_config = {"extra": "allow"}
