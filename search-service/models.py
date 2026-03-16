from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    session_id: str = "default"
    debug: bool = False


class SearchResponse(BaseModel):
    answer: str
    execution_log: Optional[List[Dict[str, Any]]] = None
