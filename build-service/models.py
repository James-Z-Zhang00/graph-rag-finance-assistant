"""
Pydantic request/response models for the build service.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel


class JobResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    stage: str
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None


class TriggerResponse(BaseModel):
    job_id: str
    message: str
