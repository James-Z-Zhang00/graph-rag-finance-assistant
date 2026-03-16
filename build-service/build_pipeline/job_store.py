"""
In-memory job store for build pipeline jobs.
Thread-safe via RLock.
"""

import threading
import time
import uuid
from typing import Dict, Any, List, Optional


class Job:
    def __init__(self, job_id: str, job_type: str):
        self.job_id = job_id
        self.job_type = job_type          # "full" | "incremental"
        self.status = "pending"           # pending | running | completed | failed
        self.stage = ""                   # human-readable current stage
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.error: Optional[str] = None
        self.stats: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "status": self.status,
            "stage": self.stage,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "stats": self.stats,
        }


class JobStore:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.RLock()

    def create(self, job_type: str) -> Job:
        job = Job(job_id=str(uuid.uuid4()), job_type=job_type)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_all(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [j.to_dict() for j in self._jobs.values()]

    def update(self, job_id: str, **kwargs):
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                for k, v in kwargs.items():
                    setattr(job, k, v)

    def mark_running(self, job_id: str, stage: str = ""):
        self.update(job_id, status="running", stage=stage, started_at=time.time())

    def mark_completed(self, job_id: str, stats: Optional[Dict] = None):
        self.update(job_id, status="completed", stage="done", completed_at=time.time(), stats=stats)

    def mark_failed(self, job_id: str, error: str):
        self.update(job_id, status="failed", completed_at=time.time(), error=error)


job_store = JobStore()
