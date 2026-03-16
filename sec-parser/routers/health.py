from fastapi import APIRouter

router = APIRouter()


@router.get("/healthz")
def healthz():
    """Liveness probe — always returns 200 if the process is up."""
    return {"status": "ok"}
