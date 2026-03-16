from fastapi import APIRouter
from config.settings import UPSTREAM_BASE_URL

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok", "upstream": UPSTREAM_BASE_URL}
