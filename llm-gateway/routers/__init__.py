from fastapi import APIRouter
from routers.health import router as health_router
from routers.chat import router as chat_router
from routers.embeddings import router as embeddings_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(chat_router, prefix="/v1")
api_router.include_router(embeddings_router, prefix="/v1")
