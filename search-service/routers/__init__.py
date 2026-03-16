from fastapi import APIRouter
from routers.health import router as health_router
from routers.hybrid import router as hybrid_router
from routers.cache import router as cache_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(hybrid_router)
api_router.include_router(cache_router)
