from fastapi import APIRouter
from routers.chat import router as chat_router
from routers.knowledge_graph import router as kg_router
from routers.source import router as source_router
from routers.feedback import router as feedback_router

api_router = APIRouter(prefix="/api")
api_router.include_router(chat_router)
api_router.include_router(kg_router)
api_router.include_router(source_router)
api_router.include_router(feedback_router)
