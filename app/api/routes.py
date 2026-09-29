from fastapi import APIRouter
from app.api.health import router as health_router
from app.api.documents import router as documents_router
from app.api.search import router as search_router
from app.api.chat import router as chat_router
from app.api.text2sql import router as text2sql_router
from app.api.cache import router as cache_router
from app.api.evaluation import router as evaluation_router

api_router = APIRouter()

api_router.include_router(health_router)
api_router.include_router(documents_router)
api_router.include_router(search_router)
api_router.include_router(chat_router)
api_router.include_router(text2sql_router)
api_router.include_router(cache_router)
api_router.include_router(evaluation_router)

