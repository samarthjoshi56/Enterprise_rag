from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging import setup_logging, logger
from app.db.postgres import init_postgres_db
from app.db.qdrant import init_qdrant_collection
from app.api.routes import api_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown events."""
    setup_logging()
    logger.info(f"Starting {settings.APP_NAME} in [{settings.APP_ENV}] mode...")
    
    try:
        await init_postgres_db()
        logger.info("PostgreSQL database tables initialized.")
    except Exception as e:
        logger.warning(f"PostgreSQL DB init deferred/warning: {e}")

    try:
        await init_qdrant_collection()
        logger.info("Qdrant vector collection initialized.")
    except Exception as e:
        logger.warning(f"Qdrant collection init deferred/warning: {e}")
    
    yield
    
    logger.info(f"Shutting down {settings.APP_NAME}...")


app = FastAPI(
    title=settings.APP_NAME,
    description="Production Enterprise RAG System Backend (Phase 1 Setup)",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")
app.include_router(api_router)  # Also expose /health at root level for convenance


@app.get("/", summary="Root endpoint")
async def root():
    """Root endpoint welcoming users and providing health check path."""
    return JSONResponse(
        content={
            "message": f"Welcome to {settings.APP_NAME} API",
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/health",
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
