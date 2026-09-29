import os
from functools import lru_cache
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    # Application
    APP_NAME: str = Field(default="Enterprise RAG")
    APP_ENV: str = Field(default="development")
    DEBUG: bool = Field(default=True)
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)

    # PostgreSQL
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: int = Field(default=5432)
    POSTGRES_USER: str = Field(default="postgres")
    POSTGRES_PASSWORD: str = Field(default="postgres_password")
    POSTGRES_DB: str = Field(default="enterprise_rag")
    POSTGRES_URL: Optional[str] = Field(default=None)

    # Qdrant
    QDRANT_HOST: str = Field(default="localhost")
    QDRANT_PORT: int = Field(default=6333)
    QDRANT_GRPC_PORT: int = Field(default=6334)
    QDRANT_URL: Optional[str] = Field(default=None)
    QDRANT_API_KEY: Optional[str] = Field(default="")

    # Redis
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379)
    REDIS_PASSWORD: Optional[str] = Field(default="")
    REDIS_URL: Optional[str] = Field(default=None)

    # Ingestion & Vector DB Configuration
    EMBEDDING_MODEL_NAME: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION: int = Field(default=384)
    QDRANT_COLLECTION_NAME: str = Field(default="enterprise_documents")
    CHUNK_SIZE: int = Field(default=1000)
    CHUNK_OVERLAP: int = Field(default=200)

    # Search & Retrieval Configuration
    VECTOR_SEARCH_TOP_K: int = Field(default=20)
    KEYWORD_SEARCH_TOP_K: int = Field(default=20)
    HYBRID_ALPHA: float = Field(default=0.7)   # 1.0 = pure vector, 0.0 = pure keyword
    RERANKER_MODEL_NAME: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANKER_TOP_N: int = Field(default=5)

    # Advanced RAG Configuration (Phase 4: HyDE, CRAG, Self-RAG)
    OPENAI_API_KEY: Optional[str] = Field(default=None)
    GOOGLE_API_KEY: Optional[str] = Field(default=None)
    LLM_PROVIDER: str = Field(default="google")  # "google" or "openai"
    LLM_MODEL_NAME: str = Field(default="gemini-2.0-flash")
    HYDE_ENABLED: bool = Field(default=True)
    CRAG_RELEVANCE_THRESHOLD: float = Field(default=-2.0)  # Cross-encoder logit threshold for "relevant"
    CRAG_MAX_REWRITES: int = Field(default=1)
    SELF_RAG_ENABLED: bool = Field(default=True)

    # Redis Cache Configuration (Phase 6)
    REDIS_CACHE_ENABLED: bool = Field(default=True)
    REDIS_CACHE_TTL: int = Field(default=3600)  # TTL in seconds (1 hour default)
    REDIS_CACHE_PREFIX: str = Field(default="rag:cache:")

    # Evaluation Configuration (Phase 7)
    EVALUATION_DATASET_PATH: str = Field(default="data/evaluation_dataset.json")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    @property
    def async_postgres_url(self) -> str:
        """Construct or return the async PostgreSQL connection string (asyncpg)."""
        if self.POSTGRES_URL:
            # Ensure it uses asyncpg scheme if postgresql:// is provided
            if self.POSTGRES_URL.startswith("postgresql://"):
                return self.POSTGRES_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
            return self.POSTGRES_URL
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def sync_postgres_url(self) -> str:
        """Construct or return the sync PostgreSQL connection string (psycopg2)."""
        if self.POSTGRES_URL:
            if self.POSTGRES_URL.startswith("postgresql+asyncpg://"):
                return self.POSTGRES_URL.replace("postgresql+asyncpg://", "postgresql://", 1)
            return self.POSTGRES_URL
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def effective_qdrant_url(self) -> str:
        """Construct or return the Qdrant HTTP URL."""
        if self.QDRANT_URL:
            return self.QDRANT_URL
        return f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"

    @property
    def effective_redis_url(self) -> str:
        """Construct or return the Redis connection URL."""
        if self.REDIS_URL:
            return self.REDIS_URL
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/0"


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton instance of Settings."""
    return Settings()
