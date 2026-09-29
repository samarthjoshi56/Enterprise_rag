from functools import lru_cache
from typing import List
from sentence_transformers import SentenceTransformer
from app.core.config import get_settings
from app.core.logging import logger

settings = get_settings()


@lru_cache()
def get_embedding_model() -> SentenceTransformer:
    """Load and cache the SentenceTransformer embedding model."""
    logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")
    return SentenceTransformer(settings.EMBEDDING_MODEL_NAME)


def generate_embedding(text: str) -> List[float]:
    """Generate a vector embedding for a single text string."""
    model = get_embedding_model()
    embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return embedding.tolist()


def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generate vector embeddings for a batch of text strings."""
    if not texts:
        return []
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.tolist()
