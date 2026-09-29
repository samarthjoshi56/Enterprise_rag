from app.ingestion.embeddings import generate_embedding, generate_embeddings_batch


def test_generate_single_embedding():
    """Test generating a single vector embedding."""
    text = "Enterprise RAG document vector embedding test."
    vector = generate_embedding(text)

    assert isinstance(vector, list)
    assert len(vector) == 384  # all-MiniLM-L6-v2 dimension
    assert all(isinstance(v, float) for v in vector)


def test_generate_embeddings_batch():
    """Test generating vector embeddings for a batch of strings."""
    texts = [
        "First document chunk text.",
        "Second document chunk text.",
        "Third document chunk text.",
    ]
    vectors = generate_embeddings_batch(texts)

    assert len(vectors) == 3
    for vec in vectors:
        assert len(vec) == 384
