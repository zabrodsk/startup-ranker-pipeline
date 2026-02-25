"""TF-IDF based chunk retrieval for evidence-grounded answering."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from agent.ingest.store import Chunk, EvidenceStore


def retrieve_chunks(
    query: str,
    store: EvidenceStore,
    k: int = 8,
) -> list[Chunk]:
    """Retrieve the top-k most relevant chunks for a query.

    Uses TF-IDF vectorization + cosine similarity. This is a zero-cost
    alternative to embedding-based retrieval suitable for MVP.

    Args:
        query: The question or search string.
        store: EvidenceStore to search over.
        k: Number of chunks to return.

    Returns:
        Top-k chunks sorted by descending similarity.
    """
    if not store.chunks:
        return []

    corpus = store.texts
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus + [query])

    query_vec = tfidf_matrix[-1]
    doc_vecs = tfidf_matrix[:-1]

    similarities = cosine_similarity(query_vec, doc_vecs).flatten()

    top_k = min(k, len(store.chunks))
    top_indices = similarities.argsort()[::-1][:top_k]

    return [store.chunks[i] for i in top_indices if similarities[i] > 0.0]
