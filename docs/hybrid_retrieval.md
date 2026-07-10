# Hybrid Retrieval

Hybrid retrieval combines lexical and semantic rankings:

- `keyword_rank` finds lexical matches between the question and document chunks.
- `embedding_rank` finds semantic similarity using embeddings.
- `hybrid_rank` combines both rankings using reciprocal rank fusion.
- `keyword_weight` and `embedding_weight` control how much each ranking influences the combined score.

This approach helps retrieve relevant chunks when either exact word matches or semantic similarity alone are insufficient.
