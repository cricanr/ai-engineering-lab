# RAG

RAG means Retrieval-Augmented Generation.

Instead of sending everything to the model, we first search for relevant chunks.
Then we send only the best chunks to the model as context.

This helps with larger documents, lower cost, smaller prompts, and more grounded answers.
