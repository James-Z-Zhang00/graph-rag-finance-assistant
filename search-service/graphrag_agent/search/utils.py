import numpy as np
from typing import List, Dict, Any, Union

class VectorUtils:
    """Unified utility class for vector search and similarity computation."""

    @staticmethod
    def cosine_similarity(vec1: Union[List[float], np.ndarray],
                         vec2: Union[List[float], np.ndarray]) -> float:
        """
        Compute the cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            float: Similarity value (0–1)
        """
        # Ensure inputs are numpy arrays
        if not isinstance(vec1, np.ndarray):
            vec1 = np.array(vec1)
        if not isinstance(vec2, np.ndarray):
            vec2 = np.array(vec2)

        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)

        # Guard against division by zero
        if norm_a == 0 or norm_b == 0:
            return 0

        return dot_product / (norm_a * norm_b)

    @staticmethod
    def rank_by_similarity(query_embedding: List[float],
                          candidates: List[Dict[str, Any]],
                          embedding_field: str = "embedding",
                          top_k: int = None) -> List[Dict[str, Any]]:
        """
        Rank candidates by their similarity to the query embedding.

        Args:
            query_embedding: Query vector
            candidates: List of candidate items, each containing the field named by embedding_field
            embedding_field: Name of the field holding the embedding vector
            top_k: Maximum number of results to return; None returns all

        Returns:
            Candidates sorted by similarity (descending), each augmented with a "score" field
        """
        scored_items = []

        for item in candidates:
            if embedding_field in item and item[embedding_field]:
                # Compute similarity
                similarity = VectorUtils.cosine_similarity(query_embedding, item[embedding_field])
                # Copy item and attach score
                scored_item = item.copy()
                scored_item["score"] = similarity
                scored_items.append(scored_item)

        # Sort by similarity (descending)
        scored_items.sort(key=lambda x: x["score"], reverse=True)

        # Return only the top_k results if specified
        if top_k is not None:
            return scored_items[:top_k]

        return scored_items

    @staticmethod
    def filter_documents_by_relevance(query_embedding: List[float],
                                     docs: List,
                                     embedding_attr: str = "embedding",
                                     threshold: float = 0.0,
                                     top_k: int = None) -> List:
        """
        Filter documents by similarity to the query embedding.

        Args:
            query_embedding: Query vector
            docs: List of documents; may be objects with an embedding attribute
            embedding_attr: Attribute name holding the embedding vector
            threshold: Minimum similarity threshold
            top_k: Maximum number of results to return

        Returns:
            Documents sorted by similarity (descending)
        """
        scored_docs = []

        for doc in docs:
            # Retrieve the document's vector representation
            doc_embedding = getattr(doc, embedding_attr, None) if hasattr(doc, embedding_attr) else None

            if doc_embedding:
                similarity = VectorUtils.cosine_similarity(query_embedding, doc_embedding)
                # Only include documents that meet the threshold
                if similarity >= threshold:
                    scored_docs.append({
                        'document': doc,
                        'score': similarity
                    })
            else:
                # Assign a baseline score when no embedding is available
                scored_docs.append({
                    'document': doc,
                    'score': 0.0
                })

        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x['score'], reverse=True)

        # Extract sorted documents
        if top_k is not None:
            top_docs = [item['document'] for item in scored_docs[:top_k]]
        else:
            top_docs = [item['document'] for item in scored_docs]

        return top_docs

    @staticmethod
    def batch_cosine_similarity(query_embedding: np.ndarray,
                            embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Compute cosine similarity in batch for improved efficiency.

        Args:
            query_embedding: Query vector
            embeddings: List of vectors to compare against

        Returns:
            NumPy array containing the similarity score for each vector
        """
        # Stack the list into a 2-D matrix
        matrix = np.vstack(embeddings)

        # Normalize the query vector
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return np.zeros(len(embeddings))
        query_normalized = query_embedding / query_norm

        # Normalize all candidate vectors (row-wise)
        matrix_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
        # Guard against division by zero
        matrix_norm[matrix_norm == 0] = 1.0
        matrix_normalized = matrix / matrix_norm

        # Compute all similarities at once via matrix multiplication
        similarities = np.dot(matrix_normalized, query_normalized)

        return similarities
