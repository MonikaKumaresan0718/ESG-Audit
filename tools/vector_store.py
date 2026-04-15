"""
FAISS Vector Store wrapper for ESG document embeddings.
Supports add, search, persist, and load operations.
"""

import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.logging import get_logger

logger = get_logger(__name__)


class FAISSVectorStore:
    """
    FAISS-based vector store for semantic ESG document search.
    Stores embeddings with associated text and metadata.
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        index_path: Optional[str] = None,
        metric: str = "l2",
    ):
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.metric = metric
        self.texts: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self._index = None

        if index_path and os.path.exists(index_path + ".faiss"):
            self._load()
        else:
            self._initialize_index()

    def _initialize_index(self):
        """Create a new FAISS index."""
        try:
            import faiss

            if self.metric == "cosine":
                self._index = faiss.IndexFlatIP(self.embedding_dim)
            else:
                self._index = faiss.IndexFlatL2(self.embedding_dim)
            logger.info(f"Initialized FAISS index (dim={self.embedding_dim}, metric={self.metric})")
        except ImportError:
            logger.warning("FAISS not installed; using numpy fallback vector store")
            self._index = None
            self._vectors: np.ndarray = np.empty((0, self.embedding_dim), dtype=np.float32)

    def add(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """
        Add embeddings to the index.

        Args:
            embeddings: numpy array of shape (N, dim).
            texts: Corresponding text strings.
            metadata: Optional list of metadata dicts.

        Returns:
            Number of vectors added.
        """
        if len(embeddings) == 0:
            return 0

        embeddings = embeddings.astype(np.float32)

        if self.metric == "cosine":
            # Normalize for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

        self.texts.extend(texts)
        self.metadata.extend(metadata or [{} for _ in texts])

        if self._index is not None:
            self._index.add(embeddings)
        else:
            # Numpy fallback
            if self._vectors.shape[0] == 0:
                self._vectors = embeddings
            else:
                self._vectors = np.vstack([self._vectors, embeddings])

        logger.debug(f"Added {len(embeddings)} vectors to store (total: {self.total_vectors})")

        if self.index_path:
            self.save()

        return len(embeddings)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query vector of shape (dim,) or (1, dim).
            top_k: Number of results to return.

        Returns:
            List of dicts with text, score, and metadata.
        """
        if self.total_vectors == 0:
            return []

        query = query_embedding.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        if self.metric == "cosine":
            norm = np.linalg.norm(query)
            if norm > 0:
                query = query / norm

        top_k = min(top_k, self.total_vectors)

        if self._index is not None:
            scores, indices = self._index.search(query, top_k)
            scores = scores[0]
            indices = indices[0]
        else:
            # Numpy fallback
            if self.metric == "cosine":
                sims = np.dot(self._vectors, query.T).flatten()
                indices = np.argsort(sims)[::-1][:top_k]
                scores = sims[indices]
            else:
                diffs = self._vectors - query
                dists = np.linalg.norm(diffs, axis=1)
                indices = np.argsort(dists)[:top_k]
                scores = dists[indices]

        results = []
        for score, idx in zip(scores, indices):
            if idx >= 0 and idx < len(self.texts):
                results.append({
                    "text": self.texts[idx],
                    "score": float(score),
                    "metadata": self.metadata[idx] if idx < len(self.metadata) else {},
                    "index": int(idx),
                })

        return results

    def save(self):
        """Persist index and metadata to disk."""
        if not self.index_path:
            return
        try:
            import faiss

            if self._index is not None:
                faiss.write_index(self._index, self.index_path + ".faiss")
            else:
                np.save(self.index_path + "_vectors.npy", self._vectors)

            with open(self.index_path + "_meta.pkl", "wb") as f:
                pickle.dump({"texts": self.texts, "metadata": self.metadata}, f)

            logger.debug(f"Saved vector store to {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")

    def _load(self):
        """Load index and metadata from disk."""
        try:
            import faiss

            self._index = faiss.read_index(self.index_path + ".faiss")
            logger.info(f"Loaded FAISS index from {self.index_path}")
        except Exception:
            self._initialize_index()

        meta_path = self.index_path + "_meta.pkl"
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self.texts = meta.get("texts", [])
            self.metadata = meta.get("metadata", [])

    @property
    def total_vectors(self) -> int:
        """Total number of indexed vectors."""
        if self._index is not None:
            return self._index.ntotal
        return getattr(self, "_vectors", np.empty((0,))).shape[0]

    def clear(self):
        """Reset the index."""
        self.texts = []
        self.metadata = []
        self._initialize_index()