"""
HuggingFace Sentence Transformer embedding engine.
Uses sentence-transformers/all-MiniLM-L6-v2 for ESG text embeddings.
"""

from typing import List, Optional

import numpy as np

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingEngine:
    """
    Sentence Transformer embedding engine for ESG text vectorization.
    Lazy-loads the model on first use.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: int = 32,
        normalize: bool = True,
    ):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None

    @property
    def model(self):
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Embedding model loaded (dim={self.embedding_dim})")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed; using TF-IDF fallback"
                )
                self._model = TFIDFFallbackModel()
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        if hasattr(self._model, "get_sentence_embedding_dimension"):
            return self._model.get_sentence_embedding_dimension()
        return settings.EMBEDDING_DIM

    def encode(
        self,
        texts: List[str],
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Encode texts into embedding vectors.

        Args:
            texts: List of text strings to encode.
            show_progress_bar: Whether to show encoding progress.

        Returns:
            numpy array of shape (N, embedding_dim).
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
            )
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Embedding encoding failed: {e}")
            # Return zero vectors as fallback
            return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text string."""
        result = self.encode([text])
        return result[0] if len(result) > 0 else np.zeros(self.embedding_dim)

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))


class TFIDFFallbackModel:
    """
    TF-IDF based fallback embedding model when sentence-transformers
    is not available. Returns fixed-dim vectors via SVD.
    """

    def __init__(self, n_components: int = 384):
        self.n_components = n_components
        self._vectorizer = None
        self._svd = None
        self._fitted = False

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        """Encode texts using TF-IDF + truncated SVD."""
        try:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer

            if not self._fitted:
                self._vectorizer = TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2),
                    stop_words="english",
                )
                n_comp = min(self.n_components, len(texts) - 1, 9999)
                self._svd = TruncatedSVD(n_components=max(1, n_comp))

                tfidf_matrix = self._vectorizer.fit_transform(texts)
                embeddings = self._svd.fit_transform(tfidf_matrix)
                self._fitted = True
            else:
                tfidf_matrix = self._vectorizer.transform(texts)
                embeddings = self._svd.transform(tfidf_matrix)

            # Pad to target dimension if needed
            if embeddings.shape[1] < self.n_components:
                pad = np.zeros((embeddings.shape[0], self.n_components - embeddings.shape[1]))
                embeddings = np.hstack([embeddings, pad])

            if normalize_embeddings:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / (norms + 1e-8)

            return embeddings.astype(np.float32)
        except Exception:
            return np.random.default_rng(42).random(
                (len(texts), self.n_components)
            ).astype(np.float32)

    def get_sentence_embedding_dimension(self) -> int:
        return self.n_components