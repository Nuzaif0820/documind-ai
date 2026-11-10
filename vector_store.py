"""
DocuMind AI - Vector Store Module
Manages FAISS index creation, incremental document addition, persistence,
document-level metadata filtering, and similarity search with score calibration.
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from embeddings import get_embedding_model
from config import (
    DEFAULT_INDEX_DIR,
    DEFAULT_TOP_K,
    DEFAULT_SIMILARITY_THRESHOLD
)


class VectorStoreManager:
    """
    Encapsulates FAISS vector store operations: indexing, persistence,
    document metadata filtering, and ranked retrieval with calibrated scores.
    """

    def __init__(self, embedding_model=None, index_path: Optional[Path] = None):
        self.embedding_model = embedding_model or get_embedding_model()
        self.index_path = Path(index_path or DEFAULT_INDEX_DIR)
        self.vector_store: Optional[FAISS] = None
        self._indexed_files_registry: Dict[str, Dict[str, Any]] = {}

    @property
    def is_indexed(self) -> bool:
        """Return True if index is loaded and ready for retrieval."""
        return self.vector_store is not None

    def add_documents(self, documents: List[Document], doc_stats: Optional[Dict[str, Any]] = None) -> int:
        """
        Add documents to the FAISS index. Creates new index if none exists.
        
        Args:
            documents: List of chunked Document objects.
            doc_stats: Optional metadata dictionary with file stats.
            
        Returns:
            Number of documents indexed.
        """
        if not documents:
            return 0

        # Update registry
        filename = documents[0].metadata.get("source", "unknown")
        total_pages = documents[0].metadata.get("total_pages", 1)
        self._indexed_files_registry[filename] = {
            "chunk_count": len(documents),
            "total_pages": total_pages,
            **(doc_stats or {})
        }

        if self.vector_store is None:
            # Create fresh FAISS index
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embedding_model
            )
        else:
            # Incrementally append to existing index
            self.vector_store.add_documents(documents=documents)

        return len(documents)

    def similarity_search_with_scores(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        score_threshold: Optional[float] = DEFAULT_SIMILARITY_THRESHOLD,
        filter_source: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform semantic similarity search against the FAISS index.
        Calibrates distance scores into a normalized [0.0, 1.0] similarity metric.
        
        Args:
            query: User's question or search query string.
            k: Top-K chunks to retrieve.
            score_threshold: Minimum normalized similarity score cutoff (0.0 to 1.0).
            filter_source: Optional document filename to restrict search to.
            
        Returns:
            List of (Document, similarity_score) tuples, sorted by highest similarity.
        """
        if self.vector_store is None:
            return []

        # Prepare metadata filter if a specific document is targeted
        search_filter = {"source": filter_source} if filter_source and filter_source != "All Documents" else None

        # Fetch top candidate chunks (request slightly more if filtering by score)
        candidate_k = min(k * 2, 20)
        
        if search_filter:
            results_with_distance = self.vector_store.similarity_search_with_score(
                query=query,
                k=candidate_k,
                filter=search_filter
            )
        else:
            results_with_distance = self.vector_store.similarity_search_with_score(
                query=query,
                k=candidate_k
            )

        scored_results: List[Tuple[Document, float]] = []

        for doc, distance in results_with_distance:
            # FAISS in LangChain defaults to L2 distance for dense embeddings:
            # For normalized unit vectors (||u|| = ||v|| = 1):
            # L2 distance squared = ||u - v||^2 = 2 - 2 * cosine_similarity
            # Therefore: cosine_similarity = 1 - (distance^2 / 2)
            # Clip between 0.0 and 1.0 for stability.
            try:
                cos_sim = 1.0 - (float(distance) ** 2 / 2.0)
                similarity_score = max(0.0, min(1.0, cos_sim))
            except Exception:
                # Fallback inverse distance mapping
                similarity_score = 1.0 / (1.0 + float(distance))

            # Apply threshold if configured
            if score_threshold is not None and similarity_score < score_threshold:
                continue

            scored_results.append((doc, round(similarity_score, 4)))

        # Sort descending by similarity score and truncate to requested top-k
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results[:k]

    def get_indexed_documents(self) -> Dict[str, Dict[str, Any]]:
        """Return metadata summary for all currently indexed documents."""
        return self._indexed_files_registry.copy()

    def clear(self) -> None:
        """Reset and erase the current in-memory index and registry."""
        self.vector_store = None
        self._indexed_files_registry.clear()
        if self.index_path.exists():
            shutil.rmtree(self.index_path, ignore_errors=True)

    def save_local(self, directory: Optional[Path] = None) -> None:
        """Persist the FAISS index to disk."""
        target_dir = Path(directory or self.index_path)
        target_dir.mkdir(parents=True, exist_ok=True)
        if self.vector_store is not None:
            self.vector_store.save_local(str(target_dir))

    def load_local(self, directory: Optional[Path] = None) -> bool:
        """
        Load a persisted FAISS index from disk.
        Returns True if successful, False otherwise.
        """
        target_dir = Path(directory or self.index_path)
        if not (target_dir / "index.faiss").exists():
            return False

        try:
            self.vector_store = FAISS.load_local(
                folder_path=str(target_dir),
                embeddings=self.embedding_model,
                allow_dangerous_deserialization=True
            )
            return True
        except Exception:
            return False
