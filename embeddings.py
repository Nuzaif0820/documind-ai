"""
DocuMind AI - Embeddings Module
Initializes and caches the Hugging Face Sentence Transformers embedding model.
Ensures normalized vector outputs for true cosine similarity comparison in FAISS.
"""

import os
from typing import Optional
from config import DEFAULT_EMBEDDING_MODEL, EMBEDDING_DEVICE

# Global singleton cache for embedding model
_EMBEDDINGS_INSTANCE = None


def get_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: Optional[str] = None
):
    """
    Load and return the HuggingFace embeddings model.
    Caches the instance to avoid repeatedly loading heavy model weights into RAM.
    Optimized for low memory consumption in cloud containers.
    """
    global _EMBEDDINGS_INSTANCE

    if _EMBEDDINGS_INSTANCE is not None:
        return _EMBEDDINGS_INSTANCE

    # Limit torch threads to save memory on shared cloud instances
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass

    chosen_device = device or EMBEDDING_DEVICE

    # Try modern langchain_huggingface first, fallback to langchain_community
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    model_kwargs = {"device": chosen_device}
    encode_kwargs = {
        # Normalizing vectors allows FAISS Inner Product (IP) index to directly compute
        # Cosine Similarity: cos_sim(u, v) = dot(u/||u||, v/||v||)
        "normalize_embeddings": True
    }

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    _EMBEDDINGS_INSTANCE = embeddings
    return embeddings
