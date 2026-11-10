"""
DocuMind AI - Embeddings Module
Initializes and caches the Hugging Face Sentence Transformers embedding model.
Ensures normalized vector outputs for true cosine similarity comparison in FAISS.
"""

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
    
    Args:
        model_name: HuggingFace model repo ID.
        device: 'cuda' or 'cpu'. Defaults to config setting.
        
    Returns:
        HuggingFaceEmbeddings instance compatible with LangChain vector stores.
    """
    global _EMBEDDINGS_INSTANCE

    if _EMBEDDINGS_INSTANCE is not None:
        return _EMBEDDINGS_INSTANCE

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
