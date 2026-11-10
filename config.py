"""
DocuMind AI - Configuration Module
Defines system parameters, model definitions, chunking strategies,
and environment settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Suppress Hugging Face Windows symlink and transfer warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# Load environment variables from .env file if available
load_dotenv()

# Base project paths
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INDEX_DIR = BASE_DIR / "storage" / "faiss_index"

# PDF Ingestion & Processing Settings
MAX_FILE_SIZE_MB = 25
ALLOWED_EXTENSIONS = [".pdf"]

# Chunking Strategy Rationale:
# - chunk_size (900 chars ~ 150-200 words): Fits 1-2 coherent paragraphs or bullet sets.
#   Avoids diluting embedding vectors with unrelated concepts.
# - chunk_overlap (150 chars ~ 25-30 words): Prevents loss of cross-boundary sentence
#   meaning and ensures search matches don't lose context.
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 150
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Embedding Configuration
# - sentence-transformers/all-MiniLM-L6-v2 produces 384-dimensional dense vectors.
# - Highly optimized for sentence semantic similarity, fast inference on CPU,
#   and ideal for free-tier Hugging Face Spaces (sub-100MB model size).
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cuda" if os.environ.get("USE_CUDA", "false").lower() == "true" else "cpu"

# Vector Store / Retrieval Settings
DEFAULT_TOP_K = 4
MIN_TOP_K = 1
MAX_TOP_K = 8
# Cosine similarity threshold (0.0 to 1.0, where 1.0 is exact match)
DEFAULT_SIMILARITY_THRESHOLD = 0.30

# LLM Providers & Models
# Groq API provides lightning-fast inference for open-weights models on free tier.
SUPPORTED_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.1
MAX_RESPONSE_TOKENS = 1500

# Grounding & Refusal Constants
REFUSAL_RESPONSE = "I couldn't find sufficient information in the uploaded documents."

# Groq API Key resolution: checks environment or Streamlit secrets
def get_groq_api_key() -> str:
    """Retrieve Groq API key from environment variables or Streamlit secrets."""
    # First check standard environment
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if api_key:
        return api_key
    
    # Try importing streamlit secrets if running inside Streamlit
    try:
        import streamlit as st
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"].strip()
    except Exception:
        pass
    
    return ""
