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

# Chunking Strategy:
# - chunk_size (1400 chars ~ 250-300 words): Captures entire promotional specification
#   cards, game flyers, or complex rule tables intact without splitting headers from content.
# - chunk_overlap (250 chars ~ 40-50 words): Ensures seamless continuity across chunk boundaries.
DEFAULT_CHUNK_SIZE = 1400
DEFAULT_CHUNK_OVERLAP = 250
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Embedding Configuration
# - sentence-transformers/all-MiniLM-L6-v2 produces 384-dimensional dense vectors.
# - Highly optimized for sentence semantic similarity, fast inference on CPU,
#   and ideal for free-tier Hugging Face Spaces (sub-100MB model size).
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cuda" if os.environ.get("USE_CUDA", "false").lower() == "true" else "cpu"

# Vector Store / Retrieval Settings
DEFAULT_TOP_K = 5
MIN_TOP_K = 1
MAX_TOP_K = 8
# Cosine similarity threshold (0.0 to 1.0, where 1.0 is exact match)
# 0.20 provides optimal balance: filters out pure noise while preserving chunks with vocabulary variance
DEFAULT_SIMILARITY_THRESHOLD = 0.20

# LLM Providers & Models
# Groq API provides lightning-fast inference for open-weights models on free tier.
SUPPORTED_GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_TEMPERATURE = 0.1
MAX_RESPONSE_TOKENS = 1500

# Grounding & Refusal Constants
REFUSAL_RESPONSE = "I couldn't find sufficient information in the uploaded documents."

# Groq API Key resolution: checks environment, .env file, or Streamlit secrets
def get_groq_api_key() -> str:
    """Retrieve Groq API key from environment variables, .env file, or Streamlit secrets."""
    # 1. First check standard environment
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if api_key:
        return api_key

    # 2. Check local .env file explicitly
    env_file = BASE_DIR / ".env"
    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GROQ_API_KEY=") and not line.startswith("#"):
                        val = line.split("=", 1)[1].strip().strip('"\'')
                        if val:
                            return val
        except Exception:
            pass

    # 3. Try importing streamlit secrets only if secrets file actually exists
    secrets_file = BASE_DIR / ".streamlit" / "secrets.toml"
    user_secrets = Path.home() / ".streamlit" / "secrets.toml"
    if secrets_file.exists() or user_secrets.exists():
        try:
            import streamlit as st
            if "GROQ_API_KEY" in st.secrets:
                return st.secrets["GROQ_API_KEY"].strip()
        except Exception:
            pass

    return ""
