"""
Central application configuration for MediAssist AI.

Loads the project-root .env file and exposes every setting from one place.
Secrets stay out of the source code and are never printed by this module.
"""

import os  # reads environment variables
from pathlib import Path  # builds file paths that work on Windows

from dotenv import load_dotenv  # loads the .env file into environment variables

# app/config.py -> app -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # folder that contains .env

# Load .env from a fixed location, not from the terminal's current folder.
load_dotenv(PROJECT_ROOT / ".env")  # puts every KEY=value line into os.environ


def required_env(name: str) -> str:  # get a setting that MUST exist
    """Return a required environment variable or raise a clear error."""
    value = os.getenv(name)  # read the value (None if the name is missing)
    if not value:  # missing or empty
        raise ValueError(  # stop with a helpful message
            f"Missing environment variable: {name}. "
            "Check your project-root .env file."
        )
    return value  # return the valid value


def _get_list(name: str, default: str) -> list[str]:  # turn "a,b,c" into ["a","b","c"]
    raw = os.getenv(name, default)  # read the comma-separated text
    return [item.strip().lower() for item in raw.split(",") if item.strip()]  # clean each item


# ===== 1. LLM =====
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()  # "groq" or "openai"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # may be None until we call the LLM
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")  # default Groq model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # optional
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # default OpenAI model
JUDGE_MODEL = os.getenv("JUDGE_MODEL", GROQ_MODEL)  # model used by the Evaluator Agent

# ===== 2. PostgreSQL (used only by the MCP server) =====
DB_HOST = os.getenv("DB_HOST")  # database server address
DB_PORT = int(os.getenv("DB_PORT", "5432"))  # port as a number
DB_NAME = os.getenv("DB_NAME")  # database name
DB_USER = os.getenv("DB_USER")  # database user
DB_PASSWORD = os.getenv("DB_PASSWORD")  # database password (secret)
DB_SSLMODE = os.getenv("DB_SSLMODE", "require")  # Aiven requires SSL

# ===== 3. Embeddings and RAG =====
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")  # Hugging Face model
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))  # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))  # characters shared between chunks
TOP_K = int(os.getenv("TOP_K", "5"))  # how many chunks to retrieve
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))  # minimum score to accept a chunk

# ===== 4. Upload rules =====
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))  # file size limit
MAX_FILES_PER_UPLOAD = int(os.getenv("MAX_FILES_PER_UPLOAD", "1"))  # files accepted at once
ALLOWED_DOC_TYPES = _get_list("ALLOWED_DOC_TYPES", "pdf,docx,txt")  # document extensions
ALLOWED_IMAGE_TYPES = _get_list("ALLOWED_IMAGE_TYPES", "png,jpg,jpeg")  # image extensions

# ===== 4b. Multimodal (OCR + vision) =====
OCR_LANGUAGES = _get_list("OCR_LANGUAGES", "en")  # languages EasyOCR reads
OCR_MAX_SIDE = int(os.getenv("OCR_MAX_SIDE", "1000"))  # longest image side in pixels (bigger = slower, 1000 tuned on real timings)
VISION_MODE = os.getenv("VISION_MODE", "auto").lower()  # auto = describe only images with no text, always, off
VISION_MODEL = os.getenv("VISION_MODEL", "Salesforce/blip-image-captioning-base")  # BLIP caption model

# ===== 4c. Login =====
AUTH_SECRET = os.getenv("AUTH_SECRET")  # signs login tokens (keep it only in .env)
TOKEN_HOURS = int(os.getenv("TOKEN_HOURS", "8"))  # how long a login lasts

# ===== 5. Chat history =====
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "5"))  # last N chats shown (FIFO)

# ===== 6. App URLs and ports =====
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")  # where Streamlit finds FastAPI
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))  # FastAPI port
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "8501"))  # Streamlit port

# ===== 7. Folders (all relative to the project root) =====
DATA_DIR = PROJECT_ROOT / "data"  # main data folder
RAW_DIR = DATA_DIR / "raw"  # original documents
IMAGES_DIR = DATA_DIR / "images"  # uploaded images
PROCESSED_DIR = DATA_DIR / "processed"  # cleaned or generated files
VECTOR_STORE_DIR = PROJECT_ROOT / os.getenv("VECTOR_STORE_PATH", "data/vector_store")  # FAISS index
EXTRACTIONS_DIR = PROCESSED_DIR / "extractions"  # saved OCR + extraction results (one JSON per image)
USERS_FILE = PROJECT_ROOT / os.getenv("USERS_FILE", "data/processed/users.json")  # login credentials

for _folder in (RAW_DIR, IMAGES_DIR, PROCESSED_DIR, EXTRACTIONS_DIR, VECTOR_STORE_DIR):  # make sure folders exist
    _folder.mkdir(parents=True, exist_ok=True)  # create if missing, no error if present

def validate_llm_config() -> None:  # call this before using the LLM
    """Raise a clear error if the chosen LLM provider has no API key."""
    if LLM_PROVIDER == "groq":  # Groq selected
        required_env("GROQ_API_KEY")  # key must exist
    elif LLM_PROVIDER == "openai":  # OpenAI selected
        required_env("OPENAI_API_KEY")  # key must exist
    else:  # anything else is a typo
        raise ValueError("LLM_PROVIDER must be 'groq' or 'openai'.")


def validate_db_config() -> None:  # call this before connecting to PostgreSQL
    """Raise a clear error if any database setting is missing."""
    for name in ("DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"):  # the four required values
        required_env(name)  # fail with the exact missing name
        