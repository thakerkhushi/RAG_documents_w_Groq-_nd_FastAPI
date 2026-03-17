from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

APP_TITLE = "RAG documents with Groq and FastAPI"

UPLOAD_DIR = Path("data/uploads")
INDEX_DIR = Path("data/index")
INDEX_FILE = INDEX_DIR / "faiss.index"
METADATA_FILE = INDEX_DIR / "metadata.json"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY is not set in environment variables.")

groq_client = Groq(api_key=groq_api_key)

faiss_index = None
metadata_store: List[Dict[str, Any]] = []


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    retrieved_chunks: List[Dict[str, Any]]