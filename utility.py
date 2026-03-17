import json
import uuid
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from PyPDF2 import PdfReader
from docx import Document

import config


def load_index_and_metadata():
    """
    Load FAISS index and metadata from disk if they exist.
    Otherwise initialize empty state.
    """
    if config.INDEX_FILE.exists():
        config.faiss_index = faiss.read_index(str(config.INDEX_FILE))
    else:
        sample_vector = config.embedding_model.encode(["hello"], convert_to_numpy=True)
        dim = sample_vector.shape[1]
        config.faiss_index = faiss.IndexFlatL2(dim)

    if config.METADATA_FILE.exists():
        with open(config.METADATA_FILE, "r", encoding="utf-8") as f:
            config.metadata_store = json.load(f)
    else:
        config.metadata_store = []


def save_index_and_metadata():
    """
    Persist FAISS index and metadata to disk.
    """
    if config.faiss_index is None:
        raise RuntimeError("FAISS index is not initialized.")

    faiss.write_index(config.faiss_index, str(config.INDEX_FILE))

    with open(config.METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(config.metadata_store, f, ensure_ascii=False, indent=2)


def extract_text_from_pdf(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)
    return "\n".join(pages_text).strip()


def extract_text_from_docx(file_path: Path) -> str:
    doc = Document(str(file_path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs).strip()


def extract_text_from_txt(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8", errors="ignore").strip()


def extract_text(file_path: Path) -> str:
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP
) -> List[str]:
    text = clean_text(text)
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]

        if end < text_len:
            last_break = max(chunk.rfind("\n"), chunk.rfind(". "), chunk.rfind(" "))
            if last_break > int(chunk_size * 0.6):
                end = start + last_break
                chunk = text[start:end]

        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        start = max(end - overlap, 0)

    return chunks


def embed_texts(texts: List[str]) -> np.ndarray:
    embeddings = config.embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return np.array(embeddings, dtype=np.float32)


def add_chunks_to_vector_store(chunks: List[str], source_file: str) -> int:
    if not chunks:
        return 0

    if config.faiss_index is None:
        raise RuntimeError("FAISS index is not initialized. Did startup run?")

    vectors = embed_texts(chunks)
    config.faiss_index.add(vectors)

    for chunk in chunks:
        config.metadata_store.append({
            "id": str(uuid.uuid4()),
            "source_file": source_file,
            "text": chunk
        })

    save_index_and_metadata()
    return len(chunks)


def search_similar_chunks(query: str, top_k: int = config.TOP_K) -> List[Dict[str, Any]]:
    if config.faiss_index is None or config.faiss_index.ntotal == 0:
        return []

    query_vector = embed_texts([query])
    distances, indices = config.faiss_index.search(query_vector, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        if idx >= len(config.metadata_store):
            continue

        item = config.metadata_store[idx].copy()
        item["score"] = float(dist)
        results.append(item)

    return results


def build_rag_prompt(question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Context {i} | Source: {chunk['source_file']}]\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a helpful RAG assistant.

Answer the user only from the provided context.
If the answer is not in the context, say:
"I could not find the answer in the uploaded document."

Context:
{context}

User Question:
{question}

Answer:
""".strip()

    return prompt


def ask_groq_llm(question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    prompt = build_rag_prompt(question, retrieved_chunks)

    response = config.groq_client.chat.completions.create(
        model=config.GROQ_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You answer questions only from retrieved document context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()