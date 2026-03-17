import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException

import config
import utility

app = FastAPI(title=config.APP_TITLE)


@app.on_event("startup")
def startup_event():
    utility.load_index_and_metadata()


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "indexed_chunks": 0 if config.faiss_index is None else config.faiss_index.ntotal
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    allowed_extensions = {".pdf", ".docx", ".txt"}
    ext = Path(file.filename).suffix.lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )

    unique_name = f"{uuid.uuid4()}_{file.filename}"
    saved_path = config.UPLOAD_DIR / unique_name

    try:
        content = await file.read()
        saved_path.write_bytes(content)

        extracted_text = utility.extract_text(saved_path)
        if not extracted_text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the uploaded file."
            )

        chunks = utility.chunk_text(extracted_text)
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Text extraction succeeded but chunking produced no chunks."
            )

        count = utility.add_chunks_to_vector_store(chunks, source_file=file.filename)

        return {
            "message": "File uploaded and indexed successfully.",
            "file_name": file.filename,
            "saved_as": unique_name,
            "chunks_added": count,
            "total_chunks_in_index": config.faiss_index.ntotal
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@app.post("/chat", response_model=config.ChatResponse)
def chat_with_document(request: config.ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if config.faiss_index is None or config.faiss_index.ntotal == 0:
        raise HTTPException(status_code=400, detail="No documents indexed yet. Upload a file first.")

    retrieved = utility.search_similar_chunks(question, top_k=config.TOP_K)

    if not retrieved:
        return config.ChatResponse(
            answer="I could not find the answer in the uploaded document.",
            retrieved_chunks=[]
        )

    answer = utility.ask_groq_llm(question, retrieved)

    return config.ChatResponse(
        answer=answer,
        retrieved_chunks=retrieved
    )