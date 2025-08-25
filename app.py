from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import uuid
import os

from rag import MultiModalRAG  # import your class

# Initialize FastAPI app
app = FastAPI(title="MultiModal RAG API", version="1.0")

# Initialize RAG system
rag_system = MultiModalRAG()

# In-memory doc registry
DOCUMENTS = {}


@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile):
    """Upload and process a PDF document."""
    try:
        # Save uploaded PDF temporarily
        temp_dir = Path("uploaded_pdfs")
        temp_dir.mkdir(exist_ok=True)
        file_path = temp_dir / file.filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Generate unique document_id
        document_id = str(uuid.uuid4())

        # Process PDF
        doc_info = rag_system.process_pdf(str(file_path), document_id=document_id)

        # Store document info
        DOCUMENTS[document_id] = doc_info

        return JSONResponse(content={
            "message": "PDF processed successfully",
            "document_info": doc_info
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/query/")
async def query_document(
    question: str = Form(...),
    k: int = Form(3),
    verbose: bool = Form(False),
):
    """Ask a question over the uploaded and processed PDFs."""
    try:
        if not DOCUMENTS:
            return JSONResponse(content={"error": "No documents processed yet"}, status_code=400)

        answer = rag_system.query(question, k=k, verbose=verbose)

        return JSONResponse(content={
            "question": question,
            "answer": answer
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/documents/")
async def list_documents():
    """List all processed documents."""
    return DOCUMENTS
