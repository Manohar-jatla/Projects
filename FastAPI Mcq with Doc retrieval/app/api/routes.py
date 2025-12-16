import os
from fastapi import APIRouter, UploadFile, Form, HTTPException
from app.services.pdf_service import extract_text_from_pdf
from app.services.gemini_service import generate_mcqs
from app.services.cache_service import get_cached_result, save_to_cache
from app.services.file_store import save_uploaded_file, list_uploaded_files, get_file_path

router = APIRouter()

# Upload and store file
@router.post("/upload-doc")
async def upload_document(file: UploadFile):
    content = await file.read()
    result = save_uploaded_file(content, file.filename)
    return {"message": "File uploaded successfully", **result}

# List stored documents
@router.get("/documents")
def list_documents():
    return {"documents": list_uploaded_files()}

# Generate MCQs from previously uploaded file
@router.post("/generate-mcqs/from-doc")
async def generate_mcqs_from_doc_id(
    doc_id: str = Form(...),
    difficulty: str = Form(...),
    num_questions: int = Form(...)
):
    file_path = get_file_path(doc_id)
    if not file_path:
        raise HTTPException(status_code=404, detail="Document not found.")

    filename = os.path.basename(file_path).split("_", 1)[-1]

    cached = get_cached_result(filename, difficulty, num_questions)
    if cached:
        return cached

    try:
        with open(file_path, "rb") as f:
            content = f.read()
        text = extract_text_from_pdf(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF processing failed: {e}")

    mcqs = generate_mcqs(text, difficulty, num_questions)
    save_to_cache(filename, difficulty, num_questions, mcqs)

    return mcqs

# Existing generate-mcqs from new upload
@router.post("/generate-mcqs")
async def generate_mcqs_from_pdf(
    file: UploadFile,
    difficulty: str = Form(...),
    num_questions: int = Form(...)
):
    content = await file.read()
    filename = file.filename

    cached = get_cached_result(filename, difficulty, num_questions)
    if cached:
        return cached

    try:
        text = extract_text_from_pdf(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF processing failed: {e}")

    mcqs = generate_mcqs(text, difficulty, num_questions)
    save_to_cache(filename, difficulty, num_questions, mcqs)

    return mcqs
