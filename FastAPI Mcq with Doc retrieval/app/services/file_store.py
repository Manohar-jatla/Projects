import os
import uuid

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'uploaded_docs')
os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_uploaded_file(file_content: bytes, filename: str) -> dict:
    doc_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{filename}")
    with open(save_path, "wb") as f:
        f.write(file_content)
    
    return {
        "doc_id": doc_id,
        "filename": filename,
        "path": save_path
    }

def list_uploaded_files() -> list:
    files = []
    for fname in os.listdir(UPLOAD_DIR):
        if "_" in fname:
            doc_id, original_name = fname.split("_", 1)
            files.append({
                "doc_id": doc_id,
                "filename": original_name
            })
    return files
def get_file_path(doc_id: str) -> str:
    for fname in os.listdir(UPLOAD_DIR):
        if fname.startswith(doc_id + "_"):
            return os.path.join(UPLOAD_DIR, fname)
    return None

