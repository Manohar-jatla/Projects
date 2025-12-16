import os

def save_file(file_bytes: bytes, filename: str, folder: str = "uploads") -> str:
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, "wb") as f:
        f.write(file_bytes)
    return path
