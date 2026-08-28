import os
import shutil
import zipfile
import io
from datetime import datetime

DB_FILE = "study_assistant.db"
CHROMA_DIR = "chroma_db"
MEDIA_DIR = os.path.join("assets", "media")

def create_system_backup_zip() -> io.BytesIO:
    """
    Bundles SQLite database, ChromaDB folder, and local media folder into an in-memory ZIP archive.
    """
    buffer = io.BytesIO()
    
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # 1. Archive SQLite Database
        if os.path.exists(DB_FILE):
            zip_file.write(DB_FILE, arcname=DB_FILE)
            
        # 2. Archive ChromaDB Directory
        if os.path.exists(CHROMA_DIR):
            for root, _, files in os.walk(CHROMA_DIR):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=".")
                    zip_file.write(file_path, arcname=arcname)

        # 3. Archive Media Assets Directory
        if os.path.exists(MEDIA_DIR):
            for root, _, files in os.walk(MEDIA_DIR):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=".")
                    zip_file.write(file_path, arcname=arcname)

    buffer.seek(0)
    return buffer

def restore_system_from_zip(uploaded_zip_file) -> bool:
    """
    Extracts a system backup ZIP file, replacing local database, ChromaDB, and media files.
    """
    try:
        with zipfile.ZipFile(uploaded_zip_file, "r") as zip_ref:
            # Safely extract all files over existing local directory structure
            zip_ref.extractall(".")
        return True
    except Exception as e:
        print(f"Failed to restore backup: {e}")
        return False