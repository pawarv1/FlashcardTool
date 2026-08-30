import os
import io
import zipfile
from db import get_db_connection

DB_FILE = "study_assistant.db"
CHROMA_DIR = "chroma_db_data"
MEDIA_DIR = os.path.join("assets", "media")

def create_system_backup_zip() -> io.BytesIO:
    """
    Bundles SQLite database (checkpointed), ChromaDB data, and local media assets into an in-memory ZIP archive.
    """
    # Force SQLite WAL checkpoint to ensure all pending writes are flushed to study_assistant.db
    if os.path.exists(DB_FILE):
        try:
            with get_db_connection() as conn:
                conn.execute("PRAGMA wal_checkpoint(FULL);")
        except Exception as e:
            print(f"Warning: WAL checkpoint failed prior to backup: {e}")

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
    Safely extracts a system backup ZIP file, replacing local database, ChromaDB, and media files.
    """
    try:
        target_dir = os.path.abspath(".")
        
        with zipfile.ZipFile(uploaded_zip_file, "r") as zip_ref:
            # Zip Slip Vulnerability Guard: Ensure target path stays strictly inside current directory
            for member in zip_ref.namelist():
                member_path = os.path.abspath(os.path.join(target_dir, member))
                if not member_path.startswith(target_dir):
                    raise PermissionError(f"Security Alert: Blocked illegal path extraction: {member}")
            
            zip_ref.extractall(target_dir)
        return True
    except Exception as e:
        print(f"Failed to restore backup: {e}")
        return False