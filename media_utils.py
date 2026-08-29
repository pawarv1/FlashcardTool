import os
import uuid
from PIL import Image

MEDIA_DIR = os.path.join(".", "assets", "media")

def ensure_media_dir():
    """Ensures the local media directory exists."""
    os.makedirs(MEDIA_DIR, exist_ok=True)

def process_and_save_media(uploaded_file, max_width: int = 1200, quality: int = 80) -> str:
    """
    Saves an uploaded media file safely using a random UUID filename.
    If it's an image, compresses it to WebP format to minimize storage size.
    Returns the normalized relative local file path.
    """
    ensure_media_dir()
    
    # Extract extension safely and generate random UUID base name
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    unique_name = uuid.uuid4().hex
    
    image_extensions = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]
    
    if file_ext in image_extensions:
        try:
            image = Image.open(uploaded_file)
            
            # Convert RGBA/P modes to RGB for standard saving if necessary
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            
            # Resize image if width exceeds max_width while maintaining aspect ratio
            if image.width > max_width:
                aspect_ratio = image.height / image.width
                new_height = int(max_width * aspect_ratio)
                image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)
            
            # Save compressed image in WebP format with secure UUID name
            output_filename = f"{unique_name}.webp"
            save_path = os.path.join(MEDIA_DIR, output_filename)
            
            image.save(save_path, "WEBP", quality=quality, optimize=True)
            return save_path.replace("\\", "/")  # Normalize path separators for cross-platform compatibility
            
        except Exception as e:
            print(f"Warning: Image compression failed, saving raw file safely instead: {e}")
    
    # Non-image files or fallback: save raw uploaded file with secure UUID name
    output_filename = f"{unique_name}{file_ext}"
    save_path = os.path.join(MEDIA_DIR, output_filename)
    
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    return save_path.replace("\\", "/")