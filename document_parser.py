import io
import os
from markitdown import MarkItDown

def extract_text_from_file(uploaded_file) -> str:
    """
    Converts uploaded files (.txt, .md, .pdf, .docx, .pptx, .csv, .html)
    into structured Markdown text ready for LLM processing.
    """
    filename = uploaded_file.name.lower()

    if filename.endswith((".txt", ".md")):
        return uploaded_file.read().decode("utf-8")

    md = MarkItDown()
    file_bytes = io.BytesIO(uploaded_file.read())
    file_ext = os.path.splitext(filename)[1]
    
    try:
        result = md.convert_stream(file_bytes, file_extension=file_ext)
        return result.text_content
    except Exception as e:
        raise ValueError(f"Failed to parse file {filename}: {str(e)}")

def chunk_markdown_by_headers(markdown_text: str, max_chars_per_chunk: int = 10000) -> list[str]:
    """Splits Markdown text by section headers to preserve context per chunk."""
    if len(markdown_text) <= max_chars_per_chunk:
        return [markdown_text]

    sections = markdown_text.split("\n#")
    chunks = []
    current_chunk = []
    current_length = 0

    for idx, sec in enumerate(sections):
        header_prefix = "\n#" if idx > 0 else ""
        full_section = f"{header_prefix}{sec}"
        
        if current_length + len(full_section) > max_chars_per_chunk and current_chunk:
            chunks.append("".join(current_chunk))
            current_chunk = [full_section]
            current_length = len(full_section)
        else:
            current_chunk.append(full_section)
            current_length += len(full_section)

    if current_chunk:
        chunks.append("".join(current_chunk))

    return chunks

def get_markdown_chunks(uploaded_file, max_chars_per_chunk: int = 10000) -> list[str]:
    """Converts uploaded file into Markdown text then splits it into context-aware chunks."""
    markdown_text = extract_text_from_file(uploaded_file)
    return chunk_markdown_by_headers(markdown_text, max_chars_per_chunk)