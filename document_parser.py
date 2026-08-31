import io
import os
import json
import pandas as pd
from markitdown import MarkItDown

md_parser = MarkItDown()

def parse_ipynb_to_markdown(uploaded_file) -> str:
    """Extracts markdown text and code blocks from a Jupyter Notebook (.ipynb)."""
    try:
        uploaded_file.seek(0)
        notebook = json.load(uploaded_file)
        
        markdown_output = []
        cells = notebook.get("cells", [])
        
        for cell in cells:
            cell_type = cell.get("cell_type")
            raw_source = cell.get("source", "")
            
            if isinstance(raw_source, list):
                source = "".join(raw_source)
            else:
                source = str(raw_source)
            
            clean_source = source.strip()
            
            if cell_type == "markdown" and clean_source:
                markdown_output.append(clean_source)
            elif cell_type == "code" and clean_source:
                # Format code cells as markdown code blocks
                markdown_output.append(f"```python\n{clean_source}\n```")
                
        return "\n\n".join(markdown_output)
        
    except Exception as e:
        print(f"Error parsing .ipynb file: {e}")
        return ""

def parse_csv_direct_to_cards(uploaded_file) -> list[dict]:
    """
    Directly reads a CSV file and converts rows into card dictionaries,
    bypassing LLM generation to preserve exact text.
    """
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        
        df.columns = [str(col).strip().lower() for col in df.columns]
        
        front_col = next((c for c in df.columns if c in ["front", "question", "prompt", "term"]), df.columns[0])
        back_col = next((c for c in df.columns if c in ["back", "answer", "definition", "summary"]), df.columns[1] if len(df.columns) > 1 else df.columns[0])
        media_col = next((c for c in df.columns if c in ["media_link", "media", "image", "url"]), None)
        tags_col = next((c for c in df.columns if c in ["tags", "tag", "categories", "category"]), None)
        
        cards = []
        for _, row in df.iterrows():
            front_text = str(row.get(front_col, "")).strip()
            back_text = str(row.get(back_col, "")).strip()
            
            if not front_text or front_text.lower() == "nan" or not back_text or back_text.lower() == "nan":
                continue
                
            code_block = str(row.get("code_block", "")).strip() if "code_block" in df.columns and pd.notna(row.get("code_block")) else None
            explanation = str(row.get("explanation", "")).strip() if "explanation" in df.columns and pd.notna(row.get("explanation")) else None
            card_type = str(row.get("card_type", "concept")).strip() if "card_type" in df.columns and pd.notna(row.get("card_type")) else "concept"
            media_link = str(row.get(media_col, "")).strip() if media_col and pd.notna(row.get(media_col)) else None
            tags = str(row.get(tags_col, "")).strip() if tags_col and pd.notna(row.get(tags_col)) else None

            cards.append({
                "card_type": card_type,
                "front": front_text,
                "back": back_text,
                "code_block": code_block,
                "explanation": explanation,
                "media_link": media_link,
                "tags": tags,
                "source_type": f"csv_import:{uploaded_file.name}",
                "mastery_level": 0
            })
            
        return cards
        
    except Exception as e:
        print(f"Error parsing direct CSV: {e}")
        return []

def extract_text_from_file(uploaded_file) -> str:
    """
    Converts uploaded files (.txt, .md, .pdf, .docx, .pptx, .html, .ipynb)
    into structured Markdown text ready for LLM processing.
    """
    filename = uploaded_file.name.lower()
    uploaded_file.seek(0)

    if filename.endswith((".txt", ".md")):
        return uploaded_file.read().decode("utf-8", errors="replace")

    # Jupyter Notebook handling
    if filename.endswith(".ipynb"):
        return parse_ipynb_to_markdown(uploaded_file)

    file_bytes = io.BytesIO(uploaded_file.read())
    file_ext = os.path.splitext(filename)[1]
    
    try:
        result = md_parser.convert_stream(file_bytes, file_extension=file_ext)
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