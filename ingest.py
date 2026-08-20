import os
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from agent import EMBEDDER, CHROMA_COLLECTION

def ingest_document(file_path, subject_name):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found. Please check the file path.")
        return

    file_name = os.path.basename(file_path)
    file_ext = os.path.splitext(file_name)[1].lower()
    clean_subject = subject_name.strip().title()
    raw_chunks = []

    # Split according to file type
    if file_ext == ".md":
        headers = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        markdown_splitter = MarkdownHeaderTextSplitter(headers)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        raw_chunks = markdown_splitter.split_text(content)

    elif file_ext == ".pdf":
        loader = PyPDFLoader(file_path)
        pdf_pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        raw_chunks = text_splitter.split_documents(pdf_pages)

    else:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        raw_chunks = text_splitter.create_documents([content])

    documents, metadatas, ids, embeddings = [], [], [], []

    # Chunking with flexible text and metadata extraction from different file types
    for i, chunk in enumerate(raw_chunks, start=1):
        if hasattr(chunk, "page_content"):
            text = chunk.page_content.strip()
        elif isinstance(chunk, str):
            text = chunk.strip()
        else:
            text = str(chunk).strip()

        if not text:
            continue

        if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
            base_meta = chunk.metadata.copy()
        else:
            base_meta = {}

        base_meta["subject"] = clean_subject
        base_meta["source_file"] = file_name

        vector = EMBEDDER.encode(text).tolist()

        documents.append(text)
        metadatas.append(base_meta)
        ids.append(f"{file_name}_chunk_{i}")
        embeddings.append(vector)

    if documents:
        CHROMA_COLLECTION.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        print(f"\n Successfully processed '{file_name}'. Added {len(documents)} chunks under subject '{clean_subject}'.")
    else:
        print(f" Warning: No valid text chunks were extracted from '{file_name}'.")

if __name__ == "__main__":
    file_name = input("\nEnter the name of your notes file:\n")
    subject = input("\nEnter the subject name these notes related to:\n")

    ingest_document(file_name, subject)

    
