import chromadb
from chromadb.utils import embedding_functions

CHROMA_DATA_PATH = "./chroma_db_data"
COLLECTION_NAME = "study_assistant"

client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

embedding_fn = embedding_functions.DefaultEmbeddingFunction()

def get_collection():
    """Returns or creates the unified ChromaDB collection."""
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

def format_card_text(card_data: dict) -> str:
    """Formats a card object into a clean text document for vector embedding."""
    parts = [f"[Front]: {card_data.get('front', '')}"]
    
    if card_data.get("code_block"):
        parts.append(f"[Code]: {card_data['code_block']}")
        
    parts.append(f"[Back]: {card_data.get('back', '')}")
    
    if card_data.get("explanation"):
        parts.append(f"[Explanation]: {card_data['explanation']}")
        
    return "\n".join(parts)

def upsert_card_to_chroma(folder_name: str, deck_name: str, card_data: dict):
    """Adds or updates a card embedding in ChromaDB."""
    collection = get_collection()
    
    card_id = card_data["card_id"]
    document_text = format_card_text(card_data)
    
    metadata = {
        "card_id": card_id,
        "folder": folder_name.strip().replace(" ", "_"),
        "deck": deck_name.strip().replace(" ", "_").lower(),
        "card_type": card_data.get("card_type", "concept"),
        "source_type": card_data.get("source_type", "manual_entry")
    }
    
    collection.upsert(documents=[document_text], metadatas=[metadata], ids=[card_id])

def delete_card_from_chroma(card_id: str):
    """Removes a single card from ChromaDB by card_id."""
    collection = get_collection()
    try:
        collection.delete(ids=[card_id])
    except Exception as e:
        print(f"Warning: Could not delete card {card_id} from ChromaDB: {e}")

def sync_deck_to_chroma(folder_name: str, deck_name: str, cards: list):
    """Bulk updates or replaces all card embeddings for a given deck."""
    clean_folder = folder_name.strip().replace(" ", "_")
    clean_deck = deck_name.strip().replace(" ", "_").lower()
    
    collection = get_collection()
    
    # 1. Delete existing vectors for this deck
    try:
        collection.delete(
            where={
                "$and": [
                    {"folder": clean_folder},
                    {"deck": clean_deck}
                ]
            }
        )
    except Exception:
        pass  # Deck might not have had vectors yet
    
    # 2. Batch upsert remaining active cards
    if not cards:
        return

    documents = [format_card_text(c) for c in cards]
    ids = [c["card_id"] for c in cards]
    metadatas = [
        {
            "card_id": c["card_id"],
            "folder": clean_folder,
            "deck": clean_deck,
            "card_type": c.get("card_type", "concept"),
            "source_type": c.get("source_type", "manual_entry")
        }
        for c in cards
    ]
    
    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

def query_relevant_cards(query_text: str, folder_name: str = None, deck_name: str = None, n_results: int = 5) -> dict:
    """Performs semantic similarity search over stored cards with optional folder/deck filters."""
    collection = get_collection()
    
    where_filter = None
    if folder_name and deck_name:
        where_filter = {
            "$and": [
                {"folder": folder_name.strip().replace(" ", "_")},
                {"deck": deck_name.strip().replace(" ", "_").lower()}
            ]
        }
    elif folder_name:
        where_filter = {"folder": folder_name.strip().replace(" ", "_")}
        
    return collection.query(query_texts=[query_text], n_results=n_results, where=where_filter)

def delete_deck_from_chroma(folder_name: str, deck_name: str):
    """Deletes all card vectors belonging to a specific deck."""
    clean_folder = folder_name.strip().replace(" ", "_")
    clean_deck = deck_name.strip().replace(" ", "_").lower()
    collection = get_collection()
    try:
        collection.delete(
            where={
                "$and": [
                    {"folder": clean_folder},
                    {"deck": clean_deck}
                ]
            }
        )
    except Exception as e:
        print(f"Warning: Could not delete deck {clean_deck} from ChromaDB: {e}")

def delete_folder_from_chroma(folder_name: str):
    """Deletes all card vectors belonging to an entire folder."""
    clean_folder = folder_name.strip().replace(" ", "_")
    collection = get_collection()
    try:
        collection.delete(where={"folder": clean_folder})
    except Exception as e:
        print(f"Warning: Could not delete folder {clean_folder} from ChromaDB: {e}")

def rename_deck_in_chroma(folder_name: str, old_deck_name: str, new_deck_name: str):
    """Updates metadata for all cards when a deck is renamed."""
    clean_folder = folder_name.strip().replace(" ", "_")
    clean_old_deck = old_deck_name.strip().replace(" ", "_").lower()
    clean_new_deck = new_deck_name.strip().replace(" ", "_").lower()
    
    collection = get_collection()
    try:
        results = collection.get(
            where={
                "$and": [
                    {"folder": clean_folder},
                    {"deck": clean_old_deck}
                ]
            }
        )
        if results and results["ids"]:
            updated_metadatas = []
            for meta in results["metadatas"]:
                meta["deck"] = clean_new_deck
                updated_metadatas.append(meta)

            collection.update(ids=results["ids"], metadatas=updated_metadatas)
    except Exception as e:
        print(f"Warning: Could not update deck metadata in ChromaDB: {e}")

def rename_folder_in_chroma(old_folder_name: str, new_folder_name: str):
    """Updates metadata for all cards when a folder is renamed."""
    clean_old_folder = old_folder_name.strip().replace(" ", "_")
    clean_new_folder = new_folder_name.strip().replace(" ", "_")
    
    collection = get_collection()
    try:
        results = collection.get(where={"folder": clean_old_folder})
        if results and results["ids"]:
            updated_metadatas = []
            for meta in results["metadatas"]:
                meta["folder"] = clean_new_folder
                updated_metadatas.append(meta)

            collection.update(ids=results["ids"], metadatas=updated_metadatas)
    except Exception as e:
        print(f"Warning: Could not update folder metadata in ChromaDB: {e}")