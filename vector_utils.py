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

def upsert_card_to_chroma(folder_name: str, deck_name: str, card_data: dict):
    """Adds or updates a card embedding in ChromaDB using only the front text."""
    collection = get_collection()
    
    card_id = card_data["card_id"]
    front_text = card_data.get("front", "").strip()
    
    metadata = {
        "card_id": card_id,
        "front": front_text,
        "back": card_data.get("back", ""),
        "folder": folder_name.strip().replace(" ", "_"),
        "deck": deck_name.strip().replace(" ", "_").lower(),
        "card_type": card_data.get("card_type", "concept"),
        "media_link": card_data.get("media_link") or "",
        "tags": card_data.get("tags") or "",
        "source_type": card_data.get("source_type", "manual_entry")
    }
    
    collection.upsert(documents=[front_text], metadatas=[metadata], ids=[card_id])

def delete_card_from_chroma(card_id: str):
    """Removes a single card from ChromaDB by card_id."""
    collection = get_collection()
    try:
        collection.delete(ids=[card_id])
    except Exception as e:
        print(f"Warning: Could not delete card {card_id} from ChromaDB: {e}")

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

def check_candidate_duplicates(candidate_cards: list, folder_name: str = None, deck_name: str = None, distance_threshold: float = 0.45) -> list:
    """
    Checks candidate cards against ChromaDB AND intra-batch using vector similarity and normalized text.
    """
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

    seen_fronts = []
    seen_backs = []

    for card in candidate_cards:
        is_duplicate = False
        matched_text = None
        
        front_text = card.get("front", "").strip()
        back_text = card.get("back", "").strip()
        
        clean_front = "".join(c for c in front_text.lower() if c.isalnum() or c.isspace())
        clean_back = "".join(c for c in back_text.lower() if c.isalnum() or c.isspace())

        # 1. Intra-batch check (Check if back answer is identical OR normalized question overlaps)
        for idx, (seen_f, seen_b) in enumerate(zip(seen_fronts, seen_backs)):
            if clean_back and clean_back == seen_b:
                is_duplicate = True
                matched_text = f"Identical answer to Card #{idx+1} in this batch"
                break
            
            if clean_front and (clean_front in seen_f or seen_f in clean_front):
                is_duplicate = True
                matched_text = f"Similar question to Card #{idx+1} in this batch"
                break

        # 2. ChromaDB search (for existing cards in database)
        if not is_duplicate and collection.count() > 0:
            try:
                results = collection.query(
                    query_texts=[front_text],
                    n_results=1,
                    where=where_filter
                )

                if results and results.get("distances") and len(results["distances"][0]) > 0:
                    top_distance = results["distances"][0][0]
                    
                    if top_distance < distance_threshold:
                        is_duplicate = True
                        matched_meta = results["metadatas"][0][0] if results.get("metadatas") and len(results["metadatas"][0]) > 0 else {}
                        doc_match = results["documents"][0][0] if results.get("documents") and len(results["documents"][0]) > 0 else ""
                        matched_text = matched_meta.get("front") or doc_match
            except Exception as e:
                print(f"Warning: Deduplication query error: {e}")

        card["is_duplicate"] = is_duplicate
        card["matched_existing_front"] = matched_text
        
        seen_fronts.append(clean_front)
        seen_backs.append(clean_back)

    return candidate_cards