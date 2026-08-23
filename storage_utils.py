import os
import json
import uuid
import shutil
from datetime import datetime
from vector_utils import (
    upsert_card_to_chroma,
    delete_card_from_chroma,
    sync_deck_to_chroma,
    delete_deck_from_chroma,
    delete_folder_from_chroma,
    rename_deck_in_chroma,
    rename_folder_in_chroma
)

BASE_DECKS_DIR = "./decks"

def ensure_deck_structure():
    """Ensures the base decks directory exists."""
    if not os.path.exists(BASE_DECKS_DIR):
        os.makedirs(BASE_DECKS_DIR)

def create_folder(folder_name: str):
    """Creates a folder"""
    ensure_deck_structure()
    clean_folder_name = folder_name.strip().replace(" ", "_")
    folder_path = os.path.join(BASE_DECKS_DIR, clean_folder_name)
    os.makedirs(folder_path, exist_ok=True)

def create_deck(folder_name: str, deck_name: str):
    """Creates a deck"""
    ensure_deck_structure()
    clean_folder_name = folder_name.strip().replace(" ", "_")
    clean_deck_name = deck_name.strip().replace(" ", "_").lower()
    folder_path = os.path.join(BASE_DECKS_DIR, clean_folder_name)
    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, f"{clean_deck_name}.json")

    if os.path.exists(file_path):
        return
    
    deck_content = {
        "deck_name": deck_name,
        "folder": folder_name,
        "cards": []
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(deck_content, f, indent=2)

def save_card_to_json(folder_name: str, deck_name: str, card_data: dict):
    """Saves or appends a card to a specific deck JSON file."""
    clean_folder_name = folder_name.strip().replace(" ", "_")
    clean_deck_name = deck_name.strip().replace(" ", "_").lower()
    file_path = os.path.join(BASE_DECKS_DIR, clean_folder_name, f"{clean_deck_name}.json")

    with open(file_path, "r", encoding="utf-8") as f:
        deck_content = json.load(f)

    full_card = {
        "card_id": card_data.get("card_id") or f"card_{uuid.uuid4().hex[:8]}",
        "card_type": card_data.get("card_type", "concept"),
        "front": card_data.get("front", ""),
        "code_block": card_data.get("code_block", None),
        "back": card_data.get("back", ""),
        "explanation": card_data.get("explanation", None),
        "source_type": card_data.get("source_type", "manual_entry"),
        "mastery_level": card_data.get("mastery_level", 0),
        "media_url": card_data.get("media_url", None),
        "created_at": card_data.get("created_at") or datetime.now().strftime("%Y-%m-%d")
    }

    deck_content["cards"].append(full_card)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(deck_content, f, indent=2)

    upsert_card_to_chroma(folder_name, deck_name, full_card)

def get_folders() -> list:
    """Returns all folder names in the base directory"""
    ensure_deck_structure()
    return [f for f in os.listdir(BASE_DECKS_DIR) if os.path.isdir(os.path.join(BASE_DECKS_DIR, f))]

def get_decks(folder_name: str) -> list:
    """Returns all decks in the folder"""
    ensure_deck_structure()
    clean_folder_name = folder_name.strip().replace(" ", "_")
    folder_path = os.path.join(BASE_DECKS_DIR, clean_folder_name)

    if not os.path.exists(folder_path):
        return []

    return [
        f[:-5] for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(".json")
    ]

def load_deck(folder_name: str, deck_name: str) -> dict:
    """Loads a full deck JSON object from disk."""
    clean_folder_name = folder_name.strip().replace(" ", "_")
    clean_deck_name = deck_name.strip().replace(" ", "_").lower()
    file_path = os.path.join(BASE_DECKS_DIR, clean_folder_name, f"{clean_deck_name}.json")

    if os.path.exists(file_path):        
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
        
    return {"deck_name": deck_name, "folder": folder_name, "cards": []}

def update_single_card(folder_name: str, deck_name: str, updated_card: dict):
    """Updates a single card in a JSON deck by matching its card_id."""
    clean_folder_name = folder_name.strip().replace(" ", "_")
    clean_deck_name = deck_name.strip().replace(" ", "_").lower()
    file_path = os.path.join(BASE_DECKS_DIR, clean_folder_name, f"{clean_deck_name}.json")

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            deck_content = json.load(f)

        cards = deck_content.get("cards", [])
        for i, card in enumerate(cards):
            if card.get("card_id") == updated_card.get("card_id"):
                cards[i] = updated_card
                break

        deck_content["cards"] = cards

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(deck_content, f, indent=2)

        upsert_card_to_chroma(folder_name, deck_name, updated_card)

def delete_single_card(folder_name: str, deck_name: str, card_id: str):
    """Deletes a single card from a JSON deck by matching its card_id."""
    clean_folder_name = folder_name.strip().replace(" ", "_")
    clean_deck_name = deck_name.strip().replace(" ", "_").lower()
    file_path = os.path.join(BASE_DECKS_DIR, clean_folder_name, f"{clean_deck_name}.json")

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            deck_content = json.load(f)

        cards = deck_content.get("cards", [])
        updated_cards = [c for c in cards if c.get("card_id") != card_id]

        deck_content["cards"] = updated_cards

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(deck_content, f, indent=2)

        delete_card_from_chroma(card_id)

def update_deck_cards(folder_name: str, deck_name: str, updated_cards: list):
    """Overwrites the cards array for a deck (used for edits, deletes, or reordering)."""
    clean_folder_name = folder_name.strip().replace(" ", "_")
    clean_deck_name = deck_name.strip().replace(" ", "_").lower()
    file_path = os.path.join(BASE_DECKS_DIR, clean_folder_name, f"{clean_deck_name}.json")

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            deck_content = json.load(f)

        deck_content["cards"] = updated_cards

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(deck_content, f, indent = 2)

        sync_deck_to_chroma(folder_name, deck_name, updated_cards)

def delete_deck(folder_name: str, deck_name: str):
    """Deletes a deck file from disk and removes its vectors from ChromaDB."""
    clean_folder = folder_name.strip().replace(" ", "_")
    clean_deck = deck_name.strip().replace(" ", "_").lower()
    file_path = os.path.join(BASE_DECKS_DIR, clean_folder, f"{clean_deck}.json")

    if os.path.exists(file_path):
        os.remove(file_path)

    delete_deck_from_chroma(folder_name, deck_name)

def delete_folder(folder_name: str):
    """Deletes an entire folder directory and removes its vectors from ChromaDB."""
    clean_folder = folder_name.strip().replace(" ", "_")
    folder_path = os.path.join(BASE_DECKS_DIR, clean_folder)

    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

    delete_folder_from_chroma(folder_name)

def rename_deck(folder_name: str, old_deck_name: str, new_deck_name: str):
    """Renames a deck JSON file and updates ChromaDB metadata."""
    clean_folder = folder_name.strip().replace(" ", "_")
    clean_old_deck = old_deck_name.strip().replace(" ", "_").lower()
    clean_new_deck = new_deck_name.strip().replace(" ", "_").lower()

    old_file_path = os.path.join(BASE_DECKS_DIR, clean_folder, f"{clean_old_deck}.json")
    new_file_path = os.path.join(BASE_DECKS_DIR, clean_folder, f"{clean_new_deck}.json")

    if not os.path.exists(old_file_path):
        return

    with open(old_file_path, "r", encoding="utf-8") as f:
        deck_content = json.load(f)

    deck_content["deck_name"] = new_deck_name

    with open(new_file_path, "w", encoding="utf-8") as f:
        json.dump(deck_content, f, indent=2)

    os.remove(old_file_path)

    rename_deck_in_chroma(folder_name, old_deck_name, new_deck_name)

def rename_folder(old_folder_name: str, new_folder_name: str):
    """Renames a folder directory and updates internal JSON and ChromaDB metadata."""
    clean_old_folder = old_folder_name.strip().replace(" ", "_")
    clean_new_folder = new_folder_name.strip().replace(" ", "_")

    old_folder_path = os.path.join(BASE_DECKS_DIR, clean_old_folder)
    new_folder_path = os.path.join(BASE_DECKS_DIR, clean_new_folder)

    if not os.path.exists(old_folder_path):
        return

    os.rename(old_folder_path, new_folder_path)

    for file_name in os.listdir(new_folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(new_folder_path, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                deck_content = json.load(f)

            deck_content["folder"] = new_folder_name

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(deck_content, f, indent=2)

    rename_folder_in_chroma(old_folder_name, new_folder_name)