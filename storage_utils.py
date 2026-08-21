import os
import json

BASE_DECKS_DIR = "./decks"

def ensure_deck_structure():
    if not os.path.exists(BASE_DECKS_DIR):
        os.makedirs(BASE_DECKS_DIR)

def create_folder(folder_name: str):
    ensure_deck_structure()
    clean_folder_name = folder_name.strip().replace(" ", "_")
    folder_path = os.path.join(BASE_DECKS_DIR, clean_folder_name)
    os.makedirs(folder_path, exist_ok=True)

def create_deck(folder_name: str, deck_name: str):
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
    clean_folder_name = folder_name.strip().replace(" ", "_")
    clean_deck_name = deck_name.strip().replace(" ", "_").lower()
    file_path = os.path.join(BASE_DECKS_DIR, clean_folder_name, f"{clean_deck_name}.json")

    with open(file_path, "r", encoding="utf-8") as f:
        deck_content = json.load(f)

    deck_content["cards"].append(card_data)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(deck_content, f, indent=2)

def get_folders() -> list:
    ensure_deck_structure()
    return [f for f in os.listdir(BASE_DECKS_DIR) if os.path.isdir(os.path.join(BASE_DECKS_DIR, f))]

def get_decks(folder_name: str) -> list:
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
    clean_folder_name = folder_name.strip().replace(" ", "_")
    clean_deck_name = deck_name.strip().replace(" ", "_").lower()
    file_path = os.path.join(BASE_DECKS_DIR, clean_folder_name, f"{clean_deck_name}.json")

    if os.path.exists(file_path):        
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
        
    return {"deck_name": deck_name, "folder": folder_name, "cards": []}