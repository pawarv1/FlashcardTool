import os
import json
import genanki
import random

BASE_DECKS_DIR = "./decks"

CARD_MODEL = genanki.Model(
    1607392319,
    'Standard Study Assistant Model',
    fields=[
        {'name': 'Front'},
        {'name': 'Back'},
        {'name': 'CodeBlock'},
        {'name': 'Explanation'},
        {'name': 'CardType'},
    ],
    templates=[
        {
            'name': 'Card 1',
            'qfmt': '''
                <div class="card-type">{{CardType}}</div>
                <div class="front">{{Front}}</div>
                {{#CodeBlock}}
                    <pre><code>{{CodeBlock}}</code></pre>
                {{/CodeBlock}}
            ''',
            'afmt': '''
                {{FrontSide}}
                <hr id="answer">
                <div class="back">{{Back}}</div>
                {{#Explanation}}
                    <div class="explanation"><strong>Explanation:</strong> {{Explanation}}</div>
                {{/Explanation}}
            ''',
        },
    ],
    css='''
        .card { font-family: arial; font-size: 18px; text-align: left; color: #333; background-color: #fcfcfc; padding: 20px; }
        .card-type { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #007acc; font-weight: bold; margin-bottom: 10px; }
        .front { font-size: 20px; font-weight: 600; margin-bottom: 12px; }
        .back { font-size: 18px; line-height: 1.5; color: #222; }
        pre { background: #282c34; color: #abb2bf; padding: 12px; border-radius: 6px; overflow-x: auto; font-family: monospace; }
        .explanation { margin-top: 15px; font-size: 14px; color: #666; font-style: italic; border-left: 3px solid #007acc; padding-left: 10px; }
    '''
)

def generate_anki_deck_bytes(folder_name: str, deck_name: str) -> bytes:
    """Reads JSON deck from disk and returns .apkg file as bytes for Streamlit downloading."""
    clean_folder = folder_name.strip().replace(" ", "_")
    clean_deck = deck_name.strip().replace(" ", "_").lower()
    
    deck_path = os.path.join(BASE_DECKS_DIR, clean_folder, f"{clean_deck}.json")
    
    if not os.path.exists(deck_path):
        raise FileNotFoundError(f"Deck file not found at {deck_path}")

    with open(deck_path, "r", encoding="utf-8") as f:
        deck_data = json.load(f)

    # Generate a unique deterministic ID for the Anki deck based on folder and deck name
    deck_id = abs(hash(f"{clean_folder}_{clean_deck}")) % (10**9)
    display_title = f"{folder_name} :: {deck_data.get('deck_name', deck_name)}"
    
    anki_deck = genanki.Deck(deck_id, display_title)

    for card in deck_data.get("cards", []):
        front = card.get("front", "")
        back = card.get("back", "")
        code_block = card.get("code_block") or ""
        explanation = card.get("explanation") or ""
        card_type = card.get("card_type", "concept")

        note = genanki.Note(
            model=CARD_MODEL,
            fields=[front, back, code_block, explanation, card_type]
        )
        anki_deck.add_note(note)

    output_path = f"/tmp/{clean_deck}.apkg"
    genanki.Package(anki_deck).write_to_file(output_path)

    with open(output_path, "rb") as f:
        file_bytes = f.read()

    if os.path.exists(output_path):
        os.remove(output_path)

    return file_bytes