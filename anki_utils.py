import os
import zlib
import tempfile
import genanki
from storage_utils import load_deck_cards

CARD_MODEL = genanki.Model(
    1607392319,
    'Standard Study Assistant Model',
    fields=[
        {'name': 'Front'},
        {'name': 'Back'},
        {'name': 'CodeBlock'},
        {'name': 'Explanation'},
        {'name': 'CardType'},
        {'name': 'MediaAttachment'},
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
                {{#MediaAttachment}}
                    <div class="media-container">{{MediaAttachment}}</div>
                {{/MediaAttachment}}
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
        .media-container { margin-top: 15px; text-align: center; }
        .media-container img { max-width: 100%; height: auto; border-radius: 6px; }
    '''
)

def generate_anki_deck_bytes(folder_name: str, deck_name: str) -> bytes:
    """Fetches active cards from SQLite and builds an in-memory .apkg file with packaged media assets."""
    cards = load_deck_cards(folder_name, deck_name)

    if not cards:
        raise ValueError(f"No active cards found in deck '{deck_name}' under folder '{folder_name}'.")

    clean_folder = folder_name.strip().replace(" ", "_")
    clean_deck = deck_name.strip().replace(" ", "_").lower()

    # Deterministic deck ID using Adler32 CRC hashing (cross-session stable)
    deck_identifier_string = f"{clean_folder}::{clean_deck}"
    deck_id = (zlib.adler32(deck_identifier_string.encode('utf-8')) & 0xffffffff) % (10**9)
    display_title = f"{folder_name} :: {deck_name}"
    
    anki_deck = genanki.Deck(deck_id, display_title)
    media_files = []

    for card in cards:
        front = card.get("front", "")
        back = card.get("back", "")
        code_block = card.get("code_block") or ""
        explanation = card.get("explanation") or ""
        card_type = card.get("card_type", "concept")
        media_link = card.get("media_link") or ""

        media_html = ""
        if media_link:
            if media_link.lower().startswith(("http://", "https://")):
                media_html = f'<img src="{media_link}">'
            elif os.path.exists(media_link):
                filename = os.path.basename(media_link)
                media_files.append(media_link)
                media_html = f'<img src="{filename}">'
            else:
                media_html = f'<a href="{media_link}">View Attachment</a>'

        tags_str = card.get("tags") or ""
        card_tags = [t.strip().replace(" ", "_") for t in tags_str.split(",") if t.strip()]

        note = genanki.Note(
            model=CARD_MODEL,
            fields=[front, back, code_block, explanation, card_type, media_html],
            tags=card_tags
        )
        anki_deck.add_note(note)

    # Safe temp file handling with guaranteed cleanup
    with tempfile.NamedTemporaryFile(suffix=".apkg", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        package = genanki.Package(anki_deck)
        if media_files:
            package.media_files = list(set(media_files))
            
        package.write_to_file(tmp_path)

        with open(tmp_path, "rb") as f:
            file_bytes = f.read()
            
        return file_bytes
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)