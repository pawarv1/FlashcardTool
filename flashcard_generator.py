import json
import random
import genanki
from agent import CHROMA_COLLECTION, llm

MODEL_ID = 1607392319
ANKI_MODEL = genanki.Model(
    MODEL_ID,
    'Simple Model',
    fields=[
        {'name': 'Question'},
        {'name': 'Answer'},
    ],
    templates=[
        {
            'name': 'Card 1',
            'qfmt': '<div style="font-size: 20px; text-align: center;">{{Question}}</div>',
            'afmt': '{{FrontSide}}<hr id="answer"><div style="font-size: 18px; text-align: center; color: #2a7ae2;">{{Answer}}</div>',
        },
    ]
)

def generate_flashcards_for_subject(subject_name: str, num_chunks: int = 5) -> str:
    results = CHROMA_COLLECTION.get(
        where={"subject": subject_name.strip().title()},
        limit=num_chunks
    )
    
    documents = results.get("documents", [])
    if not documents:
        return None

    combined_text = "\n\n---\n\n".join(documents)

    prompt = f"""
    You are an expert study assistant. Extract key concepts from the following study notes 
    and format them as 5-10 concise Flashcards (Question and Answer pairs).

    STRICT REQUIREMENT: Output valid JSON ONLY in the following format, with no markdown formatting or extra text:
    [
      {{"question": "What is encapsulation?", "answer": "Bundling data and methods into a single unit..."}},
      {{"question": "What is inheritance?", "answer": "Acquiring properties of a parent class..."}}
    ]

    NOTES:
    {combined_text}
    """

    response = llm.invoke(prompt)
    
    raw_json = response.content.strip()
    if raw_json.startswith("```json"):
        raw_json = raw_json.replace("```json", "").replace("```", "").strip()
    
    cards = json.loads(raw_json)

    deck_id = random.randrange(1 << 30, 1 << 31)
    deck = genanki.Deck(deck_id, f"Study Deck: {subject_name}")

    for card in cards:
        note = genanki.Note(
            model=ANKI_MODEL,
            fields=[card["question"], card["answer"]]
        )
        deck.add_note(note)

    output_filename = f"{subject_name.lower().replace(' ', '_')}_deck.apkg"
    genanki.Package(deck).write_to_file(output_filename)
    
    return output_filename