from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

CardTypeEnum = Literal["concept", "code_snippet", "definition", "formula", "comparison", "example"]

class ProposedCard(BaseModel):
    card_type: CardTypeEnum = Field(
        default="concept",
        description="Choose exact match: 'concept', 'code_snippet', 'definition', 'formula', 'comparison' or 'example"
    )
    front: str = Field(description="Clear, concise active-recall question or prompt")
    code_block: Optional[str] = Field(default=None, description="Optional code snippet relevant to question")
    back: str = Field(description="Direct, accurate answer")
    explanation: Optional[str] = Field(default=None, description="Memory hook or context snippet from source")

class FlashcardDraftResponse(BaseModel):
    proposed_cards: List[ProposedCard]

def generate_flashcards_from_chunks(document_chunks: list[str], user_instructions: Optional[str] = None, target_count: int = 10) -> list[dict]:
    """Generates candidate flashcards safely by batching small LLM output requests."""

    if not document_chunks:
        return []

    llm = ChatOllama(model="llama3.1", temperature=0.2)
    structured_llm = llm.with_structured_output(FlashcardDraftResponse)

    system_message = (
        "You are an expert tutor creating active-recall study flashcards. "
        "Analyze the provided source text and extract key concepts into clear flashcards."
    )
    if user_instructions and user_instructions.strip():
        system_message += f"\n\nCRITICAL USER INSTRUCTIONS:\n{user_instructions.strip()}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "Target Card Count: {target_count}\n\nDocument Text:\n{document_text}")
    ])

    chain = prompt | structured_llm

    # Cap requests to a max of 4 cards per LLM call for JSON schema reliability
    MAX_PER_CALL = 4
    all_generated_cards = []

    cards_per_chunk = max(1, target_count // len(document_chunks))

    for chunk in document_chunks:
        chunk_cards_needed = cards_per_chunk
        chunk_generated = 0
        max_attempts = 3 # Prevent infinite loops if LLM struggles
        attempt = 0

        while chunk_generated < chunk_cards_needed and attempt < max_attempts:
            attempt += 1
            needed_this_pass = min(chunk_cards_needed - chunk_generated, MAX_PER_CALL)
            
            try:
                response: FlashcardDraftResponse = chain.invoke({
                    "target_count": needed_this_pass,
                    "document_text": chunk
                })

                if response and response.proposed_cards:
                    new_cards = [card.model_dump() for card in response.proposed_cards]
                    all_generated_cards.extend(new_cards)
                    chunk_generated += len(new_cards)

                    if len(all_generated_cards) >= target_count:
                        return all_generated_cards

            except Exception as e:
                print(f"Warning: Batch generation pass failed: {e}")
                break

    return all_generated_cards

def generate_remediation_cards(question: str, reference_context: str, user_answer: str, feedback: str) -> List[dict]:
    """Generates 2-3 focused flashcards to remediate a failed quiz response."""
    
    system_prompt = (
        "You are an expert tutor helping a student master a concept they just got wrong on a quiz.\n"
        "Generate 2 to 3 high-yield remediation flashcards that specifically address the gap, "
        "misconception, or missing detail identified in the student's answer. "
        "Keep questions direct and answers structured for flashcard review."
    )

    user_prompt = (
        f"MISSED QUESTION: {question}\n\n"
        f"REFERENCE CONTEXT: {reference_context}\n\n"
        f"STUDENT'S ANSWER: {user_answer}\n\n"
        f"FEEDBACK / EVALUATION: {feedback}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt)
    ])
    
    llm = ChatOllama(model="llama3.1", temperature=0.3)
    
    structured_llm = llm.with_structured_output(FlashcardDraftResponse)
    
    chain = prompt | structured_llm
    
    try:
        response: FlashcardDraftResponse = chain.invoke({})
        if response and response.proposed_cards:
            return [card.model_dump() for card in response.proposed_cards]
    except Exception as e:
        print(f"Warning: Remediation generation failed: {e}")
        
    return []