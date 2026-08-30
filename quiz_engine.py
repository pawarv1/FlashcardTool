import random
from typing import Optional, Literal
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from vector_utils import get_collection

class QuizQuestionSchema(BaseModel):
    question: str = Field(description="Conceptual or technical short-answer quiz question based on the card front/back.")
    reference_context: str = Field(description="Summary or direct quote from the card containing the correct answer.")

class QuizGradeSchema(BaseModel):
    score: Literal["Pass", "Needs Review"] = Field(description="Pass if conceptually accurate, Needs Review otherwise.")
    grade_percent: int = Field(description="Numeric score from 0 to 100 based on answer accuracy.")
    feedback: str = Field(description="1-2 sentences of encouraging, constructive feedback explaining the score.")

llm = ChatOllama(model="llama3.1", temperature=0.2)
question_generator_llm = llm.with_structured_output(QuizQuestionSchema)
grader_llm = llm.with_structured_output(QuizGradeSchema)

def generate_quiz_question(folder_name: Optional[str] = None, deck_name: Optional[str] = None, user_focus: str = "") -> Optional[dict]:
    """Retrieves a card from ChromaDB and generates a tailored short-answer quiz question."""
    collection = get_collection()
    
    clean_folder = folder_name.strip().replace(" ", "_") if folder_name else None
    clean_deck = deck_name.strip().replace(" ", "_").lower() if deck_name else None

    where_filter = None
    if clean_folder and clean_deck:
        where_filter = {
            "$and": [
                {"folder": clean_folder},
                {"deck": clean_deck}
            ]
        }
    elif clean_folder:
        where_filter = {"folder": clean_folder}

    # Fetch candidate cards via vector search or random sampling with explicit includes
    if user_focus.strip():
        results = collection.query(
            query_texts=[user_focus.strip()],
            where=where_filter,
            n_results=3,
            include=["documents", "metadatas"]
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
    else:
        results = collection.get(
            where=where_filter, 
            limit=30,
            include=["documents", "metadatas"]
        )
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])

    if not docs or not metas or docs[0] is None:
        return None

    idx = random.randint(0, len(docs) - 1)
    selected_doc = docs[idx]
    selected_meta = metas[idx]

    focus_directive = f"The student explicitly requested to focus on: '{user_focus.strip()}'." if user_focus.strip() else ""

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert computer science professor writing an active-recall quiz question. Write 1 short-answer quiz question to test the student's understanding."),
        ("user", (
            f"{focus_directive}\n\n"
            f"FLASHCARD FRONT: {selected_meta.get('front', selected_doc)}\n"
            f"FLASHCARD BACK: {selected_meta.get('back', '')}\n"
            f"CARD TYPE: {selected_meta.get('card_type', 'concept')}"
        ))
    ])

    try:
        response: QuizQuestionSchema = question_generator_llm.invoke(prompt.format_messages())
        quiz_data = response.model_dump()
        quiz_data["source_meta"] = selected_meta
        return quiz_data
    except Exception as e:
        print(f"Error generating quiz question: {e}")
        return None

def grade_user_answer(question: str, reference_context: str, user_answer: str) -> dict:
    """Grades student short-answer input against the reference flashcard context."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an encouraging and fair computer science professor grading a short-answer quiz.\n"
            "GRADING RULES:\n"
            "1. Focus on CORE CONCEPTUAL ACCURACY. Reward full points if the core term or concept is correctly explained.\n"
            "2. Do NOT dock points for omitting optional syntax or minor extra details unless explicitly asked by the question."
        )),
        ("user", (
            f"QUESTION: {question}\n"
            f"REFERENCE CONTEXT: {reference_context}\n"
            f"STUDENT ANSWER: {user_answer}"
        ))
    ])

    try:
        response: QuizGradeSchema = grader_llm.invoke(prompt.format_messages())
        return response.model_dump()
    except Exception as e:
        print(f"Error grading quiz answer: {e}")
        return {
            "score": "Needs Review",
            "grade_percent": 0,
            "feedback": "An error occurred while evaluating your answer."
        }