import json
import random
from agent import llm, EMBEDDER, CHROMA_COLLECTION

def generate_quiz_question(subject_name: str, user_focus: str = "") -> dict:    
    where_filter = {"subject": subject_name.strip().title()} if subject_name != "All Subjects" else None

    if user_focus.strip():
        query_vec = EMBEDDER.encode(user_focus).tolist()
        results = CHROMA_COLLECTION.query(
            query_embeddings=[query_vec],
            where=where_filter,
            n_results=3
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
    else:
        results = CHROMA_COLLECTION.get(where=where_filter, limit=20)
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])

    if not docs:
        return None

    idx = random.randint(0, len(docs) - 1)
    selected_doc = docs[idx]
    selected_meta = metas[idx]

    focus_directive = f"The student explicitly requested to be tested on: '{user_focus.strip()}'." if user_focus.strip() else ""

    prompt = f"""
    You are an expert computer science professor writing a quiz question.
    Based on the following study note, write 1 conceptual or technical quiz question to test the student's memory.
    {focus_directive}

    STRICT REQUIREMENT: Return valid JSON ONLY with no markdown formatting:
    {{
      "question": "Your question here...",
      "reference_context": "Direct quote or key summary from the note containing the answer"
    }}

    STUDY NOTE:
    {selected_doc}
    """
    
    response = llm.invoke(prompt)
    raw_json = response.content.strip().replace("```json", "").replace("```", "").strip()
    
    quiz_data = json.loads(raw_json)
    quiz_data["source_meta"] = selected_meta
    return quiz_data

def grade_user_answer(question: str, reference_context: str, user_answer: str) -> dict:
    prompt = f"""
    You are an encouraging and fair computer science professor grading a short-answer quiz.

    QUESTION: {question}
    REFERENCE CONTEXT FROM NOTES: {reference_context}
    STUDENT ANSWER: {user_answer}

    GRADING RULES:
    1. Focus on CORE CONCEPTUAL ACCURACY. If the student correctly names or explains the primary term/concept asked in the question, reward them full points.
    2. Do NOT dock points if the student omitted optional programming language examples, extra metadata, or minor details present in the reference text UNLESS the question explicitly requested them.

    STRICT REQUIREMENT: Return valid JSON ONLY with no markdown formatting:
    {{
      "score": "Pass" or "Needs Review",
      "grade_percent": 85,
      "feedback": "1-2 sentences explaining what was correct or providing constructive feedback."
    }}
    """
    
    response = llm.invoke(prompt)
    raw_json = response.content.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw_json)