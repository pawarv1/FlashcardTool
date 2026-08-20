import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from ingest import ingest_document
from agent import agent_graph, CHROMA_COLLECTION
from quiz_engine import generate_quiz_question, grade_user_answer

# 1. PAGE CONFIG
st.set_page_config(
    page_title="Agentic RAG Study Assistant", 
    page_icon="📚", 
    layout="wide"
)

def get_existing_subjects():
    try:
        data = CHROMA_COLLECTION.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])
        subjects = set()
        for meta in metadatas:
            if meta and "subject" in meta:
                subjects.add(meta["subject"])
        return sorted(list(subjects))
    except Exception:
        return []

if "messages" not in st.session_state:
    st.session_state.messages = []
if "apkg_file" not in st.session_state:
    st.session_state.apkg_file = None
if "current_quiz" not in st.session_state:
    st.session_state.current_quiz = None
if "quiz_feedback" not in st.session_state:
    st.session_state.quiz_feedback = None

# ==========================================
# 2. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.header("Scope")
    existing_subjects = get_existing_subjects()
    dropdown_options = ["All Subjects"] + existing_subjects + ["➕ Add New Subject..."]
    
    selected_option = st.selectbox(
        "Active Subject Filter",
        options=dropdown_options,
        help="Select a subject to scope vector searches, flashcards, and quizzes."
    )
    
    if selected_option == "➕ Add New Subject...":
        active_subject = st.text_input("Enter New Subject Name", placeholder="e.g., Computer Architecture").strip()
    else:
        active_subject = selected_option

    st.divider()
    
    st.subheader("Ingest Study Notes")
    uploaded_file = st.file_uploader("Choose a file", type=["md", "pdf", "txt"])
    
    if st.button("Ingest Document", use_container_width=True):
        if uploaded_file is None:
            st.warning("Please upload a file first.")
        elif not active_subject or active_subject == "All Subjects":
            st.warning("Please specify or create a specific subject tag for this file.")
        else:
            temp_path = f"./temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner(f"Ingesting into '{active_subject}'..."):
                try:
                    ingest_document(temp_path, active_subject)
                    st.success(f"Added to '{active_subject}'!")
                    st.rerun()
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

    st.divider()

    st.subheader("Flashcard Generator")
    if st.button("Generate Anki Deck (.apkg)", use_container_width=True):
        if not active_subject or active_subject == "All Subjects":
            st.warning("Please select a specific subject tag to generate flashcards.")
        else:
            with st.spinner("Building deck..."):
                try:
                    from flashcard_generator import generate_flashcards_for_subject
                    generated_file = generate_flashcards_for_subject(active_subject)
                    if generated_file and os.path.exists(generated_file):
                        st.session_state.apkg_file = generated_file
                        st.success("Deck generated!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    if st.session_state.apkg_file and os.path.exists(st.session_state.apkg_file):
        with open(st.session_state.apkg_file, "rb") as f:
            st.download_button("📥 Download .apkg File", f, file_name=st.session_state.apkg_file, use_container_width=True)

    st.divider()
    st.subheader("Database Stats")
    try:
        st.metric(label="Total Stored Chunks", value=CHROMA_COLLECTION.count())
    except Exception:
        st.metric(label="Total Stored Chunks", value="0")

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.apkg_file = None
        st.rerun()

# ==========================================
# 3. MAIN INTERFACE WITH TABS
# ==========================================
st.title("📚 Local Agentic RAG Study Assistant")

if active_subject == "All Subjects":
    st.info("🌐 **Active Scope:** Searching across **ALL Subjects** in database.")
elif active_subject and active_subject != "➕ Add New Subject...":
    st.success(f"🎯 **Active Scope:** Focused on **{active_subject}**")

tab_chat, tab_quiz = st.tabs(["💬 Chat & Assistant", "📝 Interactive Quiz Mode"])

# ------------------------------------------
# TAB 1: CHAT INTERFACE
# ------------------------------------------
with tab_chat:
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage) and msg.content:
            with st.chat_message("assistant"):
                st.write(msg.content)

    placeholder_text = f"Ask a question about {active_subject}..." if active_subject != "All Subjects" else "Ask a question across all study notes..."

    if prompt := st.chat_input(placeholder_text):
        if not active_subject or active_subject == "➕ Add New Subject...":
            st.error("Please select or define an active subject in the sidebar before asking a question.")
        else:
            st.chat_message("user").write(prompt)
            scoped_prompt = f"[System Notice: The active study subject is '{active_subject}'. Focus your search_study_notes query on this subject if needed.]\n\n{prompt}" if active_subject != "All Subjects" else prompt

            st.session_state.messages.append(HumanMessage(content=prompt))

            with st.chat_message("assistant"):
                with st.spinner("Agent thinking & routing..."):
                    input_messages = st.session_state.messages[:-1] + [HumanMessage(content=scoped_prompt)]
                    response = agent_graph.invoke({"messages": input_messages})
                    all_graph_messages = response["messages"]
                    
                    for msg in all_graph_messages:
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tool_call in msg.tool_calls:
                                tool_name = tool_call.get("name", "tool")
                                tool_args = tool_call.get("args", {})
                                with st.status(f"🛠️ Tool Executed: `{tool_name}`", expanded=False):
                                    st.json(tool_args)
                    
                    final_answer = all_graph_messages[-1].content
                    st.write(final_answer)
                    st.session_state.messages.append(AIMessage(content=final_answer))

# ------------------------------------------
# TAB 2: INTERACTIVE QUIZ MODE
# ------------------------------------------
with tab_quiz:
    st.subheader("🧠 Test Your Knowledge")
    st.caption("Customize your quiz topic or let Llama 3.1 pick a random question from your active subject.")

    # Custom Focus Input
    user_focus_topic = st.text_input(
        "Custom Quiz Focus (Optional)", 
        placeholder="e.g., Memory leak prevention, Method overriding syntax, Recursion base cases...",
        help="Leave blank for a general question, or type a topic to generate a targeted question."
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🎲 Generate Question", use_container_width=True):
            with st.spinner("Searching notes & drafting custom question..."):
                st.session_state.current_quiz = generate_quiz_question(
                    subject_name=active_subject,
                    user_focus=user_focus_topic
                )
                st.session_state.quiz_feedback = None

    if st.session_state.current_quiz:
        quiz = st.session_state.current_quiz
        
        st.markdown("---")
        st.markdown(f"**Question:** {quiz['question']}")
        st.caption(f"Source Document: `{quiz['source_meta'].get('source_file', 'Notes')}`")
        
        user_answer = st.text_area("Your Answer (from memory):", placeholder="Type your answer here...")
        
        if st.button("Submit for Grading", type="primary"):
            if not user_answer.strip():
                st.warning("Please type an answer before submitting.")
            else:
                with st.spinner("Llama 3.1 grading answer..."):
                    st.session_state.quiz_feedback = grade_user_answer(
                        question=quiz["question"],
                        reference_context=quiz["reference_context"],
                        user_answer=user_answer
                    )

        if st.session_state.quiz_feedback:
            feedback = st.session_state.quiz_feedback
            st.markdown("---")
            
            if feedback["score"] == "Pass":
                st.success(f"✅ **{feedback['score']}** (Score: {feedback['grade_percent']}%)")
            else:
                st.error(f"⚠️ **{feedback['score']}** (Score: {feedback['grade_percent']}%)")
            
            st.write(f"**Feedback:** {feedback['feedback']}")
            
            with st.expander("📖 View Ground Truth Reference from Notes"):
                st.write(quiz["reference_context"])
    else:
        st.info("Click **'🎲 Generate Question'** above to start a practice question based on your active subject.")