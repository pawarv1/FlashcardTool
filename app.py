import os
import random
from typing import get_args
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from document_parser import get_markdown_chunks
from card_generator import generate_flashcards_from_chunks, CardTypeEnum
from vector_utils import check_candidate_duplicates
from storage_utils import (
    create_folder, 
    create_deck, 
    get_all_folders, 
    get_decks_in_folder, 
    rename_folder,
    delete_folder,
    rename_deck,
    delete_deck,
    save_card,
    save_card_batch,
    load_deck_cards, 
    delete_card,
)
from anki_utils import generate_anki_deck_bytes
from agent import agent_graph
from quiz_engine import generate_quiz_question, grade_user_answer
from db import init_db, auto_heal_chroma_sync
from sm2_utils import calculate_sm2

# 1. DATABASE & VECTOR INITIALIZATION
@st.cache_resource
def setup_database():
    """Runs database initialization and vector sync healing once per app startup."""
    init_db()
    auto_heal_chroma_sync()
    return True

setup_database()

CARD_TYPES = get_args(CardTypeEnum) if get_args(CardTypeEnum) else ["concept", "code_snippet", "definition", "formula", "comparison", "example"]

# PAGE CONFIG
st.set_page_config(
    page_title="Agentic RAG Study Assistant", 
    page_icon="📚", 
    layout="wide"
)

st.title("Flashcard tool")

if "current_folder" not in st.session_state:
    st.session_state.current_folder = None

if "current_deck" not in st.session_state:
    st.session_state.current_deck = None

if "study_card_index" not in st.session_state:
    st.session_state.study_card_index = 0

if "study_is_flipped" not in st.session_state:
    st.session_state.study_is_flipped = False

def render_media(media_url: str):
    if not media_url:
        return
    clean_url = media_url.lower()
    if any(clean_url.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif", ".webp"]):
        st.image(media_url, use_column_width=True)
    else:
        st.markdown(f"🔗 [View Attached Media]({media_url})")

# DIALOG POPUPS
@st.dialog("New Folder")
def new_folder_popup():
    new_folder_name = st.text_input("Enter New Folder Name")
    if st.button("Create"):
        if new_folder_name.strip():
            create_folder(new_folder_name.strip())
            st.rerun()

@st.dialog("Rename Folder")
def rename_folder_popup(folder_name: str):
    new_name = st.text_input("New Folder Name", value=folder_name)
    if st.button("Save Name", type="primary"):
        if new_name.strip() and new_name.strip() != folder_name:
            rename_folder(folder_name, new_name.strip())
            if st.session_state.current_folder == folder_name:
                st.session_state.current_folder = new_name.strip()
            st.rerun()

@st.dialog("New Deck")
def new_deck_popup():
    new_deck_name = st.text_input("Enter New Deck Name")
    if st.button("Create"):
        if new_deck_name.strip():
            create_deck(st.session_state.current_folder, new_deck_name.strip())
            st.rerun()

@st.dialog("Rename Deck")
def rename_deck_popup(deck_name: str):
    new_name = st.text_input("New Deck Name", value=deck_name)
    if st.button("Save Name", type="primary"):
        if new_name.strip() and new_name.strip() != deck_name:
            rename_deck(st.session_state.current_folder, deck_name, new_name.strip())
            if st.session_state.current_deck == deck_name:
                st.session_state.current_deck = new_name.strip()
            st.rerun()

@st.dialog("Add New Card")
def new_card_popup():
    card_type = st.selectbox("Card Type", CARD_TYPES, index=0)
    front = st.text_area("Front (Question / Prompt)")
    back = st.text_area("Back (Answer / Summary)")
    code_block = st.text_area("Code Block (Optional)", help="Paste any relevant code snippet here")
    explanation = st.text_area("Explanation (Optional)", help="Additional detail or context for review")
    media_link = st.text_input("Media Link / URL (Optional)", help="Paste image or reference URL")

    if st.button("Save Card", type="primary"):
        if front.strip() and back.strip():
            card_data = {
                "card_type": card_type,
                "front": front.strip(),
                "back": back.strip(),
                "code_block": code_block.strip() if code_block.strip() else None,
                "explanation": explanation.strip() if explanation.strip() else None,
                "media_link": media_link.strip() if media_link.strip() else None,
                "source_type": "manual_entry",
                "mastery_level": 0
            }
            save_card(
                folder_name=st.session_state.current_folder,
                deck_name=st.session_state.current_deck,
                card_data=card_data
            )
            st.success("Card added.")
            st.rerun()
        else:
            st.warning("Both Front and Back text are required.")

@st.dialog("Edit Card")
def edit_card_popup(card: dict):
    curr_type = card.get("card_type", "concept")
    type_idx = CARD_TYPES.index(curr_type) if curr_type in CARD_TYPES else 0

    updated_type = st.selectbox("Card Type", CARD_TYPES, index=type_idx)
    updated_front = st.text_area("Front (Question / Prompt)", value=card.get("front", ""))
    updated_back = st.text_area("Back (Answer / Summary)", value=card.get("back", ""))
    updated_code = st.text_area("Code Block (Optional)", value=card.get("code_block") or "")
    updated_explanation = st.text_area("Explanation (Optional)", value=card.get("explanation") or "")
    updated_media = st.text_input("Media Link / URL (Optional)", value=card.get("media_link") or "")

    st.divider()

    col_save, col_delete = st.columns([1, 1])

    with col_save:
        if st.button("Save Changes", type="primary", use_container_width=True):
            if updated_front.strip() and updated_back.strip():
                updated_card = {
                    "card_id": card.get("card_id"),
                    "card_type": updated_type,
                    "front": updated_front.strip(),
                    "back": updated_back.strip(),
                    "code_block": updated_code.strip() if updated_code.strip() else None,
                    "explanation": updated_explanation.strip() if updated_explanation.strip() else None,
                    "media_link": updated_media.strip() if updated_media.strip() else None,
                    "source_type": card.get("source_type", "manual_entry"),
                    "mastery_level": card.get("mastery_level", 0)
                }
                save_card(
                    folder_name=st.session_state.current_folder,
                    deck_name=st.session_state.current_deck,
                    card_data=updated_card
                )
                st.rerun()
            else:
                st.warning("Both Front and Back text are required.")

    with col_delete:
        with st.expander("🗑️ Delete Card"):
            st.warning("Delete this card permanently?")
            if st.button("Yes, Delete", type="primary", use_container_width=True, key=f"del_confirm_{card.get('card_id')}"):
                delete_card(card_id=card.get("card_id"))
                st.rerun()

# DOCUMENT INGESTION & AI DRAFTING DIALOG
@st.dialog("📄 Ingest Document & Generate Cards", width="large")
def ingest_document_popup():
    uploaded_file = st.file_uploader(
        "Upload Study Document", 
        type=["pdf", "docx", "pptx", "txt", "md", "csv", "html"]
    )
    instructions = st.text_input(
        "Focus Directive (Optional)", 
        placeholder="e.g. Focus on search algorithms and time complexity"
    )
    num_cards = st.slider("Target Number of Cards", min_value=3, max_value=20, value=8)

    if "draft_cards" not in st.session_state:
        st.session_state.draft_cards = None

    if st.button("⚙️ Parse & Generate Drafts", type="primary", disabled=not uploaded_file):
        with st.spinner("Parsing document and drafting flashcards via Llama 3.1..."):
            document_chunks = get_markdown_chunks(uploaded_file)
            
            raw_drafts = generate_flashcards_from_chunks(
                document_chunks=document_chunks, 
                user_instructions=instructions, 
                target_count=num_cards
            )
            
            flagged_drafts = check_candidate_duplicates(
                candidate_cards=raw_drafts,
                folder_name=st.session_state.current_folder,
                deck_name=st.session_state.current_deck
            )
            st.session_state.draft_cards = flagged_drafts

    # DRAFTING TABLE (HUMAN-IN-THE-LOOP REVIEW)
    if st.session_state.draft_cards:
        st.divider()
        st.subheader("📋 Drafting Table (Review & Edit)")
        
        selected_indices = []
        edited_drafts = []

        for i, card in enumerate(st.session_state.draft_cards):
            is_dup = card.get("is_duplicate", False)
            matched_q = card.get("matched_existing_front", "")

            with st.expander(f"Card {i+1}: {card.get('front', '')[:50]}...", expanded=not is_dup):
                if is_dup:
                    st.warning(f"⚠️ Potential Duplicate of: *\"{matched_q}\"*")

                col_check, col_content = st.columns([1, 10])
                with col_check:
                    should_import = st.checkbox("Import", value=not is_dup, key=f"import_chk_{i}")
                    if should_import:
                        selected_indices.append(i)

                with col_content:
                    c_front = st.text_area("Front", value=card.get("front", ""), key=f"draft_f_{i}")
                    c_back = st.text_area("Back", value=card.get("back", ""), key=f"draft_b_{i}")
                    
                    curr_c_type = card.get("card_type", "concept")
                    type_c_idx = CARD_TYPES.index(curr_c_type) if curr_c_type in CARD_TYPES else 0
                    c_type = st.selectbox("Type", CARD_TYPES, index=type_c_idx, key=f"draft_t_{i}")
                    
                    c_code = st.text_area("Code Block", value=card.get("code_block") or "", key=f"draft_c_{i}")
                    c_exp = st.text_area("Explanation", value=card.get("explanation") or "", key=f"draft_e_{i}")

                    edited_drafts.append({
                        "card_type": c_type,
                        "front": c_front.strip(),
                        "back": c_back.strip(),
                        "code_block": c_code.strip() if c_code.strip() else None,
                        "explanation": c_exp.strip() if c_exp.strip() else None,
                        "media_link": None,
                        "source_type": f"doc_ingest:{uploaded_file.name if uploaded_file else 'file'}",
                        "mastery_level": 0
                    })

        st.divider()
        if st.button(f"💾 Save {len(selected_indices)} Selected Cards", type="primary", use_container_width=True):
            cards_to_import = [edited_drafts[idx] for idx in selected_indices]
            save_card_batch(
                folder_name=st.session_state.current_folder,
                deck_name=st.session_state.current_deck,
                cards=cards_to_import
            )
            st.session_state.draft_cards = None
            st.success(f"Successfully imported {len(selected_indices)} cards!")
            st.rerun()

# NAVIGATION LOGIC

# VIEW 3: INDIVIDUAL DECK VIEW
if st.session_state.current_folder is not None and st.session_state.current_deck is not None:
    col_nav, col_actions = st.columns([3, 1])
    with col_nav:
        if st.button(f"⬅️ Back to Folder: {st.session_state.current_folder}"):
            st.session_state.current_deck = None
            st.session_state.study_card_index = 0
            st.session_state.study_is_flipped = False
            st.rerun()

    with col_actions:
        col_ren, col_del = st.columns(2)
        with col_ren:
            if st.button("✏️ Rename Deck", use_container_width=True):
                rename_deck_popup(st.session_state.current_deck)
        with col_del:
            with st.popover("🗑️ Delete Deck", use_container_width=True):
                st.warning("Delete this entire deck and all cards?")
                if st.button("Yes, Delete Deck", type="primary", use_container_width=True):
                    delete_deck(st.session_state.current_folder, st.session_state.current_deck)
                    st.session_state.current_deck = None
                    st.rerun()

    st.header(f"Deck: {st.session_state.current_deck}")
    st.caption(f"Folder: {st.session_state.current_folder}")
    st.divider()

    deck_tab_cards, deck_tab_study, deck_tab_chat, deck_tab_quiz = st.tabs(["🎴 Cards", "📖 Study Mode", "💬 Chat Assistant", "📝 AI Quiz"])

    cards = load_deck_cards(st.session_state.current_folder, st.session_state.current_deck)

    # SUB-TAB 1: CARD LIST & MANAGEMENT
    with deck_tab_cards:
        col_title, col_btn1, col_btn2, col_btn3 = st.columns([2.5, 1.3, 1, 1.2])
        with col_btn1:
            if st.button("📄 Ingest Document", use_container_width=True, type="primary"):
                st.session_state.draft_cards = None
                ingest_document_popup()
        with col_btn2:
            if st.button("➕ Add Card", use_container_width=True):
                new_card_popup()
        with col_btn3:
            if not cards:
                st.button("📦 Export to Anki", use_container_width=True, disabled=True, help="Add cards before exporting.")
            else:
                try:
                    anki_bytes = generate_anki_deck_bytes(
                        folder_name=st.session_state.current_folder,
                        deck_name=st.session_state.current_deck
                    )
                    clean_deck_filename = st.session_state.current_deck.strip().replace(" ", "_").lower()
                    st.download_button(
                        label="📦 Export to Anki",
                        data=anki_bytes,
                        file_name=f"{clean_deck_filename}.apkg",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error generating Anki package: {e}")

        st.subheader(f"Cards ({len(cards)})")

        if not cards:
            st.info("No cards in this deck yet. Click '➕ Add Card' or '📄 Ingest Document' above to add some.")
        else:
            for idx, card in enumerate(cards, start=1):
                card_id = card.get("card_id", f"card_{idx}")
                card_type = card.get("card_type", "concept").upper()
                mastery = card.get("mastery_level", 0)

                with st.container(border=True):
                    col_card_header, col_edit_btn = st.columns([5, 1])
                    with col_card_header:
                        st.markdown(f"**Card {idx}** · `{card_type}` · *Mastery Level: {mastery}*")
                    with col_edit_btn:
                        if st.button("✏️ Edit", key=f"edit_{card_id}"):
                            edit_card_popup(card)
                        
                    c_front, c_back = st.columns(2)
                    with c_front:
                        st.markdown("**Front:**")
                        st.write(card.get("front", ""))
                        if card.get("code_block"):
                            st.code(card.get("code_block"))

                    with c_back:
                        st.markdown("**Back:**")
                        st.write(card.get("back", ""))
                        if card.get("explanation"):
                            st.caption(f"**Explanation:** {card.get('explanation')}")
                        if card.get("media_link"):
                            render_media(card.get("media_link"))

    # SUB-TAB 2: INTERACTIVE STUDY MODE
    with deck_tab_study:
        # 1. Study Mode Controls Header
        col_mode_title, col_toggle = st.columns([2, 1])
        with col_mode_title:
            st.subheader("📖 Spaced Repetition Study")
        with col_toggle:
            only_due = st.toggle("⏰ Only Due Cards", value=False, help="Filter out cards that are not due for review yet")

        # 2. Fetch cards based on toggle selection
        study_cards = load_deck_cards(
            st.session_state.current_folder, 
            st.session_state.current_deck, 
            due_only=only_due
        )

        if not study_cards:
            if only_due:
                st.success("🎉 All caught up! No cards in this deck are currently due for review.")
            else:
                st.info("No cards in this deck yet. Add cards to start studying!")
        else:
            total_cards = len(study_cards)
            
            # Reset index if out of bounds when switching toggles
            if st.session_state.study_card_index >= total_cards:
                st.session_state.study_card_index = 0

            curr_idx = st.session_state.study_card_index
            curr_card = study_cards[curr_idx]

            st.markdown(f"**Card {curr_idx + 1} of {total_cards}**")
            st.progress((curr_idx + 1) / total_cards)

            # 3. Flashcard Render Container
            with st.container(border=True):
                card_type = curr_card.get("card_type", "concept").upper()
                interval = curr_card.get("interval_days", 0)
                reps = curr_card.get("repetition_count", 0)
                
                st.caption(f"Type: `{card_type}` | Repetitions: {reps} | Next Interval: {interval}d")
                
                st.markdown("### Question / Prompt")
                st.write(curr_card.get("front", ""))
                if curr_card.get("code_block"):
                    st.code(curr_card.get("code_block"))

                if st.session_state.study_is_flipped:
                    st.divider()
                    st.markdown("### Answer / Explanation")
                    st.write(curr_card.get("back", ""))
                    if curr_card.get("explanation"):
                        st.info(curr_card.get("explanation"))
                    if curr_card.get("media_link"):
                        render_media(curr_card.get("media_link"))

            # 4. Navigation Buttons
            col_prev, col_flip, col_next = st.columns([1, 2, 1])

            with col_prev:
                if st.button("⬅️ Previous", use_container_width=True, disabled=(curr_idx == 0)):
                    st.session_state.study_card_index -= 1
                    st.session_state.study_is_flipped = False
                    st.rerun()

            with col_flip:
                flip_label = "🙈 Hide Answer" if st.session_state.study_is_flipped else "🔄 Flip Card (Show Answer)"
                if st.button(flip_label, type="primary", use_container_width=True):
                    st.session_state.study_is_flipped = not st.session_state.study_is_flipped
                    st.rerun()

            with col_next:
                if st.button("Next ➡️", use_container_width=True, disabled=(curr_idx == total_cards - 1)):
                    st.session_state.study_card_index += 1
                    st.session_state.study_is_flipped = False
                    st.rerun()

            # 5. SM-2 Self-Grading Buttons
            if st.session_state.study_is_flipped:
                st.markdown("---")
                st.markdown("#### Rate Your Recall (SM-2):")
                col_again, col_hard, col_good, col_easy = st.columns(4)

                def process_sm2_review(quality_score: int):
                    new_rep, new_ef, new_interval, next_review = calculate_sm2(
                        quality=quality_score,
                        repetition_count=curr_card.get("repetition_count", 0),
                        ease_factor=curr_card.get("ease_factor", 2.5),
                        interval_days=curr_card.get("interval_days", 0)
                    )
                    
                    updated_card = curr_card.copy()
                    updated_card["repetition_count"] = new_rep
                    updated_card["ease_factor"] = new_ef
                    updated_card["interval_days"] = new_interval
                    updated_card["next_review_at"] = next_review
                    updated_card["mastery_level"] = new_rep
                    
                    save_card(
                        folder_name=st.session_state.current_folder,
                        deck_name=st.session_state.current_deck,
                        card_data=updated_card
                    )
                    
                    if st.session_state.study_card_index < total_cards - 1:
                        st.session_state.study_card_index += 1
                    st.session_state.study_is_flipped = False
                    st.rerun()

                with col_again:
                    if st.button("🔴 Blackout (0)", use_container_width=True, help="Complete failure to recall"):
                        process_sm2_review(0)

                with col_hard:
                    if st.button("🟠 Hard (1)", use_container_width=True, help="Remembered only upon seeing answer"):
                        process_sm2_review(1)

                with col_good:
                    if st.button("🟡 Good (2)", use_container_width=True, help="Correct response with hesitation"):
                        process_sm2_review(2)

                with col_easy:
                    if st.button("🟢 Easy (3)", use_container_width=True, help="Perfect, instant recall"):
                        process_sm2_review(3)

    # SUB-TAB 3: AGENTIC RAG CHAT ASSISTANT
    with deck_tab_chat:
        col_chat_title, col_chat_clear = st.columns([4, 1])
        with col_chat_title:
            st.subheader(f"💬 Study Assistant ({st.session_state.current_deck})")
        with col_chat_clear:
            chat_key = f"messages_{st.session_state.current_folder}_{st.session_state.current_deck}"
            if st.button("🧹 Clear Chat", use_container_width=True):
                st.session_state[chat_key] = []
                st.rerun()

        if chat_key not in st.session_state:
            st.session_state[chat_key] = []

        for msg in st.session_state[chat_key]:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.write(msg.content)

        if prompt := st.chat_input("Ask a question about this deck..."):
            user_msg = HumanMessage(content=prompt)
            st.session_state[chat_key].append(user_msg)
            
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    graph_input = {
                        "messages": st.session_state[chat_key],
                        "folder_name": st.session_state.current_folder,
                        "deck_name": st.session_state.current_deck
                    }
                    
                    response = agent_graph.invoke(graph_input)
                    assistant_messages = response.get("messages", [])
                    
                    if assistant_messages:
                        final_response = assistant_messages[-1]
                        st.write(final_response.content)
                        st.session_state[chat_key].append(final_response)

    # SUB-TAB 4: AI QUIZ ENGINE
    with deck_tab_quiz:
        st.subheader("📝 Active Recall Quiz Generator")
        
        quiz_key = f"quiz_data_{st.session_state.current_folder}_{st.session_state.current_deck}"
        grade_key = f"quiz_grade_{st.session_state.current_folder}_{st.session_state.current_deck}"

        if quiz_key not in st.session_state:
            st.session_state[quiz_key] = None
        if grade_key not in st.session_state:
            st.session_state[grade_key] = None

        col_q_focus, col_q_gen = st.columns([3, 1])
        with col_q_focus:
            quiz_focus = st.text_input(
                "Focus Topic (Optional)", 
                placeholder="e.g. Focus on encapsulation or Big-O analysis",
                key="quiz_focus_input"
            )
        with col_q_gen:
            st.write(" ")
            if st.button("🎲 Generate New Quiz", type="primary", use_container_width=True):
                with st.spinner("Retrieving card and generating short-answer quiz..."):
                    q_data = generate_quiz_question(
                        folder_name=st.session_state.current_folder,
                        deck_name=st.session_state.current_deck,
                        user_focus=quiz_focus
                    )
                    st.session_state[quiz_key] = q_data
                    st.session_state[grade_key] = None
                    st.rerun()

        st.divider()

        current_quiz = st.session_state[quiz_key]
        if not current_quiz:
            st.info("Click '🎲 Generate New Quiz' above to start a short-answer active recall test.")
        else:
            with st.container(border=True):
                st.markdown("### Question")
                st.write(current_quiz.get("question", ""))
                
                source_meta = current_quiz.get("source_meta", {})
                if source_meta.get("card_type"):
                    st.caption(f"Based on Card Type: `{source_meta.get('card_type')}`")

            user_quiz_answer = st.text_area("Your Answer", height=120, key=f"quiz_ans_input_{current_quiz.get('question')[:15]}")

            col_sub, col_reset = st.columns([1, 1])
            with col_sub:
                if st.button("Submit Answer", type="primary", use_container_width=True, disabled=not user_quiz_answer.strip()):
                    with st.spinner("Grading answer against reference material via Llama 3.1..."):
                        evaluation = grade_user_answer(
                            question=current_quiz.get("question", ""),
                            reference_context=current_quiz.get("reference_context", ""),
                            user_answer=user_quiz_answer.strip()
                        )
                        st.session_state[grade_key] = evaluation
                        st.rerun()

            current_grade = st.session_state[grade_key]
            if current_grade:
                st.divider()
                st.markdown("### Evaluation & Feedback")
                
                score_str = current_grade.get("score", "Needs Review")
                grade_pct = current_grade.get("grade_percent", 0)
                feedback = current_grade.get("feedback", "")

                if score_str == "Pass":
                    st.success(f"**Score:** {score_str} ({grade_pct}%)")
                else:
                    st.warning(f"**Score:** {score_str} ({grade_pct}%)")

                st.markdown(f"**Feedback:** {feedback}")
                
                with st.expander("📖 View Reference Context"):
                    st.write(current_quiz.get("reference_context", ""))

# VIEW 2: INDIVIDUAL FOLDER VIEW (List of Decks)
elif st.session_state.current_folder is not None:
    col_header, col_f_actions = st.columns([3, 1])
    with col_header:
        st.header(f"Folder: {st.session_state.current_folder}")
    with col_f_actions:
        col_f_ren, col_f_del = st.columns(2)
        with col_f_ren:
            if st.button("✏️ Rename", use_container_width=True):
                rename_folder_popup(st.session_state.current_folder)
        with col_f_del:
            with st.popover("🗑️ Delete", use_container_width=True):
                st.warning("Delete folder and all decks?")
                if st.button("Yes, Delete Folder", type="primary", use_container_width=True):
                    delete_folder(st.session_state.current_folder)
                    st.session_state.current_folder = None
                    st.session_state.current_deck = None
                    st.rerun()

    if st.button("⬅️ Back to All Folders"):
        st.session_state.current_folder = None
        st.session_state.current_deck = None
        st.rerun()

    st.divider()

    if st.button("➕ Add New Deck"):
        new_deck_popup()

    st.subheader("Decks")
    decks = get_decks_in_folder(st.session_state.current_folder)

    for d in decks:
        col_d_btn, col_d_ren, col_d_del = st.columns([4, 1, 1])
        with col_d_btn:
            if st.button(d, key=f"deck_btn_{d}", use_container_width=True):
                st.session_state.current_deck = d
                st.session_state.study_card_index = 0
                st.session_state.study_is_flipped = False
                st.rerun()
        with col_d_ren:
            if st.button("✏️", key=f"ren_deck_{d}"):
                rename_deck_popup(d)
        with col_d_del:
            with st.popover("🗑️", key=f"del_deck_pop_{d}"):
                st.warning(f"Delete deck '{d}'?")
                if st.button("Confirm Delete", key=f"confirm_del_deck_{d}", type="primary"):
                    delete_deck(st.session_state.current_folder, d)
                    st.rerun()

# VIEW 1: ALL FOLDERS VIEW
else:
    if st.button("➕ Add New Folder"):
        new_folder_popup()

    st.subheader("All Folders")
    folders = get_all_folders()

    for f in folders:
        col_f_btn, col_f_ren, col_f_del = st.columns([4, 1, 1])
        with col_f_btn:
            if st.button(f, key=f"folder_btn_{f}", use_container_width=True):
                st.session_state.current_folder = f
                st.rerun()
        with col_f_ren:
            if st.button("✏️", key=f"ren_fold_{f}"):
                rename_folder_popup(f)
        with col_f_del:
            with st.popover("🗑️", key=f"del_fold_pop_{f}"):
                st.warning(f"Delete folder '{f}' and contents?")
                if st.button("Confirm Delete", key=f"confirm_del_fold_{f}", type="primary"):
                    delete_folder(f)
                    st.rerun()