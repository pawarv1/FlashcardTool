import os
import io
import pandas as pd
from typing import get_args
from datetime import datetime
from gtts import gTTS
import streamlit as st
from langchain_core.messages import HumanMessage

# Local Modules
from db import init_db, auto_heal_chroma_sync
from document_parser import get_markdown_chunks, parse_csv_direct_to_cards
from card_generator import generate_flashcards_from_chunks, CardTypeEnum, generate_remediation_cards
from vector_utils import check_candidate_duplicates
from storage_utils import (
    create_folder, create_deck, get_all_folders, get_decks_in_folder, 
    rename_folder, delete_folder, rename_deck, delete_deck, save_card, 
    save_card_batch, load_deck_cards, delete_card, get_deck_analytics, 
    log_quiz_attempt, get_quiz_analytics, get_review_forecast, get_all_deck_tags, 
    load_cram_cards, get_all_folder_tags
)
from anki_utils import generate_anki_deck_bytes
from agent import agent_graph
from quiz_engine import generate_quiz_question, grade_user_answer
from sm2_utils import calculate_sm2
from graph_utils import generate_knowledge_graph_html
from media_utils import process_and_save_media
from backup_utils import create_system_backup_zip, restore_system_from_zip

# -------------------------------------------------------------------
# INITIALIZATION & SESSION STATE
# -------------------------------------------------------------------

@st.cache_resource
def setup_database():
    """Runs database initialization and vector sync healing once per app startup."""
    init_db()
    auto_heal_chroma_sync()
    return True

setup_database()

CARD_TYPES = get_args(CardTypeEnum) if get_args(CardTypeEnum) else [
    "concept", "code_snippet", "definition", "formula", "comparison", "example"
]

st.set_page_config(
    page_title="Agentic RAG Study Assistant", 
    page_icon="📚", 
    layout="wide"
)

def init_session_state():
    """Consolidates session state initialization."""
    defaults = {
        "current_folder": None,
        "current_deck": None,
        "study_card_index": 0,
        "study_is_flipped": False,
        "cram_mode_active": False,
        "cram_card_index": 0,
        "cram_is_flipped": False,
        "cram_cards": [],
        "draft_cards": None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# -------------------------------------------------------------------
# HELPER RENDERERS
# -------------------------------------------------------------------

def generate_tts_audio_bytes(text: str, lang: str = "en") -> io.BytesIO:
    """Converts a text string into MP3 audio bytes in memory."""
    clean_text = text.strip()
    if not clean_text:
        return None
    try:
        tts = gTTS(text=clean_text, lang=lang, slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except Exception as e:
        print(f"TTS Generation failed: {e}")
        return None

def render_media(media_path: str):
    """Renders local or remote media attachments."""
    if not media_path:
        return
    
    clean_path = media_path.lower()
    image_exts = [".png", ".jpg", ".jpeg", ".gif", ".webp"]
    is_web_url = clean_path.startswith("http://") or clean_path.startswith("https://")
    
    if any(clean_path.endswith(ext) for ext in image_exts) or is_web_url:
        if os.path.exists(media_path) or is_web_url:
            st.image(media_path, use_container_width=True)
        else:
            st.caption(f"⚠️ Attachment not found: `{media_path}`")
    else:
        st.markdown(f"🔗 [View Attached Media]({media_path})")

# -------------------------------------------------------------------
# SIDEBAR: BACKUP & RESTORE
# -------------------------------------------------------------------

with st.sidebar:
    st.title("📚 Study Assistant")
    st.divider()
    with st.expander("⚙️ Backup & Restore Data", expanded=False):
        st.caption("Export or restore your complete database, vector collections, and media attachments.")

        st.markdown("#### 📦 Export System Backup")
        if st.button("Generate Backup ZIP", type="primary", use_container_width=True):
            with st.spinner("Bundling database, vectors, and media assets..."):
                zip_buffer = create_system_backup_zip()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_filename = f"flashcard_app_backup_{timestamp}.zip"
                
                st.download_button(
                    label="💾 Download Backup Archive",
                    data=zip_buffer,
                    file_name=backup_filename,
                    mime="application/zip",
                    use_container_width=True
                )

        st.divider()

        st.markdown("#### 📂 Restore System Backup")
        uploaded_backup = st.file_uploader(
            "Upload Backup ZIP", 
            type=["zip"], 
            key="restore_zip_uploader"
        )
        
        if uploaded_backup:
            st.warning("⚠️ Restoring will overwrite existing card data, media, and study statistics!")
            if st.button("Confirm & Restore Data", type="primary", use_container_width=True):
                with st.spinner("Restoring files from backup archive..."):
                    if restore_system_from_zip(uploaded_backup):
                        st.success("System restored successfully! Reloading...")
                        st.rerun()
                    else:
                        st.error("Failed to restore system backup. Ensure it is a valid backup ZIP.")

# -------------------------------------------------------------------
# DIALOG MODALS
# -------------------------------------------------------------------

@st.dialog("New Folder")
def new_folder_popup():
    new_folder_name = st.text_input("Enter New Folder Name")
    if st.button("Create", type="primary"):
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
    if st.button("Create", type="primary"):
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
    tags_input = st.text_input("Tags (Optional, comma-separated)", placeholder="e.g. midterm, algorithms, priority")
    front = st.text_area("Front (Question / Prompt)")
    back = st.text_area("Back (Answer / Summary)")
    code_block = st.text_area("Code Block (Optional)")
    explanation = st.text_area("Explanation (Optional)")

    uploaded_media_file = st.file_uploader("Upload Media Attachment", type=["png", "jpg", "jpeg", "webp", "gif"])
    media_link_input = st.text_input("OR Media Link / URL (Optional)")

    if st.button("Save Card", type="primary"):
        if front.strip() and back.strip():
            final_media_path = None
            if uploaded_media_file:
                final_media_path = process_and_save_media(uploaded_media_file)
            elif media_link_input.strip():
                final_media_path = media_link_input.strip()

            clean_tags = ",".join([t.strip().lower() for t in tags_input.split(",") if t.strip()]) if tags_input.strip() else None

            card_data = {
                "card_type": card_type,
                "front": front.strip(),
                "back": back.strip(),
                "code_block": code_block.strip() if code_block.strip() else None,
                "explanation": explanation.strip() if explanation.strip() else None,
                "media_link": final_media_path,
                "tags": clean_tags,
                "source_type": "manual_entry",
                "mastery_level": 0
            }
            save_card(
                folder_name=st.session_state.current_folder,
                deck_name=st.session_state.current_deck,
                card_data=card_data
            )
            st.rerun()
        else:
            st.warning("Both Front and Back text are required.")

@st.dialog("Edit Card")
def edit_card_popup(card: dict):
    curr_type = card.get("card_type", "concept")
    type_idx = CARD_TYPES.index(curr_type) if curr_type in CARD_TYPES else 0
    existing_tags = card.get("tags") or ""

    updated_type = st.selectbox("Card Type", CARD_TYPES, index=type_idx)
    updated_tags_input = st.text_input("Tags (Optional, comma-separated)", value=existing_tags)
    updated_front = st.text_area("Front (Question / Prompt)", value=card.get("front", ""))
    updated_back = st.text_area("Back (Answer / Summary)", value=card.get("back", ""))
    updated_code = st.text_area("Code Block (Optional)", value=card.get("code_block") or "")
    updated_explanation = st.text_area("Explanation (Optional)", value=card.get("explanation") or "")
    
    existing_media = card.get("media_link")
    if existing_media:
        st.markdown("**Current Media Attachment:**")
        render_media(existing_media)
    
    uploaded_media_file = st.file_uploader("Replace / Upload Media", type=["png", "jpg", "jpeg", "webp", "gif"], key=f"edit_file_{card.get('card_id')}")
    updated_media_url = st.text_input("OR Media Link / URL", value=existing_media or "", key=f"edit_url_{card.get('card_id')}")

    st.divider()
    col_save, col_delete = st.columns([1, 1])

    with col_save:
        if st.button("Save Changes", type="primary", use_container_width=True):
            if updated_front.strip() and updated_back.strip():
                final_media_path = existing_media
                if uploaded_media_file:
                    final_media_path = process_and_save_media(uploaded_media_file)
                elif updated_media_url.strip():
                    final_media_path = updated_media_url.strip()
                else:
                    final_media_path = None

                clean_updated_tags = ",".join([t.strip().lower() for t in updated_tags_input.split(",") if t.strip()]) if updated_tags_input.strip() else None

                updated_card = {
                    "card_id": card.get("card_id"),
                    "card_type": updated_type,
                    "front": updated_front.strip(),
                    "back": updated_back.strip(),
                    "code_block": updated_code.strip() if updated_code.strip() else None,
                    "explanation": updated_explanation.strip() if updated_explanation.strip() else None,
                    "media_link": final_media_path,
                    "tags": clean_updated_tags,
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

@st.dialog("📄 Ingest Document & Generate Cards", width="large")
def ingest_document_popup():
    uploaded_file = st.file_uploader("Upload Document", type=["pdf", "docx", "pptx", "txt", "md", "csv", "html", "ipynb"])
    instructions = st.text_input("Focus Directive (Optional)", placeholder="e.g. Focus on search algorithms")
    num_cards = st.slider("Target Number of Cards", min_value=3, max_value=20, value=8)

    if st.button("⚙️ Parse & Generate Drafts", type="primary", disabled=not uploaded_file):
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()

        if file_ext == ".csv":
            with st.spinner("Parsing CSV rows directly into flashcards..."):
                raw_drafts = parse_csv_direct_to_cards(uploaded_file)
                st.session_state.draft_cards = check_candidate_duplicates(
                    candidate_cards=raw_drafts,
                    folder_name=st.session_state.current_folder,
                    deck_name=st.session_state.current_deck
                )
        else:
            with st.spinner("Parsing document and drafting flashcards via Llama 3.1..."):
                document_chunks = get_markdown_chunks(uploaded_file)
                raw_drafts = generate_flashcards_from_chunks(
                    document_chunks=document_chunks, 
                    user_instructions=instructions, 
                    target_count=num_cards
                )
                st.session_state.draft_cards = check_candidate_duplicates(
                    candidate_cards=raw_drafts,
                    folder_name=st.session_state.current_folder,
                    deck_name=st.session_state.current_deck
                )

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

# -------------------------------------------------------------------
# VIEW 3: DECK VIEW
# -------------------------------------------------------------------

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

    st.title(f"Deck: {st.session_state.current_deck}")
    st.caption(f"Folder: {st.session_state.current_folder}")

    # Analytics Banner
    analytics = get_deck_analytics(st.session_state.current_folder, st.session_state.current_deck)
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric(label="Total Cards", value=analytics["total_cards"])
    with col_m2:
        st.metric(
            label="Due for Review", 
            value=analytics["due_cards"], 
            delta=f"{analytics['due_cards']} Actionable" if analytics["due_cards"] > 0 else "All Caught Up!",
            delta_color="inverse" if analytics["due_cards"] > 0 else "normal"
        )
    with col_m3:
        st.metric(
            label="Avg. Ease Factor", 
            value=f"{analytics['avg_ease_factor']:.2f}",
            help="Higher values indicate material that is easier to recall (Standard baseline is 2.50)"
        )

    with st.expander("📅 Review Workload Forecast", expanded=False):
        col_fc_info, col_fc_select = st.columns([3, 1])
        with col_fc_info:
            st.caption("Distribution of cards scheduled for review based on SM-2 interval calculations:")
        with col_fc_select:
            forecast_days = st.selectbox(
                "Forecast Window",
                options=[7, 14, 30],
                format_func=lambda x: f"Next {x} Days",
                index=0,
                key=f"fc_days_{st.session_state.current_folder}_{st.session_state.current_deck}"
            )

        forecast_data = get_review_forecast(
            st.session_state.current_folder, 
            st.session_state.current_deck, 
            days=forecast_days
        )

        if not forecast_data:
            st.info("No cards in this deck to forecast.")
        else:
            df_forecast = pd.DataFrame(list(forecast_data.items()), columns=["Date", "Due Cards"])
            df_forecast["Date"] = pd.Categorical(df_forecast["Date"], categories=list(forecast_data.keys()), ordered=True)
            df_forecast = df_forecast.set_index("Date")
            st.bar_chart(df_forecast)

    st.divider()

    deck_tab_cards, deck_tab_study, deck_tab_chat, deck_tab_quiz, deck_tab_graph = st.tabs([
        "🎴 Cards", "📖 Study Mode", "💬 Chat Assistant", "📝 AI Quiz", "🕸️ Knowledge Graph"
    ])

    cards = load_deck_cards(st.session_state.current_folder, st.session_state.current_deck)
    deck_tags = get_all_deck_tags(st.session_state.current_folder, st.session_state.current_deck)

    selected_tags = []
    if deck_tags:
        selected_tags = st.multiselect(
            "🏷️ Filter by Tag:",
            options=deck_tags,
            format_func=lambda t: f"#{t}",
            key=f"tag_filter_{st.session_state.current_folder}_{st.session_state.current_deck}"
        )

    if selected_tags:
        cards = [
            c for c in cards 
            if c.get("tags") and any(t.strip().lower() in selected_tags for t in c.get("tags").split(","))
        ]

    # TAB 1: CARD LIST
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
                st.button("📦 Export to Anki", use_container_width=True, disabled=True)
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
            st.info("No cards in this deck yet. Click '➕ Add Card' or '📄 Ingest Document' to add some.")
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
                        if st.button("✏️ Edit", key=f"edit_btn_{card_id}"):
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

    # TAB 2: STUDY & CRAM ENGINE
    with deck_tab_study:
        col_mode_title, col_session_type = st.columns([2, 1])
        with col_mode_title:
            st.subheader("📖 Study & Review Engine")
        with col_session_type:
            study_mode_type = st.radio(
                "Study Mode",
                options=["Spaced Repetition (SM-2)", "⚡ Cram Session"],
                horizontal=True,
                key=f"mode_select_{st.session_state.current_folder}_{st.session_state.current_deck}"
            )

        st.divider()

        # BRANCH A: SM-2 SPACED REPETITION
        if study_mode_type == "Spaced Repetition (SM-2)":
            col_due_toggle, _ = st.columns([1, 2])
            with col_due_toggle:
                only_due = st.toggle("⏰ Only Due Cards", value=False)

            study_cards = load_deck_cards(
                st.session_state.current_folder, 
                st.session_state.current_deck, 
                due_only=only_due
            )

            if selected_tags:
                study_cards = [
                    c for c in study_cards 
                    if c.get("tags") and any(t.strip().lower() in selected_tags for t in c.get("tags").split(","))
                ]

            if not study_cards:
                if only_due:
                    st.success("🎉 All caught up! No cards in this deck are currently due for review.")
                else:
                    st.info("No cards match your current filters.")
            else:
                total_cards = len(study_cards)
                if st.session_state.study_card_index >= total_cards:
                    st.session_state.study_card_index = 0

                curr_idx = st.session_state.study_card_index
                curr_card = study_cards[curr_idx]

                st.markdown(f"**Card {curr_idx + 1} of {total_cards}**")
                st.progress((curr_idx + 1) / total_cards)

                with st.container(border=True):
                    card_type = curr_card.get("card_type", "concept").upper()
                    interval = curr_card.get("interval_days", 0)
                    reps = curr_card.get("repetition_count", 0)
                    
                    st.caption(f"Type: `{card_type}` | Repetitions: {reps} | Next Interval: {interval}d")
                    
                    col_q_text, col_q_audio = st.columns([4, 1])
                    with col_q_text:
                        st.markdown("### Question / Prompt")
                        st.write(curr_card.get("front", ""))
                        if curr_card.get("code_block"):
                            st.code(curr_card.get("code_block"))

                    with col_q_audio:
                        if st.button("🔊 Read Question", key=f"tts_q_{curr_card.get('card_id')}"):
                            with st.spinner("Generating audio..."):
                                q_audio = generate_tts_audio_bytes(curr_card.get("front", ""))
                                if q_audio:
                                    st.audio(q_audio, format="audio/mp3", autoplay=True)

                    if st.session_state.study_is_flipped:
                        st.divider()
                        col_a_text, col_a_audio = st.columns([4, 1])
                        with col_a_text:
                            st.markdown("### Answer / Explanation")
                            st.write(curr_card.get("back", ""))
                            if curr_card.get("explanation"):
                                st.info(curr_card.get("explanation"))
                            if curr_card.get("media_link"):
                                render_media(curr_card.get("media_link"))

                        with col_a_audio:
                            if st.button("🔊 Read Answer", key=f"tts_a_{curr_card.get('card_id')}"):
                                with st.spinner("Generating audio..."):
                                    full_text = curr_card.get("back", "")
                                    if curr_card.get("explanation"):
                                        full_text += f". Explanation: {curr_card.get('explanation')}"
                                    a_audio = generate_tts_audio_bytes(full_text)
                                    if a_audio:
                                        st.audio(a_audio, format="audio/mp3", autoplay=True)

                col_prev, col_flip, col_next = st.columns([1, 2, 1])
                with col_prev:
                    if st.button("⬅️ Previous", use_container_width=True, disabled=(curr_idx == 0)):
                        st.session_state.study_card_index -= 1
                        st.session_state.study_is_flipped = False
                        st.rerun()

                with col_flip:
                    flip_label = "🙈 Hide Answer" if st.session_state.study_is_flipped else "🔄 Flip Card"
                    if st.button(flip_label, type="primary", use_container_width=True):
                        st.session_state.study_is_flipped = not st.session_state.study_is_flipped
                        st.rerun()

                with col_next:
                    if st.button("Next ➡️", use_container_width=True, disabled=(curr_idx == total_cards - 1)):
                        st.session_state.study_card_index += 1
                        st.session_state.study_is_flipped = False
                        st.rerun()

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
                        if st.button("🔴 Blackout (0)", use_container_width=True): process_sm2_review(0)
                    with col_hard:
                        if st.button("🟠 Hard (1)", use_container_width=True): process_sm2_review(1)
                    with col_good:
                        if st.button("🟡 Good (2)", use_container_width=True): process_sm2_review(2)
                    with col_easy:
                        if st.button("🟢 Easy (3)", use_container_width=True): process_sm2_review(3)

        # BRANCH B: CRAM MODE
        else:
            with st.expander("⚡ Configure Focused Cram Session", expanded=not st.session_state.cram_cards):
                c1, c2, c3 = st.columns(3)
                with c1:
                    cram_tags = st.multiselect(
                        "Target Tags",
                        options=deck_tags,
                        default=selected_tags if selected_tags else [],
                        format_func=lambda t: f"#{t}",
                        key=f"cram_tags_{st.session_state.current_deck}"
                    )
                with c2:
                    cram_difficulty = st.selectbox(
                        "Card Focus",
                        options=["all", "hard", "unseen"],
                        format_func=lambda x: {"all": "All Matching Cards", "hard": "Hard Cards (EF < 2.3)", "unseen": "Unseen / New Cards"}[x],
                        key=f"cram_diff_{st.session_state.current_deck}"
                    )
                with c3:
                    cram_limit = st.number_input(
                        "Max Cards Limit (0 = All)", 
                        min_value=0, max_value=100, value=0,
                        key=f"cram_lim_{st.session_state.current_deck}"
                    )

                update_sm2_in_cram = st.checkbox(
                    "🔄 Update SM-2 Spaced Repetition Intervals",
                    value=False,
                    help="If unchecked, cramming won't impact long-term review dates."
                )

                if st.button("🚀 Start Cram Session", type="primary", use_container_width=True):
                    fetched_cram_cards = load_cram_cards(
                        folder_name=st.session_state.current_folder,
                        deck_name=st.session_state.current_deck,
                        selected_tags=cram_tags,
                        difficulty_filter=cram_difficulty,
                        card_limit=cram_limit
                    )
                    st.session_state.cram_cards = fetched_cram_cards
                    st.session_state.cram_card_index = 0
                    st.session_state.cram_is_flipped = False
                    st.rerun()

            cram_cards = st.session_state.cram_cards

            if not cram_cards:
                st.info("Configure your filters above and click '🚀 Start Cram Session' to begin!")
            else:
                total_cram = len(cram_cards)
                if st.session_state.cram_card_index >= total_cram:
                    st.session_state.cram_card_index = 0

                curr_c_idx = st.session_state.cram_card_index
                curr_c_card = cram_cards[curr_c_idx]

                st.markdown(f"**Cram Card {curr_c_idx + 1} of {total_cram}**")
                st.progress((curr_c_idx + 1) / total_cram)

                with st.container(border=True):
                    st.caption(f"Cram Mode | Type: `{curr_c_card.get('card_type', 'concept').upper()}`")
                    st.markdown("### Question / Prompt")
                    st.write(curr_c_card.get("front", ""))
                    if curr_c_card.get("code_block"):
                        st.code(curr_c_card.get("code_block"))

                    if st.session_state.cram_is_flipped:
                        st.divider()
                        st.markdown("### Answer / Explanation")
                        st.write(curr_c_card.get("back", ""))
                        if curr_c_card.get("explanation"):
                            st.info(curr_c_card.get("explanation"))
                        if curr_c_card.get("media_link"):
                            render_media(curr_c_card.get("media_link"))

                col_c_prev, col_c_flip, col_c_next = st.columns([1, 2, 1])
                with col_c_prev:
                    if st.button("⬅️ Previous", key="cram_prev_btn", use_container_width=True, disabled=(curr_c_idx == 0)):
                        st.session_state.cram_card_index -= 1
                        st.session_state.cram_is_flipped = False
                        st.rerun()

                with col_c_flip:
                    c_flip_label = "🙈 Hide Answer" if st.session_state.cram_is_flipped else "🔄 Flip Card"
                    if st.button(c_flip_label, key="cram_flip_btn", type="primary", use_container_width=True):
                        st.session_state.cram_is_flipped = not st.session_state.cram_is_flipped
                        st.rerun()

                with col_c_next:
                    if st.button("Next ➡️", key="cram_next_btn", use_container_width=True, disabled=(curr_c_idx == total_cram - 1)):
                        st.session_state.cram_card_index += 1
                        st.session_state.cram_is_flipped = False
                        st.rerun()

                if st.session_state.cram_is_flipped and update_sm2_in_cram:
                    st.markdown("---")
                    st.markdown("#### Optional SM-2 Rating:")
                    col_ca, col_ch, col_cg, col_ce = st.columns(4)
                    
                    def process_cram_sm2(q_score: int):
                        n_rep, n_ef, n_int, n_rev = calculate_sm2(
                            quality=q_score,
                            repetition_count=curr_c_card.get("repetition_count", 0),
                            ease_factor=curr_c_card.get("ease_factor", 2.5),
                            interval_days=curr_c_card.get("interval_days", 0)
                        )
                        u_card = curr_c_card.copy()
                        u_card["repetition_count"] = n_rep
                        u_card["ease_factor"] = n_ef
                        u_card["interval_days"] = n_int
                        u_card["next_review_at"] = n_rev
                        u_card["mastery_level"] = n_rep
                        save_card(st.session_state.current_folder, st.session_state.current_deck, u_card)
                        
                        if st.session_state.cram_card_index < total_cram - 1:
                            st.session_state.cram_card_index += 1
                        st.session_state.cram_is_flipped = False
                        st.rerun()

                    with col_ca:
                        if st.button("🔴 Blackout", key="cram_sm2_0", use_container_width=True): process_cram_sm2(0)
                    with col_ch:
                        if st.button("🟠 Hard", key="cram_sm2_1", use_container_width=True): process_cram_sm2(1)
                    with col_cg:
                        if st.button("🟡 Good", key="cram_sm2_2", use_container_width=True): process_cram_sm2(2)
                    with col_ce:
                        if st.button("🟢 Easy", key="cram_sm2_3", use_container_width=True): process_cram_sm2(3)

    # TAB 3: AGENTIC CHAT ASSISTANT
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

    # TAB 4: AI QUIZ ENGINE
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
            quiz_focus = st.text_input("Focus Topic (Optional)", placeholder="e.g. Focus on encapsulation", key="quiz_focus_input")
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
                        
                        log_quiz_attempt(
                            folder_name=st.session_state.current_folder,
                            deck_name=st.session_state.current_deck,
                            question=current_quiz.get("question", ""),
                            user_answer=user_quiz_answer.strip(),
                            score_label=evaluation.get("score", "Needs Review"),
                            grade_percent=evaluation.get("grade_percent", 0),
                            feedback=evaluation.get("feedback", "")
                        )
                        st.rerun()

            current_grade = st.session_state[grade_key]
            if current_grade:
                st.divider()
                st.markdown("### Evaluation & Feedback")
                
                score_str = current_grade.get("score", "Needs Review")
                grade_pct = current_grade.get("grade_percent", 0)
                feedback = current_grade.get("feedback", "")

                if score_str == "Pass" and grade_pct >= 70:
                    st.success(f"**Score:** {score_str} ({grade_pct}%)")
                else:
                    st.warning(f"**Score:** {score_str} ({grade_pct}%)")

                st.markdown(f"**Feedback:** {feedback}")
                
                with st.expander("📖 View Reference Context"):
                    st.write(current_quiz.get("reference_context", ""))

                if score_str != "Pass" or grade_pct < 70:
                    st.divider()
                    st.subheader("🩹 Weak Concept Remediation")
                    st.info("Generate targeted flashcards to patch this concept?")

                    remed_key = f"remed_cards_{st.session_state.current_folder}_{st.session_state.current_deck}"
                    if remed_key not in st.session_state:
                        st.session_state[remed_key] = None

                    if st.button("⚡ Generate Remediation Cards", type="primary"):
                        with st.spinner("Analyzing knowledge gap and drafting targeted cards..."):
                            raw_remed_cards = generate_remediation_cards(
                                question=current_quiz.get("question", ""),
                                reference_context=current_quiz.get("reference_context", ""),
                                user_answer=user_quiz_answer.strip(),
                                feedback=feedback
                            )
                            flagged_remed_cards = check_candidate_duplicates(
                                candidate_cards=raw_remed_cards,
                                folder_name=st.session_state.current_folder,
                                deck_name=st.session_state.current_deck
                            )
                            st.session_state[remed_key] = flagged_remed_cards
                            st.rerun()

                    if st.session_state[remed_key]:
                        st.markdown("#### Drafted Remediation Cards")
                        selected_remed_indices = []
                        
                        for i, card in enumerate(st.session_state[remed_key]):
                            is_dup = card.get("is_duplicate", False)
                            matched_q = card.get("matched_existing_front", "")

                            with st.container(border=True):
                                if is_dup:
                                    st.warning(f"⚠️ Potential Duplicate of: *\"{matched_q}\"*")

                                col_chk, col_card = st.columns([1, 10])
                                with col_chk:
                                    inc = st.checkbox("Import", value=not is_dup, key=f"remed_chk_{i}")
                                    if inc:
                                        selected_remed_indices.append(i)
                                with col_card:
                                    st.markdown(f"**Front:** {card.get('front', '')}")
                                    st.markdown(f"**Back:** {card.get('back', '')}")
                                    if card.get("code_block"):
                                        st.code(card.get("code_block"))
                                    if card.get("explanation"):
                                        st.caption(f"**Explanation:** {card.get('explanation')}")

                        if st.button("💾 Add Selected Remediation Cards to Deck", type="primary", disabled=not selected_remed_indices):
                            cards_to_add = [st.session_state[remed_key][idx] for idx in selected_remed_indices]
                            for c in cards_to_add:
                                c["source_type"] = "quiz_remediation"
                                c["mastery_level"] = 0
                            
                            save_card_batch(
                                folder_name=st.session_state.current_folder,
                                deck_name=st.session_state.current_deck,
                                cards=cards_to_add
                            )
                            st.session_state[remed_key] = None
                            st.success(f"Added {len(cards_to_add)} remediation cards to your deck!")
                            st.rerun()

        st.divider()
        with st.expander("📈 Quiz Performance Dashboard & History", expanded=False):
            quiz_stats = get_quiz_analytics(st.session_state.current_folder, st.session_state.current_deck)

            if quiz_stats["total_quizzes"] == 0:
                st.info("No quiz attempts recorded yet. Submit your first answer above to start tracking performance!")
            else:
                col_q1, col_q2, col_q3 = st.columns(3)
                with col_q1:
                    st.metric("Total Quizzes Taken", quiz_stats["total_quizzes"])
                with col_q2:
                    st.metric("Average Score", f"{quiz_stats['avg_score']}%")
                with col_q3:
                    st.metric("Pass Rate (≥70%)", f"{quiz_stats['pass_rate']}%")

                scores = [h["grade_percent"] for h in reversed(quiz_stats["history"])]
                if len(scores) > 1:
                    st.markdown("#### Score Progression")
                    st.line_chart(scores)

                st.markdown("#### Recent Attempts")
                for attempt in quiz_stats["history"]:
                    with st.container(border=True):
                        c_head, c_score = st.columns([4, 1])
                        with c_head:
                            st.markdown(f"**Q:** {attempt['question']}")
                            st.caption(f"Date: {attempt['created_at']}")
                        with c_score:
                            pct = attempt["grade_percent"]
                            badge_color = "🟢" if pct >= 70 else "🔴"
                            st.markdown(f"### {badge_color} {pct}%")

                        st.markdown(f"**Your Answer:** {attempt['user_answer']}")
                        st.caption(f"**Feedback:** {attempt['feedback']}")

    # TAB 5: KNOWLEDGE GRAPH
    with deck_tab_graph:
        st.subheader("🕸️ Concept Knowledge Graph")
        st.caption("Nodes represent flashcards; edges represent semantic similarity derived from vector embeddings.")

        threshold = st.slider(
            "Similarity Connection Threshold", 
            min_value=0.20, max_value=0.80, value=0.45, step=0.05,
            key=f"thresh_{st.session_state.current_folder}_{st.session_state.current_deck}"
        )

        st.divider()
        with st.spinner("Calculating vector distances and building interactive graph..."):
            graph_html = generate_knowledge_graph_html(
                folder_name=st.session_state.current_folder,
                deck_name=st.session_state.current_deck,
                similarity_threshold=threshold
            )
            st.iframe(src=graph_html, height=580)

# -------------------------------------------------------------------
# VIEW 2: FOLDER VIEW
# -------------------------------------------------------------------

elif st.session_state.current_folder is not None:
    col_header, col_f_actions = st.columns([3, 1])
    with col_header:
        st.title(f"Folder: {st.session_state.current_folder}")
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

    folder_tab_decks, folder_tab_cram = st.tabs(["📁 Decks Overview", "⚡ Folder Cram Session"])

    # SUB-TAB 1: DECKS LIST
    with folder_tab_decks:
        if st.button("➕ Add New Deck", type="primary"):
            new_deck_popup()

        st.subheader("Decks")
        decks = get_decks_in_folder(st.session_state.current_folder)

        if not decks:
            st.info("No decks in this folder yet. Click '➕ Add New Deck' above to create one!")
        else:
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

    # SUB-TAB 2: FOLDER CRAM SESSION
    with folder_tab_cram:
        st.subheader(f"⚡ Cross-Deck Cramming ({st.session_state.current_folder})")
        st.caption("Review cards pulled from ALL decks inside this folder.")

        folder_tags = get_all_folder_tags(st.session_state.current_folder)

        with st.expander("⚙️ Configure Folder Cram Session", expanded=not st.session_state.cram_cards):
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                f_cram_tags = st.multiselect(
                    "Target Tags (Across Decks)",
                    options=folder_tags,
                    format_func=lambda t: f"#{t}",
                    key=f"folder_cram_tags_{st.session_state.current_folder}"
                )
            with fc2:
                f_cram_difficulty = st.selectbox(
                    "Card Focus",
                    options=["all", "hard", "unseen"],
                    format_func=lambda x: {"all": "All Matching Cards", "hard": "Hard Cards (EF < 2.3)", "unseen": "Unseen / New Cards"}[x],
                    key=f"folder_cram_diff_{st.session_state.current_folder}"
                )
            with fc3:
                f_cram_limit = st.number_input(
                    "Max Cards Limit (0 = All)", 
                    min_value=0, max_value=200, value=0,
                    key=f"folder_cram_lim_{st.session_state.current_folder}"
                )

            f_update_sm2 = st.checkbox(
                "🔄 Update SM-2 Spaced Repetition Intervals",
                value=False,
                key=f"folder_sm2_chk_{st.session_state.current_folder}",
                help="If unchecked, cramming won't impact long-term review dates."
            )

            if st.button("🚀 Start Folder Cram Session", type="primary", use_container_width=True):
                fetched_folder_cards = load_cram_cards(
                    folder_name=st.session_state.current_folder,
                    deck_name=None,
                    selected_tags=f_cram_tags,
                    difficulty_filter=f_cram_difficulty,
                    card_limit=f_cram_limit
                )
                st.session_state.cram_cards = fetched_folder_cards
                st.session_state.cram_card_index = 0
                st.session_state.cram_is_flipped = False
                st.rerun()

        f_cram_cards = st.session_state.cram_cards

        if not f_cram_cards:
            st.info("Configure your filters above and click '🚀 Start Folder Cram Session' to begin!")
        else:
            total_f_cram = len(f_cram_cards)
            if st.session_state.cram_card_index >= total_f_cram:
                st.session_state.cram_card_index = 0

            curr_fc_idx = st.session_state.cram_card_index
            curr_fc_card = f_cram_cards[curr_fc_idx]

            st.markdown(f"**Cram Card {curr_fc_idx + 1} of {total_f_cram}**")
            st.progress((curr_fc_idx + 1) / total_f_cram)

            with st.container(border=True):
                card_deck_origin = curr_fc_card.get("deck_name", "Unknown Deck")
                card_type = curr_fc_card.get("card_type", "concept").upper()
                
                st.caption(f"Deck: `{card_deck_origin}` | Type: `{card_type}`")
                
                st.markdown("### Question / Prompt")
                st.write(curr_fc_card.get("front", ""))
                if curr_fc_card.get("code_block"):
                    st.code(curr_fc_card.get("code_block"))

                if st.session_state.cram_is_flipped:
                    st.divider()
                    st.markdown("### Answer / Explanation")
                    st.write(curr_fc_card.get("back", ""))
                    if curr_fc_card.get("explanation"):
                        st.info(curr_fc_card.get("explanation"))
                    if curr_fc_card.get("media_link"):
                        render_media(curr_fc_card.get("media_link"))

            col_fc_prev, col_fc_flip, col_fc_next = st.columns([1, 2, 1])

            with col_fc_prev:
                if st.button("⬅️ Previous", key="fcram_prev_btn", use_container_width=True, disabled=(curr_fc_idx == 0)):
                    st.session_state.cram_card_index -= 1
                    st.session_state.cram_is_flipped = False
                    st.rerun()

            with col_fc_flip:
                fc_flip_label = "🙈 Hide Answer" if st.session_state.cram_is_flipped else "🔄 Flip Card"
                if st.button(fc_flip_label, key="fcram_flip_btn", type="primary", use_container_width=True):
                    st.session_state.cram_is_flipped = not st.session_state.cram_is_flipped
                    st.rerun()

            with col_fc_next:
                if st.button("Next ➡️", key="fcram_next_btn", use_container_width=True, disabled=(curr_fc_idx == total_f_cram - 1)):
                    st.session_state.cram_card_index += 1
                    st.session_state.cram_is_flipped = False
                    st.rerun()

            if st.session_state.cram_is_flipped and f_update_sm2:
                st.markdown("---")
                st.markdown("#### Optional SM-2 Rating:")
                col_fca, col_fch, col_fcg, col_fce = st.columns(4)
                
                def process_folder_cram_sm2(q_score: int):
                    n_rep, n_ef, n_int, n_rev = calculate_sm2(
                        quality=q_score,
                        repetition_count=curr_fc_card.get("repetition_count", 0),
                        ease_factor=curr_fc_card.get("ease_factor", 2.5),
                        interval_days=curr_fc_card.get("interval_days", 0)
                    )
                    u_card = curr_fc_card.copy()
                    u_card["repetition_count"] = n_rep
                    u_card["ease_factor"] = n_ef
                    u_card["interval_days"] = n_int
                    u_card["next_review_at"] = n_rev
                    u_card["mastery_level"] = n_rep
                    
                    save_card(st.session_state.current_folder, curr_fc_card.get("deck_name"), u_card)
                    
                    if st.session_state.cram_card_index < total_f_cram - 1:
                        st.session_state.cram_card_index += 1
                    st.session_state.cram_is_flipped = False
                    st.rerun()

                with col_fca:
                    if st.button("🔴 Blackout", key="fcram_sm2_0", use_container_width=True): process_folder_cram_sm2(0)
                with col_fch:
                    if st.button("🟠 Hard", key="fcram_sm2_1", use_container_width=True): process_folder_cram_sm2(1)
                with col_fcg:
                    if st.button("🟡 Good", key="fcram_sm2_2", use_container_width=True): process_folder_cram_sm2(2)
                with col_fce:
                    if st.button("🟢 Easy", key="fcram_sm2_3", use_container_width=True): process_folder_cram_sm2(3)

# -------------------------------------------------------------------
# VIEW 1: ALL FOLDERS VIEW
# -------------------------------------------------------------------

else:
    if st.button("➕ Add New Folder", type="primary"):
        new_folder_popup()

    st.subheader("All Folders")
    folders = get_all_folders()

    if not folders:
        st.info("No folders created yet. Click '➕ Add New Folder' above to get started!")
    else:
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