import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from ingest import ingest_document
from agent import agent_graph, CHROMA_COLLECTION
from quiz_engine import generate_quiz_question, grade_user_answer
from storage_utils import create_folder, create_deck, get_folders, get_decks, save_card_to_json, load_deck, update_deck_cards, update_single_card, delete_single_card


# 1. PAGE CONFIG
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

CARD_TYPES = ["concept", "code", "term", "problem"]

# Helper to display media link as image if direct URL, or standard link
def render_media(media_url: str):
    if not media_url:
        return
    clean_url = media_url.lower()
    if any(clean_url.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif", ".webp"]):
        st.image(media_url, use_column_width=True)
    else:
        st.markdown(f"🔗 [View Attached Media]({media_url})")

# 2. DIALOG POPUPS
@st.dialog("New Folder")
def new_folder_popup():
    new_folder_name = st.text_input("Enter New Folder Name")
    if st.button("Create"):
        if new_folder_name.strip():
            create_folder(new_folder_name.strip())
            st.rerun()

@st.dialog("New Deck")
def new_deck_popup():
    new_deck_name = st.text_input("Enter New Deck Name")
    if st.button("Create"):
        if new_deck_name.strip():
            create_deck(st.session_state.current_folder, new_deck_name.strip())
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
            save_card_to_json(
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
                update_single_card(
                    folder_name=st.session_state.current_folder,
                    deck_name=st.session_state.current_deck,
                    updated_card=updated_card
                )
                st.rerun()
            else:
                st.warning("Both Front and Back text are required.")

    with col_delete:
        with st.expander("🗑️ Delete Card"):
            st.warning("Delete this card permanently?")
            if st.button("Yes, Delete", type="primary", use_container_width=True, key=f"del_confirm_{card.get('card_id')}"):
                delete_single_card(
                    folder_name=st.session_state.current_folder,
                    deck_name=st.session_state.current_deck,
                    card_id=card.get("card_id")
                )
                st.rerun()

tab_folders, tab_sets = st.tabs(["folders", "sets"])

# 3. NAVIGATION LOGIC
with tab_folders:

    # VIEW 3: INDIVIDUAL DECK VIEW
    if st.session_state.current_folder is not None and st.session_state.current_deck is not None:
        if st.button(f"⬅️ Back to Folder: {st.session_state.current_folder}"):
            st.session_state.current_deck = None
            st.rerun()

        st.header(f"Deck: {st.session_state.current_deck}")
        st.caption(f"Folder: {st.session_state.current_folder}")
        st.divider()

        # SUB-TABS INSIDE DECK VIEW
        deck_tab_cards, deck_tab_study = st.tabs(["🎴 Cards", "📖 Study Mode"])

        # SUB-TAB 1: CARD LIST & MANAGEMENT
        with deck_tab_cards:
            col_title, col_btn = st.columns([4, 1])
            with col_btn:
                if st.button("➕ Add Card", use_container_width=True):
                    new_card_popup()

            deck_content = load_deck(st.session_state.current_folder, st.session_state.current_deck)
            cards = deck_content.get("cards", [])
            st.subheader(f"Cards ({len(cards)})")

            if not cards:
                st.info("No cards in this deck yet. Click '➕ Add Card' above to create one.")
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

        # SUB-TAB 2: STUDY MODE (PLACEHOLDER FOR ACTIVE RECALL)
        with deck_tab_study:
            st.subheader("Interactive Study Mode")
            st.info("Interactive card flipping / active recall study session will go here.")                

    # VIEW 2: INDIVIDUAL FOLDER VIEW (List of Decks)
    elif st.session_state.current_folder is not None:
        st.header(f"Folder: {st.session_state.current_folder}")
        
        if st.button("⬅️ Back to All Folders"):
            st.session_state.current_folder = None
            st.session_state.current_deck = None
            st.rerun()

        st.divider()

        if st.button("➕ Add New Deck"):
            new_deck_popup()

        st.subheader("Decks")
        decks = get_decks(st.session_state.current_folder)

        for d in decks:
            if st.button(d, key=f"deck_btn_{d}"):
                st.session_state.current_deck = d
                st.rerun()

    # VIEW 1: ALL FOLDERS VIEW
    else:
        if st.button("➕ Add New Folder"):
            new_folder_popup()

        st.subheader("All Folders")
        folders = get_folders()

        for f in folders:
            if st.button(f, key=f"folder_btn_{f}"):
                st.session_state.current_folder = f
                st.rerun()