import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from ingest import ingest_document
from agent import agent_graph, CHROMA_COLLECTION
from quiz_engine import generate_quiz_question, grade_user_answer
from storage_utils import create_folder, create_deck, get_folders, get_decks, save_card_to_json, load_deck

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

tab_folders, tab_sets = st.tabs(["folders", "sets"])

# 3. NAVIGATION LOGIC
with tab_folders:

    # VIEW 3: INDIVIDUAL DECK VIEW
    if st.session_state.current_folder is not None and st.session_state.current_deck is not None:
        if st.button(f"⬅️ Back to Folder: {st.session_state.current_folder}"):
            st.session_state.current_deck = None
            st.rerun()

        st.header(f"Deck: {st.session_state.current_deck}")
        st.write(f"Folder: {st.session_state.current_folder}")

        st.divider()

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