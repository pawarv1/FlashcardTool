# RecallMind RAG 📚

*An Agentic, Privacy-First Local RAG Flashcard & Study Assistant Engine*

**RecallMind RAG** is an intelligent, privacy-focused study application that turns notes and documents into flashcards, automates review schedules with the **SuperMemo-2 (SM-2)** algorithm, generates AI active-recall quizzes, and constructs interactive concept maps—all running 100% locally on your machine.

---

## 🌟 Key Features

* **🧠 Agentic RAG Study Assistant:** Chat with an agentic workflow powered by **LangGraph** and local **Llama 3.1** via **Ollama**. Context queries dynamically pull from your card collection using vector search.
* **📄 Automated Document Ingestion:** Upload PDFs, Word documents, PowerPoint decks, CSVs, Jupyter Notebooks, or Markdown files. The app auto-chunks and drafts structured flashcards while screening for duplicates.
* **📈 Adaptive SM-2 Spaced Repetition:** Calculates recall intervals and ease factors based on your study ratings, featuring a **7 to 30-day Review Workload Forecast**.
* **⚡ Focused Cram Sessions:** Filter cards across individual decks or entire folders by custom tags (`#midterm`, `#algorithms`), difficulty levels, or unseen status.
* **📝 Active Recall Quiz Engine & Weak Concept Remediation:** Generates short-answer quizzes graded by local LLMs. If you miss a concept, the engine drafts targeted remediation flashcards to patch your knowledge gaps.
* **🕸️ Interactive Concept Knowledge Graph:** Renders dynamic PyVis networks showing semantic relationships between cards based on vector embedding distances, optimized with vector matrix caching.
* **🔊 Audio TTS, Media Attachments & Anki Export:** Listen to prompts with `gTTS`, attach image/URL resources, or export your decks natively to `.apkg` packages for Anki.
* **⚙️ Automated Lifecycle & Backups:** Auto-manages the background `ollama serve` process and provides full system export/restore `.zip` archives.

---

## 🛠️ System Architecture & Tech Stack

| Layer | Technology / Library | Purpose |
| --- | --- | --- |
| **User Interface** | Streamlit | Responsive multi-view dashboard & modals |
| **Relational Database** | SQLite (WAL Mode) | Persistent storage for folders, decks, cards, and quiz history |
| **Vector Database** | ChromaDB | Local vector embeddings and similarity queries |
| **Local Inference & Agent** | Ollama (`llama3.1`) + LangGraph | Local LLM execution & agentic state graph routing |
| **Graph Visualizer** | PyVis + NetworkX + Scikit-Learn | Interactive concept network & cosine similarity math |
| **Audio & Media** | gTTS + PIL | Text-To-Speech generation & image optimization |

---

## 🗄️ Database & Vector Alignment

RecallMind RAG utilizes a **Harmonized `card_id` Primary Key** architecture:

* **Data Integrity:** `cards.card_id` (UUIDv4) is shared identically across SQLite and ChromaDB vector metadata.
* **Automatic Healing:** On startup, `auto_heal_chroma_sync()` detects and synchronizes any un-indexed cards into ChromaDB without user intervention.
* **Batch Performance:** Bulk operations utilize parameterized batching (`executemany`) to ensure high database throughput.

---

## 🚀 Getting Started

### 1. Prerequisites

* **Python 3.10+**
* **Ollama:** Install Ollama from [ollama.ai](https://ollama.ai/)

### 2. Download Llama 3.1 Model

Open a terminal and pull the default model:

```bash
ollama pull llama3.1

```

### 3. Clone Repository & Setup Virtual Environment

```bash
git clone https://github.com/YOUR_USERNAME/FlashcardTool.git
cd FlashcardTool

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

```

### 4. Install Dependencies

Install all required Python libraries:

```bash
pip install streamlit chromadb langgraph langchain-core langchain-community gtts pyvis networkx scikit-learn pandas numpy pypdf python-docx python-pptx genanki

```

---

## 🏁 Running the Application

### 1. Initialize Database Schema

Run the database setup script to initialize SQLite WAL mode and build the required tables:

```bash
python db.py

```

### 2. Launch Streamlit

Start the application interface:

```bash
streamlit run app.py

```

> **Note:** The application will automatically detect and start `ollama serve` in the background if it is not already running, and terminate it cleanly upon exiting.

---

## 📂 Repository Structure

```text
FlashcardTool/
├── app.py                  # Main Streamlit UI dashboard and tab views
├── db.py                   # SQLite database connection, WAL configuration, & Chroma sync auto-heal
├── storage_utils.py        # SQLite CRUD operations, cram queries, & batch saving
├── vector_utils.py         # ChromaDB collection management and vector operations
├── card_generator.py       # LLM flashcard generation from document chunks
├── document_parser.py      # Document ingestion (PDF, DOCX, PPTX, CSV, MD, IPYNB)
├── quiz_engine.py          # AI quiz generation and evaluation logic
├── agent.py                # LangGraph state machine for local RAG chat
├── graph_utils.py          # Cached vector similarity matrix & PyVis graph generator
├── sm2_utils.py            # SuperMemo-2 spaced repetition mathematical algorithm
├── media_utils.py          # Image compression (WebP) & asset management
├── backup_utils.py         # Complete ZIP backup creation and restore logic
├── anki_utils.py           # Anki (.apkg) package exporter
└── ollama_service.py       # Automated lifecycle management for 'ollama serve'

```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
