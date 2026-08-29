import sqlite3
from vector_utils import get_collection

DB_PATH = "./study_assistant.db"

def get_db_connection():
    """Returns a SQLite connection object with dict-like row formatting and FK constraints enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # Enforce Foreign Keys & Enable Write-Ahead Logging
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn

def init_db():
    """Initializes normalized SQLite tables, schema indices, schema migrations, and soft-delete defaults."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # 1. Folders Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS folders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                is_deleted INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # 2. Decks Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS decks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                folder_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                is_deleted INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (folder_id) REFERENCES folders(id) ON DELETE CASCADE,
                UNIQUE(folder_id, name)
            );
        """)
        
        # 3. Cards Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cards (
                id TEXT PRIMARY KEY,
                deck_id INTEGER NOT NULL,
                card_type TEXT DEFAULT 'concept',
                front TEXT NOT NULL,
                back TEXT NOT NULL,
                code_block TEXT,
                explanation TEXT,
                media_link TEXT,
                tags TEXT,
                source_type TEXT DEFAULT 'manual_entry',
                mastery_level INTEGER DEFAULT 0,
                ease_factor REAL DEFAULT 2.5,
                interval_days INTEGER DEFAULT 0,
                repetition_count INTEGER DEFAULT 0,
                next_review_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                synced_to_chroma INTEGER DEFAULT 0,
                is_deleted INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (deck_id) REFERENCES decks(id) ON DELETE CASCADE
            );
        """)

        # 4. Quiz History Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS quiz_history (
            id TEXT PRIMARY KEY,
            folder_name TEXT NOT NULL,
            deck_name TEXT NOT NULL,
            question TEXT NOT NULL,
            user_answer TEXT,
            score_label TEXT,
            grade_percent INTEGER NOT NULL,
            feedback TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Speed up common queries with indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_decks_folder ON decks(folder_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cards_deck ON cards(deck_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cards_sync ON cards(synced_to_chroma);")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_quiz_history_lookup 
            ON quiz_history(folder_name, deck_name, created_at DESC);
        """)
        conn.commit()
    print("SQLite database initialized successfully.")

def auto_heal_chroma_sync():
    """Scans SQLite for unsynced active cards and auto-upserts them to ChromaDB."""
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        query = """
            SELECT 
            f.name AS folder_name,
            d.name AS deck_name,
            c.*
            FROM folders f
            JOIN decks d ON d.folder_id = f.id
            JOIN cards c ON c.deck_id = d.id
            WHERE c.synced_to_chroma = 0
            AND c.is_deleted = 0
            AND d.is_deleted = 0
            AND f.is_deleted = 0;
        """
        unsynced_cards = cursor.execute(query).fetchall()
        
        if not unsynced_cards:
            return

        collection = get_collection()
        documents = []
        metadatas = []
        ids = []

        for row in unsynced_cards:
            card = dict(row)
            front_text = card["front"].strip()
            card_id = card["id"]
            
            clean_folder = card["folder_name"].strip().replace(" ", "_")
            clean_deck = card["deck_name"].strip().replace(" ", "_").lower()
            
            metadata = {
                "card_id": card_id,
                "front": front_text,
                "back": card["back"],
                "folder": clean_folder,
                "deck": clean_deck,
                "card_type": card.get("card_type", "concept"),
                "media_link": card.get("media_link") or "",
                "tags": card.get("tags") or "",
                "source_type": card.get("source_type", "manual_entry")
            }
            
            documents.append(front_text)
            metadatas.append(metadata)
            ids.append(card_id)

        try:
            collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
            # Mark as synced in SQLite
            cursor.executemany(
                "UPDATE cards SET synced_to_chroma = 1 WHERE id = ?;",
                [(cid,) for cid in ids]
            )
            conn.commit()
            print(f"Auto-healed and synced {len(ids)} cards to ChromaDB.")
        except Exception as e:
            print(f"Warning: Auto-heal Chroma sync failed: {e}")

if __name__ == "__main__":
    init_db()
    auto_heal_chroma_sync()