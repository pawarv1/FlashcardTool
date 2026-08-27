import uuid
from typing import List, Dict, Optional
from db import get_db_connection
from vector_utils import (
    upsert_card_to_chroma,
    delete_card_from_chroma,
    delete_deck_from_chroma,
    delete_folder_from_chroma,
    rename_deck_in_chroma,
    rename_folder_in_chroma,
    get_collection
)

# -------------------------------------------------------------------
# FOLDER OPERATIONS
# -------------------------------------------------------------------

def get_all_folders() -> List[str]:
    """Returns a list of active (non-deleted) folder names."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        rows = cursor.execute(
            "SELECT name FROM folders WHERE is_deleted = 0 ORDER BY name ASC;"
        ).fetchall()
        return [r["name"] for r in rows]

def create_folder(folder_name: str) -> bool:
    """Creates a new folder or restores a soft-deleted one."""
    clean_name = folder_name.strip()
    if not clean_name:
        return False

    with get_db_connection() as conn:
        cursor = conn.cursor()
        existing = cursor.execute(
            "SELECT id, is_deleted FROM folders WHERE name = ?;", (clean_name,)
        ).fetchone()

        if existing:
            if existing["is_deleted"] == 1:
                # Restore the folder
                cursor.execute(
                    "UPDATE folders SET is_deleted = 0 WHERE id = ?;", (existing["id"],)
                )
                conn.commit()
                return True
            return False

        cursor.execute("INSERT INTO folders (name) VALUES (?);", (clean_name,))
        conn.commit()
        return True

def rename_folder(old_name: str, new_name: str) -> bool:
    """Renames an existing folder and updates vector metadata in ChromaDB."""
    clean_old = old_name.strip()
    clean_new = new_name.strip()

    if not clean_new or clean_old == clean_new:
        return False

    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "UPDATE folders SET name = ? WHERE name = ? AND is_deleted = 0;",
                (clean_new, clean_old)
            )
            if cursor.rowcount > 0:
                conn.commit()
                rename_folder_in_chroma(clean_old, clean_new)
                return True
        except Exception as e:
            print(f"Error renaming folder: {e}")
    return False

def delete_folder(folder_name: str) -> bool:
    """Soft-deletes a folder and purges its associated vectors from ChromaDB."""
    clean_name = folder_name.strip()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        folder = cursor.execute(
            "SELECT id FROM folders WHERE name = ? AND is_deleted = 0;", (clean_name,)
        ).fetchone()

        if not folder:
            return False

        folder_id = folder["id"]

        cursor.execute("UPDATE folders SET is_deleted = 1 WHERE id = ?;", (folder_id,))
        cursor.execute("UPDATE decks SET is_deleted = 1 WHERE folder_id = ?;", (folder_id,))
        cursor.execute("""
            UPDATE cards 
            SET is_deleted = 1 
            WHERE deck_id IN (SELECT id FROM decks WHERE folder_id = ?);
        """, (folder_id,))
        
        conn.commit()

        delete_folder_from_chroma(clean_name)
        return True

# -------------------------------------------------------------------
# DECK OPERATIONS
# -------------------------------------------------------------------

def get_decks_in_folder(folder_name: str) -> List[str]:
    """Returns a list of active deck names within a specific folder."""
    clean_folder = folder_name.strip()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        query = """
            SELECT d.name 
            FROM decks d
            JOIN folders f ON d.folder_id = f.id
            WHERE f.name = ? AND f.is_deleted = 0 AND d.is_deleted = 0
            ORDER BY d.name ASC;
        """
        rows = cursor.execute(query, (clean_folder,)).fetchall()
        return [r["name"] for r in rows]

def create_deck(folder_name: str, deck_name: str) -> bool:
    """Creates a new deck inside a parent folder or restores a soft-deleted one."""
    clean_folder = folder_name.strip()
    clean_deck = deck_name.strip()

    if not clean_deck:
        return False

    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        folder = cursor.execute(
            "SELECT id FROM folders WHERE name = ? AND is_deleted = 0;", (clean_folder,)
        ).fetchone()

        if not folder:
            return False

        folder_id = folder["id"]

        existing = cursor.execute(
            "SELECT id, is_deleted FROM decks WHERE folder_id = ? AND name = ?;",
            (folder_id, clean_deck)
        ).fetchone()

        if existing:
            if existing["is_deleted"] == 1:
                # Restore the deck
                cursor.execute(
                    "UPDATE decks SET is_deleted = 0 WHERE id = ?;", (existing["id"],)
                )
                conn.commit()
                return True
            return False

        cursor.execute(
            "INSERT INTO decks (folder_id, name) VALUES (?, ?);",
            (folder_id, clean_deck)
        )
        conn.commit()
        return True

def rename_deck(folder_name: str, old_deck_name: str, new_deck_name: str) -> bool:
    """Renames a deck and updates metadata in ChromaDB."""
    clean_folder = folder_name.strip()
    clean_old = old_deck_name.strip()
    clean_new = new_deck_name.strip()

    if not clean_new or clean_old == clean_new:
        return False

    with get_db_connection() as conn:
        cursor = conn.cursor()
        query = """
            UPDATE decks 
            SET name = ? 
            WHERE name = ? AND folder_id = (SELECT id FROM folders WHERE name = ? AND is_deleted = 0)
              AND is_deleted = 0;
        """
        cursor.execute(query, (clean_new, clean_old, clean_folder))
        if cursor.rowcount > 0:
            conn.commit()
            rename_deck_in_chroma(clean_folder, clean_old, clean_new)
            return True
    return False

def delete_deck(folder_name: str, deck_name: str) -> bool:
    """Soft-deletes a deck and purges its cards from ChromaDB."""
    clean_folder = folder_name.strip()
    clean_deck = deck_name.strip()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        deck = cursor.execute("""
            SELECT d.id 
            FROM decks d
            JOIN folders f ON d.folder_id = f.id
            WHERE f.name = ? AND d.name = ? AND f.is_deleted = 0 AND d.is_deleted = 0;
        """, (clean_folder, clean_deck)).fetchone()

        if not deck:
            return False

        deck_id = deck["id"]

        cursor.execute("UPDATE decks SET is_deleted = 1 WHERE id = ?;", (deck_id,))
        cursor.execute("UPDATE cards SET is_deleted = 1 WHERE deck_id = ?;", (deck_id,))
        conn.commit()

        delete_deck_from_chroma(clean_folder, clean_deck)
        return True

def get_deck_analytics(folder_name: str, deck_name: str) -> Dict[str, any]:
    """Returns total active cards, due count, and average ease factor for a deck."""
    clean_folder = folder_name.strip()
    clean_deck = deck_name.strip()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        query = """
            SELECT 
                COUNT(c.id) AS total_cards,
                SUM(CASE WHEN c.next_review_at IS NULL OR c.next_review_at <= CURRENT_TIMESTAMP THEN 1 ELSE 0 END) AS due_cards,
                AVG(c.ease_factor) AS avg_ease_factor
            FROM cards c
            JOIN decks d ON c.deck_id = d.id
            JOIN folders f ON d.folder_id = f.id
            WHERE f.name = ? AND d.name = ? 
              AND f.is_deleted = 0 AND d.is_deleted = 0 AND c.is_deleted = 0;
        """
        row = cursor.execute(query, (clean_folder, clean_deck)).fetchone()
        
        return {
            "total_cards": row["total_cards"] or 0,
            "due_cards": row["due_cards"] or 0,
            "avg_ease_factor": round(row["avg_ease_factor"] or 2.5, 2)
        }
    
# -------------------------------------------------------------------
# CARD OPERATIONS
# -------------------------------------------------------------------

def load_deck_cards(folder_name: str, deck_name: str, due_only: bool = False) -> List[Dict]:
    """Loads active cards for a deck. If due_only=True, filters for cards needing review."""
    clean_folder = folder_name.strip()
    clean_deck = deck_name.strip()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        base_query = """
            SELECT c.* 
            FROM cards c
            JOIN decks d ON c.deck_id = d.id
            JOIN folders f ON d.folder_id = f.id
            WHERE f.name = ? AND d.name = ? 
              AND f.is_deleted = 0 AND d.is_deleted = 0 AND c.is_deleted = 0
        """
        
        if due_only:
            base_query += " AND (c.next_review_at IS NULL OR c.next_review_at <= CURRENT_TIMESTAMP)"

        base_query += " ORDER BY c.next_review_at ASC, c.created_at ASC;"

        rows = cursor.execute(base_query, (clean_folder, clean_deck)).fetchall()
        
        cards = []
        for r in rows:
            card_dict = dict(r)
            card_dict["card_id"] = card_dict.pop("id")
            cards.append(card_dict)
        return cards

def save_card(folder_name: str, deck_name: str, card_data: Dict) -> bool:
    """Inserts or updates a single card in SQLite and syncs to ChromaDB."""
    clean_folder = folder_name.strip()
    clean_deck = deck_name.strip()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        deck = cursor.execute("""
            SELECT d.id 
            FROM decks d
            JOIN folders f ON d.folder_id = f.id
            WHERE f.name = ? AND d.name = ? AND f.is_deleted = 0 AND d.is_deleted = 0;
        """, (clean_folder, clean_deck)).fetchone()

        if not deck:
            return False

        deck_id = deck["id"]
        card_id = card_data.get("card_id") or str(uuid.uuid4())

        query = """
            INSERT INTO cards (
                id, deck_id, card_type, front, back, code_block, explanation, 
                source_type, mastery_level, ease_factor, interval_days, 
                repetition_count, next_review_at, synced_to_chroma, is_deleted
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0)
            ON CONFLICT(id) DO UPDATE SET
                card_type = excluded.card_type,
                front = excluded.front,
                back = excluded.back,
                code_block = excluded.code_block,
                explanation = excluded.explanation,
                source_type = excluded.source_type,
                mastery_level = excluded.mastery_level,
                ease_factor = excluded.ease_factor,
                interval_days = excluded.interval_days,
                repetition_count = excluded.repetition_count,
                next_review_at = excluded.next_review_at,
                synced_to_chroma = 0,
                is_deleted = 0;
        """

        cursor.execute(query, (
            card_id,
            deck_id,
            card_data.get("card_type", "concept"),
            card_data.get("front", "").strip(),
            card_data.get("back", "").strip(),
            card_data.get("code_block"),
            card_data.get("explanation"),
            card_data.get("source_type", "manual_entry"),
            card_data.get("mastery_level", 0),
            card_data.get("ease_factor", 2.5),
            card_data.get("interval_days", 0),
            card_data.get("repetition_count", 0),
            card_data.get("next_review_at")
        ))
        conn.commit()

        # Update ChromaDB vector store
        card_data["card_id"] = card_id
        try:
            upsert_card_to_chroma(clean_folder, clean_deck, card_data)
            cursor.execute("UPDATE cards SET synced_to_chroma = 1 WHERE id = ?;", (card_id,))
            conn.commit()
        except Exception as e:
            print(f"Warning: Failed to sync card {card_id} to ChromaDB: {e}")

        return True

def save_card_batch(folder_name: str, deck_name: str, cards: List[Dict]) -> bool:
    """Saves multiple cards in a single database transaction and bulk upserts to ChromaDB."""
    if not cards:
        return True

    clean_folder = folder_name.strip()
    clean_deck = deck_name.strip()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        deck = cursor.execute("""
            SELECT d.id 
            FROM decks d
            JOIN folders f ON d.folder_id = f.id
            WHERE f.name = ? AND d.name = ? AND f.is_deleted = 0 AND d.is_deleted = 0;
        """, (clean_folder, clean_deck)).fetchone()

        if not deck:
            return False

        deck_id = deck["id"]
        
        query = """
            INSERT INTO cards (
                id, deck_id, card_type, front, back, code_block, explanation, 
                source_type, mastery_level, ease_factor, interval_days, 
                repetition_count, next_review_at, synced_to_chroma, is_deleted
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0)
            ON CONFLICT(id) DO UPDATE SET
                card_type = excluded.card_type,
                front = excluded.front,
                back = excluded.back,
                code_block = excluded.code_block,
                explanation = excluded.explanation,
                source_type = excluded.source_type,
                mastery_level = excluded.mastery_level,
                ease_factor = excluded.ease_factor,
                interval_days = excluded.interval_days,
                repetition_count = excluded.repetition_count,
                next_review_at = excluded.next_review_at,
                synced_to_chroma = 0,
                is_deleted = 0;
        """

        ids_to_sync = []
        documents = []
        metadatas = []

        for card in cards:
            card_id = card.get("card_id") or str(uuid.uuid4())
            card["card_id"] = card_id
            
            cursor.execute(query, (
                card_id,
                deck_id,
                card.get("card_type", "concept"),
                card.get("front", "").strip(),
                card.get("back", "").strip(),
                card.get("code_block"),
                card.get("explanation"),
                card.get("source_type", "manual_entry"),
                card.get("mastery_level", 0),
                card.get("ease_factor", 2.5),
                card.get("interval_days", 0),
                card.get("repetition_count", 0),
                card.get("next_review_at")
            ))

            ids_to_sync.append(card_id)
            documents.append(card.get("front", "").strip())
            metadatas.append({
                "card_id": card_id,
                "front": card.get("front", "").strip(),
                "back": card.get("back", "").strip(),
                "folder": clean_folder.replace(" ", "_"),
                "deck": clean_deck.replace(" ", "_").lower(),
                "card_type": card.get("card_type", "concept"),
                "source_type": card.get("source_type", "manual_entry")
            })

        # Commit ALL cards to SQLite in a single transaction
        conn.commit()

        # Batch upsert to ChromaDB
        try:
            collection = get_collection()
            collection.upsert(documents=documents, metadatas=metadatas, ids=ids_to_sync)

            cursor.execute(
                f"UPDATE cards SET synced_to_chroma = 1 WHERE id IN ({','.join(['?']*len(ids_to_sync))});", 
                ids_to_sync
            )
            conn.commit()
        except Exception as e:
            print(f"Warning: Batch ChromaDB sync failed, auto-heal will handle it: {e}")

    return True

def delete_card(card_id: str) -> bool:
    """Soft-deletes a card from SQLite and removes its vector from ChromaDB."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE cards SET is_deleted = 1 WHERE id = ?;", (card_id,))
        if cursor.rowcount > 0:
            conn.commit()
            delete_card_from_chroma(card_id)
            return True
    return False