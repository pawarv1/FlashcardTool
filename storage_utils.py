import uuid
from typing import List, Dict, Any
from datetime import datetime, timedelta, date
from db import get_db_connection
from vector_utils import upsert_card_to_chroma, delete_card_from_chroma, delete_deck_from_chroma, delete_folder_from_chroma, rename_deck_in_chroma, rename_folder_in_chroma, get_collection

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
        existing = cursor.execute("SELECT id, is_deleted FROM folders WHERE name = ?;", (clean_name,)).fetchone()

        if existing:
            if existing["is_deleted"] == 1:
                cursor.execute("UPDATE folders SET is_deleted = 0 WHERE id = ?;", (existing["id"],))
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
            cursor.execute("UPDATE folders SET name = ? WHERE name = ? AND is_deleted = 0;", (clean_new, clean_old))
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

        folder = cursor.execute("SELECT id FROM folders WHERE name = ? AND is_deleted = 0;", (clean_name,)).fetchone()

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

def get_all_folder_tags(folder_name: str) -> List[str]:
    """Returns a sorted list of unique tags across ALL decks in a folder."""
    clean_folder = folder_name.strip()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        query = """
            SELECT c.tags 
            FROM cards c
            JOIN decks d ON c.deck_id = d.id
            JOIN folders f ON d.folder_id = f.id
            WHERE f.name = ? 
              AND f.is_deleted = 0 AND d.is_deleted = 0 AND c.is_deleted = 0
              AND c.tags IS NOT NULL AND c.tags != '';
        """
        rows = cursor.execute(query, (clean_folder,)).fetchall()

    unique_tags = set()
    for row in rows:
        if row["tags"]:
            for tag in row["tags"].split(","):
                clean_tag = tag.strip().lower()
                if clean_tag:
                    unique_tags.add(clean_tag)

    return sorted(list(unique_tags))

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

def get_all_deck_tags(folder_name: str, deck_name: str) -> List[str]:
    """Returns a sorted list of unique tags used in a specific deck."""
    clean_folder = folder_name.strip()
    clean_deck = deck_name.strip()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        query = """
            SELECT c.tags 
            FROM cards c
            JOIN decks d ON c.deck_id = d.id
            JOIN folders f ON d.folder_id = f.id
            WHERE f.name = ? AND d.name = ? 
              AND f.is_deleted = 0 AND d.is_deleted = 0 AND c.is_deleted = 0
              AND c.tags IS NOT NULL AND c.tags != '';
        """
        rows = cursor.execute(query, (clean_folder, clean_deck)).fetchall()

    unique_tags = set()
    for row in rows:
        if row["tags"]:
            for tag in row["tags"].split(","):
                clean_tag = tag.strip().lower()
                if clean_tag:
                    unique_tags.add(clean_tag)

    return sorted(list(unique_tags))

def create_deck(folder_name: str, deck_name: str) -> bool:
    """Creates a new deck inside a parent folder or restores a soft-deleted one."""
    clean_folder = folder_name.strip()
    clean_deck = deck_name.strip()

    if not clean_deck:
        return False

    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        folder = cursor.execute("SELECT id FROM folders WHERE name = ? AND is_deleted = 0;", (clean_folder,)).fetchone()

        if not folder:
            return False

        folder_id = folder["id"]

        existing = cursor.execute("SELECT id, is_deleted FROM decks WHERE folder_id = ? AND name = ?;", (folder_id, clean_deck)).fetchone()

        if existing:
            if existing["is_deleted"] == 1:
                cursor.execute("UPDATE decks SET is_deleted = 0 WHERE id = ?;", (existing["id"],))
                conn.commit()
                return True
            return False

        cursor.execute("INSERT INTO decks (folder_id, name) VALUES (?, ?);", (folder_id, clean_deck))
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

def get_deck_analytics(folder_name: str, deck_name: str) -> Dict[str, Any]:
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

def get_review_forecast(folder_name: str, deck_name: str, days: int = 7) -> Dict[str, int]:
    """
    Groups active cards by their next_review_at date over a future window (default 7 days).
    Returns a dictionary mapping date labels to due card counts.
    """
    clean_folder = folder_name.strip()
    clean_deck = deck_name.strip()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        query = """
            SELECT c.next_review_at 
            FROM cards c
            JOIN decks d ON c.deck_id = d.id
            JOIN folders f ON d.folder_id = f.id
            WHERE f.name = ? AND d.name = ? 
              AND f.is_deleted = 0 AND d.is_deleted = 0 AND c.is_deleted = 0;
        """
        rows = cursor.execute(query, (clean_folder, clean_deck)).fetchall()

    if not rows:
        return {}

    today = date.today()
    forecast_buckets = {"Overdue / Today": 0}
    
    for i in range(1, days):
        day_date = today + timedelta(days=i)
        day_label = day_date.strftime("%a (%m/%d)")
        forecast_buckets[day_label] = 0

    forecast_buckets[f"+{days} Days+"] = 0

    for row in rows:
        next_review_str = row["next_review_at"]
        
        if not next_review_str:
            forecast_buckets["Overdue / Today"] += 1
            continue

        try:
            card_date = datetime.strptime(next_review_str.split(".")[0].split("T")[0], "%Y-%m-%d").date()
        except Exception:
            forecast_buckets["Overdue / Today"] += 1
            continue

        days_diff = (card_date - today).days

        if days_diff <= 0:
            forecast_buckets["Overdue / Today"] += 1
        elif 1 <= days_diff < days:
            day_label = card_date.strftime("%a (%m/%d)")
            if day_label in forecast_buckets:
                forecast_buckets[day_label] += 1
            else:
                forecast_buckets["Overdue / Today"] += 1
        else:
            forecast_buckets[f"+{days} Days+"] += 1

    return forecast_buckets
    
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

def load_cram_cards(folder_name: str, deck_name: str = None, selected_tags: List[str] = None, difficulty_filter: str = "all", card_limit: int = 0) -> List[Dict]:
    """
    Fetches active cards for a custom cram session based on folder/deck scope,
    tags, difficulty metrics, and optional card limits.
    """
    clean_folder = folder_name.strip()
    clean_deck = deck_name.strip() if deck_name else None

    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        query_params = [clean_folder]
        base_query = """
            SELECT c.*, d.name AS deck_name, f.name AS folder_name
            FROM cards c
            JOIN decks d ON c.deck_id = d.id
            JOIN folders f ON d.folder_id = f.id
            WHERE f.name = ? 
              AND f.is_deleted = 0 AND d.is_deleted = 0 AND c.is_deleted = 0
        """

        if clean_deck:
            base_query += " AND d.name = ?"
            query_params.append(clean_deck)

        if difficulty_filter == "hard":
            base_query += " AND (c.ease_factor < 2.3 OR c.mastery_level = 0)"
        elif difficulty_filter == "unseen":
            base_query += " AND c.repetition_count = 0"

        base_query += " ORDER BY c.ease_factor ASC, c.created_at ASC"

        rows = cursor.execute(base_query, query_params).fetchall()

    cards = []
    normalized_tags = [t.strip().lower() for t in selected_tags] if selected_tags else []

    for r in rows:
        card_dict = dict(r)
        card_dict["card_id"] = card_dict.pop("id")
        
        card_tags_str = card_dict.get("tags") or ""
        card_tags = [t.strip().lower() for t in card_tags_str.split(",") if t.strip()]

        if normalized_tags:
            if not any(tag in card_tags for tag in normalized_tags):
                continue

        cards.append(card_dict)

    if card_limit > 0 and len(cards) > card_limit:
        cards = cards[:card_limit]

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
                id, deck_id, card_type, front, back, code_block, explanation, media_link,
                source_type, mastery_level, tags, ease_factor, interval_days, 
                repetition_count, next_review_at, synced_to_chroma, is_deleted
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0)
            ON CONFLICT(id) DO UPDATE SET
                card_type = excluded.card_type,
                front = excluded.front,
                back = excluded.back,
                code_block = excluded.code_block,
                explanation = excluded.explanation,
                media_link = excluded.media_link,
                tags = excluded.tags,
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
            card_data.get("media_link"),
            card_data.get("source_type", "manual_entry"),
            card_data.get("mastery_level", 0),
            card_data.get("tags"),
            card_data.get("ease_factor", 2.5),
            card_data.get("interval_days", 0),
            card_data.get("repetition_count", 0),
            card_data.get("next_review_at")
        ))
        conn.commit()

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
                id, deck_id, card_type, front, back, code_block, explanation, media_link,
                source_type, mastery_level, tags, ease_factor, interval_days, 
                repetition_count, next_review_at, synced_to_chroma, is_deleted
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0)
            ON CONFLICT(id) DO UPDATE SET
                card_type = excluded.card_type,
                front = excluded.front,
                back = excluded.back,
                code_block = excluded.code_block,
                explanation = excluded.explanation,
                media_link = excluded.media_link,
                tags = excluded.tags,
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
                card.get("media_link"),
                card.get("source_type", "manual_entry"),
                card.get("mastery_level", 0),
                card.get("tags"),
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
                "media_link": card.get("media_link") or "",
                "tags": card.get("tags") or "",
                "source_type": card.get("source_type", "manual_entry")
            })

        conn.commit()

        try:
            collection = get_collection()
            collection.upsert(documents=documents, metadatas=metadatas, ids=ids_to_sync)

            # Chunk parameter updates to stay under SQLite variable limits (max 900 per batch)
            chunk_size = 900
            for i in range(0, len(ids_to_sync), chunk_size):
                chunk_ids = ids_to_sync[i:i + chunk_size]
                cursor.execute(
                    f"UPDATE cards SET synced_to_chroma = 1 WHERE id IN ({','.join(['?']*len(chunk_ids))});", 
                    chunk_ids
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

# -------------------------------------------------------------------
# QUIZ OPERATIONS
# -------------------------------------------------------------------

def log_quiz_attempt(folder_name: str, deck_name: str, question: str, user_answer: str, score_label: str, grade_percent: int, feedback: str) -> bool:
    """Logs a completed quiz attempt into SQLite."""
    clean_folder = folder_name.strip()
    clean_deck = deck_name.strip()
    attempt_id = str(uuid.uuid4())

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO quiz_history (
                id, folder_name, deck_name, question, user_answer, 
                score_label, grade_percent, feedback
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            attempt_id, clean_folder, clean_deck, question, 
            user_answer, score_label, grade_percent, feedback
        ))
        conn.commit()
        return True

def get_quiz_analytics(folder_name: str, deck_name: str) -> Dict[str, Any]:
    """Retrieves aggregated quiz statistics and history logs for a specific deck."""
    clean_folder = folder_name.strip()
    clean_deck = deck_name.strip()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        agg_row = cursor.execute("""
            SELECT 
                COUNT(*) AS total_quizzes,
                AVG(grade_percent) AS avg_score,
                SUM(CASE WHEN grade_percent >= 70 THEN 1 ELSE 0 END) AS passed_quizzes
            FROM quiz_history
            WHERE folder_name = ? AND deck_name = ?;
        """, (clean_folder, clean_deck)).fetchone()

        history_rows = cursor.execute("""
            SELECT question, user_answer, score_label, grade_percent, feedback, created_at
            FROM quiz_history
            WHERE folder_name = ? AND deck_name = ?
            ORDER BY created_at DESC
            LIMIT 20;
        """, (clean_folder, clean_deck)).fetchall()

        total = agg_row["total_quizzes"] or 0
        avg_score = round(agg_row["avg_score"] or 0, 1)
        passed = agg_row["passed_quizzes"] or 0
        pass_rate = round((passed / total) * 100, 1) if total > 0 else 0.0

        return {
            "total_quizzes": total,
            "avg_score": avg_score,
            "pass_rate": pass_rate,
            "history": [dict(r) for r in history_rows]
        }