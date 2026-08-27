import html
import tempfile
import numpy as np
import networkx as nx
from pyvis.network import Network
from vector_utils import get_collection
from storage_utils import load_deck_cards

# Color scheme per card type
TYPE_COLORS = {
    "concept": "#007acc",      # Blue
    "code_snippet": "#28a745", # Green
    "definition": "#fd7e14",   # Orange
    "formula": "#dc3545",      # Red
    "comparison": "#6f42c1",   # Purple
    "example": "#17a2b8"       # Cyan
}

def cosine_similarity(v1: list, v2: list) -> float:
    """Computes cosine similarity between two vector embeddings."""
    a = np.array(v1)
    b = np.array(v2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def generate_knowledge_graph_html(folder_name: str, deck_name: str, similarity_threshold: float = 0.45) -> str:
    """
    Fetches active card vectors from ChromaDB, computes pairwise similarity,
    and returns a PyVis interactive HTML string.
    """
    
    # 1. Fetch active cards from SQLite using exact folder/deck names
    cards = load_deck_cards(folder_name, deck_name)
    if not cards or len(cards) < 2:
        return "<p style='color: #666; font-family: sans-serif; text-align: center; padding-top: 40px;'>Add at least 2 cards to this deck to generate a visual knowledge graph.</p>"

    card_map = {c["card_id"]: c for c in cards}
    card_ids = list(card_map.keys())

    # 2. Get embeddings from ChromaDB
    collection = get_collection()
    chroma_data = collection.get(
        ids=card_ids,
        include=["embeddings"]
    )

    if not chroma_data:
        return "<p style='color: #666; font-family: sans-serif; text-align: center; padding-top: 40px;'>No vector data returned from ChromaDB.</p>"

    ids = chroma_data.get("ids")
    embeddings = chroma_data.get("embeddings")

    if ids is None or embeddings is None or len(ids) == 0 or len(embeddings) == 0:
        return "<p style='color: #666; font-family: sans-serif; text-align: center; padding-top: 40px;'>No vector embeddings found. Ensure cards are synced to ChromaDB.</p>"

    # Map ID to embedding vector
    id_to_vec = {cid: vec for cid, vec in zip(ids, embeddings)}

    # 3. Build NetworkX Graph
    G = nx.Graph()

    # Add Nodes
    for cid in card_ids:
        card = card_map[cid]
        front_text = card.get("front", "Untitled")
        label_text = front_text[:30] + "..." if len(front_text) > 30 else front_text
        c_type = card.get("card_type", "concept")
        color = TYPE_COLORS.get(c_type, "#007acc")

        # Safely escape HTML characters in card text to prevent tooltip corruption
        safe_front = html.escape(front_text)
        safe_back = html.escape(card.get("back", "")[:100])

        tooltip_html = (
            f"<b>Type:</b> {html.escape(c_type.upper())}<br>"
            f"<b>Front:</b> {safe_front}<br>"
            f"<b>Back:</b> {safe_back}...<br>"
            f"<b>Repetitions:</b> {card.get('repetition_count', 0)}"
        )

        G.add_node(
            cid,
            label=label_text,
            title=tooltip_html,
            color=color,
            shape="dot",
            size=18
        )

    # Add Edges based on cosine similarity threshold
    valid_ids = [cid for cid in card_ids if cid in id_to_vec]
    for i in range(len(valid_ids)):
        for j in range(i + 1, len(valid_ids)):
            id1, id2 = valid_ids[i], valid_ids[j]
            sim = cosine_similarity(id_to_vec[id1], id_to_vec[id2])

            if sim >= similarity_threshold:
                G.add_edge(
                    id1, 
                    id2, 
                    value=round(sim, 2), 
                    title=f"Similarity: {sim:.2%}",
                    color="#cccccc"
                )

    # 4. Render PyVis Network
    net = Network(height="550px", width="100%", bgcolor="#ffffff", font_color="#333333")
    net.from_nx(G)

    # Physics options to keep network centered and clean
    net.set_options("""
    var options = {
      "nodes": {
        "font": { "size": 14, "face": "arial" }
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -3000,
          "centralGravity": 0.3,
          "springLength": 120
        },
        "minVelocity": 0.75
      }
    }
    """)

    # Cross-platform safe temp file handling
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        tmp_path = tmp_file.name

    net.save_graph(tmp_path)
    
    with open(tmp_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    return html_content