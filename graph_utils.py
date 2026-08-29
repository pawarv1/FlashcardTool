import tempfile
import networkx as nx
from pyvis.network import Network
from vector_utils import get_collection
import numpy as np
import html

def generate_knowledge_graph_html(folder_name: str, deck_name: str, similarity_threshold: float = 0.45) -> str:
    """
    Queries ChromaDB for all card embeddings in a deck, computes pairwise vector distances,
    and generates a dark-mode interactive PyVis graph HTML string.
    """
    clean_folder = folder_name.strip().replace(" ", "_")
    clean_deck = deck_name.strip().replace(" ", "_").lower()

    collection = get_collection()
    
    try:
        results = collection.get(
            where={
                "$and": [
                    {"folder": clean_folder},
                    {"deck": clean_deck}
                ]
            },
            include=["embeddings", "metadatas", "documents"]
        )
    except Exception as e:
        print(f"Error querying vectors for graph: {e}")
        return "<h4 style='color: #e6edf3;'>Unable to load knowledge graph vectors.</h4>"

    if not results or not results["ids"]:
        return "<div style='color: #90a4ae; padding: 20px; text-align: center;'>No cards found in this deck to map. Add flashcards to generate a graph!</div>"

    ids = results["ids"]
    metadatas = results["metadatas"]
    documents = results["documents"]

    # Initialize PyVis Network with Dark Mode Settings
    net = Network(
        height="550px", 
        width="100%", 
        bgcolor="#0e1117",
        font_color="#e6edf3",
        directed=False
    )

    # 1. Add Nodes
    for idx, card_id in enumerate(ids):
        meta = metadatas[idx] if metadatas else {}
        front_text = meta.get("front") or (documents[idx] if documents else f"Card {idx+1}")
        card_type = meta.get("card_type", "concept").upper()
        clean_front_text = html.escape(front_text)
        
        short_label = front_text[:30] + "..." if len(front_text) > 30 else front_text

        tooltip_html = f"<b>Type:</b> {html.escape(card_type)}<br><b>Prompt:</b> {clean_front_text}"

        net.add_node(
            n_id=card_id,
            label=short_label,
            title=tooltip_html,
            color="#007acc",
            borderWidth=2,
            shape="dot",
            size=18
        )

    # 2. Compute Cosine Distance Edges
    embeddings = results.get("embeddings")
    if embeddings is not None and len(embeddings) > 1:

        emb_matrix = np.array(embeddings)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        norm_emb = emb_matrix / norms

        similarity_matrix = np.dot(norm_emb, norm_emb.T)

        num_nodes = len(ids)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                sim_score = float(similarity_matrix[i, j])
                
                # Connect nodes if similarity meets or exceeds user threshold
                if sim_score >= similarity_threshold:
                    # Scale edge width and opacity by similarity score
                    edge_width = max(1.0, (sim_score - similarity_threshold) * 8.0)
                    
                    net.add_edge(
                        source=ids[i],
                        to=ids[j],
                        value=edge_width,
                        color={"color": "#4fc3f7", "highlight": "#00e5ff", "opacity": 0.6},  # High-contrast cyan
                        title=f"Similarity: {sim_score:.2f}"
                    )

    # Physics Engine Styling (Smooth Dark Mode Forces)
    net.set_options("""
    {
      "nodes": {
        "font": { "size": 13, "face": "arial", "color": "#e6edf3" }
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

    # Export to HTML string
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        with open(tmp_file.name, "r", encoding="utf-8") as f:
            html_content = f.read()

    return html_content