from typing import Annotated, TypedDict, Optional
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from vector_utils import get_collection

@tool
def search_deck_cards(query: str, folder_name: Optional[str] = None, deck_name: Optional[str] = None, num_results: int = 5) -> str:
    """Use this tool to search through flashcards in the user's active deck or folder for definitions, formulas, and concepts."""
    collection = get_collection()
    
    clean_folder = folder_name.strip().replace(" ", "_") if folder_name else None
    clean_deck = deck_name.strip().replace(" ", "_").lower() if deck_name else None
    
    where_filter = None
    if clean_folder and clean_deck:
        where_filter = {
            "$and": [
                {"folder": clean_folder},
                {"deck": clean_deck}
            ]
        }
    elif clean_folder:
        where_filter = {"folder": clean_folder}

    try:
        results = collection.query(
            query_texts=[query],
            n_results=num_results,
            where=where_filter
        )
        
        if results and results["documents"] and results["documents"][0]:
            formatted_cards = []
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                card_str = f"**Front:** {meta.get('front', doc)}\n**Back:** {meta.get('back', '')}"
                formatted_cards.append(card_str)
            
            return "\n\n---\n\n".join(formatted_cards)
    except Exception as e:
        return f"Error querying vector collection: {e}"
        
    return "No relevant cards found for this query in the specified deck."

tools = [search_deck_cards]

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    folder_name: str
    deck_name: str

llm = ChatOllama(model="llama3.1", temperature=0)
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: State):
    """Brain node: Enforces scope around the active deck and decides tool usage."""
    folder = state.get("folder_name", "General")
    deck = state.get("deck_name", "Default Deck")
    
    system_prompt = SystemMessage(content=(
        f"You are a helpful study assistant AI. The user currently has the deck '{deck}' "
        f"inside folder '{folder}' open.\n"
        "Use the search_deck_cards tool whenever the user asks about specific flashcards, "
        "concepts, or practice questions related to their current study material."
    ))
    
    messages = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Build StateGraph
builder = StateGraph(State)

builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

agent_graph = builder.compile()