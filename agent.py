from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from sentence_transformers import SentenceTransformer
import chromadb

EMBEDDER = SentenceTransformer("BAAI/bge-small-en-v1.5")
CHROMA_CLIENT = chromadb.PersistentClient(path="./chroma_db_data")
CHROMA_COLLECTION = CHROMA_CLIENT.get_or_create_collection(name="study_notes")

@tool
def search_study_notes(query: str, subject = None, num_results = 10) -> str:
    """Use this tool to search the user's personal study notes for concepts, 
    definitions, diagrams, algorithms, or code examples."""

    query_vec = EMBEDDER.encode(query).tolist()
    where_filter = {"subject": subject.strip().title()} if subject else None
    results = CHROMA_COLLECTION.query(
        query_embeddings=[query_vec],
        where=where_filter,
        n_results=num_results
    )
    
    if results["documents"] and results["documents"][0]:
        formatted_chunks = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            formatted_chunks.append(f"[Source Metadata: {meta}]\n{doc}")
        
        return "\n\n---\n\n".join(formatted_chunks)
        
    return "No relevant notes found for this query."

tools = [search_study_notes]

# Save conversation context by adding messages instead of overwritting it
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatOllama(model="llama3.1", temperature=0)
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: State):
    """The central brain node: decides whether to respond directly or call a tool."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Create the Langgraph
builder = StateGraph(State)

builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
# Conditional edge: if LLM output contains tool calls -> go to 'tools', else -> go to END
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")  # Loop tool output back to agent for reasoning

agent_graph = builder.compile()