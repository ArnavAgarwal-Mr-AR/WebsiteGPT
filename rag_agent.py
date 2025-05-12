from dataclasses import dataclass
from typing import Optional
import asyncio
import chromadb
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from openai import AsyncOpenAI
from utils import (
    get_chroma_client,
    get_or_create_collection,
    query_collection,
    format_results_as_context
)

@dataclass
class RAGDeps:
    """Dependencies for the RAG agent."""
    chroma_client: chromadb.PersistentClient
    collection_name: str
    embedding_model: str
    model_choice: str
    api_key: str

# Create the RAG agent with explicit API key handling
agent = Agent(
    model=None,  # Model will be set via RAGDeps
    deps_type=RAGDeps,
    system_prompt="You are a helpful assistant that answers questions based on the provided documentation. "
                  "Use the retrieve tool to get relevant information from the documentation before answering. "
                  "If the documentation doesn't contain the answer, clearly state that the information isn't available "
                  "in the current documentation and provide your best general knowledge response.",
    client=lambda deps: AsyncOpenAI(api_key=deps.api_key)  # Pass API key to OpenAI client
)

@agent.tool
async def retrieve(context: RunContext[RAGDeps], search_query: str, n_results: int = 5) -> str:
    """Retrieve relevant documents from ChromaDB based on a search query.
    
    Args:
        context: The run context containing dependencies.
        search_query: The search query to find relevant documents.
        n_results: Number of results to return (default: 5).
        
    Returns:
        Formatted context information from the retrieved documents.
    """
    # Get ChromaDB client and collection
    collection = get_or_create_collection(
        context.deps.chroma_client,
        context.deps.collection_name,
        embedding_model_name=context.deps.embedding_model
    )
    
    # Query the collection
    query_results = query_collection(
        collection,
        search_query,
        n_results=n_results
    )
    
    # Format the results as context
    return format_results_as_context(query_results)

async def run_rag_agent(
    question: str,
    collection_name: str = "docs",
    db_directory: str = "./chroma_db",
    embedding_model: str = "all-MiniLM-L6-v2",
    model_choice: str = "gpt-4.1-mini",
    api_key: str = None,
    n_results: int = 5
) -> str:
    """Run the RAG agent to answer a question about Pydantic AI.
    
    Args:
        question: The question to answer.
        collection_name: Name of the ChromaDB collection to use.
        db_directory: Directory where ChromaDB data is stored.
        embedding_model: Name of the embedding model to use.
        model_choice: The model to use for the agent.
        api_key: The OpenAI API key.
        n_results: Number of results to return from the retrieval.
        
    Returns:
        The agent's response.
    """
    if not api_key:
        raise ValueError("API key must be provided.")

    # Create dependencies
    deps = RAGDeps(
        chroma_client=get_chroma_client(db_directory),
        collection_name=collection_name,
        embedding_model=embedding_model,
        model_choice=model_choice,
        api_key=api_key
    )
    
    # Run the agent
    result = await agent.run(question, deps=deps, model=deps.model_choice)
    
    return result.data