import streamlit as st
import asyncio
import platform

# Set Windows Proactor event loop policy for Playwright compatibility
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from insert_docs import insert_docs

# Lazy import message parts
def get_message_parts():
    from pydantic_ai.messages import (
        ModelMessage,
        ModelRequest,
        ModelResponse,
        SystemPromptPart,
        UserPromptPart,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
        RetryPromptPart,
        ModelMessagesTypeAdapter
    )
    return {
        "ModelMessage": ModelMessage,
        "ModelRequest": ModelRequest,
        "ModelResponse": ModelResponse,
        "SystemPromptPart": SystemPromptPart,
        "UserPromptPart": UserPromptPart,
        "TextPart": TextPart,
        "ToolCallPart": ToolCallPart,
        "ToolReturnPart": ToolReturnPart,
        "RetryPromptPart": RetryPromptPart,
        "ModelMessagesTypeAdapter": ModelMessagesTypeAdapter
    }

# Lazy import rag_agent
def get_rag_agent():
    from rag_agent import agent, RAGDeps
    return agent, RAGDeps

# Lazy import utils
def get_utils():
    from utils import get_chroma_client
    return get_chroma_client

MODEL_CHOICE = 'gpt-4.1-mini'

async def get_agent_deps(api_key):
    get_chroma_client = get_utils()
    _, RAGDeps = get_rag_agent()
    return RAGDeps(
        chroma_client=get_chroma_client("./chroma_db"),
        collection_name="docs",
        embedding_model="all-MiniLM-L6-v2",
        model_choice=MODEL_CHOICE,
        api_key=api_key
    )

def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    """
    if part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)

async def run_agent_with_streaming(user_input):
    agent, _ = get_rag_agent()
    async with agent.run_stream(
        user_input,
        deps=st.session_state.agent_deps,
        message_history=st.session_state.messages
    ) as result:
        async for message in result.stream_text(delta=True):
            yield message

    # Add new messages to chat history
    st.session_state.messages.extend(result.new_messages())

async def main():
    st.title("ChromaDB Crawl4AI RAG AI Agent")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "website_url" not in st.session_state:
        st.session_state.website_url = ""
    if "agent_deps" not in st.session_state:
        st.session_state.agent_deps = None

    # Sidebar for configuration
    st.sidebar.header("Configuration")

    # API key input
    api_key = st.sidebar.text_input("Enter API Key", type="password", value=st.session_state.api_key)

    # Website URL input
    website_url = st.sidebar.text_input("Enter Website URL", value=st.session_state.website_url)

    # Update session state
    if api_key and api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
        try:
            st.session_state.agent_deps = await get_agent_deps(api_key)
        except Exception as e:
            st.sidebar.error(f"Failed to initialize agent: {str(e)}")
            st.session_state.agent_deps = None

    if website_url:
        st.session_state.website_url = website_url

    # Button to trigger document insertion
    if st.sidebar.button("Crawl and Insert Documents"):
        if not website_url:
            st.sidebar.error("Please enter a website URL.")
        else:
            with st.spinner("Crawling and inserting documents..."):
                try:
                    result = await insert_docs(
                        url=website_url,
                        collection="docs",
                        db_dir="./chroma_db",
                        embedding_model="all-MiniLM-L6-v2",
                        chunk_size=1000,
                        max_depth=3,
                        max_concurrent=10,
                        batch_size=100
                    )
                    st.sidebar.success(f"Successfully inserted {result['chunk_count']} chunks from {website_url}")
                except Exception as e:
                    st.sidebar.error(f"Error inserting documents: {str(e)}")

    # Check if API key and agent_deps are ready
    if not st.session_state.api_key:
        st.warning("Please enter an API key to continue.")
        return
    if not st.session_state.agent_deps:
        st.warning("Agent initialization failed. Please check your API key.")
        return

    # Load message parts for rendering
    message_parts = get_message_parts()

    # Display all messages from the conversation
    for msg in st.session_state.messages:
        if isinstance(msg, message_parts["ModelRequest"]) or isinstance(msg, message_parts["ModelResponse"]):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user
    user_input = st.chat_input("What do you want to know?")

    if user_input:
        # Display user prompt
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display assistant's streaming response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            generator = run_agent_with_streaming(user_input)
            async for message in generator:
                full_response += message
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

# Run the main coroutine using Streamlit's async context
if __name__ == "__main__":
    import streamlit.runtime.scriptrunner as scriptrunner
    scriptrunner.exec_async_in_loop(main())