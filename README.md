# WebsiteGPT - RAG-based Web Crawler and Q&A System

## Overview
WebsiteGPT is a sophisticated web crawling and question-answering system that combines web crawling capabilities with Retrieval-Augmented Generation (RAG) to create an intelligent Q&A system. The system can crawl websites, process their content, and answer questions based on the crawled information using advanced language models. The best part is it generates llms.txt and full-llm.txt files which you can download and use with other GPT models of your own.

## Key Features
- Web crawling with configurable depth and concurrency
- Document processing and chunking
- Vector storage using ChromaDB
- RAG-based question answering
- Support for multiple LLM backends (OpenAI and Llama)
- Interactive web interface using Streamlit
- Real-time streaming responses

## System Architecture

### 1. Web Crawler Component
- Crawls websites up to a specified depth
- Extracts and processes content from web pages
- Handles concurrent requests for efficient crawling
- Supports configurable batch processing

### 2. Document Processing
- Chunks documents into manageable pieces
- Generates embeddings for semantic search
- Stores processed documents in ChromaDB
- Maintains document metadata and relationships

### 3. RAG Agent
- Implements Retrieval-Augmented Generation
- Uses vector similarity search for relevant context
- Integrates with multiple LLM backends
- Provides streaming responses for better UX

### 4. Web Interface
- Built with Streamlit for easy interaction
- Real-time document crawling and processing
- Interactive Q&A interface
- Streaming response display

## Key Concepts Explained

### 1. RAG (Retrieval-Augmented Generation)
RAG is a technique that combines retrieval-based and generation-based approaches:
- **Retrieval**: Searches for relevant information from a knowledge base
- **Augmentation**: Enhances the LLM's response with retrieved information
- **Generation**: Produces contextually relevant answers

### 2. Vector Embeddings
- Converts text into numerical vectors
- Enables semantic search capabilities
- Uses the `all-MiniLM-L6-v2` model by default
- Allows for similarity-based document retrieval

### 3. ChromaDB
- Vector database for storing embeddings
- Enables efficient similarity search
- Maintains document collections
- Supports persistent storage

### 4. Streaming Responses
- Provides real-time feedback to users
- Improves user experience
- Allows for progressive response generation
- Reduces perceived latency

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/ArnavAgarwal-Mr-AR/WebsiteGPT
cd WebsiteGPT
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install Playwright and its dependencies:
```bash
playwright install
crawl4AI-setup
```
This step is crucial as the system uses Playwright for web crawling capabilities.

4. Set up your OpenAI API key

5. Run the application:
```bash
streamlit run streamlit_rag_app.py
```

Note: If you're using Windows, the system will automatically configure the appropriate event loop policy for Playwright compatibility. For other operating systems, no additional configuration is needed.

## Usage

1. **Crawling Websites**:
   - Enter the target website URL
   - Configure crawl depth and other parameters
   - Click "Crawl and Insert Documents"

2. **Asking Questions**:
   - Enter your question in the chat interface
   - Select the preferred model (OpenAI/Llama)
   - View the streaming response

## Configuration Options

- **Crawl Settings**:
  - Depth: How deep to crawl the website
  - Chunk size: Size of document chunks
  - Max concurrent requests
  - Batch size for processing

- **Model Settings**:
  - Model choice (OpenAI/Llama)
  - Embedding model
  - API keys

- **Database Settings**:
  - Collection name
  - Database directory
  - Embedding model

## Technical Details

### Dependencies
- Streamlit: Web interface
- ChromaDB: Vector database
- OpenAI: Language model integration
- Pydantic: Data validation
- AsyncIO: Asynchronous operations

### File Structure
- `streamlit_rag_app.py`: Main application
- `rag_agent.py`: RAG implementation
- `insert_docs.py`: Document processing
- `utils.py`: Utility functions
- `chroma_db/`: Vector database storage

## Best Practices

1. **API Key Management**:
   - Never commit API keys to version control
   - Use environment variables or secure storage
   - Rotate keys regularly

2. **Crawling Considerations**:
   - Respect robots.txt
   - Implement rate limiting
   - Handle errors gracefully
   - Monitor resource usage

3. **Performance Optimization**:
   - Use appropriate chunk sizes
   - Configure concurrent requests
   - Monitor memory usage
   - Implement caching where appropriate

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
[Specify your license here]

## Acknowledgments
- OpenAI for the language models
- ChromaDB team for the vector database
- Streamlit for the web interface framework
