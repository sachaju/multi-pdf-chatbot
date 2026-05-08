# Multi-PDF Chatbot

An AI-powered conversational tool that allows users to chat 
with multiple PDF documents simultaneously using LangChain and LLaMA 3.

## What it does
- Upload multiple PDFs at once
- Asks questions across all documents naturally
- Remembers conversation history
- Returns accurate answers based on document content

## Tech Stack
- Python
- LangChain
- FAISS vector store
- LLaMA 3 (Ollama)
- HuggingFace Embeddings
- Streamlit

## How to run
1. Clone this repository
2. Install Ollama and pull LLaMA 3
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `streamlit run app.py`