# Multi-PDF Chatbot

A local, AI-powered chatbot that lets you chat with several PDF documents at once,
using LangChain, FAISS and LLaMA 3 (via Ollama). No API key is needed and your
documents never leave your machine.

## What it does
- Upload several PDFs and ask questions about them
- Handles three kinds of questions automatically (see below)
- Shows the filename and page of every passage used to answer
- Keeps the recent conversation in context
- Caches the index of each PDF, so a document is only processed once

## How it works
1. Text is extracted page by page (PyPDF2), keeping the filename and page number.
2. Pages are split into chunks (1000 characters, 200 overlap), embedded with
   `all-MiniLM-L6-v2` and stored in a FAISS index (one index per PDF, then merged).
3. Each question is routed to the right strategy.
4. LLaMA 3 writes the answer from the retrieved passages.

| Question type | Example | What the app does |
|---|---|---|
| Specific | "How does the Ecosystem Lens differ from the Structural Elements Lens?" | Similarity search across all PDFs |
| Comparison | "Compare the two documents on productivity" | Retrieves the best passages from **each** PDF, so every document is represented |
| Overview | "What are these documents about?" | Answers from a stored summary of each PDF |

Why the routing? A plain similarity search works well for precise questions, but it
cannot answer "what is this document about?" (no single passage contains the answer)
and it tends to pull passages from only one PDF when you ask for a comparison.

## Application Screenshots

### Main Interface

![Main Interface](screenshots/home%20page.png)

### Document Processing

![Document Processing](screenshots/Document%20Processing.png)

### Chatbot

![Chatbot](screenshots/Conversational%20Interface.png)

### Sources

![Retrieved Sources](screenshots/sources%20retrieved.png)

## Tech Stack
- Python
- LangChain
- FAISS vector store
- LLaMA 3 and LLaMA 3.2 3B (Ollama)
- HuggingFace Embeddings
- Streamlit

## How to run
1. Clone this repository
2. Install [Ollama](https://ollama.com) and pull the models:
   `ollama pull llama3` and `ollama pull llama3.2:3b`
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `streamlit run app.py`
5. Upload your PDFs in the sidebar and click **Process**

The first processing computes the embeddings and takes a little while (about a minute
for two 36-page PDFs on a CPU-only laptop). After that, the same files load from the
`pdf_cache` folder almost instantly. Summaries are generated the first time you ask an
overview question, then cached.

## Limitations
- Scanned PDFs without a text layer need OCR first
- Summaries are based on the beginning and end of each document
- Questions work best when they can be understood on their own, since the search uses
  the question as typed
- On a CPU, local models take time to answer