import hashlib
import io
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from htmlTemplates import css, bot_template, user_template


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

CACHE_DIR = Path("pdf_cache")
CACHE_DIR.mkdir(exist_ok=True)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Model used to answer questions
ANSWER_MODEL = "llama3"

# Model used to write the document summaries.
# On a CPU-only machine, a small model is much faster:
#   ollama pull llama3.2:3b
SUMMARY_MODEL = "llama3.2:3b"

# Context window given to Ollama (prevents long prompts from being truncated)
NUM_CTX = 8192


# --------------------------------------------------
# MODELS (loaded once and reused)
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource
def get_llm(model_name):
    return ChatOllama(model=model_name, num_ctx=NUM_CTX)


# --------------------------------------------------
# PDF: identifier, text extraction, chunking
# --------------------------------------------------

def file_key(name, data):
    """Identifier of a PDF: a hash of its name and its content."""
    hasher = hashlib.sha256()
    hasher.update(name.encode("utf-8"))
    hasher.update(data)
    return hasher.hexdigest()


def get_pdf_pages(name, data):
    """One Document per page, tagged with the filename and page number."""
    reader = PdfReader(io.BytesIO(data))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append(
                Document(
                    page_content=text,
                    metadata={"source": name, "page": page_number}
                )
            )

    return pages


def get_text_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)


# --------------------------------------------------
# FAISS INDEX: one index per PDF, then merged
# --------------------------------------------------

def load_or_build_index(name, data, key, embeddings):
    """Load the PDF's saved index if it exists, otherwise build and save it."""
    index_path = CACHE_DIR / key / "faiss_index"

    if (index_path / "index.faiss").exists() and (
        index_path / "index.pkl"
    ).exists():
        return FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True
        )

    pages = get_pdf_pages(name, data)

    if not pages:
        return None

    chunks = get_text_chunks(pages)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    index_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_path))

    return vectorstore


def build_vectorstore(files):
    """files = {pdf_name: content_as_bytes}.
    Returns the merged index and a {pdf_name: file_key} dictionary."""
    embeddings = get_embeddings()
    merged = None
    keys = {}

    for name, data in files.items():
        key = file_key(name, data)
        vectorstore = load_or_build_index(name, data, key, embeddings)

        if vectorstore is None:
            continue

        keys[name] = key

        if merged is None:
            merged = vectorstore
        else:
            merged.merge_from(vectorstore)

    return merged, keys


# --------------------------------------------------
# SUMMARIES (generated only when needed, then cached)
# --------------------------------------------------

def final_prompt(pdf_name, material):
    return f"""
Create a structured, informative summary of this report.
Write the summary in the same language as the document.

Filename: {pdf_name}

Include the purpose, main topics, findings, evidence,
opportunities, challenges, limitations and conclusions
where supported by the supplied material.

The middle of a long document may be omitted from the material.
Do not guess what the omitted part contains.
Do not invent facts, statistics or page references.
Use only the material below.

Material:
{material}

Final summary:
"""


QUICK_HEAD_CHARS = 15000   # characters kept from the start of the document
QUICK_TAIL_CHARS = 6000    # characters kept from the end of the document


def summarize_pdf(pdf_name, pages, llm):
    full_text = "\n\n".join(
        f"[Page {doc.metadata['page']}]\n{doc.page_content}"
        for doc in pages
    )

    if not full_text.strip():
        return ""

    # Keep the beginning and the end, skip the middle (one fast LLM call).
    if len(full_text) > QUICK_HEAD_CHARS + QUICK_TAIL_CHARS:
        material = (
            full_text[:QUICK_HEAD_CHARS]
            + "\n\n[... middle of the document omitted ...]\n\n"
            + full_text[-QUICK_TAIL_CHARS:]
        )
    else:
        material = full_text

    start = time.time()
    result = llm.invoke(final_prompt(pdf_name, material)).content
    print(
        f"[{pdf_name}] summary finished in {time.time() - start:.1f}s",
        flush=True
    )

    return result


def get_summary(name):
    """Summary of one PDF: read from the cache, or generated and saved."""
    key = st.session_state.pdf_keys[name]
    data = st.session_state.pdf_files[name]
    path = CACHE_DIR / key / "summary.txt"

    if path.exists():
        return path.read_text(encoding="utf-8")

    pages = get_pdf_pages(name, data)

    with st.spinner(
        f"Summarising {name} (first time only, this can take a while)..."
    ):
        summary = summarize_pdf(name, pages, get_llm(SUMMARY_MODEL))

    if summary.strip():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(summary, encoding="utf-8")

    return summary


# --------------------------------------------------
# PROMPT AND ANSWER HELPERS
# --------------------------------------------------

def format_docs(docs):
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')} | "
        f"Page: {doc.metadata.get('page', '?')}\n"
        f"Content: {doc.page_content}"
        for doc in docs
    )


def format_history(max_messages=6):
    """The last few messages, so the model can follow the conversation."""
    history = st.session_state.chat_history[-max_messages:]

    if not history:
        return "(none)"

    lines = []
    for role, content in history:
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content[:600]}")

    return "\n".join(lines)


def qa_prompt(context, question):
    return f"""
You answer questions about uploaded PDF documents.
Answer in the same language as the question.

Use the retrieved passages to answer the question.
Base factual claims on those passages.
Preserve qualifications and uncertainty.
Do not invent facts, statistics, sources or page numbers.
If the passages are insufficient, say so.
Identify the source using its filename and page number.

Recent conversation:
{format_history()}

Retrieved passages:
{context}

Question:
{question}

Answer:
"""


# --------------------------------------------------
# QUESTION ROUTING
# --------------------------------------------------

OVERVIEW_PHRASES = [
    "what are these documents about",
    "what are the documents about",
    "what are these pdfs about",
    "what is this document about",
    "overview of the documents",
    "overview of these documents",
    "summarize each", "summarise each",
    "summarize all", "summarise all",
    "summary of each", "summaries of the",
    "de quoi parle", "de quoi parlent",
    "résume chaque", "resume chaque",
    "résumer chaque", "résumé de chaque",
    "résumés des", "résumez",
    "vue d'ensemble",
    "présente les documents", "presente les documents",
]

COMPARISON_WORDS = [
    "compar", "différence", "difference", "versus", " vs ",
    "contrast", "similarit", "similaire", "ressemblance",
]


def is_overview_question(question):
    q = question.lower().strip()
    return any(phrase in q for phrase in OVERVIEW_PHRASES)


def is_comparison_question(question):
    q = f" {question.lower().strip()} "
    return any(word in q for word in COMPARISON_WORDS)


# --------------------------------------------------
# THREE TYPES OF ANSWER
# --------------------------------------------------

def answer_overview(question):
    """General questions: answer from the summary of each PDF."""
    summaries = {}

    for name in st.session_state.pdf_files:
        summary = get_summary(name)
        if summary.strip():
            summaries[name] = summary

    summary_context = "\n\n".join(
        f"Filename: {name}\nSummary:\n{summary}"
        for name, summary in summaries.items()
    )

    prompt = f"""
Answer the user's general question about the uploaded documents.
Answer in the same language as the question.

Use only the summaries below.
Write one paragraph per document, each starting with its filename.
In each paragraph, give the main topic, the key findings and the main conclusion.
Do not merge the documents into one description.
Do not invent facts, statistics, citations or page numbers.
If the summaries do not contain enough information, say so.

Recent conversation:
{format_history()}

Document summaries:
{summary_context}

Question:
{question}

Answer:
"""

    answer = get_llm(ANSWER_MODEL).invoke(prompt).content
    return answer, summaries


def answer_comparison(question, k_per_doc=3):
    """Comparisons: retrieve the best passages from EACH PDF."""
    vectorstore = st.session_state.vectorstore
    docs = []

    for name in st.session_state.pdf_files:
        docs += vectorstore.similarity_search(
            question,
            k=k_per_doc,
            filter={"source": name},
            fetch_k=300
        )

    prompt = qa_prompt(format_docs(docs), question)
    answer = get_llm(ANSWER_MODEL).invoke(prompt).content
    return answer, docs


def answer_specific(question, k=4):
    """Specific questions: classic similarity search over all PDFs."""
    docs = st.session_state.vectorstore.similarity_search(question, k=k)
    prompt = qa_prompt(format_docs(docs), question)
    answer = get_llm(ANSWER_MODEL).invoke(prompt).content
    return answer, docs


# --------------------------------------------------
# HANDLING A QUESTION
# --------------------------------------------------

def handle_userinput(question):
    if st.session_state.vectorstore is None:
        st.warning("Please upload PDFs and click Process first.")
        return

    with st.spinner("Thinking..."):
        if is_overview_question(question):
            mode = "overview"
            answer, details = answer_overview(question)
        elif (
            is_comparison_question(question)
            and len(st.session_state.pdf_files) > 1
        ):
            mode = "comparison"
            answer, details = answer_comparison(question)
        else:
            mode = "specific"
            answer, details = answer_specific(question)

    st.session_state.chat_history.extend([
        ("user", question),
        ("assistant", answer)
    ])
    st.session_state.last_mode = mode
    st.session_state.last_details = details


def display_chat():
    for role, content in st.session_state.chat_history:
        template = user_template if role == "user" else bot_template
        st.write(
            template.replace("{{MSG}}", content),
            unsafe_allow_html=True
        )

    details = st.session_state.last_details

    if not details:
        return

    if st.session_state.last_mode == "overview":
        with st.expander("Document summaries used"):
            for name, summary in details.items():
                st.markdown(f"**{name}**")
                st.write(summary)
                st.divider()
    else:
        with st.expander("Sources retrieved"):
            seen = set()

            for doc in details:
                source = doc.metadata.get("source", "Unknown PDF")
                page = doc.metadata.get("page", "?")

                if (source, page) not in seen:
                    st.markdown(f"**{source} — page {page}**")
                    st.caption(doc.page_content[:700])
                    st.divider()
                    seen.add((source, page))


# --------------------------------------------------
# STREAMLIT APP
# --------------------------------------------------

def init_state():
    defaults = {
        "vectorstore": None,
        "pdf_files": {},      # pdf name -> content (bytes)
        "pdf_keys": {},       # pdf name -> file key
        "chat_history": [],
        "last_mode": None,
        "last_details": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main():
    load_dotenv()

    st.set_page_config(
        page_title="Chat with multiple PDFs",
        page_icon=":books:"
    )

    st.write(css, unsafe_allow_html=True)
    init_state()

    st.header("Chat with multiple PDFs :books:")

    with st.form("question_form"):
        question = st.text_input("Ask a question about your documents:")
        submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        handle_userinput(question.strip())

    display_chat()

    with st.sidebar:
        st.subheader("Your documents")

        pdf_docs = st.file_uploader(
            "Upload your PDFs and click Process",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                try:
                    files = {pdf.name: pdf.getvalue() for pdf in pdf_docs}
                    start = time.time()

                    with st.spinner("Processing PDFs..."):
                        vectorstore, keys = build_vectorstore(files)

                    if vectorstore is None:
                        st.error(
                            "No extractable text found in the PDFs. "
                            "Scanned PDFs may require OCR."
                        )
                    else:
                        st.session_state.vectorstore = vectorstore
                        st.session_state.pdf_files = {
                            name: files[name] for name in keys
                        }
                        st.session_state.pdf_keys = keys
                        st.session_state.chat_history = []
                        st.session_state.last_mode = None
                        st.session_state.last_details = None

                        st.success(
                            f"Processing complete in "
                            f"{time.time() - start:.1f}s."
                        )
                        st.write(f"PDFs indexed: {len(keys)}")
                        st.write(
                            f"Chunks in index: {vectorstore.index.ntotal}"
                        )
                        st.caption(
                            "Summaries are generated the first time you ask "
                            "a general question, then cached."
                        )

                except Exception as error:
                    st.error("Processing failed.")
                    st.exception(error)


if __name__ == "__main__":
    main()