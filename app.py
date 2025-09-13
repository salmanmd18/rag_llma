"""
app.py
-------
Simple Streamlit UI for the Healthcare RAG Q&A system.

Loads a persistent ChromaDB collection and a lightweight Flan‑T5 model
wrapped via LangChain's HuggingFacePipeline into a RetrievalQA chain.
"""

from __future__ import annotations

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Prefer modern provider packages; fall back to community modules if needed
try:  # Embeddings
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
except Exception:  # pragma: no cover
    from langchain_community.embeddings import (  # type: ignore
        HuggingFaceEmbeddings,  # deprecated location
    )

try:  # Chroma vector store
    from langchain_chroma import Chroma  # type: ignore
except Exception:  # pragma: no cover
    from langchain_community.vectorstores import Chroma  # type: ignore

try:  # HF pipeline LLM wrapper
    from langchain_huggingface import HuggingFacePipeline  # type: ignore
except Exception:  # pragma: no cover
    from langchain_community.llms import HuggingFacePipeline  # type: ignore

from langchain.chains import RetrievalQA


DEFAULT_DB_PATH = "chroma_db"
# Use pubmed_abstracts to match the ingestion/build scripts
DEFAULT_COLLECTION = "pubmed_abstracts"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "google/flan-t5-base"


@st.cache_resource(show_spinner=False)
def load_pipeline():
    # Embeddings + Vector Store
    embedding_fn = HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)
    vectordb = Chroma(
        collection_name=DEFAULT_COLLECTION,
        persist_directory=DEFAULT_DB_PATH,
        embedding_function=embedding_fn,
    )

    # Hugging Face model (Flan‑T5)
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(DEFAULT_LLM_MODEL)

    generate_pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.2,
    )

    llm = HuggingFacePipeline(pipeline=generate_pipe)

    # LangChain QA Chain — 'stuff' places retrieved chunks directly into prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
    )
    return qa_chain


st.set_page_config(page_title="Healthcare RAG Q&A", page_icon="🩺", layout="wide")
st.title("🩺 Healthcare RAG Q&A")
st.write("Ask questions about PubMed abstracts and WHO fact sheets loaded into this system.")

qa_chain = load_pipeline()

user_query = st.text_input("Enter your question:")

if st.button("Submit") and user_query:
    with st.spinner("Retrieving and generating answer..."):
        result = qa_chain.run(user_query)
    st.markdown("### ✅ Answer")
    st.write(result)

st.markdown("---")
st.caption("Powered by LangChain, Hugging Face, and ChromaDB")
