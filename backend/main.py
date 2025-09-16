"""
FastAPI backend for Healthcare RAG Q&A.

Exposes:
- GET /           → health check {"status": "ok"}
- POST /ask       → {"question": str} → {"answer": str}

Loads on startup:
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Vector store: Chroma persistent DB from ./chroma_db (collection: pubmed_abstracts)
- LLM: google/flan-t5-base via Hugging Face pipeline
- Chain: LangChain RetrievalQA (chain_type="stuff")
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Prefer modern provider packages with graceful fallbacks
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
DEFAULT_COLLECTION = "pubmed_abstracts"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "google/flan-t5-base"


def _device_for_pipeline() -> int:
    """Return pipeline device id: 0 for CUDA, -1 for CPU."""
    return 0 if torch.cuda.is_available() else -1


def _build_rag_chain() -> RetrievalQA:
    # Embeddings + Vector Store
    embedding_fn = HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)
    vectordb = Chroma(
        collection_name=DEFAULT_COLLECTION,
        persist_directory=DEFAULT_DB_PATH,
        embedding_function=embedding_fn,
    )

    # LLM pipeline (Flan‑T5)
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(DEFAULT_LLM_MODEL)
    generate_pipe = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=_device_for_pipeline(),
        max_new_tokens=256,
        temperature=0.2,
    )
    llm = HuggingFacePipeline(pipeline=generate_pipe)

    # RetrievalQA chain — 'stuff' inserts chunks directly into the prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
    )
    return qa_chain


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Initialize heavy resources once
    app.state.qa_chain = _build_rag_chain()
    yield
    # Optional: add teardown if needed


app = FastAPI(title="Healthcare RAG Backend", lifespan=lifespan)

# Allow local Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


@app.get("/")
async def healthcheck():
    return {"status": "ok"}


@app.post("/ask")
async def ask(req: AskRequest):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question must be a non-empty string")

    try:
        answer = app.state.qa_chain.run(q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference failed: {e}")

    return {"answer": answer}


# For local run: uvicorn backend.main:app --reload --port 8000
