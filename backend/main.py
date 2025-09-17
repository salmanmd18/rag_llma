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
import os
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import chromadb
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel


DEFAULT_DB_PATH = "chroma_db"
DEFAULT_COLLECTION = "pubmed_abstracts"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "google/flan-t5-base"


def _device_for_pipeline() -> int:
    """Return pipeline device id: 0 for CUDA, -1 for CPU."""
    return 0 if torch.cuda.is_available() else -1


class SimpleChromaRetriever:
    """Minimal retriever that queries Chroma directly via embeddings."""

    def __init__(self, collection: chromadb.api.models.Collection.Collection, embedder: SentenceTransformer, k: int = 4):
        self.collection = collection
        self.embedder = embedder
        self.k = k

    def get_relevant_documents(self, query: str):
        vec = self.embedder.encode([query]).tolist()
        res = self.collection.query(query_embeddings=vec, n_results=self.k, include=["documents", "metadatas"])
        docs = []
        for doc_text, meta in zip(res.get("documents", [[]])[0], res.get("metadatas", [[]])[0]):
            docs.append(Document(page_content=doc_text, metadata=meta or {}))
        return docs


def _build_rag_chain() -> RetrievalQA:
    # Embedding model + Chroma client/collection
    embedder = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=DEFAULT_DB_PATH)
    collection = client.get_or_create_collection(name=DEFAULT_COLLECTION)

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
    # Wrap pipeline in a lightweight shim that satisfies LangChain's LLM interface
    # Using Runnable or BaseLanguageModel wrappers would be ideal, but a simple adapter works.
    class PipelineLLM(BaseLanguageModel):
        def __init__(self, pipe):
            super().__init__()
            self.pipe = pipe

        def _call(self, prompt: str, stop=None, run_manager=None):
            out = self.pipe(prompt)[0]["generated_text"]
            return out

        @property
        def _llm_type(self) -> str:
            return "hf_pipeline"

    llm = PipelineLLM(generate_pipe)

    # RetrievalQA chain — 'stuff' inserts chunks directly into the prompt
    retriever = SimpleChromaRetriever(collection, embedder, k=4)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa_chain


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Initialize heavy resources once
    app.state.qa_chain = _build_rag_chain()
    yield
    # Optional: add teardown if needed


app = FastAPI(title="Healthcare RAG Backend", lifespan=lifespan)

# CORS: read allowed origins from env (comma‑separated). Defaults include localhost.
_cors_env = os.getenv(
    "BACKEND_CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
)
_origins = [o.strip() for o in _cors_env.split(",") if o.strip()]
allow_all = "*" in _origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all else _origins,
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
