"""
rag_pipeline.py
----------------
Wire up ChromaDB with a lightweight Hugging Face model (Flan‑T5) using
LangChain to form a Retrieval‑Augmented Generation (RAG) QA system.

- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Vector store: Chroma persistent DB (default path: ./chroma_db)
- LLM: google/flan-t5-base wrapped by LangChain's HuggingFacePipeline

Notes
- The 'stuff' chain type inserts retrieved chunks directly into the prompt.
- `k=4` retrieves the top 4 most similar chunks; tweak via CLI.
- You can swap to a larger Flan‑T5 variant if hardware allows.
"""

from __future__ import annotations

import argparse
from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Prefer modern provider packages; fall back to community modules if needed.
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


def device_for_pipeline() -> int:
    """Return pipeline device id: 0 for CUDA, -1 for CPU."""
    return 0 if torch.cuda.is_available() else -1


def build_embedding_fn(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name)


def build_vectorstore(
    db_path: str, collection: str, embedding_fn: HuggingFaceEmbeddings
) -> Chroma:
    return Chroma(
        collection_name=collection,
        persist_directory=db_path,
        embedding_function=embedding_fn,
    )


def build_llm_pipeline(model_name: str) -> HuggingFacePipeline:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device_for_pipeline(),
        max_new_tokens=256,
        temperature=0.2,
    )
    return HuggingFacePipeline(pipeline=pipe)


def build_rag_chain(db_path: str, collection: str, k: int, model_name: str) -> RetrievalQA:
    embedding_fn = build_embedding_fn(DEFAULT_EMBEDDING_MODEL)
    vectordb = build_vectorstore(db_path, collection, embedding_fn)
    llm = build_llm_pipeline(model_name)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Inserts retrieved chunks into the prompt template
        retriever=vectordb.as_retriever(search_kwargs={"k": k}),
        return_source_documents=False,
    )
    return qa_chain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a RAG QA query over a ChromaDB collection")
    parser.add_argument("--db-path", type=str, default=DEFAULT_DB_PATH, help="Path to Chroma persistent DB")
    parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION,
        help="Chroma collection name (default: pubmed_abstracts)",
    )
    parser.add_argument("--k", type=int, default=4, help="Top-k documents to retrieve (default: 4)")
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help="HF model for generation (default: google/flan-t5-base)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What are the benefits of metformin in diabetes treatment?",
        help="Question to ask the RAG system",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qa = build_rag_chain(args.db_path, args.collection, args.k, args.model_name)
    print(f"Running RAG query (k={args.k}) against collection='{args.collection}' at '{args.db_path}'...")
    answer = qa.run(args.query)
    print("\nQuery:", args.query)
    print("Answer:", answer)


if __name__ == "__main__":
    main()
