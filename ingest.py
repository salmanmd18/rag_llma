"""
Healthcare RAG Q&A - Ingest and persist chunks into ChromaDB.

Reads .txt and .md files from ./data, splits into overlapping chunks,
embeds with Sentence-Transformers, and persists vectors to ./chroma_db.

Usage:
  python ingest.py --chunk-size 1000 --chunk-overlap 100 \
    --persist-dir chroma_db --collection pubmed_abstracts
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
from typing import Iterable, List

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    # Preferred, deprecation-safe import
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
except Exception:  # pragma: no cover
    # Fallback to older location if extra isn't installed
    from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings  # type: ignore
    print(
        "Warning: Using deprecated HuggingFaceEmbeddings from langchain_community. "
        "Install 'langchain-huggingface' and switch to the new import to silence this."
    )
from langchain_chroma import Chroma
from chromadb.config import Settings
import shutil


DEFAULT_DATA_DIR = Path("data")
DEFAULT_PERSIST_DIR = Path("chroma_db")

# Force Chroma to use DuckDB+Parquet (works on environments with old sqlite)
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet",
    anonymized_telemetry=False,
)


def find_source_files(data_dir: Path) -> List[Path]:
    patterns = ["**/*.txt", "**/*.md"]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(sorted(data_dir.glob(pattern)))
    seen = set()
    unique_files = []
    for f in files:
        if f not in seen and f.is_file():
            unique_files.append(f)
            seen.add(f)
    return unique_files


def load_text_documents(paths: Iterable[Path]) -> List[Document]:
    documents: List[Document] = []
    for p in paths:
        try:
            docs = TextLoader(str(p), encoding="utf-8").load()
        except UnicodeDecodeError:
            docs = TextLoader(str(p), encoding="latin-1").load()
        for d in docs:
            d.metadata.setdefault("source", str(p))
        documents.extend(docs)
    return documents


def split_documents(
    documents: List[Document], *, chunk_size: int, chunk_overlap: int
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def build_embeddings(model_name: str, device: str | None) -> HuggingFaceEmbeddings:
    model_kwargs = {"device": device} if device else {}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
    )


def persist_to_chroma(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
    persist_dir: Path,
    collection_name: str,
    reset: bool = False,
) -> Chroma:
    if reset and persist_dir.exists():
        shutil.rmtree(persist_dir, ignore_errors=True)
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Initialize (creates or loads) the collection
    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,  # alias: embedding
        persist_directory=str(persist_dir),
        client_settings=CHROMA_SETTINGS,
    )

    # Build stable IDs per chunk: source::start_index
    ids = [
        f"{doc.metadata.get('source','unknown')}::{doc.metadata.get('start_index', 0)}"
        for doc in chunks
    ]

    # Avoid duplicates on re-ingest by checking existing ids
    try:
        existing = set(vectordb._collection.get(include=[])  # type: ignore[attr-defined]
                       .get("ids", []))
    except Exception:
        existing = set()

    to_add_docs: List[Document] = []
    to_add_ids: List[str] = []
    for d, i in zip(chunks, ids):
        if i not in existing:
            to_add_docs.append(d)
            to_add_ids.append(i)

    if to_add_docs:
        vectordb.add_documents(to_add_docs, ids=to_add_ids)

    return vectordb


def summarize(docs: List[Document], chunks: List[Document], db: Chroma) -> None:
    sources = [d.metadata.get("source") for d in docs]
    unique_sources = sorted({s for s in sources if s})
    print(
        f"Loaded {len(docs)} document(s) from {len(unique_sources)} file(s).\n"
        f"Split into {len(chunks)} chunk(s)."
    )
    try:
        count = db._collection.count()  # type: ignore[attr-defined]
        print(f"Chroma collection contains {count} vector(s).")
    except Exception:
        print("Persisted to Chroma collection (count unavailable).")
    if chunks:
        sample = chunks[0]
        text_preview = sample.page_content[:300].replace("\n", " ")
        print("\nSample chunk:")
        print(f"- source: {sample.metadata.get('source')}")
        if "start_index" in sample.metadata:
            start = sample.metadata["start_index"]
            print(f"- span: {start}..{start + len(sample.page_content)}")
        print(f"- length: {len(sample.page_content)} chars")
        print(f"- preview: {text_preview!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest and chunk text documents into ChromaDB")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing .txt/.md files (default: ./data)",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=DEFAULT_PERSIST_DIR,
        help="ChromaDB persistence directory (default: ./chroma_db)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="pubmed_abstracts",
        help="Chroma collection name (default: pubmed_abstracts)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size in characters (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Overlap between chunks in characters (default: 100)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformers model name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "cuda:0", "mps"],
        help="Torch device to use (default: auto)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the persist directory before ingest to avoid duplicates",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.persist_dir.mkdir(parents=True, exist_ok=True)

    files = find_source_files(args.data_dir)
    if not files:
        print(
            f"No .txt or .md files found in '{args.data_dir}'. "
            "Add files to the data directory and re-run."
        )
        return

    docs = load_text_documents(files)
    chunks = split_documents(
        docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    embeddings = build_embeddings(args.embedding_model, args.device)
    db = persist_to_chroma(
        chunks, embeddings, args.persist_dir, args.collection, reset=args.reset
    )
    summarize(docs, chunks, db)
    # Save chunks to a pickle so other scripts (e.g., build_vector_db.py) can reuse them
    try:
        with open("chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)
        print("\nSaved chunk list to chunks.pkl")
    except Exception as e:
        print(f"\nWarning: Failed to save chunks.pkl: {e}")
    print(f"\nPersisted ChromaDB at: {args.persist_dir}")


if __name__ == "__main__":
    main()
