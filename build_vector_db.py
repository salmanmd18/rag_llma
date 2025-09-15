"""
build_vector_db.py
-------------------
Create embeddings for document chunks and store them in a persistent ChromaDB.

Workflow:
1) Load chunked documents from chunks.pkl (produced by ingest.py)
2) Encode chunks with SentenceTransformer 'all-MiniLM-L6-v2'
3) Persist into a ChromaDB collection using chromadb.PersistentClient

Run:
  python build_vector_db.py --chunks-file chunks.pkl --db-path chroma_db --collection pubmed_docs
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import List

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import shutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Chroma vector DB from pickled chunks")
    parser.add_argument(
        "--chunks-file",
        type=str,
        default="chunks.pkl",
        help="Path to pickle file with a list of LangChain Document objects",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="chroma_db",
        help="Directory for persistent ChromaDB",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="pubmed_docs",
        help="ChromaDB collection name",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model to use for embeddings",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the DB path before indexing to avoid duplicates",
    )
    return parser.parse_args()


def load_chunks(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"chunks file not found: {path}. Run 'python ingest.py' first to create it."
        )
    with open(path, "rb") as f:
        chunks = pickle.load(f)
    return chunks


def main() -> None:
    args = parse_args()

    # 1) Load chunked documents
    chunks = load_chunks(args.chunks_file)
    if not isinstance(chunks, list) or len(chunks) == 0:
        print("No chunks found in pickle — nothing to index.")
        return

    # 2) Initialize embedding model
    print(f"Loading embedding model: {args.model}")
    model = SentenceTransformer(args.model)

    # Prepare inputs for embedding
    documents: List[str] = [c.page_content for c in chunks]
    metadatas: List[dict] = [getattr(c, "metadata", {}) for c in chunks]
    ids: List[str] = [f"chunk_{i}" for i in range(len(chunks))]

    # 3) Initialize Chroma persistent client and collection
    if args.reset and os.path.exists(args.db_path):
        print(f"Resetting DB path: {args.db_path}")
        shutil.rmtree(args.db_path, ignore_errors=True)
    print(f"Initializing persistent ChromaDB at: {args.db_path}")
    client = chromadb.PersistentClient(
        path=args.db_path,
        settings=Settings(chroma_db_impl="duckdb+parquet", anonymized_telemetry=False),
    )
    # Unify with ingest.py defaults: use 'pubmed_abstracts' collection by default
    if args.collection == "pubmed_docs":
        args.collection = "pubmed_abstracts"
    collection = client.get_or_create_collection(name=args.collection)

    # 4) Compute embeddings and add to collection
    print(f"Encoding {len(documents)} document chunk(s)...")
    # Build stable IDs matching ingest.py: source::start_index
    stable_ids = [
        f"{md.get('source','unknown')}::{md.get('start_index', 0)}" for md in metadatas
    ]

    # Determine which IDs already exist to avoid duplicates
    try:
        existing_resp = collection.get(include=[])
        existing_ids = set(existing_resp.get("ids", []))
    except Exception:
        existing_ids = set()

    to_add_idx = [i for i, sid in enumerate(stable_ids) if sid not in existing_ids]
    if not to_add_idx:
        print("No new chunks to add (all IDs already present).")
    else:
        add_docs = [documents[i] for i in to_add_idx]
        add_metas = [metadatas[i] for i in to_add_idx]
        add_ids = [stable_ids[i] for i in to_add_idx]

        embeddings = model.encode(add_docs).tolist()
        collection.add(
            documents=add_docs,
            embeddings=embeddings,
            metadatas=add_metas,
            ids=add_ids,
        )

    # 5) Confirm (persistence is automatic in modern Chroma clients)
    # Older clients had client.persist(); guard for it if present.
    persist_fn = getattr(client, "persist", None)
    if callable(persist_fn):
        try:
            persist_fn()
        except Exception as e:
            print(f"Skipping client.persist(): {e}")

    count = collection.count()
    print(f"Stored {count} vector(s) in collection '{args.collection}'.")

    # Show a sample metadata entry for verification
    print("Example metadata:")
    print(metadatas[0] if metadatas else {})


if __name__ == "__main__":
    main()
