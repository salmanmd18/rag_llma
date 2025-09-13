# Healthcare RAG Q&A

Retrieval-Augmented Generation system using PubMed abstracts and WHO fact sheets with LangChain, ChromaDB, Flan‑T5, and Streamlit.

## Setup

- Create and activate a virtual environment (Python 3.10+ recommended):
  - Windows (PowerShell):
    - `python -m venv .venv`
    - `./.venv/Scripts/Activate.ps1`
  - macOS/Linux:
    - `python3 -m venv .venv`
    - `source .venv/bin/activate`
- Install dependencies:
  - `pip install --upgrade pip`
  - `pip install -r requirements.txt`

## Usage

1. Place `.txt` or `.md` files (PubMed abstracts or WHO fact sheets) into `data/`.
2. Build the persistent vector DB with Chroma (choose one):
   - One‑step ingest (chunks + vectors): `python ingest.py --persist-dir chroma_db`
   - Two‑step path:
     - Create chunks: `python ingest.py --persist-dir chroma_db`
     - Build vectors from chunks: `python build_vector_db.py --chunks-file chunks.pkl --db-path chroma_db --collection pubmed_abstracts`
3. Run the Streamlit app: `streamlit run app.py`

## Repository Tree

```
data/
chroma_db/                # persisted Chroma (created on first ingest)
ingest.py                 # load + chunk raw documents and (optionally) index
build_vector_db.py        # generate embeddings + store in ChromaDB
rag_pipeline.py           # retrieval + QA pipeline (LangChain + Flan‑T5)
app.py                    # Streamlit frontend
load_model.py             # test Flan‑T5 loading locally
requirements.txt          # pinned dependencies
Dockerfile                # container build instructions
.dockerignore             # avoid unnecessary files in image
.gitignore                # ignore caches, venv, DB, artifacts
README.md                 # documentation
```

## Docker

Build and run the Streamlit app in a container:

- Build image:
  - `docker build -t healthcare-rag .`
- Run container (maps host port 8501 → container 8501):
  - `docker run -p 8501:8501 healthcare-rag`

Notes:
- The image uses `python:3.10-slim`, installs dependencies from `requirements.txt`, copies the project, exposes port `8501`, and starts Streamlit.
- `.dockerignore` excludes local caches (`__pycache__/`, `.venv/`), pickles/logs, and local `chroma_db/` to keep images small. Mount a volume if you want persistence across runs.

## Example Query

- Query: "What are the benefits of metformin in diabetes treatment?"
- Expected answer style (Flan‑T5): concise summary referencing improved glycemic control and potential cardiovascular benefits (exact wording varies).

## Notes

- Model: `google/flan-t5-base` (instruction‑tuned, CPU‑friendly; swap to larger variants if hardware allows).
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`.
- Vector store: ChromaDB persisted under `chroma_db/`.
- Credits: Hugging Face, LangChain, Chroma, Streamlit.

## Project Structure

```
data/            # .txt/.md files to index
chroma_db/       # persistent ChromaDB directory (created on first ingest)
ingest.py        # loads, chunks, embeds, and indexes documents
app.py           # Streamlit UI (placeholder)
requirements.txt # dependencies (kept as-is per project)
README.md        # overview and usage
Dockerfile       # containerization (placeholder)
```
