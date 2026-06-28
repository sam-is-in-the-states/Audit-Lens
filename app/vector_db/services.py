"""
Usage
-----
1. Pull an embedding model (one-time):
 
    ollama pull nomic-embed-text
 
2. Create a DB from a JSONL file:
 
    from vector_db_service import VectorDBService
 
    svc = VectorDBService()                         # defaults to nomic-embed-text
    svc.create_db_from_jsonl(
        jsonl_path="KB1_asc606_guidance.jsonl",
        collection_name="asc606",
    )
 
3. Query:
 
    results = svc.query(
        collection_name="asc606",
        query_text="How do I identify a performance obligation?",
        top_k=3,
    )
    for r in results:
        print(r["id"], r["score"])
        print(r["text"])
        print(r["metadata"])
 
Notes
-----
- Every field in each JSONL record *except* `text_field` is stored as metadata.
- The embedding model is configurable; pass a different Ollama model name or
  swap in any chromadb-compatible EmbeddingFunction via `embedding_fn=`.
- Ollama must be running locally (`ollama serve`) before you call any method.
"""

from __future__ import annotations
 
from dotenv import load_dotenv
import json
import os
from pathlib import Path
from typing import Any
 
import chromadb
from chromadb.utils.embedding_functions import (
    EmbeddingFunction,
    OllamaEmbeddingFunction,
)


load_dotenv()

class VectorDBService:
 
    def __init__(
        self,
        persist_dir: str | Path = "./chroma_store",
        ollama_url: str = os.getenv("OLLAMA_BASE_URL"),
        embed_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL"),
        embedding_fn: EmbeddingFunction | None = None,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
 
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
 
        # Resolve embedding function — custom > Ollama default
        if embedding_fn is not None:
            self._embedding_fn: EmbeddingFunction = embedding_fn
        else:
            self._embedding_fn = OllamaEmbeddingFunction(
                url=ollama_url,
                model_name=embed_model,
            )
            print(
                f"[VectorDBService] Using Ollama embedding model '{embed_model}' "
                f"at {ollama_url}"
            )
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
 
    def create_db_from_jsonl(
        self,
        jsonl_path: str | Path,
        collection_name: str,
        text_field: str = "text",
        id_field: str = "id",
        overwrite: bool = False,
    ) -> chromadb.Collection:
        """
        Load a JSONL file into a ChromaDB collection.
 
        Each record's `text_field` value is embedded and stored as the
        document content. All other keys are stored as metadata.
 
        Parameters
        ----------
        jsonl_path : str | Path
            Path to the ``.jsonl`` source file.
        collection_name : str
            Name of the ChromaDB collection to create or populate.
        text_field : str
            JSON key whose value will be embedded. Default: ``"text"``.
        id_field : str
            JSON key to use as the document ID. Default: ``"id"``.
        overwrite : bool
            Delete and recreate the collection if it already exists.
 
        Returns
        -------
        chromadb.Collection
        """
        jsonl_path = Path(jsonl_path)
        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
 
        # ── handle existing collection ────────────────────────────────────
        existing = [c.name for c in self._client.list_collections()]
        if collection_name in existing:
            if overwrite:
                self._client.delete_collection(collection_name)
                print(f"[VectorDBService] Deleted existing collection '{collection_name}'.")
            else:
                print(
                    f"[VectorDBService] Collection '{collection_name}' already exists. "
                    "Pass overwrite=True to re-create it."
                )
                return self._client.get_collection(
                    name=collection_name,
                    embedding_function=self._embedding_fn,
                )
 
        collection = self._client.create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
 
        # ── parse JSONL ───────────────────────────────────────────────────
        records: list[dict] = []
        with jsonl_path.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
 
        if not records:
            raise ValueError(f"No records found in {jsonl_path}")
 
        # ── validate required fields ──────────────────────────────────────
        for i, rec in enumerate(records):
            if text_field not in rec:
                raise ValueError(
                    f"Record {i} is missing text field '{text_field}'. Keys present: {list(rec.keys())}"
                )
            if id_field not in rec:
                raise ValueError(
                    f"Record {i} is missing id field '{id_field}'. Keys present: {list(rec.keys())}"
                )
 
        # ── build parallel lists ──────────────────────────────────────────
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
 
        for rec in records:
            meta: dict[str, Any] = {
                key: _coerce_metadata_value(val)
                for key, val in rec.items()
                if key != text_field          # text is stored as the document
            }
            ids.append(str(rec[id_field]))
            documents.append(str(rec[text_field]))
            metadatas.append(meta)
 
        # ── insert in batches ─────────────────────────────────────────────
        batch_size = 50
        total = len(ids)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
            print(f"[VectorDBService]   Embedded {end}/{total} documents...")
 
        print(
            f"[VectorDBService] ✓ Collection '{collection_name}' ready — "
            f"{total} documents from '{jsonl_path.name}'."
        )
        return collection
 
    # ------------------------------------------------------------------
 
    def query(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Find the top-K most semantically similar documents.
 
        Parameters
        ----------
        collection_name : str
            The collection to search.
        query_text : str
            Natural-language query string.
        top_k : int
            Number of results to return.
 
        Returns
        -------
        list[dict]  — each dict has:
            ``id``        document ID
            ``score``     cosine distance (0 = identical, 1 = unrelated)
            ``text``      original document text
            ``metadata``  all other fields from the original JSONL record
        """
        existing = [c.name for c in self._client.list_collections()]
        if collection_name not in existing:
            raise ValueError(
                f"Collection '{collection_name}' not found. "
                f"Available collections: {existing}"
            )
 
        collection = self._client.get_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
        )
 
        raw = collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
 
        return [
            {
                "id": doc_id,
                "score": round(dist, 6),
                "text": text,
                "metadata": meta,
            }
            for doc_id, text, meta, dist in zip(
                raw["ids"][0],
                raw["documents"][0],
                raw["metadatas"][0],
                raw["distances"][0],
            )
        ]
 
    # ------------------------------------------------------------------
 
    def list_collections(self) -> list[str]:
        """Return the names of all collections in the store."""
        return [c.name for c in self._client.list_collections()]
 
    def delete_collection(self, collection_name: str) -> None:
        """Permanently delete a collection and all its data."""
        self._client.delete_collection(collection_name)
        print(f"[VectorDBService] Deleted collection '{collection_name}'.")
 
    def get_collection_count(self, collection_name: str) -> int:
        """Return the number of documents in a collection."""
        col = self._client.get_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
        )
        return col.count()
 
 
# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
 
def _coerce_metadata_value(val: Any) -> str | int | float | bool:
    """
    ChromaDB metadata values must be str | int | float | bool.
    Coerce None and complex types (lists, dicts) to a JSON string.
    """
    if isinstance(val, (str, int, float, bool)):
        return val
    if val is None:
        return ""
    return json.dumps(val)
