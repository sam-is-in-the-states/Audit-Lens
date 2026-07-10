import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
KB_DIR = PROJECT_ROOT / "documents" / "knowledge_base"

# Default retrieval hyperparameters
DEFAULT_TOP_K = 20
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 75

# Default per-KB quotas. These are applied only when that KB exists.
DEFAULT_KB_QUOTAS = {
    "KB1": 8,
    "KB2": 7,
    "KB3": 5,
}

class PolicyRetrievalAgent:
    def __init__(
        self,
        kb_dir: Path = KB_DIR,
        top_k: int = DEFAULT_TOP_K,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int = 10000,
        kb_quotas: dict[str, int] | None = None,
    ):
        self.kb_dir = Path(kb_dir)
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.kb_quotas = kb_quotas or DEFAULT_KB_QUOTAS.copy()

        # Validate retrieval hyperparameters
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if self.chunk_size < 1:
            raise ValueError("chunk_size must be at least 1.")

        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self.documents: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=True,
        )

        self.matrix = None

        # Load knowledge base and build retrieval index
        self.load_knowledge_base()
        self.build_index()

    # Load JSON and JSONL policy knowledge-base files
    def load_knowledge_base(self) -> None:
        if not self.kb_dir.exists():
            raise FileNotFoundError(
                f"Knowledge base folder not found: {self.kb_dir}"
            )

        json_files = sorted(self.kb_dir.glob("*.json"))
        jsonl_files = sorted(self.kb_dir.glob("*.jsonl"))
        kb_files = json_files + jsonl_files

        if not kb_files:
            raise FileNotFoundError(
                "No JSON or JSONL knowledge-base files found in: "
                f"{self.kb_dir}"
            )

        for file_path in kb_files:
            if file_path.suffix.lower() == ".json":
                try:
                    data = json.loads(
                        file_path.read_text(encoding="utf-8")
                    )
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {file_path.name}: {exc}"
                    ) from exc

                self._add_kb_data(
                    data=data,
                    file_path=file_path,
                )

            elif file_path.suffix.lower() == ".jsonl":
                with file_path.open("r", encoding="utf-8") as file:
                    for line_number, line in enumerate(file, start=1):
                        line = line.strip()

                        if not line:
                            continue

                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"Invalid JSON on line {line_number} "
                                f"of {file_path.name}: {exc}"
                            ) from exc

                        self._add_kb_item(
                            item=item,
                            file_path=file_path,
                            default_id=f"{file_path.stem}_{line_number}",
                        )

        if not self.documents:
            raise ValueError(
                "Knowledge-base files were found, but no policy documents "
                f"could be loaded from {self.kb_dir}"
            )

        loaded_prefixes = sorted(
            {
                self._get_kb_prefix(document["doc_id"])
                for document in self.documents
                if self._get_kb_prefix(document["doc_id"])
            }
        )

        print(
            f"Loaded {len(self.documents)} policy documents "
            f"from {len(kb_files)} knowledge-base files."
        )
        print(f"Detected knowledge bases: {', '.join(loaded_prefixes)}")

        if "KB3" not in loaded_prefixes:
            print(
                "Warning: No KB3 records were loaded. "
                "Any KB3 ground-truth policies will be impossible to retrieve."
            )

    # Add JSON file content to the document collection
    def _add_kb_data(
        self,
        data: Any,
        file_path: Path,
    ) -> None:
        if isinstance(data, list):
            for index, item in enumerate(data, start=1):
                self._add_kb_item(
                    item=item,
                    file_path=file_path,
                    default_id=f"{file_path.stem}_{index}",
                )

        elif isinstance(data, dict):
            if self._looks_like_policy_record(data):
                self._add_kb_item(
                    item=data,
                    file_path=file_path,
                    default_id=file_path.stem,
                )
            else:
                for key, value in data.items():
                    if isinstance(value, dict):
                        item = {
                            "id": (
                                value.get("id")
                                or value.get("policy_id")
                                or value.get("section_id")
                                or str(key)
                            ),
                            **value,
                        }
                    else:
                        item = {
                            "id": str(key),
                            "title": str(key),
                            "text": value,
                        }

                    self._add_kb_item(
                        item=item,
                        file_path=file_path,
                        default_id=str(key),
                    )

        else:
            self._add_kb_item(
                item=data,
                file_path=file_path,
                default_id=file_path.stem,
            )

    # Add one policy record to the retrieval collection
    def _add_kb_item(
        self,
        item: Any,
        file_path: Path,
        default_id: str,
    ) -> None:
        if isinstance(item, dict):
            doc_id = str(
                item.get("id")
                or item.get("policy_id")
                or item.get("section_id")
                or item.get("chunk_id")
                or default_id
            ).strip()

            title = str(
                item.get("title")
                or item.get("topic")
                or item.get("section")
                or item.get("name")
                or doc_id
            ).strip()

            text = self.stringify_kb_item(item)
            metadata = item
        else:
            doc_id = default_id
            title = default_id
            text = str(item)
            metadata = {"value": item}

        if not text.strip():
            return

        self.documents.append(
            {
                "source_file": file_path.name,
                "doc_id": doc_id,
                "kb_prefix": self._get_kb_prefix(doc_id),
                "title": title,
                "text": text,
                "metadata": metadata,
            }
        )

    # Determine whether a dictionary represents one policy record
    def _looks_like_policy_record(
        self,
        data: dict[str, Any],
    ) -> bool:
        record_fields = {
            "id",
            "policy_id",
            "section_id",
            "chunk_id",
            "title",
            "topic",
            "section",
            "citation",
            "text",
            "content",
            "guidance",
            "policy",
            "summary",
            "keywords",
            "conditions",
            "treatment",
        }

        return bool(record_fields.intersection(data.keys()))

    # Extract KB1, KB2, or KB3 from a policy document ID
    def _get_kb_prefix(self, doc_id: str) -> str | None:
        match = re.match(
            r"^(KB\d+)_",
            str(doc_id).strip(),
            flags=re.IGNORECASE,
        )

        return match.group(1).upper() if match else None

    # Convert a knowledge-base item into searchable text
    def stringify_kb_item(self, item: Any) -> str:
        if isinstance(item, str):
            return item

        if isinstance(item, dict):
            preferred_fields = [
                "title",
                "topic",
                "section",
                "citation",
                "text",
                "content",
                "guidance",
                "policy",
                "summary",
                "keywords",
                "conditions",
                "treatment",
            ]

            parts: list[str] = []

            for field in preferred_fields:
                value = item.get(field)

                if value not in (None, "", [], {}):
                    if isinstance(value, (dict, list)):
                        value = json.dumps(
                            value,
                            ensure_ascii=False,
                        )

                    parts.append(f"{field}: {value}")

            for key, value in item.items():
                if key in preferred_fields:
                    continue

                if value not in (None, "", [], {}):
                    if isinstance(value, (dict, list)):
                        value = json.dumps(
                            value,
                            ensure_ascii=False,
                        )

                    parts.append(f"{key}: {value}")

            return "\n".join(parts)

        if isinstance(item, list):
            return "\n".join(
                self.stringify_kb_item(value)
                for value in item
            )

        return str(item)

    # Split knowledge-base documents into overlapping chunks
    def chunk_text(self, text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0
        step_size = self.chunk_size - self.chunk_overlap

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= len(text):
                break

            start += step_size

        return chunks

    # Build the TF-IDF retrieval index
    def build_index(self) -> None:
        self.chunks = []

        for document in self.documents:
            text_chunks = self.chunk_text(document["text"])

            for index, chunk in enumerate(text_chunks):
                self.chunks.append(
                    {
                        "chunk_id": f'{document["doc_id"]}_chunk_{index}',
                        "source_file": document["source_file"],
                        "doc_id": document["doc_id"],
                        "kb_prefix": document["kb_prefix"],
                        "title": document["title"],
                        "text": chunk,
                        "metadata": document["metadata"],
                    }
                )

        if not self.chunks:
            raise ValueError(
                "No searchable chunks were generated from the knowledge base."
            )

        chunk_texts = [chunk["text"] for chunk in self.chunks]
        self.matrix = self.vectorizer.fit_transform(chunk_texts)

        print(f"Built retrieval index with {len(self.chunks)} chunks.")

    # Build targeted policy queries only for active contract features
    def build_policy_queries(
        self,
        contract: dict[str, Any],
    ) -> list[str]:
        queries = [
            (
                "ASC 606 contract existence enforceable rights "
                "collectibility transaction price performance obligations"
            ),
            (
                "ASC 606 hosted SaaS subscription stand-ready service "
                "revenue recognized over time"
            ),
        ]

        services_summary = str(
            contract.get("services_summary", "")
        ).strip()

        if services_summary:
            queries.append(
                f"SaaS contract services: {services_summary}"
            )

        if contract.get("implementation_services"):
            queries.append(
                "ASC 606 implementation onboarding setup services "
                "distinct performance obligation nonrefundable upfront fee"
            )

        if contract.get("implementation_required_for_platform"):
            queries.append(
                "implementation required for SaaS platform "
                "not distinct combined performance obligation"
            )

        discount = contract.get("discount_terms") or {}

        if discount.get("has_discount"):
            queries.append(
                "ASC 606 discount allocation relative standalone selling price "
                f"{json.dumps(discount, ensure_ascii=False)}"
            )

        usage_fee = contract.get("usage_fee") or {}

        if usage_fee and (
            usage_fee.get("overage_rate")
            or usage_fee.get("included_units") is not None
        ):
            queries.append(
                "ASC 606 usage overage variable consideration "
                "recognized as usage occurs "
                f"{json.dumps(usage_fee, ensure_ascii=False)}"
            )

        onboarding = contract.get("onboarding_terms") or {}

        if onboarding.get("required") or onboarding.get("fee"):
            queries.append(
                "onboarding upfront fee implementation revenue recognition "
                f"{json.dumps(onboarding, ensure_ascii=False)}"
            )

        renewal = contract.get("renewal_terms") or {}

        if renewal.get("auto_renewal"):
            queries.append(
                "ASC 606 renewal option material right renewal pricing "
                f"{json.dumps(renewal, ensure_ascii=False)}"
            )

        termination = contract.get("termination_terms") or {}

        if termination:
            queries.append(
                "ASC 606 termination rights enforceable contract term "
                f"{json.dumps(termination, ensure_ascii=False)}"
            )

        ambiguous = contract.get("ambiguous_clauses") or []

        if ambiguous:
            queries.append(
                "revenue recognition accounting risk ambiguity escalation "
                f"{json.dumps(ambiguous, ensure_ascii=False)}"
            )

        return queries

    # Score all chunks for a query
    def _score_chunks(
        self,
        query: str,
    ) -> list[tuple[int, float]]:
        if self.matrix is None:
            raise RuntimeError("Retrieval index has not been built.")

        query_vector = self.vectorizer.transform([query])

        scores = cosine_similarity(
            query_vector,
            self.matrix,
        ).flatten()

        return sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )

    # Retrieve unique policy documents using multiple issue-specific queries
    def retrieve(
        self,
        queries: str | list[str],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if isinstance(queries, str):
            query_list = [queries]
        else:
            query_list = [
                query
                for query in queries
                if query and query.strip()
            ]

        if not query_list:
            raise ValueError("At least one retrieval query is required.")

        k = top_k if top_k is not None else self.top_k

        # Aggregate evidence across multiple targeted queries.
        # A document keeps its strongest chunk score for each query, and those
        # scores are summed across queries.
        doc_scores: dict[str, float] = defaultdict(float)
        best_chunk_by_doc: dict[str, dict[str, Any]] = {}
        best_score_by_doc: dict[str, float] = defaultdict(float)

        for query in query_list:
            per_query_best: dict[str, tuple[int, float]] = {}

            for chunk_index, score in self._score_chunks(query):
                chunk = self.chunks[chunk_index]
                doc_id = chunk["doc_id"]

                if doc_id not in per_query_best:
                    per_query_best[doc_id] = (
                        chunk_index,
                        float(score),
                    )

            for doc_id, (
                chunk_index,
                score,
            ) in per_query_best.items():
                doc_scores[doc_id] += score

                if score > best_score_by_doc[doc_id]:
                    best_score_by_doc[doc_id] = score
                    best_chunk_by_doc[doc_id] = self.chunks[
                        chunk_index
                    ].copy()

        ranked_doc_ids = sorted(
            doc_scores,
            key=doc_scores.get,
            reverse=True,
        )

        # Apply per-KB quotas when that KB exists
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        counts_by_kb: dict[str, int] = defaultdict(int)

        for doc_id in ranked_doc_ids:
            chunk = best_chunk_by_doc[doc_id]
            kb_prefix = chunk.get("kb_prefix")

            quota = self.kb_quotas.get(
                kb_prefix,
                k,
            )

            if counts_by_kb[kb_prefix] >= quota:
                continue

            result = chunk.copy()
            result["score"] = float(doc_scores[doc_id])
            result["best_single_query_score"] = float(
                best_score_by_doc[doc_id]
            )
            result["rank"] = len(selected) + 1

            selected.append(result)
            selected_ids.add(doc_id)
            counts_by_kb[kb_prefix] += 1

            if len(selected) >= k:
                break

        # Fill any remaining slots from the global ranking while preserving
        # one result per policy document.
        if len(selected) < k:
            for doc_id in ranked_doc_ids:
                if doc_id in selected_ids:
                    continue

                chunk = best_chunk_by_doc[doc_id].copy()
                chunk["score"] = float(doc_scores[doc_id])
                chunk["best_single_query_score"] = float(
                    best_score_by_doc[doc_id]
                )
                chunk["rank"] = len(selected) + 1

                selected.append(chunk)
                selected_ids.add(doc_id)

                if len(selected) >= k:
                    break

        return selected

    # Format retrieved document IDs like the Ground Truth columns
    def format_kb_chunk_response(
        self,
        retrieved_chunks: list[dict[str, Any]],
        kb_prefix: str,
    ) -> str:
        matching_ids = [
            str(chunk["doc_id"])
            for chunk in retrieved_chunks
            if str(chunk.get("kb_prefix", "")).upper()
            == kb_prefix.upper()
        ]

        return "; ".join(matching_ids) if matching_ids else "(none)"

    # Build compact Ground Truth-style KB responses
    def build_kb_chunk_responses(
        self,
        retrieved_chunks: list[dict[str, Any]],
    ) -> dict[str, str]:
        return {
            "KB1 Chunks": self.format_kb_chunk_response(
                retrieved_chunks,
                "KB1",
            ),
            "KB2 Chunks": self.format_kb_chunk_response(
                retrieved_chunks,
                "KB2",
            ),
            "KB3 Chunks": self.format_kb_chunk_response(
                retrieved_chunks,
                "KB3",
            ),
        }

    # Run Agent 2 on Agent 1 structured contract output
    def run(
        self,
        agent1_output: dict[str, Any],
        top_k: int | None = None,
    ) -> dict[str, Any]:
        queries = self.build_policy_queries(agent1_output)

        retrieved_chunks = self.retrieve(
            queries=queries,
            top_k=top_k,
        )

        kb_chunk_responses = self.build_kb_chunk_responses(
            retrieved_chunks
        )

        effective_top_k = (
            top_k
            if top_k is not None
            else self.top_k
        )

        return {
            "agent": (
                "Agent 2 - Accounting Standards "
                "Policy Retrieval"
            ),
            "queries": queries,
            "top_k": effective_top_k,
            "KB1 Chunks": kb_chunk_responses["KB1 Chunks"],
            "KB2 Chunks": kb_chunk_responses["KB2 Chunks"],
            "KB3 Chunks": kb_chunk_responses["KB3 Chunks"],
            "retrieved_policy_chunks": retrieved_chunks,
            "policy_prompt": self.build_policy_prompt(
                contract=agent1_output,
                retrieved_chunks=retrieved_chunks,
            ),
        }

    # Build the policy prompt for downstream agents
    def build_policy_prompt(
        self,
        contract: dict[str, Any],
        retrieved_chunks: list[dict[str, Any]],
    ) -> str:
        policy_context = "\n\n".join(
            [
                (
                    f"[Policy Source: {chunk['source_file']} | "
                    f"Document: {chunk['doc_id']} | "
                    f"Chunk: {chunk['chunk_id']} | "
                    f"Aggregate Score: {chunk['score']:.3f}]\n"
                    f"{chunk['text']}"
                )
                for chunk in retrieved_chunks
            ]
        )

        return f"""
You are Agent 2: Accounting Standards Policy Retrieval Agent.

Your task is to identify and organize the ASC 606 guidance, company
revenue-recognition policy, and risk guidance needed to evaluate the
supplied contract.

Do not make the final accounting conclusion.
Do not calculate the revenue schedule.
Do not invent policy guidance that is not supported by the retrieved context.

Contract Information:
{json.dumps(contract, indent=2)}

Retrieved Policy Context:
{policy_context}

Return the result in the following structure:

1. Relevant ASC 606 Guidance
2. Relevant Company Revenue Policy
3. Relevant Risk Guidance
4. Rules Needed for Schedule Calculation
5. Rules Needed for Risk Flagging
6. Missing or Ambiguous Policy Areas
""".strip()

# Demonstration runner
def main() -> None:
    sample_contract = {
        "contract_id": "RL-2026-0001",
        "services_summary": (
            "Hosted SaaS subscription platform with implementation services."
        ),
        "hosted_service": True,
        "implementation_services": True,
        "implementation_required_for_platform": True,
        "discount_terms": {
            "has_discount": True,
            "discount_type": "percentage",
            "percentage": 10,
        },
        "usage_fee": {
            "included_units": 10000,
            "overage_rate": {
                "amount": 0.25,
                "currency": "USD",
            },
        },
        "onboarding_terms": {
            "required": True,
            "fee": {
                "amount": 10000,
                "currency": "USD",
            },
        },
        "renewal_terms": {
            "auto_renewal": True,
            "renewal_term_months": 12,
            "notice_days": 60,
        },
        "termination_terms": {
            "termination_for_breach": True,
            "notice_days": 30,
        },
        "ambiguous_clauses": [],
    }

    agent = PolicyRetrievalAgent(
        top_k=20,
        chunk_size=500,
        chunk_overlap=75,
        kb_quotas={
            "KB1": 8,
            "KB2": 7,
            "KB3": 5,
        },
    )

    output = agent.run(sample_contract)

    print("\nAgent 2 Ground Truth-Style Response")
    print("-----------------------------------")
    print(f"KB1 Chunks: {output['KB1 Chunks']}")
    print(f"KB2 Chunks: {output['KB2 Chunks']}")
    print(f"KB3 Chunks: {output['KB3 Chunks']}")

    print("\nRanked Retrieval Results")
    print("------------------------")

    for chunk in output["retrieved_policy_chunks"]:
        print(
            f"Rank {chunk['rank']}: "
            f"{chunk['doc_id']} | "
            f"{chunk['title']} | "
            f"Score: {chunk['score']:.4f}"
        )

if __name__ == "__main__":
    main()