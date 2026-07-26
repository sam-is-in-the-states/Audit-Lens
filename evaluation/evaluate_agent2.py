import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.llm.policy_retrieval import PolicyRetrievalAgent

GROUND_TRUTH_PATH = PROJECT_ROOT / "evaluation" / "GroundTruth.xlsx"
EXTRACTED_CONTRACTS_PATH = (
    PROJECT_ROOT / "app" / "llm" / "extracted_contracts.json"
)
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

FINAL_TOP_K = 20
CANDIDATE_TOP_K = 30
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75

KB_QUOTAS = {
    "KB1": 12,
    "KB2": 10,
    "KB3": 8,
}

# Final-run design:
# - one paid LLM call per contract;
# - cache all candidate + LLM scoring data;
# - tune parameters offline on a deterministic tuning split;
# - evaluate selected parameters on untouched held-out contracts.
TUNE_FRACTION = 0.70
SPLIT_SALT = "audit-lens-agent2-final-v1"

THRESHOLD_GRID = [
    0.25, 0.30, 0.35, 0.40,
    0.45, 0.50, 0.55, 0.60,
]

MIN_RESULTS_GRID = [
    8, 10, 12, 14, 16, 18,
]

MAX_RESULTS_GRID = [
    14, 16, 18, 20, 22, 24,
]

RETRIEVAL_WEIGHT_GRID = [
    0.00, 0.10, 0.20, 0.30,
    0.40, 0.50, 0.60, 0.70,
]

# Agent 2 is a retrieval stage, so choose among high-F1 settings that
# preserve reasonable recall on the tuning set.
RECALL_FLOOR = 0.65

CACHE_PATH = RESULTS_DIR / "agent2_finalrun_cache.json"
SWEEP_PATH = RESULTS_DIR / "agent2_parameter_sweep.csv"
FINAL_JSON_PATH = RESULTS_DIR / "agent2_final_evaluation.json"
FINAL_CSV_PATH = RESULTS_DIR / "agent2_final_evaluation.csv"
CONFIG_PATH = RESULTS_DIR / "agent2_selected_config.json"


def load_extracted_contracts() -> list[dict[str, Any]]:
    if not EXTRACTED_CONTRACTS_PATH.exists():
        raise FileNotFoundError(
            f"Agent 1 output not found: {EXTRACTED_CONTRACTS_PATH}"
        )

    data = json.loads(
        EXTRACTED_CONTRACTS_PATH.read_text(encoding="utf-8")
    )

    if isinstance(data, dict):
        records = data.get("contracts", [])
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(
            "extracted_contracts.json must contain a list or object."
        )

    contracts: list[dict[str, Any]] = []

    for item in records:
        if not isinstance(item, dict):
            continue

        output = (
            item.get("validated_output")
            or item.get("candidate_output")
            or item
        )

        if not isinstance(output, dict):
            continue

        contract = dict(output)

        if not contract.get("source_file"):
            contract["source_file"] = (
                item.get("file_name")
                or item.get("source_file")
            )

        contracts.append(contract)

    print(f"Loaded {len(contracts)} extracted contracts.")
    return contracts


def load_ground_truth() -> pd.DataFrame:
    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(
            f"Ground truth file not found: {GROUND_TRUTH_PATH}"
        )

    ground_truth = pd.read_excel(
        GROUND_TRUTH_PATH,
        header=3,
    )

    ground_truth = ground_truth.dropna(axis=0, how="all")
    ground_truth = ground_truth.dropna(axis=1, how="all")

    ground_truth.columns = [
        str(column).strip()
        for column in ground_truth.columns
    ]

    required_columns = {
        "contract_id",
        "KB1 Chunks",
        "KB2 Chunks",
        "KB3 Chunks",
    }

    missing = required_columns - set(ground_truth.columns)

    if missing:
        raise ValueError(
            "Ground truth is missing required columns: "
            f"{sorted(missing)}\n"
            f"Available columns: {ground_truth.columns.tolist()}"
        )

    ground_truth["contract_id"] = (
        ground_truth["contract_id"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    print(f"Loaded {len(ground_truth)} ground-truth contracts.")
    return ground_truth


def extract_contract_id(source_file: str | None) -> str | None:
    if not source_file:
        return None

    match = re.match(
        r"(C\d+)",
        Path(source_file).stem,
        flags=re.IGNORECASE,
    )

    return match.group(1).upper() if match else None


def parse_kb_ids(value: Any) -> set[str]:
    if pd.isna(value):
        return set()

    text = str(value).strip()

    if not text or text.lower().startswith("(none"):
        return set()

    matches = re.findall(
        r"\bKB\d+_\d+\b",
        text,
        flags=re.IGNORECASE,
    )

    return {match.upper() for match in matches}


def get_ids(chunks: list[dict[str, Any]]) -> set[str]:
    return {
        str(chunk.get("doc_id", "")).strip().upper()
        for chunk in chunks
        if str(chunk.get("doc_id", "")).strip()
    }


def calculate_metrics(
    expected: set[str],
    predicted: set[str],
) -> dict[str, Any]:
    matched = expected & predicted
    missing = expected - predicted
    unexpected = predicted - expected

    recall = (
        len(matched) / len(expected)
        if expected
        else (1.0 if not predicted else 0.0)
    )

    precision = (
        len(matched) / len(predicted)
        if predicted
        else (1.0 if not expected else 0.0)
    )

    f1 = (
        2 * recall * precision / (recall + precision)
        if recall + precision > 0
        else 0.0
    )

    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "matched": sorted(matched),
        "missing": sorted(missing),
        "unexpected": sorted(unexpected),
        "exact_match": predicted == expected,
        "any_match": bool(matched),
    }


def family_ids(ids: set[str], prefix: str) -> set[str]:
    return {
        policy_id
        for policy_id in ids
        if policy_id.startswith(prefix + "_")
    }


def family_recall(
    expected: set[str],
    predicted: set[str],
) -> float:
    if not expected:
        return 1.0

    return len(expected & predicted) / len(expected)


def deterministic_split(contract_id: str) -> str:
    digest = hashlib.sha256(
        f"{SPLIT_SALT}:{contract_id}".encode("utf-8")
    ).hexdigest()

    bucket = int(digest[:8], 16) / 0xFFFFFFFF

    return "tune" if bucket < TUNE_FRACTION else "holdout"


def normalize_retrieval_scores_by_kb(
    candidate_chunks: list[dict[str, Any]],
) -> dict[str, float]:
    by_kb: dict[str, list[float]] = {}

    for chunk in candidate_chunks:
        kb = str(chunk.get("kb_prefix", ""))
        by_kb.setdefault(kb, []).append(
            float(chunk.get("score", 0.0))
        )

    normalized: dict[str, float] = {}

    for chunk in candidate_chunks:
        kb = str(chunk.get("kb_prefix", ""))
        values = by_kb[kb]

        low = min(values)
        high = max(values)

        raw = float(chunk.get("score", 0.0))

        normalized[
            str(chunk.get("doc_id", "")).strip().upper()
        ] = (
            (raw - low) / (high - low)
            if high > low
            else 1.0
        )

    return normalized


def offline_select(
    candidate_chunks: list[dict[str, Any]],
    llm_scores: list[dict[str, Any]],
    threshold: float,
    min_results: int,
    max_results: int,
    retrieval_weight: float,
) -> list[dict[str, Any]]:
    llm_weight = 1.0 - retrieval_weight

    relevance_by_id = {
        str(item.get("doc_id", "")).strip().upper(): float(
            item.get("relevance", 0.5)
        )
        for item in llm_scores
    }

    normalized = normalize_retrieval_scores_by_kb(
        candidate_chunks
    )

    reranked: list[dict[str, Any]] = []

    for chunk in candidate_chunks:
        result = dict(chunk)

        doc_id = str(
            chunk.get("doc_id", "")
        ).strip().upper()

        retrieval_score = normalized.get(doc_id, 0.0)
        llm_score = relevance_by_id.get(doc_id, 0.5)

        final_score = (
            retrieval_weight * retrieval_score
            + llm_weight * llm_score
        )

        result["normalized_retrieval_score"] = retrieval_score
        result["llm_relevance_score"] = llm_score
        result["final_score"] = final_score

        reranked.append(result)

    reranked.sort(
        key=lambda chunk: float(
            chunk.get("final_score", 0.0)
        ),
        reverse=True,
    )

    strong_count = sum(
        float(chunk.get("final_score", 0.0)) >= threshold
        for chunk in reranked
    )

    target_count = max(
        min_results,
        min(strong_count, max_results),
    )

    target_count = min(
        target_count,
        len(reranked),
        max_results,
    )

    final = reranked[:target_count]

    for rank, chunk in enumerate(final, start=1):
        chunk["rank"] = rank
        chunk["passed_score_threshold"] = (
            float(chunk.get("final_score", 0.0))
            >= threshold
        )

    return final


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "avg_result_count": 0.0,
            "kb1_recall": 0.0,
            "kb2_recall": 0.0,
            "kb3_recall": 0.0,
            "exact_match": 0.0,
            "any_match": 0.0,
        }

    def mean(key: str) -> float:
        return sum(float(row[key]) for row in rows) / len(rows)

    return {
        "recall": mean("recall"),
        "precision": mean("precision"),
        "f1": mean("f1"),
        "avg_result_count": mean("result_count"),
        "kb1_recall": mean("kb1_recall"),
        "kb2_recall": mean("kb2_recall"),
        "kb3_recall": mean("kb3_recall"),
        "exact_match": mean("exact_match"),
        "any_match": mean("any_match"),
    }


def evaluate_config(
    cache_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    rows = []

    for cached in cache_rows:
        selected = offline_select(
            candidate_chunks=cached["candidate_chunks"],
            llm_scores=cached["llm_policy_scores"],
            threshold=float(config["threshold"]),
            min_results=int(config["min_results"]),
            max_results=int(config["max_results"]),
            retrieval_weight=float(
                config["retrieval_weight"]
            ),
        )

        predicted = get_ids(selected)
        expected = set(cached["expected_kb_ids"])

        metrics = calculate_metrics(
            expected,
            predicted,
        )

        expected_kb1 = set(cached["expected_kb1_ids"])
        expected_kb2 = set(cached["expected_kb2_ids"])
        expected_kb3 = set(cached["expected_kb3_ids"])

        rows.append(
            {
                "recall": metrics["recall"],
                "precision": metrics["precision"],
                "f1": metrics["f1"],
                "result_count": len(selected),
                "kb1_recall": family_recall(
                    expected_kb1,
                    family_ids(predicted, "KB1"),
                ),
                "kb2_recall": family_recall(
                    expected_kb2,
                    family_ids(predicted, "KB2"),
                ),
                "kb3_recall": family_recall(
                    expected_kb3,
                    family_ids(predicted, "KB3"),
                ),
                "exact_match": metrics["exact_match"],
                "any_match": metrics["any_match"],
            }
        )

    aggregate = aggregate_rows(rows)

    return {
        **config,
        **aggregate,
    }


def parameter_sweep(
    tune_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sweep: list[dict[str, Any]] = []

    for threshold in THRESHOLD_GRID:
        for min_results in MIN_RESULTS_GRID:
            for max_results in MAX_RESULTS_GRID:
                if min_results > max_results:
                    continue

                for retrieval_weight in RETRIEVAL_WEIGHT_GRID:
                    config = {
                        "threshold": threshold,
                        "min_results": min_results,
                        "max_results": max_results,
                        "retrieval_weight": retrieval_weight,
                        "llm_weight": 1.0 - retrieval_weight,
                    }

                    sweep.append(
                        evaluate_config(
                            tune_rows,
                            config,
                        )
                    )

    sweep.sort(
        key=lambda row: (
            row["recall"] >= RECALL_FLOOR,
            row["f1"],
            row["recall"],
            row["precision"],
        ),
        reverse=True,
    )

    return sweep


def choose_config(
    sweep: list[dict[str, Any]],
) -> dict[str, Any]:
    eligible = [
        row
        for row in sweep
        if row["recall"] >= RECALL_FLOOR
    ]

    pool = eligible if eligible else sweep

    return max(
        pool,
        key=lambda row: (
            row["f1"],
            row["recall"],
            row["precision"],
        ),
    )


def build_cache(
    contracts: list[dict[str, Any]],
    ground_truth: pd.DataFrame,
) -> list[dict[str, Any]]:
    agent = PolicyRetrievalAgent(
        top_k=FINAL_TOP_K,
        candidate_top_k=CANDIDATE_TOP_K,
        min_results=14,
        score_threshold=0.45,
        retrieval_weight=0.40,
        llm_weight=0.60,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        ngram_range=(1, 2),
        max_features=10000,
        kb_quotas=KB_QUOTAS,
        use_llm=True,
    )

    cache_rows: list[dict[str, Any]] = []

    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0

    for index, contract in enumerate(contracts, start=1):
        source_file = contract.get("source_file")
        contract_id = extract_contract_id(source_file)

        print(
            f"Evaluating {index}/{len(contracts)}: "
            f"{source_file} -> {contract_id}"
        )

        if not contract_id:
            continue

        gt_match = ground_truth[
            ground_truth["contract_id"] == contract_id
        ]

        if gt_match.empty:
            print("  No Ground Truth match found.")
            continue

        gt_row = gt_match.iloc[0]

        expected_kb1 = parse_kb_ids(
            gt_row.get("KB1 Chunks")
        )
        expected_kb2 = parse_kb_ids(
            gt_row.get("KB2 Chunks")
        )
        expected_kb3 = parse_kb_ids(
            gt_row.get("KB3 Chunks")
        )

        expected_ids = (
            expected_kb1
            | expected_kb2
            | expected_kb3
        )

        output = agent.run(contract)

        usage = output["llm_refinement"].get("usage")

        if usage:
            total_input_tokens += int(
                usage.get("input_tokens") or 0
            )
            total_output_tokens += int(
                usage.get("output_tokens") or 0
            )
            total_tokens += int(
                usage.get("total_tokens") or 0
            )

        candidate_chunks = output[
            "candidate_policy_chunks"
        ]

        # Store only what is needed for unlimited future offline tuning.
        compact_candidates = [
            {
                "doc_id": chunk.get("doc_id"),
                "kb_prefix": chunk.get("kb_prefix"),
                "score": chunk.get("score"),
                "best_single_query_score": chunk.get(
                    "best_single_query_score"
                ),
                "candidate_rank": chunk.get(
                    "candidate_rank"
                ),
            }
            for chunk in candidate_chunks
        ]

        cache_rows.append(
            {
                "source_file": source_file,
                "contract_id": contract_id,
                "split": deterministic_split(contract_id),

                "expected_kb_ids": sorted(expected_ids),
                "expected_kb1_ids": sorted(expected_kb1),
                "expected_kb2_ids": sorted(expected_kb2),
                "expected_kb3_ids": sorted(expected_kb3),

                "candidate_chunks": compact_candidates,
                "llm_policy_scores": output[
                    "llm_refinement"
                ].get("policy_scores", []),

                "llm_model": output[
                    "llm_refinement"
                ].get("model"),
                "llm_used": output[
                    "llm_refinement"
                ].get("used_llm"),
                "llm_error": output[
                    "llm_refinement"
                ].get("error"),
                "llm_reasoning": output[
                    "llm_refinement"
                ].get("reasoning"),
                "usage": usage,
            }
        )

    CACHE_PATH.write_text(
        json.dumps(
            {
                "metadata": {
                    "candidate_top_k": CANDIDATE_TOP_K,
                    "kb_quotas": KB_QUOTAS,
                    "tune_fraction": TUNE_FRACTION,
                    "split_salt": SPLIT_SALT,
                    "llm_calls": len(cache_rows),
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                    "total_tokens": total_tokens,
                },
                "contracts": cache_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\nReusable final-run cache: {CACHE_PATH}")
    print(
        "Token usage captured: "
        f"input={total_input_tokens}, "
        f"output={total_output_tokens}, "
        f"total={total_tokens}"
    )

    return cache_rows


def load_cache() -> list[dict[str, Any]]:
    data = json.loads(
        CACHE_PATH.read_text(encoding="utf-8")
    )

    return data["contracts"]


def final_rows_for_config(
    cache_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []

    for cached in cache_rows:
        selected = offline_select(
            candidate_chunks=cached["candidate_chunks"],
            llm_scores=cached["llm_policy_scores"],
            threshold=float(config["threshold"]),
            min_results=int(config["min_results"]),
            max_results=int(config["max_results"]),
            retrieval_weight=float(
                config["retrieval_weight"]
            ),
        )

        predicted = get_ids(selected)
        expected = set(cached["expected_kb_ids"])

        metrics = calculate_metrics(
            expected,
            predicted,
        )

        output_rows.append(
            {
                "source_file": cached["source_file"],
                "contract_id": cached["contract_id"],
                "split": cached["split"],

                "expected_kb_ids": sorted(expected),
                "predicted_kb_ids": sorted(predicted),

                "matched_kb_ids": metrics["matched"],
                "missing_kb_ids": metrics["missing"],
                "unexpected_kb_ids": metrics["unexpected"],

                "recall": metrics["recall"],
                "precision": metrics["precision"],
                "f1": metrics["f1"],
                "exact_match": metrics["exact_match"],
                "any_match": metrics["any_match"],

                "result_count": len(selected),

                "kb1_recall": family_recall(
                    set(cached["expected_kb1_ids"]),
                    family_ids(predicted, "KB1"),
                ),
                "kb2_recall": family_recall(
                    set(cached["expected_kb2_ids"]),
                    family_ids(predicted, "KB2"),
                ),
                "kb3_recall": family_recall(
                    set(cached["expected_kb3_ids"]),
                    family_ids(predicted, "KB3"),
                ),

                "ranked_policies": [
                    {
                        "rank": chunk.get("rank"),
                        "doc_id": chunk.get("doc_id"),
                        "kb_prefix": chunk.get("kb_prefix"),
                        "retrieval_score": chunk.get("score"),
                        "normalized_retrieval_score": chunk.get(
                            "normalized_retrieval_score"
                        ),
                        "llm_relevance_score": chunk.get(
                            "llm_relevance_score"
                        ),
                        "final_score": chunk.get("final_score"),
                        "passed_score_threshold": chunk.get(
                            "passed_score_threshold"
                        ),
                    }
                    for chunk in selected
                ],
            }
        )

    return output_rows


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Safety: once the paid cache exists, do not call the model again.
    if CACHE_PATH.exists():
        print(
            "Existing Agent 2 final-run cache found. "
            "Skipping all LLM/API calls."
        )
        cache_rows = load_cache()
    else:
        contracts = load_extracted_contracts()
        ground_truth = load_ground_truth()

        cache_rows = build_cache(
            contracts,
            ground_truth,
        )

    tune_rows = [
        row
        for row in cache_rows
        if row["split"] == "tune"
    ]

    holdout_rows = [
        row
        for row in cache_rows
        if row["split"] == "holdout"
    ]

    print(
        f"\nSplit: {len(tune_rows)} tuning contracts, "
        f"{len(holdout_rows)} held-out contracts."
    )

    if not tune_rows or not holdout_rows:
        raise ValueError(
            "Deterministic split produced an empty partition. "
            "Change SPLIT_SALT and rebuild the cache only if necessary."
        )

    sweep = parameter_sweep(tune_rows)

    pd.DataFrame(sweep).to_csv(
        SWEEP_PATH,
        index=False,
    )

    selected_config = choose_config(sweep)

    CONFIG_PATH.write_text(
        json.dumps(
            {
                "selection_basis": (
                    "Highest tuning-set F1 among configurations "
                    f"with recall >= {RECALL_FLOOR:.2f}; "
                    "falls back to best F1 if none meet floor."
                ),
                "selected_config": selected_config,
                "tune_contracts": len(tune_rows),
                "holdout_contracts": len(holdout_rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    tune_metrics = evaluate_config(
        tune_rows,
        selected_config,
    )

    holdout_metrics = evaluate_config(
        holdout_rows,
        selected_config,
    )

    final_rows = final_rows_for_config(
        cache_rows,
        selected_config,
    )

    FINAL_JSON_PATH.write_text(
        json.dumps(
            {
                "selected_config": selected_config,
                "tuning_metrics": tune_metrics,
                "heldout_metrics": holdout_metrics,
                "results": final_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    csv_rows = []

    for row in final_rows:
        csv_rows.append(
            {
                "source_file": row["source_file"],
                "contract_id": row["contract_id"],
                "split": row["split"],
                "expected_kb_ids": "; ".join(
                    row["expected_kb_ids"]
                ),
                "predicted_kb_ids": "; ".join(
                    row["predicted_kb_ids"]
                ),
                "matched_kb_ids": "; ".join(
                    row["matched_kb_ids"]
                ),
                "missing_kb_ids": "; ".join(
                    row["missing_kb_ids"]
                ),
                "unexpected_kb_ids": "; ".join(
                    row["unexpected_kb_ids"]
                ),
                "recall": row["recall"],
                "precision": row["precision"],
                "f1": row["f1"],
                "result_count": row["result_count"],
                "kb1_recall": row["kb1_recall"],
                "kb2_recall": row["kb2_recall"],
                "kb3_recall": row["kb3_recall"],
                "exact_match": row["exact_match"],
                "any_match": row["any_match"],
            }
        )

    pd.DataFrame(csv_rows).to_csv(
        FINAL_CSV_PATH,
        index=False,
    )

    best_overall_tune = max(
        sweep,
        key=lambda row: row["f1"],
    )

    print("\nAutomatic Offline Tuning Complete")
    print("---------------------------------")
    print(
        "Selected parameters: "
        f"threshold={selected_config['threshold']:.2f}, "
        f"min_results={selected_config['min_results']}, "
        f"max_results={selected_config['max_results']}, "
        f"retrieval_weight="
        f"{selected_config['retrieval_weight']:.2f}, "
        f"llm_weight={selected_config['llm_weight']:.2f}"
    )

    print("\nTuning-set performance")
    print("----------------------")
    print(
        f"Recall: {tune_metrics['recall']:.3f}"
    )
    print(
        f"Precision: {tune_metrics['precision']:.3f}"
    )
    print(
        f"F1: {tune_metrics['f1']:.3f}"
    )
    print(
        "Average policies returned: "
        f"{tune_metrics['avg_result_count']:.1f}"
    )

    print("\nHELD-OUT FINAL PERFORMANCE")
    print("--------------------------")
    print(
        f"Recall: {holdout_metrics['recall']:.3f}"
    )
    print(
        f"Precision: {holdout_metrics['precision']:.3f}"
    )
    print(
        f"F1: {holdout_metrics['f1']:.3f}"
    )
    print(
        f"KB1 Recall: {holdout_metrics['kb1_recall']:.3f}"
    )
    print(
        f"KB2 Recall: {holdout_metrics['kb2_recall']:.3f}"
    )
    print(
        f"KB3 Recall: {holdout_metrics['kb3_recall']:.3f}"
    )
    print(
        "Average policies returned: "
        f"{holdout_metrics['avg_result_count']:.1f}"
    )
    print(
        "Exact-match accuracy: "
        f"{holdout_metrics['exact_match']:.3f}"
    )
    print(
        "At-least-one-policy accuracy: "
        f"{holdout_metrics['any_match']:.3f}"
    )

    print("\nSaved artifacts")
    print("---------------")
    print(f"Reusable cache: {CACHE_PATH}")
    print(f"Parameter sweep: {SWEEP_PATH}")
    print(f"Selected config: {CONFIG_PATH}")
    print(f"Final JSON: {FINAL_JSON_PATH}")
    print(f"Final CSV: {FINAL_CSV_PATH}")

    print(
        "\nFuture reruns of this evaluator will detect the cache "
        "and make ZERO LLM/API calls."
    )


if __name__ == "__main__":
    main()
