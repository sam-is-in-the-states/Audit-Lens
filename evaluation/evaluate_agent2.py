import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.llm.policy_retrieval import PolicyRetrievalAgent

GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "GroundTruth.xlsx"
)

EXTRACTED_CONTRACTS_PATH = (
    PROJECT_ROOT
    / "app"
    / "llm"
    / "extracted_contracts.json"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)

# Retrieval settings
TOP_K = 20
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75
KB_QUOTAS = {
    "KB1": 8,
    "KB2": 7,
    "KB3": 5,
}

# Load Agent 1 extracted contract outputs
def load_extracted_contracts() -> list[dict[str, Any]]:
    if not EXTRACTED_CONTRACTS_PATH.exists():
        raise FileNotFoundError(
            "Agent 1 output not found: "
            f"{EXTRACTED_CONTRACTS_PATH}"
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

# Load GroundTruth.xlsx using the actual fourth row as the header
def load_ground_truth() -> pd.DataFrame:
    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(
            f"Ground truth file not found: {GROUND_TRUTH_PATH}"
        )

    ground_truth = pd.read_excel(
        GROUND_TRUTH_PATH,
        header=3,
    )

    ground_truth = ground_truth.dropna(
        axis=0,
        how="all",
    )
    ground_truth = ground_truth.dropna(
        axis=1,
        how="all",
    )

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

    missing_columns = required_columns - set(
        ground_truth.columns
    )

    if missing_columns:
        raise ValueError(
            "Ground truth is missing required columns: "
            f"{sorted(missing_columns)}\n"
            f"Available columns: {ground_truth.columns.tolist()}"
        )

    ground_truth["contract_id"] = (
        ground_truth["contract_id"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    print(
        f"Loaded {len(ground_truth)} ground-truth contracts."
    )

    return ground_truth

# Extract C01, C02, etc. from filenames such as C01_Brightwater.pdf
def extract_ground_truth_contract_id(
    source_file: str | None,
) -> str | None:
    if not source_file:
        return None

    match = re.match(
        r"(C\d+)",
        Path(source_file).stem,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return match.group(1).upper()

# Parse a semicolon-separated Ground Truth KB cell into IDs
def parse_kb_ids(value: Any) -> set[str]:
    if pd.isna(value):
        return set()

    text = str(value).strip()

    if not text:
        return set()

    if text.lower().startswith("(none"):
        return set()

    matches = re.findall(
        r"\bKB\d+_\d+\b",
        text,
        flags=re.IGNORECASE,
    )

    return {
        match.upper()
        for match in matches
    }

# Convert retrieved chunks into unique policy document IDs
def get_retrieved_ids(
    retrieved_chunks: list[dict[str, Any]],
) -> set[str]:
    return {
        str(chunk.get("doc_id", "")).strip().upper()
        for chunk in retrieved_chunks
        if str(chunk.get("doc_id", "")).strip()
    }

# Calculate retrieval metrics for one contract
def calculate_metrics(
    expected_ids: set[str],
    retrieved_ids: set[str],
) -> dict[str, Any]:
    matched_ids = expected_ids & retrieved_ids
    missing_ids = expected_ids - retrieved_ids
    unexpected_ids = retrieved_ids - expected_ids

    if expected_ids:
        recall = len(matched_ids) / len(expected_ids)
    else:
        recall = 1.0 if not retrieved_ids else 0.0

    if retrieved_ids:
        precision = len(matched_ids) / len(retrieved_ids)
    else:
        precision = 1.0 if not expected_ids else 0.0

    if recall + precision > 0:
        f1 = 2 * recall * precision / (recall + precision)
    else:
        f1 = 0.0

    return {
        "expected_kb_ids": sorted(expected_ids),
        "retrieved_kb_ids": sorted(retrieved_ids),
        "matched_kb_ids": sorted(matched_ids),
        "missing_kb_ids": sorted(missing_ids),
        "unexpected_kb_ids": sorted(unexpected_ids),
        "recall_at_k": recall,
        "precision_at_k": precision,
        "f1_at_k": f1,
        "exact_match": retrieved_ids == expected_ids,
        "any_match": bool(matched_ids),
    }

# Calculate metrics for one KB family
def calculate_kb_family_metrics(
    expected_ids: set[str],
    retrieved_ids: set[str],
) -> dict[str, float]:
    matched = expected_ids & retrieved_ids

    recall = (
        len(matched) / len(expected_ids)
        if expected_ids
        else 1.0
    )

    precision = (
        len(matched) / len(retrieved_ids)
        if retrieved_ids
        else (1.0 if not expected_ids else 0.0)
    )

    return {
        "recall": recall,
        "precision": precision,
    }

# Evaluate all contracts
def main() -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    contracts = load_extracted_contracts()
    ground_truth = load_ground_truth()

    agent = PolicyRetrievalAgent(
        top_k=TOP_K,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        ngram_range=(1, 2),
        max_features=10000,
        kb_quotas=KB_QUOTAS,
    )

    results: list[dict[str, Any]] = []
    unmatched_contracts: list[str] = []

    for index, contract in enumerate(
        contracts,
        start=1,
    ):
        source_file = contract.get("source_file")
        gt_contract_id = extract_ground_truth_contract_id(
            source_file
        )

        print(
            f"Evaluating {index}/{len(contracts)}: "
            f"{source_file} -> {gt_contract_id}"
        )

        if not gt_contract_id:
            unmatched_contracts.append(
                str(source_file)
            )
            continue

        gt_match = ground_truth[
            ground_truth["contract_id"]
            == gt_contract_id
        ]

        if gt_match.empty:
            print("  No Ground Truth match found.")
            unmatched_contracts.append(
                str(source_file)
            )
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

        agent2_output = agent.run(contract)
        retrieved_chunks = agent2_output[
            "retrieved_policy_chunks"
        ]
        retrieved_ids = get_retrieved_ids(
            retrieved_chunks
        )

        retrieved_kb1 = {
            policy_id
            for policy_id in retrieved_ids
            if policy_id.startswith("KB1_")
        }
        retrieved_kb2 = {
            policy_id
            for policy_id in retrieved_ids
            if policy_id.startswith("KB2_")
        }
        retrieved_kb3 = {
            policy_id
            for policy_id in retrieved_ids
            if policy_id.startswith("KB3_")
        }

        metrics = calculate_metrics(
            expected_ids=expected_ids,
            retrieved_ids=retrieved_ids,
        )

        kb1_metrics = calculate_kb_family_metrics(
            expected_kb1,
            retrieved_kb1,
        )
        kb2_metrics = calculate_kb_family_metrics(
            expected_kb2,
            retrieved_kb2,
        )
        kb3_metrics = calculate_kb_family_metrics(
            expected_kb3,
            retrieved_kb3,
        )

        result = {
            "source_file": source_file,
            "ground_truth_contract_id": gt_contract_id,
            "top_k": agent.top_k,
            "chunk_size": agent.chunk_size,
            "chunk_overlap": agent.chunk_overlap,
            "kb_quotas": agent.kb_quotas,
            "expected_kb1_ids": sorted(expected_kb1),
            "expected_kb2_ids": sorted(expected_kb2),
            "expected_kb3_ids": sorted(expected_kb3),
            **metrics,
            "kb1_recall": kb1_metrics["recall"],
            "kb1_precision": kb1_metrics["precision"],
            "kb2_recall": kb2_metrics["recall"],
            "kb2_precision": kb2_metrics["precision"],
            "kb3_recall": kb3_metrics["recall"],
            "kb3_precision": kb3_metrics["precision"],
            "retrieved_chunk_ids": [
                chunk["chunk_id"]
                for chunk in retrieved_chunks
            ],
            "retrieved_scores": [
                chunk["score"]
                for chunk in retrieved_chunks
            ],
            "agent2_compact_output": {
                "KB1 Chunks": agent2_output[
                    "KB1 Chunks"
                ],
                "KB2 Chunks": agent2_output[
                    "KB2 Chunks"
                ],
                "KB3 Chunks": agent2_output[
                    "KB3 Chunks"
                ],
            },
        }

        results.append(result)

    # Save detailed JSON results
    json_output_path = (
        RESULTS_DIR
        / "policy_retrieval_results.json"
    )

    json_output_path.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Save compact CSV for easier review
    csv_rows = []

    for result in results:
        csv_rows.append(
            {
                "source_file": result["source_file"],
                "ground_truth_contract_id": (
                    result["ground_truth_contract_id"]
                ),
                "top_k": result["top_k"],
                "expected_kb_ids": "; ".join(
                    result["expected_kb_ids"]
                ),
                "retrieved_kb_ids": "; ".join(
                    result["retrieved_kb_ids"]
                ),
                "matched_kb_ids": "; ".join(
                    result["matched_kb_ids"]
                ),
                "missing_kb_ids": "; ".join(
                    result["missing_kb_ids"]
                ),
                "unexpected_kb_ids": "; ".join(
                    result["unexpected_kb_ids"]
                ),
                "recall_at_k": result["recall_at_k"],
                "precision_at_k": result["precision_at_k"],
                "f1_at_k": result["f1_at_k"],
                "kb1_recall": result["kb1_recall"],
                "kb1_precision": result["kb1_precision"],
                "kb2_recall": result["kb2_recall"],
                "kb2_precision": result["kb2_precision"],
                "kb3_recall": result["kb3_recall"],
                "kb3_precision": result["kb3_precision"],
                "exact_match": result["exact_match"],
                "any_match": result["any_match"],
            }
        )

    csv_output_path = (
        RESULTS_DIR
        / "policy_retrieval_summary.csv"
    )

    pd.DataFrame(csv_rows).to_csv(
        csv_output_path,
        index=False,
    )

    # Print aggregate metrics
    if results:
        average_recall = sum(
            result["recall_at_k"]
            for result in results
        ) / len(results)

        average_precision = sum(
            result["precision_at_k"]
            for result in results
        ) / len(results)

        average_f1 = sum(
            result["f1_at_k"]
            for result in results
        ) / len(results)

        average_kb1_recall = sum(
            result["kb1_recall"]
            for result in results
        ) / len(results)

        average_kb2_recall = sum(
            result["kb2_recall"]
            for result in results
        ) / len(results)

        average_kb3_recall = sum(
            result["kb3_recall"]
            for result in results
        ) / len(results)

        exact_match_accuracy = sum(
            result["exact_match"]
            for result in results
        ) / len(results)

        any_match_accuracy = sum(
            result["any_match"]
            for result in results
        ) / len(results)

        print("\nAgent 2 Evaluation Summary")
        print("--------------------------")
        print(f"Evaluated contracts: {len(results)}")
        print(
            f"Average Recall@{TOP_K}: "
            f"{average_recall:.3f}"
        )
        print(
            f"Average Precision@{TOP_K}: "
            f"{average_precision:.3f}"
        )
        print(
            f"Average F1@{TOP_K}: "
            f"{average_f1:.3f}"
        )
        print(
            f"Average KB1 Recall: "
            f"{average_kb1_recall:.3f}"
        )
        print(
            f"Average KB2 Recall: "
            f"{average_kb2_recall:.3f}"
        )
        print(
            f"Average KB3 Recall: "
            f"{average_kb3_recall:.3f}"
        )
        print(
            "Exact-match accuracy: "
            f"{exact_match_accuracy:.3f}"
        )
        print(
            "At-least-one-policy accuracy: "
            f"{any_match_accuracy:.3f}"
        )

    if unmatched_contracts:
        print("\nUnmatched contracts")
        print("-------------------")

        for source_file in unmatched_contracts:
            print(source_file)

    print(f"\nDetailed results: {json_output_path}")
    print(f"Summary CSV: {csv_output_path}")

if __name__ == "__main__":
    main()