"""Evaluate Agent 5 (Memo Agent).

The Memo Agent assembles a deliverable, so there is no ground-truth memo to score against.
Its correctness is checked with programmatic integrity tests instead:
  - journal entries balance (total debits == total credits)
  - the recognition schedule totals to the transaction price
  - the disposition follows from the review-point severities
  - the LLM narrative passed the hallucination/citation guard (else it fell back to template)
plus a portfolio view (how many contracts can be booked vs need review).

It reads the cached upstream outputs (Agent 1 / 3 / 4); it does not re-run the pipeline.

    py evaluate_agent5.py               # run with the LLM narrative
    py evaluate_agent5.py --no-llm      # template prose only (no LLM calls)
    py evaluate_agent5.py --limit 5
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from app.llm.memo_agent import (MemoAgent, portfolio_summary, _num, determine_disposition,
                                materiality_threshold)

EXTRACTED = ROOT / "app" / "llm" / "extracted_contracts.json"
TREATMENT = HERE / "results" / "treatment_outputs.json"
AUDIT = HERE / "results" / "audit_agent_results.json"
OUT_JSON = HERE / "results" / "memo_outputs.json"
OUT_MEMOS = HERE / "results" / "memos.txt"


def _cid(name):
    return (name or "").split("_")[0].upper()


def _load():
    ext = json.loads(EXTRACTED.read_text(encoding="utf-8"))
    treat = json.loads(TREATMENT.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    a1 = {_cid(r.get("file_name", "")): r for r in ext}
    a3 = {_cid(o.get("source_file", "")): o for o in treat}
    a4 = {(_cid(r.get("source_file", "")) or r.get("contract_id")): r for r in audit}
    return a1, a3, a4


def _je_balances(result) -> bool:
    for entry in result.get("journal_entries", []):
        dr = sum(l["amount"] for l in entry["lines"] if l.get("dr") and l.get("amount"))
        cr = sum(l["amount"] for l in entry["lines"] if l.get("cr") and l.get("amount"))
        if (dr or cr) and abs(dr - cr) > 0.01:
            return False
    return True


def _schedule_ok(result) -> bool:
    rows = result.get("schedule", {}).get("rows") or []
    if not rows:
        return True  # no schedule (e.g. missing term) is not a failure
    price = _num(result["treatment"].get("transaction_price")) or 0.0
    total = sum(r["revenue_recognised"] for r in rows)
    return abs(total - price) <= 1.0


def _disposition_ok(result) -> bool:
    return result["disposition"] == determine_disposition(result.get("review_points", []))


def run_all(limit=None, use_llm=True):
    agent = MemoAgent(use_llm=use_llm)
    a1, a3, a4 = _load()
    contract_ids = sorted(a3)
    if limit is not None:
        contract_ids = contract_ids[:limit]

    results = []
    for cid in contract_ids:
        rec = a1.get(cid) or {}
        facts = rec.get("validated_output") or rec.get("candidate_output") or {}
        facts = {**facts, "contract_id": cid, "source_file": rec.get("file_name")}
        agent1 = {**rec, "validated_output": facts}
        out = agent.run(agent1, a3.get(cid) or {}, a4.get(cid) or {})
        results.append(out)
        print(f"  {cid}: {out['disposition']:<26} "
              f"review_points={len(out['review_points'])} narrative={out.get('narrative_source')}")

    # persist: keep the full memo text in memos.txt, the rest in json
    OUT_JSON.write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "memo"} for r in results], indent=2, default=str),
        encoding="utf-8")
    OUT_MEMOS.write_text("\n\n\n".join(r["memo"] for r in results), encoding="utf-8")
    print(f"\nWrote {OUT_JSON} and {OUT_MEMOS}")
    _report(results)
    return results


def _report(results):
    n = len(results)
    assessable = [r for r in results if r["disposition"] != "cannot_assess"]
    denom = len(assessable) or 1
    je_ok = sum(1 for r in assessable if _je_balances(r))
    sch_ok = sum(1 for r in assessable if _schedule_ok(r))
    disp_ok = sum(1 for r in assessable if _disposition_ok(r))

    print("\n=== AGENT 5 - memo integrity checks (must stay 50/50) ===")
    print(f"  journal entries balance (Dr=Cr):        {je_ok}/{denom}")
    print(f"  schedule totals to transaction price:   {sch_ok}/{denom}")
    print(f"  disposition follows severities:         {disp_ok}/{denom}")

    # disposition by count AND by transaction price
    summ = portfolio_summary(assessable)
    total_tp = summ["total_transaction_price"] or 1.0
    print("\n=== disposition distribution (by count and by transaction price) ===")
    print(f"  {'disposition':<28}{'count':>7}{'amount':>16}{'% $':>8}")
    for disp in ("cannot_assess", "escalate", "review_before_recognition", "review_before_close", "proceed"):
        info = summ["by_disposition"].get(disp)
        cnt = Counter(r["disposition"] for r in results).get(disp, 0)
        amt = info["amount"] if info else 0.0
        if cnt:
            amt_s = f"${amt:,.0f}"
            print(f"  {disp:<28}{cnt:>7}{amt_s:>16}{100 * amt / total_tp:>7.1f}%")
    print(f"  portfolio: total price ${summ['total_transaction_price']:,.0f}  |  "
          f"recognisable ${summ['total_revenue_recognisable']:,.0f}")
    print(f"  opening deferred ${summ['opening_deferred_total']:,.0f}  |  "
          f"deferred at reporting date ${summ['deferred_at_reporting_date']:,.0f}")
    exc = summ["excluded_from_schedule"]
    if exc["count"]:
        print(f"  excluded from schedule (variable/usage): ${exc['amount']:,.0f} "
              f"across {exc['count']} contract(s): {', '.join(exc['contracts'])}")

    # materiality outcomes
    immat = [(r["contract_id"], p) for r in results for p in r.get("immaterial_matters", [])]
    print(f"\n=== materiality: {len(immat)} matters concluded immaterial ===")
    for kb, c in sorted(Counter(p["kb3_id"] for _, p in immat).items()):
        print(f"  {kb:<8} {c}")

    # LLM narrative pass rate + rejection-reason breakdown
    reasons = Counter(r.get("narrative_reject_reason") for r in results)
    llm = reasons.get("ok", 0)
    print(f"\n=== LLM narrative: {llm}/{n} passed the guard ===")
    for reason, c in sorted(reasons.items()):
        if reason and reason != "ok":
            print(f"  fell back ({reason}): {c}")

    print("\n=== review workload by KB3 item, ranked by exposure ($) ===")
    for kb, info in list(summ["by_review_item"].items())[:8]:
        print(f"  {kb:<8} {info['count']} contracts  ${info['amount']:,.0f}")

    fails = [r["contract_id"] for r in assessable
             if not (_je_balances(r) and _schedule_ok(r) and _disposition_ok(r))]
    if fails:
        print(f"\n  integrity failures: {fails}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-llm", action="store_true", help="template prose only, no LLM calls")
    args = ap.parse_args()
    run_all(limit=args.limit, use_llm=not args.no_llm)
