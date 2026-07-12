import json
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).parent
RES = ROOT / "results" / "audit_agent_results.json"
GT = ROOT / "GroundTruth.xlsx"

if not RES.exists():
    print("Missing results file:", RES)
    raise SystemExit(1)

data = json.loads(RES.read_text(encoding='utf-8'))

pred_count = Counter()
exp_count = Counter()
true_pos = defaultdict(int)
false_pos = defaultdict(int)
false_neg = defaultdict(int)

for rec in data:
    expected = set(rec.get('expected_kb3_ids', []))
    predicted = set(rec.get('retrieved_kb3_ids', []))

    for p in predicted:
        pred_count[p] += 1
    for e in expected:
        exp_count[e] += 1

    for p in predicted:
        if p in expected:
            true_pos[p] += 1
        else:
            false_pos[p] += 1
    for e in expected:
        if e not in predicted:
            false_neg[e] += 1

# Aggregate
all_ids = sorted(set(list(pred_count.keys()) + list(exp_count.keys())))

summary = {
    'total_contracts': len(data),
    'per_id': {}
}

for kid in all_ids:
    summary['per_id'][kid] = {
        'expected_count': exp_count.get(kid, 0),
        'predicted_count': pred_count.get(kid, 0),
        'TP': true_pos.get(kid, 0),
        'FP': false_pos.get(kid, 0),
        'FN': false_neg.get(kid, 0),
    }

# Top false positives
top_fp = sorted(summary['per_id'].items(), key=lambda kv: kv[1]['FP'], reverse=True)[:10]
# Top false negatives
top_fn = sorted(summary['per_id'].items(), key=lambda kv: kv[1]['FN'], reverse=True)[:10]

out = {
    'top_false_positives': top_fp,
    'top_false_negatives': top_fn,
    'summary': summary,
}

OUT_PATH = ROOT / 'results' / 'audit_agent_analysis.json'
OUT_PATH.write_text(json.dumps(out, indent=2), encoding='utf-8')
print('Wrote analysis ->', OUT_PATH)
print('\nTop false positives:')
for kid, stats in top_fp:
    print(kid, stats)
print('\nTop false negatives:')
for kid, stats in top_fn:
    print(kid, stats)
