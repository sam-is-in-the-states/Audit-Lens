import os

from app.vector_db.services import VectorDBService

BASE_PATH = os.path.join("documents", "knowledge_base")

svc = VectorDBService()
svc.create_db_from_jsonl(
    jsonl_path=os.path.join(BASE_PATH, "KB1_asc606_guidance.jsonl"),
    collection_name="asc606_guidance",
    text_field="text",
    id_field="id",
    overwrite=True,
)
svc.create_db_from_jsonl(
    jsonl_path=os.path.join(BASE_PATH, "KB2_revlens_policy.jsonl"),
    collection_name="revlens_policy",
    text_field="text",
    id_field="id",
    overwrite=True,
)
print("\nCollections:", svc.list_collections())
print("asc606_guidance docs :", svc.get_collection_count("asc606_guidance"))
print("revlens_policy docs  :", svc.get_collection_count("revlens_policy"))
 
# ── 4. Query DB 1 ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Query → asc606_guidance")
print("=" * 60)
 
results = svc.query(
    collection_name="asc606_guidance",
    query_text="How do I determine whether goods or services are distinct performance obligations?",
    top_k=3,
)
 
for rank, r in enumerate(results, 1):
    print(f"\n[{rank}] id={r['id']}  cosine_distance={r['score']}")
    print(f"    name   : {r['metadata'].get('name', '')}")
    print(f"    topic  : {r['metadata'].get('topic', '')}")
    print(f"    asc_ref: {r['metadata'].get('asc_ref', '')}")
    print(f"    text   : {r['text'][:220]}...")
 
# ── 5. Query DB 2 ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Query → revlens_policy")
print("=" * 60)
 
results = svc.query(
    collection_name="revlens_policy",
    query_text="When does RevLens recognise onboarding or implementation fees?",
    top_k=3,
)
 
for rank, r in enumerate(results, 1):
    print(f"\n[{rank}] id={r['id']}  cosine_distance={r['score']}")
    print(f"    name        : {r['metadata'].get('name', '')}")
    print(f"    topic       : {r['metadata'].get('topic', '')}")
    print(f"    related_asc : {r['metadata'].get('related_asc', '')}")
    print(f"    text        : {r['text'][:220]}...")

