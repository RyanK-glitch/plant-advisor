"""Retrieval evaluation for RegulatoryRAG: does the right disease record come back?

Gold labels need no hand labelling: every knowledge-base chunk carries a record_id
(ADV-001 ... ADV-014) - PDFs via "Record ID", spreadsheet rows via their record_id column -
and the PPQS/CIBRC table links each record_id to one disease_key. A retrieved chunk is
RELEVANT when its record_id is the record of the disease being asked about.

Queries: 14 diseases x 4 phrasings (the app's own query plus three farmer-style questions).

Reported for three retrieval paths:
    semantic      - plain similarity search over all chunks (k = 1..10)
    filtered      - the metadata-filtered branch (disease_key filter, k=2)
    app pipeline  - query_knowledge_base() exactly as app.py calls it (filtered + semantic + post-filter)
Metrics: Hit@k, Recall@k, Precision@k, MRR, nDCG@k, and for the app pipeline the share of
returned records that belong to ANOTHER disease (cross-disease contamination) and the
share of queries with no evidence at all (the app then refuses to write a report).

Usage (Mac venv, after `python src/rag_pipeline.py` has built chroma_db):
    python scripts/evaluation/rag_retrieval_eval.py
Outputs: reports/rag_eval/retrieval_per_query.csv, retrieval_summary.csv, retrieval_at_k.png
"""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
from src.rag_pipeline import RegulatoryRAG  # noqa: E402

OUT = PROJECT_DIR / "reports/rag_eval"
TABLE = PROJECT_DIR / "knowledge_base/India_PPQS_CIBRC_Verified_Treatment_Table.xlsx"
KS = (1, 2, 3, 4, 5, 10)


def queries_for(crop, condition, disease_key):
    plain = condition.lower()
    return {
        "app_query": (f"{crop} {disease_key.replace('_', ' ')} symptoms management "
                      "verified chemical dose water timing PHI re-entry PPQS CIBRC"),
        "symptoms": f"What does {plain} look like on {crop} leaves?",
        "management": f"How do I control and prevent {plain} in my {crop} field?",
        "treatment": f"Which fungicide and dose is approved in India for {plain} of {crop}?",
    }


def ranking_metrics(retrieved_ids, gold, n_relevant):
    rels = [1 if rid == gold else 0 for rid in retrieved_ids]
    out = {}
    first = next((i for i, r in enumerate(rels) if r), None)
    out["rr"] = 0.0 if first is None else 1.0 / (first + 1)
    for k in KS:
        top = rels[:k]
        out[f"hit@{k}"] = float(any(top))
        out[f"precision@{k}"] = sum(top) / k
        out[f"recall@{k}"] = sum(top) / n_relevant if n_relevant else 0.0
        dcg = sum(r / math.log2(i + 2) for i, r in enumerate(top))
        ideal = sum(1 / math.log2(i + 2) for i in range(min(k, n_relevant)))
        out[f"ndcg@{k}"] = dcg / ideal if ideal else 0.0
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rag = RegulatoryRAG()
    if rag.vector_store is None:
        sys.exit("chroma_db not found. Build it first: python src/rag_pipeline.py")

    table = pd.read_excel(TABLE, sheet_name="Regulatory_Treatments")
    diseases = table[["record_id", "crop", "condition", "disease_key"]].dropna()

    # How many chunks belong to each record (denominator for recall)
    stored = rag.vector_store.get(include=["metadatas"])["metadatas"]
    per_record = pd.Series([m.get("record_id", "") for m in stored]).value_counts()
    print(f"Chunks in store: {len(stored)}; with a record_id: {int(per_record.drop('', errors='ignore').sum())}")

    rows = []
    for _, d in diseases.iterrows():
        crop, gold, key = d.crop.lower().split(" ")[0], d.record_id, d.disease_key
        n_rel = int(per_record.get(gold, 0))
        for qtype, query in queries_for(crop, d.condition, key).items():
            semantic = rag.vector_store.similarity_search(query, k=max(KS))
            sem_ids = [doc.metadata.get("record_id", "") for doc in semantic]
            filtered = rag.vector_store.similarity_search(query, k=2, filter={"disease_key": key})
            fil_ids = [doc.metadata.get("record_id", "") for doc in filtered]
            app = rag.query_knowledge_base(query, crop_key=crop, disease_key=key, k=4)
            app_ids = [r["metadata"].get("record_id", "") for r in app["records"]]

            row = {"disease_key": key, "record_id": gold, "query_type": qtype, "query": query,
                   "relevant_chunks": n_rel, "semantic_top5": ";".join(sem_ids[:5]),
                   **{f"semantic_{m}": v for m, v in ranking_metrics(sem_ids, gold, n_rel).items()},
                   "filtered_returned": len(fil_ids),
                   "filtered_precision": (sum(i == gold for i in fil_ids) / len(fil_ids)) if fil_ids else float("nan"),
                   "app_returned": len(app_ids),
                   "app_hit": float(gold in app_ids),
                   "app_precision": (sum(i == gold for i in app_ids) / len(app_ids)) if app_ids else float("nan"),
                   "app_other_disease_records": sum(1 for i in app_ids if i and i != gold),
                   "app_no_evidence": float(not app["context"]),
                   "app_has_verified": float(app["has_verified"])}
            rows.append(row)

    per_query = pd.DataFrame(rows)
    per_query.to_csv(OUT / "retrieval_per_query.csv", index=False)

    summary = {"queries": len(per_query), "semantic_MRR": per_query.semantic_rr.mean()}
    for k in KS:
        for m in ("hit", "recall", "precision", "ndcg"):
            summary[f"semantic_{m}@{k}"] = per_query[f"semantic_{m}@{k}"].mean()
    summary.update({
        "filtered_precision": per_query.filtered_precision.mean(),
        "app_hit_rate": per_query.app_hit.mean(),
        "app_context_precision": per_query.app_precision.mean(),
        "app_queries_with_other_disease_records": (per_query.app_other_disease_records > 0).mean(),
        "app_no_evidence_rate": per_query.app_no_evidence.mean(),
        "app_has_verified_rate": per_query.app_has_verified.mean(),
    })
    pd.Series(summary).to_csv(OUT / "retrieval_summary.csv", header=["value"])
    print(pd.Series(summary).round(4).to_string())
    print("\nBy query type (semantic MRR, app hit):")
    print(per_query.groupby("query_type")[["semantic_rr", "semantic_hit@4", "app_hit", "app_precision"]].mean().round(3))

    fig, ax = plt.subplots(figsize=(6, 4))
    for m in ("hit", "recall", "ndcg"):
        ax.plot(KS, [summary[f"semantic_{m}@{k}"] for k in KS], marker="o", label=f"{m.upper()}@k")
    ax.set_xlabel("k (chunks retrieved)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Semantic retrieval quality (56 queries)", fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "retrieval_at_k.png", dpi=200)
    print(f"\nSaved results in {OUT}")


if __name__ == "__main__":
    main()
