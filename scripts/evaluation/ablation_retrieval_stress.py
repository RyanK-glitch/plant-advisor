"""Retrieval-failure stress test: does the rule layer (C) protect the advice when RAG goes wrong?

In the main A/B/C ablation every card gets correct retrieval, so B (RAG only) and C (RAG + rules)
tie. Here retrieval is deliberately broken and B and C are compared again:

    none   retrieval returns nothing (empty or missing knowledge-base entry)
    wrong  retrieval returns the evidence for a DIFFERENT disease of the same crop
           (e.g. rice_leaf_blast evidence for a rice_brown_spot card)

The safety rule in C always comes from the TRUE label, as in the app, where the rule is looked up
from the classifier label and not from retrieval. So only the retrieval step fails.

Same 70 cards and prompts as ablation_abc.py. Each mode is written to its own folder so it can be
scored on its own; the script skips cards already done and can be restarted.

Usage:
    python scripts/evaluation/ablation_retrieval_stress.py --modes none --limit 15
    python scripts/evaluation/ablation_retrieval_stress.py                  # both modes, all 70 cards
    python scripts/evaluation/score_compliance.py --dir reports/ablation_stress/none
    python scripts/evaluation/score_compliance.py --dir reports/ablation_stress/wrong
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation_abc import CLASSES, CONFIDENCE, PROJECT_DIR, QUESTIONS, load_api_key  # noqa: E402
from src.advisory_prompt import build_rag_only_prompt, build_report_prompt, with_farmer_question  # noqa: E402
from src.gemini_client import generate_with_fallback  # noqa: E402
from src.rag_pipeline import RegulatoryRAG  # noqa: E402
from src.rule_engine import AgriculturalRuleEngine  # noqa: E402

OUT = PROJECT_DIR / "reports/ablation_stress"
NO_EVIDENCE = "No evidence was retrieved for this disease."


def wrong_disease(disease_key):
    """Another disease of the same crop: the next one in CLASSES order, skipping healthy classes."""
    crop = disease_key.split("_")[0]
    diseases = [c for c in CLASSES if c.startswith(crop + "_") and not c.endswith("_healthy")]
    others = [d for d in diseases if d != disease_key]
    if disease_key in diseases:
        return others[diseases.index(disease_key) % len(others)]
    return others[0]


def retrieve(rag, disease_key):
    crop = disease_key.split("_")[0]
    query = (f"{crop} {disease_key.replace('_', ' ')} symptoms management "
             "verified chemical dose water timing PHI re-entry PPQS CIBRC")
    return rag.query_knowledge_base(query, crop_key=crop, disease_key=disease_key, k=4)["context"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default="none,wrong")
    parser.add_argument("--limit", type=int, default=0, help="only the first N cards (0 = all 70)")
    parser.add_argument("--language", default="English")
    parser.add_argument("--sleep", type=float, default=4.0, help="seconds between calls (free-tier limits)")
    args = parser.parse_args()

    api_key = load_api_key()
    if not api_key:
        sys.exit("GEMINI_API_KEY not found in the environment or .env")

    rag, engine = RegulatoryRAG(), AgriculturalRuleEngine()
    cards = [(f"{key}__{q}", key, q) for key in CLASSES for q in QUESTIONS]
    if args.limit:
        cards = cards[:args.limit]

    for mode in args.modes.split(","):
        out = OUT / mode
        out.mkdir(parents=True, exist_ok=True)
        results = out / "ablation_reports.jsonl"
        done = set()
        if results.is_file():
            for line in results.read_text().splitlines():
                r = json.loads(line)
                if r.get("report"):  # failed calls (e.g. quota) are retried on the next run
                    done.add((r["card_id"], r["condition"]))

        for card_id, disease_key, qtype in cards:
            question = QUESTIONS[qtype]
            retrieved_for = wrong_disease(disease_key) if mode == "wrong" else None
            context = retrieve(rag, retrieved_for) if retrieved_for else NO_EVIDENCE
            rule = engine.evaluate(disease_key.split("_")[0], disease_key)

            prompts = {
                "B": build_rag_only_prompt(disease_key, CONFIDENCE, context, args.language, question),
                "C": with_farmer_question(
                    build_report_prompt(disease_key, CONFIDENCE, context, rule, "", args.language), question),
            }
            for condition in "BC":
                if (card_id, condition) in done:
                    continue
                record = {"card_id": card_id, "disease_key": disease_key, "question_type": qtype,
                          "question": question, "condition": condition, "language": args.language,
                          "retrieval_mode": mode, "retrieved_for": retrieved_for,
                          "rule_status": rule.get("engine_status"),
                          "rule_regulatory_status": rule.get("regulatory_status"),
                          "rule_chemical": rule.get("chemical"), "rule_dose": rule.get("dose"),
                          "context_chars": len(context)}
                try:
                    text, model, failures = generate_with_fallback(api_key, prompts[condition])
                    record.update({"report": text, "model": model, "error": None})
                except Exception as error:
                    record.update({"report": "", "model": None, "error": f"{type(error).__name__}: {error}"})
                with results.open("a") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"[{mode}] {card_id} [{condition}] {'ok' if record['report'] else record['error']}")
                time.sleep(args.sleep)

        print(f"\nSaved {results}")


if __name__ == "__main__":
    main()
