# Reliability and system evaluation

This page explains how to reproduce every number in the evaluation chapter: calibration, abstention, unknown-input (OOD) detection, the cross-dataset field test, RAG retrieval quality, the A/B/C ablation and the safety test suite.

There are two classifiers. The **stage 1 lab model** (`yolo11m_staples_480_split70`) was trained on lab-style images. The **stage 2 field model** (`yolo11m_field_ft`) was fine-tuned from it on field photos, and **it is the model the app uses**. Sections 1–4 below are the individual steps, shown for the stage 1 model. For the app's model, `evaluate_field_model.sh` runs the same steps (section 0). The RAG, ablation and safety-test results do not depend on the classifier: the ablation gives every condition the true label.

All commands run from the project folder with the virtual environment active:

```bash
source .venv/bin/activate
pip install -r requirements-eval.txt
```

## Files added for evaluation

| File | Purpose |
|---|---|
| `src/reliability.py` | Logit/feature capture from YOLO11-cls, temperature scaling, ECE, risk–coverage, kNN / energy / MSP OOD scores, and the runtime `ReliabilityGate` |
| `src/input_guard.py` | Now checks, in order: unknown input → crop mismatch → low confidence. The reliability checks switch on only when `models/reliability.json` exists |
| `src/advisory_prompt.py` | The app's report prompt (moved out of `app.py`, text unchanged) plus the reduced A and B prompts for the ablation |
| `src/gemini_client.py` | `generate_with_fallback` (moved out of `app.py`, unchanged) so the scripts can call Gemini the same way |
| `scripts/evaluation/extract_outputs.py` | One slow model pass per image set; caches raw logits, 1280-d features and labels in `reports/reliability/cache/*.npz` |
| `scripts/evaluation/calibration_abstention.py` | Fits the temperature on val, reports ECE/NLL/Brier, chooses abstention thresholds, writes `models/reliability.json` |
| `scripts/evaluation/ood_detection.py` | Compares MSP, MaxLogit, Energy, Mahalanobis and kNN; sets the unknown-input threshold; writes `models/ood_feature_bank.npy` |
| `scripts/evaluation/field_test.py` | Cross-dataset test with pHash de-duplication, bootstrap CIs and gate behaviour on field photos |
| `scripts/evaluation/rag_retrieval_eval.py` | Hit@k, Recall@k, MRR, nDCG and contamination rate for the RAG pipeline |
| `scripts/evaluation/ablation_abc.py` | Generates reports for 70 scenario cards under A (LLM only), B (+RAG) and C (+RAG + rule engine) |
| `scripts/evaluation/score_compliance.py` | Automatic compliance scoring, Wilson CIs, Cochran's Q, exact McNemar with Holm, blinded rating sheet |
| `scripts/evaluation/ablation_retrieval_stress.py` | Re-runs B and C with retrieval deliberately broken (no evidence / another disease's evidence) |
| `scripts/evaluation/ablation_summary.py` | Combines the ablation and the stress test into one table, applying the hand-checked corrections |
| `scripts/evaluation/config/` | PlantDoc class mapping, the banned-pesticide list used by the compliance scorer, and `manual_corrections.csv` |
| `tests/test_safety_scenarios.py` | 29 behaviour tests: healthy → no chemical, partial → no dose, unknown disease → rejected, gate maths, compliance scorer |

## 0. The app's model in one step

```bash
bash scripts/evaluation/evaluate_field_model.sh
```

It needs the field datasets in `../plant-field-training/` (`dataset/`, `test_field/`, `train_images/`). It caches outputs for the lab and field val/test/train sets and two unknown-input sets, fits calibration and abstention on lab + field val, fits the kNN gate, runs the field test, and compares the stage 1 and stage 2 models. Results go to `reports/reliability_field_ft/` and `reports/model_comparison/`, and `models/reliability.json` / `ood_feature_bank.npy` are rewritten for the field model (the stage 1 versions are kept as `*_staples.*`).

## 1. Cache model outputs (once per image set)

```bash
python scripts/evaluation/extract_outputs.py --images data/processed/val   --name val
python scripts/evaluation/extract_outputs.py --images data/processed/test  --name test
python scripts/evaluation/extract_outputs.py --images data/processed/train --name train
# Images the model was never trained for (any folder of images; labels not needed)
python scripts/evaluation/extract_outputs.py --images path/to/other_crops --name ood_other_crops --ood
# A labelled external field set, mapped onto the 14 classes
python scripts/evaluation/extract_outputs.py --images path/to/PlantDoc-Dataset/test --name field_plantdoc \
    --mapping scripts/evaluation/config/plantdoc_mapping.json
```

The script checks that softmax(captured logits) equals Ultralytics' own probabilities and warns if they differ.

## 2. Calibration and abstention

```bash
python scripts/evaluation/calibration_abstention.py --target-risk 0.01
```

- One temperature *T* is fitted on **val** by minimising negative log-likelihood. Accuracy does not change, because dividing all logits by the same number keeps their order.
- The abstention threshold is the lowest calibrated confidence at which the **val** error rate among answered images is at most the target risk (default 1%). It is then reported on **test**, which plays no part in choosing it.
- Two thresholds are saved: one for "Not sure: detect automatically" and one for when the farmer selects the crop, since restricting to one crop raises confidence.

## 3. Unknown-input detection

```bash
python scripts/evaluation/ood_detection.py --k 10 --id-keep 0.95
```

The kNN score (distance from the image's normalised feature to its 10th nearest training feature) is compared with softmax-based scores. The gate threshold keeps 95% of genuine **val** images. `gate_outcomes.csv` shows the whole decision chain on test and OOD images.

## 4. Field test

```bash
python scripts/evaluation/field_test.py --field field_plantdoc
```

Field images within 8 bits (pHash) of any train/val/test image are removed first. Results are given for 14-way prediction and for the crop-selected mode, with stratified bootstrap 95% CIs and the in-domain score on the same classes for comparison.

## 5. RAG retrieval

```bash
python src/rag_pipeline.py                       # (re)build chroma_db if the knowledge base changed
python scripts/evaluation/rag_retrieval_eval.py
```

Relevance is defined by `record_id`: a retrieved chunk is relevant when it belongs to the record (ADV-001 … ADV-014) of the disease asked about.

## 6. A/B/C ablation and compliance

```bash
python scripts/evaluation/ablation_abc.py --limit 14   # quick check first
python scripts/evaluation/ablation_abc.py              # all 70 cards x 3 conditions (resumable)
python scripts/evaluation/score_compliance.py
```

Every condition gets the **true** label so classifier errors do not affect the comparison. The farmer questions include two adversarial ones (asking for a double dose, and for banned products). The automatic checks are heuristics, so confirm every row of `flagged_sentences.csv` by hand and report the confirmed rates. Give `blinded_rating_sheet.xlsx` to two raters and report Cohen's κ between them.

Retrieval stress test and combined table:

```bash
python scripts/evaluation/ablation_retrieval_stress.py --limit 15     # quick check first
python scripts/evaluation/ablation_retrieval_stress.py                # both modes, all 70 cards (resumable)
python scripts/evaluation/score_compliance.py --dir reports/ablation_stress/none
python scripts/evaluation/score_compliance.py --dir reports/ablation_stress/wrong
python scripts/evaluation/ablation_summary.py
```

In the stress test the safety rule still comes from the true label (as in the app, where the rule is looked up from the classifier label), so only retrieval fails. Flagged sentences confirmed as refusals are listed, with quotes, in `config/manual_corrections.csv`; `ablation_summary.py` reports both the automatic and the confirmed counts.

Before submission, check `scripts/evaluation/config/banned_pesticides_india.csv` against the current PPQS list of pesticides banned, refused registration or restricted in use.

## 7. Safety test suite

```bash
python -m pytest tests -q
```

## Results: stage 2 field model (the app's model)

From the run on 24 October 2026 with `runs/classify/crop_model_runs/yolo11m_field_ft/weights/best.pt`. Full tables and figures are in `reports/reliability_field_ft/` and `reports/model_comparison/`.

Test sets: **lab test** = the 1,779-image held-out split of the original dataset; **field test** = 847 field photos from five public datasets, de-duplicated against all lab images (pHash) and never trained on. The field test covers 12 of the 14 classes (no field photos exist for maize healthy or rice neck blast).

### Stage 1 vs stage 2 (95% bootstrap CIs)

| Model | Test set | Images | Accuracy | Macro-F1 |
|---|---|---|---|---|
| Stage 1 lab model | Lab test | 1,779 | 97.13% (96.4–97.8) | 0.968 (0.959–0.977) |
| **Stage 2 field model** | Lab test | 1,779 | 94.77% (93.8–95.7) | 0.941 (0.930–0.952) |
| Stage 1 lab model | Field test | 847 | 2.13% (1.3–3.1) | 0.104 (0.050–0.150) |
| **Stage 2 field model** | Field test | 847 | **92.44% (90.8–94.2)** | **0.831 (0.773–0.884)** |

Fine-tuning costs 2.4 points on lab images and gains 90 points on field photos. Field F1 by class: potato healthy 1.00, wheat brown rust 0.99, wheat healthy 0.99, wheat yellow rust 0.98, rice healthy 0.97, potato late blight 0.96, rice leaf blast 0.90, rice brown spot 0.85, maize common rust 0.84, potato early blight 0.71 (7 photos), maize northern leaf blight 0.53 (10 photos), maize gray leaf spot 0.25 (3 photos). With crop selected, field accuracy is 92.56% (90.9–94.3). Per-class table: `reports/model_comparison/per_class_f1.csv`.

### Calibration (temperature fitted on lab + field val)

| Test set | Accuracy | ECE raw → scaled (T = 1.359) | NLL raw → scaled |
|---|---|---|---|
| Lab + field test (2,626) | 0.9402 | 0.0269 → 0.0124 | 0.196 → 0.176 |
| Lab test (1,779) | 0.9477 | 0.0287 → 0.0149 | 0.180 → 0.157 |
| Field test (847) | 0.9244 | 0.0253 → 0.0168 | 0.228 → 0.217 |

Unlike the stage 1 model, calibration now holds on field photos as well as lab images, because field images were part of training and calibration.

### Abstention (threshold 0.930, chosen on val for at most 1% error)

| Test set | Coverage | Error rate when answering | Images withheld | Errors caught |
|---|---|---|---|---|
| Lab + field test (2,626) | 80.6% | 1.61% (from 5.98%) | 509 | 123 of 157 |
| Lab test (1,779) | 86.2% | 1.70% | 246 | 67 of 93 |
| Field test (847) | 68.9% | 1.37% | 263 | 56 of 64 |

The 1% target set on val becomes about 1.6% on test. Field photos are withheld much more often than lab images (31% vs 14%).

### Unknown-input detection (ID = lab + field test, 2,626 images)

| Score | AUROC, other crops (PlantVillage, n = 930) | FPR@95TPR | AUROC, unseen rice problems (field, n = 210) | FPR@95TPR |
|---|---|---|---|---|
| MSP (T = 1.36) | 0.876 | 65.9% | 0.889 | 61.9% |
| Energy | 0.941 | 25.7% | 0.826 | 68.1% |
| Mahalanobis | 0.917 | 59.5% | 0.567 | 99.0% |
| **kNN, k = 10 (used by the app)** | **0.940** | **43.9%** | **0.762** | **90.5%** |

"Unseen rice problems" are Paddy Doctor field photos of seven rice conditions the model was not trained on (e.g. bacterial leaf blight, tungro, hispa). The kNN gate separates other crops well, but it barely separates unseen rice diseases from the rice classes it knows. Those are caught mainly by the low-confidence check instead (below). The feature bank holds 3,241 training images (120 lab and up to 150 field per class).

### Whole gate (unknown → low confidence → answer)

| Image set | Refused as unknown | Withheld (low confidence) | Answered | Wrong answers given |
|---|---|---|---|---|
| Lab test (1,779) | 4.8% | 10.6% | 84.7% (98.6% correct) | 1.2% |
| Field test (847) | 3.4% | 29.6% | 66.9% (99.3% correct) | 0.5% |
| Other crops, PlantVillage (930) | 51.7% | 31.9% | 16.3% | 16.3% |
| Unseen rice problems (210) | 8.1% | 75.7% | 16.2% | 16.2% |

Compared with the stage 1 model, the stage 2 model with its gate is usable on field photos: two-thirds of them get an answer, and 99% of those answers are correct. On the PlantVillage other-crops set, fewer photos get through than with stage 1 (16.3% answered vs 27.7%). Still, about 1 in 6 unknown inputs receive a wrong diagnosis, and for unseen rice diseases the low-confidence check does most of the work.

## Results: stage 1 lab model (baseline)

These results come from the run on 22 October 2026 with the stage 1 model (`yolo11m_staples_480_split70/weights/best.pt`, YOLO11m-cls, imgsz 480). The app no longer uses this model. It is kept as the baseline that motivated field fine-tuning. Full tables and figures are in `reports/reliability/`.

### Calibration (temperature fitted on val, reported on test, n = 1,779)

| | Accuracy | ECE (15 bins) | NLL | Brier | Mean confidence |
|---|---|---|---|---|---|
| Raw (T = 1) | 0.9713 | 0.0138 | 0.1005 | 0.0459 | 0.984 |
| Temperature scaled (T = 1.569) | 0.9713 | 0.0099 | 0.0890 | 0.0455 | 0.969 |

The model was slightly over-confident. Scaling lowers ECE by 28% and NLL by 11% without changing any prediction. Figure: `reliability_diagram.png`.

### Abstention (threshold chosen on val for at most 1% error, reported on test)

| Mode | Threshold | Coverage | Error rate when answering | Images withheld | Errors caught |
|---|---|---|---|---|---|
| Auto-detect | 0.862 | 91.9% | 0.92% (from 2.87%) | 144 | 36 of 51 |
| Crop selected | 0.865 | 91.9% | 0.92% | 144 | 36 of 51 |

AURC = 2.26 × 10⁻³. Withholding falls mostly on the weakest classes: rice healthy 37.5%, rice brown spot 30.4%, rice leaf blast 19.4%, maize northern leaf blight 13.3%; most other classes are under 3%. Figures: `risk_coverage.png`, `abstention_per_class.csv`.

### Unknown-input detection (ID = test split; OOD = crops the model was never trained on)

| Score | AUROC, PlantDoc other crops (n = 194, field) | FPR@95TPR | AUROC, PlantVillage other crops (n = 930, lab) | FPR@95TPR |
|---|---|---|---|---|
| MSP (T = 1.57) | 0.916 | 43.3% | 0.864 | 63.0% |
| Energy | 0.964 | 12.4% | 0.848 | 50.7% |
| Mahalanobis | 0.992 | 2.1% | 0.920 | 51.9% |
| **kNN, k = 10 (used by the app)** | **0.989** | **4.6%** | **0.927** | **37.3%** |

The kNN feature bank in this run is a random 1,680-image sample of the training set (120 per class). Re-run `extract_outputs.py --name train` on the full training split and then `ood_detection.py` to use all 8,323 images.

### Whole gate (unknown → low confidence → answer)

| Image set | Refused as unknown | Withheld (low confidence) | Answered | Wrong answers given |
|---|---|---|---|---|
| ID test (1,779) | 6.2% | 5.9% | 87.9% (99.2% correct) | 0.7% |
| PlantDoc other crops (194) | 95.4% | 0.0% | 4.6% | 4.6% |
| PlantVillage other crops (930) | 66.0% | 6.2% | 27.7% | 27.7% |

### Field test: PlantDoc maize and potato (Singh et al., 2020; CC BY 4.0)

598 images in 5 shared classes; 2 near-duplicates of training data removed (pHash ≤ 8 bits) → 596 scored.

| | Accuracy (95% CI) | Macro-F1 (95% CI) |
|---|---|---|
| In-domain test, same 5 classes | 98.8% (97.8–99.5) | 0.981 |
| PlantDoc, auto-detect | **9.4% (7.0–11.7)** | 0.164 (0.126–0.202) |
| PlantDoc, crop selected | 34.6% (31.2–38.1) | 0.421 (0.380–0.460) |

- Most field photos are predicted as wheat classes. For example, 68 of 116 corn-rust photos become wheat brown rust. The model seems to have learnt "field background → wheat", because the wheat training images are the most field-like.
- Confidence stays high on field photos (mean 0.74, ECE 0.64), so calibration fitted on lab images does not carry over to field images.
- With the gate on, 96.1% of field photos are refused as unknown, and only 3.2% receive a wrong diagnosis (90.6% without the gate).
- So the gate makes the failure safe. It does not make the system useful in the field: that needs field images in training (e.g. the PlantDoc train split, Paddy Doctor for rice).

### RAG retrieval (14 diseases x 4 query phrasings = 56 queries)

| Path | Result |
|---|---|
| Semantic search | Hit@1 0.929, Hit@2–10 1.000, nDCG@4 0.752 |
| Metadata-filtered branch | Precision 1.000 |
| App pipeline (`query_knowledge_base`) | Hit rate 1.000, context precision 0.964, no-evidence rate 0.000 |
| App pipeline, queries with another disease's records mixed in | **14.3%** |

The app always retrieves the right disease, but about one query in seven also brings in another disease's record. That is the failure the wrong-disease stress test below exaggerates. Full tables: `reports/rag_eval/`.

### A/B/C ablation and retrieval stress test (70 scenario cards, Gemini, true label given)

A = LLM only, B = + RAG, C = + RAG + rule engine (the app). A uses no retrieval, so its row applies to every retrieval mode. Confirmed counts (after the hand check); full tables with Wilson CIs in `reports/ablation_summary/summary.md`.

| Retrieval | Metric | A | B | C |
|---|---|---|---|---|
| Correct | Chemical or dose outside the verified rule | 34/70 (49%) | 0/70 | 0/70 |
| Correct | Gives the verified rule dose (max 45) | 11/70 | 45/70 | 45/70 |
| None | Chemical or dose outside the verified rule | 34/70 (49%) | 0/70 | 0/70 |
| None | Gives the verified rule dose (max 45) | 11/70 | **1/70** | **45/70** |
| Another disease's evidence | Chemical or dose outside the verified rule | 34/70 (49%) | **44/70 (63%)** | **0/70** |
| Another disease's evidence | Gives the verified rule dose (max 45) | 11/70 | 25/70 | 45/70 |

- With correct retrieval, RAG removes off-label advice (A vs B, exact McNemar p < 0.0001) and the rule engine adds nothing measurable (B vs C, p = 1.0).
- When retrieval fails, the rule engine is what keeps the advice correct: with no evidence B can give the verified dose in 1 case and C in all 45 (p < 0.0001); with another disease's evidence B repeats it in 63% of cards and C in none (p < 0.0001).
- No condition recommended a banned pesticide or agreed to a double dose once flagged refusals were checked (9 corrections, all refusals of Methyl Parathion / Phorate).
- "Outside the verified rule" means outside the PPQS/CIBRC rule for that disease, not necessarily unregistered. Most of B's 17 healthy-crop cases in the wrong-evidence mode are conditional ("if rust appears, use ...").
- Human rating of `blinded_rating_sheet.xlsx` was not carried out; the results rest on the automatic checks and the hand check of every flagged sentence.

### Safety test suite

29 of 29 tests pass (`python -m pytest tests -q`, 29 October 2026).
