# 🌿 Plant Disease Safe-RAG Advisory System (India)

An AI-assisted crop disease advisor for Indian farmers. A farmer uploads a photo of a leaf; the system identifies the disease, checks treatment advice against verified Indian regulatory records (PPQS / CIBRC), and explains what to do in plain language, in English or one of ten Indian languages.

> Final-year Computer Science dissertation project by **Ryan K**.

**Live app:** <https://plant-advisor.victoriousbeach-0c1589c9.southeastasia.azurecontainerapps.io>
**Docker image:** [`ryankglitch/plant-advisor`](https://hub.docker.com/r/ryankglitch/plant-advisor) (linux/amd64 and linux/arm64)

---

## Contents

1. [Motivation](#motivation)
2. [How it works](#how-it-works)
3. [Features](#features)
4. [Supported crops and diseases](#supported-crops-and-diseases)
5. [Model and results](#model-and-results)
6. [Project structure](#project-structure)
7. [Installation](#installation)
8. [Running the app](#running-the-app)
9. [Running with Docker](#running-with-docker)
10. [Deploying to Azure](#deploying-to-azure)
11. [Configuration and maintenance](#configuration-and-maintenance)
12. [Re-training and evaluation](#re-training-and-evaluation)
13. [Safety design](#safety-design)
14. [Limitations](#limitations)
15. [Technology stack](#technology-stack)
16. [Acknowledgements and data sources](#acknowledgements-and-data-sources)
17. [Disclaimer](#disclaimer)

---

## Motivation

General-purpose chatbots can give confident but wrong pesticide advice: invented products, incorrect doses, or chemicals not registered for that crop in India. For a farmer, following that advice can damage the crop, harm health, or leave unsafe residues on food.

This project tests a **"Safe-RAG"** approach. The language model is never trusted to decide *what* chemical or dose to recommend. Those values come only from a verified table built from official PPQS / CIBRC sources and are enforced by a deterministic rule engine. The language model's role is limited to *explaining* the verified evidence clearly.

## How it works

```
 Leaf photo
     │
     ▼
┌───────────────────────┐
│ 1. YOLO11m classifier │  → crop + disease label, confidence score
└───────────────────────┘
     │
     ▼
┌───────────────────────┐
│ 2. RAG retrieval      │  → matching evidence from the knowledge base
│    (ChromaDB + BGE)   │     (advisory PDFs + verified treatment tables)
└───────────────────────┘
     │
     ▼
┌───────────────────────┐
│ 3. Rule engine        │  → APPROVED / MANUAL CHECK / REJECTED
│    (deterministic)    │     with exact chemical, dose, PHI, re-entry
└───────────────────────┘
     │
     ▼
┌───────────────────────┐
│ 4. Gemini LLM         │  → farmer-friendly report, grounded only in
│    (with fallback)    │     steps 2 and 3, in the chosen language
└───────────────────────┘
```

1. **Image classification.** A YOLO11m classification model, fine-tuned on 14 crop-disease classes, predicts the disease and a confidence score.
2. **Retrieval.** The prediction is used to query a ChromaDB vector store. Records tagged with the exact disease key are fetched first, then supporting semantic matches; results are filtered so only evidence for the detected crop and disease is kept.
3. **Rule engine.** `src/rule_engine.py` looks up the disease in a hard-coded table of audited PPQS / CIBRC rules and returns one of three statuses:
   - **APPROVED**: a verified chemical, dose, timing, maximum applications, pre-harvest interval (PHI) and re-entry period, with the source URL. Healthy classes are also approved, with *no chemical advised*.
   - **WARNING (manual check)**: a partial match where the source does not exactly cover the disease; no automatic dose is given.
   - **REJECTED**: no verified rule exists.
4. **Report generation.** Gemini writes an action-focused report using only the retrieved evidence and the rule output. If no evidence was retrieved, no report is generated.

## Features

- 📷 **Leaf image diagnosis** with a visual confidence meter (high / moderate / low).
- 🛡️ **Regulatory safety card** showing status, chemical, dose, timing, PHI, re-entry period and a link to the official source.
- 📚 **Transparent evidence**: the retrieved RAG passages can be expanded and inspected.
- 🌐 **Multilingual reports** in English, Hindi, Tamil, Telugu, Kannada, Malayalam, Marathi, Bengali, Gujarati, Punjabi and Odia. Chemical names, doses, numbers and URLs are kept exactly as verified, so they still match the product label.
- 🔊 **Read-aloud reports** in the selected language (all except Odia).
- 🌾 **Crop selector**: the farmer can state the crop, which limits the diagnosis to that crop's diseases and blocks advice when the photo looks like a different crop.
- 🌦️ **Weather-based disease risk** *(prototype)*: an optional location gives a 4-day Low / Moderate / High risk outlook and the next suitable spray window, from the free Open-Meteo forecast.
- ✅ **Field feedback**: farmers can confirm or correct each diagnosis, building a dataset for retraining.
- 🔁 **Automatic model fallback**: if a Gemini model is busy, rate-limited or retired, the app immediately tries the next free-tier model and shows which one produced the report.
- 🎨 **Custom HTML / CSS interface** on top of Streamlit, with a responsive layout for phones.
- 🌗 **Dark mode by default**, with Light / Dark / System in Streamlit's Settings menu.
- 🛰️ **Vision-scan interface**: an "AI console" hero with the four crops' leaves and real evaluation numbers, and an animated background showing the pipeline's real values alongside farmer-friendly lines in the report languages. A Blender-made 3D leaf scans while the model and Gemini work. Everything holds still with reduced motion. See [`docs/UI_3D.md`](docs/UI_3D.md).

## Supported crops and diseases

| Crop   | Classes                                                    |
|--------|------------------------------------------------------------|
| Maize  | Common rust, Gray leaf spot, Northern leaf blight, Healthy |
| Potato | Early blight, Late blight, Healthy                         |
| Rice   | Brown spot, Leaf blast, Neck blast, Healthy                |
| Wheat  | Brown rust, Yellow rust, Healthy                           |

**14 classes** in total.

## Model and results

The model was trained in two stages. Stage 1 trained YOLO11m on lab-style images. It reached 97 % on its own test split but only 2 % on real field photos, so stage 2 fine-tuned it on field photos. **The app uses the stage 2 model** (`runs/classify/crop_model_runs/yolo11m_field_ft/weights/best.pt`).

### Summary: lab model vs field model (the app's model)

| Model | Lab test (1,779 images) | Field test (847 images) |
|---|---|---|
| Stage 1: lab model | **97.13 %** (96.4–97.8), macro-F1 0.968 | 2.13 % (1.3–3.1), macro-F1 0.104 |
| **Stage 2: field model (used by the app)** | 94.77 % (93.8–95.7), macro-F1 0.941 | **92.44 %** (90.8–94.2), macro-F1 0.831 |

95 % bootstrap confidence intervals in brackets. Fine-tuning gave up 2.4 points on lab images and gained 90 points on field photos. Full tables: `reports/model_comparison/`.

### Stage 1: lab model

#### Dataset

15,216 source images were pooled, **3,335 exact duplicates were removed** so that no image can appear in two splits, and the remaining 11,881 unique images were given a stratified 70 / 15 / 15 split (seed 42).

| Split      | Images |
|------------|-------:|
| Train      | 8,323  |
| Validation | 1,779  |
| Test       | 1,779  |

Per-class counts are in `reports/tables/split_summary.csv`.

#### Training

| Setting        | Value                                             |
|----------------|---------------------------------------------------|
| Base model     | `yolo11m-cls.pt` (Ultralytics, pretrained)        |
| Image size     | 480 × 480                                         |
| Epochs         | 40 (early-stopping patience 10)                   |
| Batch size     | 16                                                |
| Optimiser      | AdamW, lr₀ = 0.0005, 2 warm-up epochs, cosine LR  |
| Regularisation | Dropout 0.1, weight decay 0.0005                  |
| Augmentation   | RandAugment, rotation, flips, HSV, erasing        |
| Hardware       | Apple Silicon (MPS)                               |
| Seed           | 42 (deterministic)                                |

Best validation top-1 accuracy was **97.58 %** (top-5: 100 %) at epoch 40.

#### Results on the held-out lab test set

The test split was not used for training or model selection.

| Metric             | Value       |
|--------------------|-------------|
| Accuracy           | **97.13 %** |
| Macro F1           | 0.968       |
| Weighted F1        | 0.971       |
| Misclassified      | 51 / 1,779  |

Most errors are between the three rice classes, which are the weakest:

| Class            | Precision | Recall | F1    |
|------------------|----------:|-------:|------:|
| Rice brown spot  | 0.878     | 0.859  | 0.868 |
| Rice leaf blast  | 0.879     | 0.885  | 0.882 |
| Rice healthy     | 0.911     | 0.917  | 0.914 |

Four classes score a perfect 1.000 F1 (maize common rust, maize healthy, rice neck blast and wheat healthy) and every other class is above 0.94. The full per-class report, confusion matrix and list of misclassified images are in `reports/evaluation_yolo11m_staples_480_split70/`; training curves are in `reports/figures/`.

**It did not work on field photos.** On the 847-photo field test set below it scored 2.13 %, and on 596 PlantDoc maize and potato photos 9.4 %, predicting most of them as wheat classes (68 of 116 corn-rust photos became wheat brown rust). It seems to have learnt "field background → wheat", because the wheat training images were the most field-like. That is why stage 2 was needed.

### Stage 2: field fine-tuning (the app's model)

#### Field dataset

Field photos of the supported classes were collected from five public datasets (see [Acknowledgements](#acknowledgements-and-data-sources)) and cleaned before splitting:

- Perceptual hashing (pHash) of all 6,780 field images and the 11,881 lab images: **no field image was within 6 bits of a lab image**, so there is no leakage from the stage 1 data.
- 836 near-identical field images removed, and 38 pairs with conflicting labels dropped. Look-alike images were kept together in one split.
- 70 / 15 / 15 split per source and class: **4,208 train, 889 val, 847 test**. The field test set was never trained on.
- Lab images topped up each class (100 per class in train, 20 in val; 300 in train for maize healthy and rice neck blast, which have no field photos), so the model keeps its lab accuracy. Field classes with fewer than 300 training photos were oversampled up to 4×.

The field test set covers 12 of the 14 classes (no field photos exist for maize healthy or rice neck blast) and is uneven: rice has 574 of the 847 images, and maize gray leaf spot only 3.

#### Training

Fine-tuned from the stage 1 weights with the same image size and augmentation, but a lower learning rate (AdamW, lr₀ = 0.0002, cosine LR, 1 warm-up epoch) so the weights are adjusted rather than overwritten. Training ran for 20 of 25 epochs before early stopping (patience 7); the best validation top-1 accuracy was **93.84 %** at epoch 15, on a validation set that mixes field and lab images.

#### Results

| Test set | Accuracy (95% CI) | Macro-F1 (95% CI) |
|---|---|---|
| Field test (847) | **92.44 %** (90.8–94.2) | 0.831 (0.773–0.884) |
| Lab test (1,779) | 94.77 % (93.8–95.7) | 0.941 (0.930–0.952) |

- **Field F1 per class:** 0.96–1.00 for potato healthy, potato late blight, rice healthy and all three wheat classes; 0.84–0.90 for maize common rust, rice brown spot and rice leaf blast; weaker on classes with very few field test photos: potato early blight 0.71 (7 photos), maize northern leaf blight 0.53 (10) and maize gray leaf spot 0.25 (3).
- **On lab images** every class dropped slightly; the rice classes remain the weakest (brown spot 0.80, leaf blast 0.81, healthy 0.89).
- Per-class tables and confusion matrices: `reports/model_comparison/` and `reports/reliability_field_ft/test_field/`.

#### Reliability checks for this model

The confidence calibration, abstention threshold and unknown-input gate were re-fitted for this model (`models/reliability.json`, `models/ood_feature_bank.npy`):

| Image set | Refused as unknown | Withheld (low confidence) | Answered | Correct when answered |
|---|---|---|---|---|
| Lab test (1,779) | 4.8 % | 10.6 % | 84.7 % | 98.6 % |
| Field test (847) | 3.4 % | 29.6 % | 66.9 % | 99.3 % |
| Other crops, PlantVillage (930) | 51.7 % | 31.9 % | 16.3 % (all wrong) | – |
| Unseen rice problems, field (210) | 8.1 % | 75.7 % | 16.2 % (all wrong) | – |

When the app does answer, it is right about 99 % of the time. The cost is that it withholds about a third of field photos and asks for a better photo or an extension officer. Unknown inputs are mostly stopped by the two checks together, but about 1 in 6 still receive a (wrong) diagnosis. Details: [`docs/EVALUATION.md`](docs/EVALUATION.md).

## Project structure

Folders marked *(local only)* are listed in `.gitignore` and are not in the repository.

```
.
├── .devcontainer/
│   └── devcontainer.json          # VS Code / Codespaces dev container
├── .streamlit/
│   ├── config.toml                # Streamlit theme
│   └── secrets.toml               # GEMINI_API_KEY (local only)
├── assets/
│   └── style.css                  # Custom styling for the interface
├── chroma_db/                     # Persisted vector store, built from knowledge_base/
├── docs/
│   └── EVALUATION.md              # How to reproduce every evaluation result, and the results
├── data/                          # (local only)
│   ├── raw/                       # Source datasets (PlantVillage, Jain 2026)
│   ├── processed/                 # Final 14-class dataset
│   │   ├── train/                 #   70 %
│   │   ├── val/                   #   15 %
│   │   └── test/                  #   15 %
│   ├── feedback/                  # Farmer feedback database and consented photos
│   └── feedback_dataset/          # Feedback photos exported for retraining
├── knowledge_base/
│   ├── ADV-001 … ADV-014_*.pdf    # One advisory per class (14 PDFs)
│   ├── Crop_Disease_RAG_Advisory_India.xlsx
│   └── India_PPQS_CIBRC_Verified_Treatment_Table.xlsx
├── models/
│   ├── yolo11m-cls.pt             # Pretrained base model used for stage 1 training
│   ├── reliability.json           # Calibration, abstention and unknown-input settings for the app's model
│   ├── ood_feature_bank.npy       # Training-image features for the unknown-input check
│   └── *_staples.*                # The same two files for the stage 1 lab model (backup)
├── reports/
│   ├── ablation/                  # A/B/C ablation reports and compliance scores
│   ├── ablation_stress/           # Retrieval-failure stress test (no / wrong evidence)
│   ├── ablation_summary/          # Combined ablation table (summary.md)
│   ├── evaluation_yolo11m_staples_480_split70/   # Stage 1 lab model, lab test set
│   ├── figures/                   # Dataset distribution and training curves
│   ├── model_comparison/          # Stage 1 vs stage 2 on lab and field test sets
│   ├── rag_eval/                  # Retrieval metrics
│   ├── reliability/               # Stage 1 model: calibration, abstention, unknown input, PlantDoc test
│   ├── reliability_field_ft/      # App's model: calibration, abstention, unknown input, field test
│   └── tables/                    # Class counts, split manifest, dataset issues, feedback summary
├── runs/classify/crop_model_runs/
│   ├── yolo11m_field_ft/
│   │   └── weights/
│   │       └── best.pt            # Stage 2 field model, loaded by the app
│   └── yolo11m_staples_480_split70/
│       └── weights/
│           └── best.pt            # Stage 1 lab model (starting point for stage 2)
├── scripts/                       # Offline pipeline, run from the project folder
│   ├── analysis/
│   │   ├── audit_dataset.py       # Class distribution chart
│   │   ├── count_classes.py       # Images per class and split
│   │   └── export_feedback.py     # Field accuracy and retraining photos from feedback
│   ├── data_prep/
│   │   ├── check_images.py        # Find unreadable images
│   │   ├── clean_dataset.py       # CleanVision quality checks
│   │   ├── filter_dataset.py      # Keep the four target crops
│   │   ├── fix_dataset_classes.py # Merge and rename class folders
│   │   ├── organize_dataset.py    # Copy raw datasets into data/processed
│   │   ├── prune_dataset.py       # Optional: cap images per class
│   │   └── split_dataset.py       # Remove duplicates, stratified 70/15/15 split
│   ├── evaluation/
│   │   ├── evaluate_model.py      # Test-set metrics and confusion matrix
│   │   ├── plot_training_history.py
│   │   ├── visualize_dataset.py
│   │   ├── evaluate_field_model.sh   # Full re-evaluation of the field model + comparison
│   │   ├── extract_outputs.py, calibration_abstention.py, ood_detection.py, field_test.py, compare_models.py
│   │   ├── rag_retrieval_eval.py
│   │   ├── ablation_abc.py, ablation_retrieval_stress.py, score_compliance.py, ablation_summary.py
│   │   └── config/                # Class mapping, banned-pesticide list, manual corrections
│   └── training/
│       └── train.py               # Fine-tune YOLO11m on data/processed
├── src/                           # Application modules
│   ├── advisory_prompt.py         # Report prompt for Gemini (plus reduced prompts for the ablation)
│   ├── feedback.py                # Stores "Was this correct?" answers (SQLite)
│   ├── gemini_client.py           # Gemini call with automatic model fallback
│   ├── input_guard.py             # Unknown input → crop mismatch → low confidence checks
│   ├── labels.py                  # Shared label parsing
│   ├── rag_pipeline.py            # Knowledge-base ingestion and retrieval (ChromaDB)
│   ├── read_aloud.py              # Reads the report aloud with gTTS
│   ├── reliability.py             # Temperature calibration, abstention and kNN unknown-input gate
│   ├── rule_engine.py             # Deterministic PPQS / CIBRC safety rules
│   ├── ui_components.py           # HTML components (hero, cards, sidebar, footer)
│   └── weather_risk.py            # Open-Meteo forecast, disease risk and spray window
├── tests/
│   └── test_safety_scenarios.py   # 29 safety behaviour tests
├── .dockerignore
├── .env                           # GEMINI_API_KEY for Docker (local only)
├── .env.example                   # Template for .env
├── .gitignore
├── Dockerfile                     # Container image for the app
├── README.md
├── app.py                         # Streamlit entry point: diagnosis → RAG → rules → Gemini report
├── docker-compose.yml             # Local container run with a persistent feedback volume
├── requirements.txt
└── requirements-eval.txt          # Extra packages for the evaluation scripts and tests
```

## Installation

**Requirements:** Python 3.9 or newer (tested on 3.11 and 3.12) and about 5 GB of free disk space for the dependencies, models and embeddings.

```bash
# 1. Clone the repository
git clone https://github.com/RyanK-glitch/plant-advisor.git
cd plant-advisor

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**4. Add a Gemini API key.** Get a free key from [Google AI Studio](https://aistudio.google.com/apikey) and add it to `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-key-here"
```

Alternatively, set it as an environment variable: `export GEMINI_API_KEY="your-key-here"`. Both `secrets.toml` and `.env` are already in `.gitignore`; never commit a real key.

**5. Check the trained model is present** at `runs/classify/crop_model_runs/yolo11m_field_ft/weights/best.pt`. The reliability checks only switch on when the model is at this path, because `models/reliability.json` was calibrated for it. To use a model stored elsewhere, set:

```bash
export YOLO_MODEL_PATH="/full/path/to/best.pt"
```

The app will not start without a trained model; it deliberately refuses to fall back to the untrained base model.

> **Moved or copied the project folder?** A virtual environment stores its own absolute path, so a copied `.venv` breaks with `bad interpreter: no such file or directory`. Delete `.venv` and repeat steps 2–3, or start the app with `python -m streamlit run app.py`.

## Running the app

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually <http://localhost:8501>):

1. Choose a report language.
2. Optionally choose the crop and enter a location for the weather risk.
3. Upload a clear photo of a single leaf (JPG or PNG).
4. Click **🔍 Run Safe-RAG Analysis**.
5. Review the diagnosis, the regulatory safety check, the retrieved evidence and the advisory report.
6. Optionally press **🔊 Read the report aloud** or answer **Was this correct?**

On first launch, the BGE embedding model (`BAAI/bge-large-en-v1.5`, about 1.3 GB) is downloaded from Hugging Face, which can take a few minutes.

## Running with Docker

The image uses CPU-only PyTorch, runs as a non-root user, includes a health check, and has the embedding model built in, so it starts without downloading anything. The Gemini key is passed in at runtime and is never stored in the image.

**Option A: pull the published image**

```bash
docker run -d -p 8501:8501 -e GEMINI_API_KEY=your-key-here ryankglitch/plant-advisor:v7
```

**Option B: build from source with Docker Compose**

```bash
cp .env.example .env              # then put your real key in .env
docker compose up -d --build      # app at http://localhost:8501
docker compose logs -f app        # follow the logs
docker compose down               # stop (feedback volume is kept)
```

Compose stores farmer feedback in a named volume (`feedback`), so it survives rebuilds. If port 8501 is taken, run with `HOST_PORT=8502 docker compose up -d`.

**Publishing a multi-platform image** (Intel/AMD servers and Apple Silicon):

```bash
docker buildx build --platform linux/amd64,linux/arm64 \
  -t ryankglitch/plant-advisor:v7 -t ryankglitch/plant-advisor:latest --push .
```

## Deploying to Azure

The live app runs on **Azure Container Apps** from the Docker Hub image.

| Setting          | Value                                                       |
|------------------|-------------------------------------------------------------|
| Resource group   | `plant-rg` (Southeast Asia)                                 |
| Environment      | `plant-env`                                                 |
| Container app    | `plant-advisor`, 2 vCPU / 4 GiB, port 8501, 0–1 replicas    |
| Image            | `docker.io/ryankglitch/plant-advisor:v7` (field model, dark-mode vision-scan UI, app-style phone layout)|
| Feedback storage | Azure Files share `feedback` in storage account `plantadvisorfeedback`, mounted at `/app/data/feedback` |

Because feedback is stored on Azure Files, it survives restarts, scale-to-zero and new revisions. Browse it in the portal under **Storage accounts → plantadvisorfeedback → File shares → feedback**; `feedback.db` appears after the first submission.

To deploy a new image version:

```bash
az containerapp update -n plant-advisor -g plant-rg \
  --image docker.io/ryankglitch/plant-advisor:<tag>
```

With a minimum of 0 replicas the app shuts down when idle, so the first visit after a quiet period takes longer while the models load.

## Configuration and maintenance

### Environment variables

| Variable          | Purpose                                            | Default                                    |
|-------------------|----------------------------------------------------|--------------------------------------------|
| `GEMINI_API_KEY`  | Google Gemini key for report generation (required) | read from `.streamlit/secrets.toml`        |
| `YOLO_MODEL_PATH` | Path to the trained classifier                     | `runs/…/yolo11m_field_ft/weights/best.pt`  |
| `FEEDBACK_DIR`    | Where the feedback database and photos are stored  | `data/feedback`                            |

### Rebuilding the knowledge base

Run this whenever files in `knowledge_base/` change:

```bash
python src/rag_pipeline.py
```

It deletes the old `chroma_db/`, loads every PDF, CSV and XLSX file, splits them into chunks of 900 characters (120 overlap) and stores them in a fresh vector database. Each spreadsheet row is stored as its own record with its `crop_key`, `disease_key`, `regulatory_status` and `source_url`, so exact matches can be retrieved by disease. Rebuild the Docker image afterwards, since `chroma_db/` is copied into it.

### Weather risk

Type a village, town or district under **Location** and pick the match. The app fetches an hourly forecast from [Open-Meteo](https://open-meteo.com/) (free, no API key) and shows:

- **Disease risk** for the next 4 days, from hours of high humidity or rain within each disease's favourable temperature range (`DISEASE_WEATHER` in `src/weather_risk.py`). Potato late blight uses the Hutton criteria, with a mean-temperature ceiling of 25 °C added for Indian conditions.
- **Next spray window** (only when a chemical is advised): daytime, wind under 15 km/h, below 35 °C and no rain for the following 6 hours.

The weather only affects risk and timing advice. It never changes the chemical, dose or rule status. The thresholds are indicative and should be checked against ICAR guidance.

### Read-aloud reports

Audio comes from Google Text-to-Speech (gTTS) in `src/read_aloud.py`. It needs an internet connection and covers every report language except Odia. Markdown and web addresses are removed before the text is spoken.

### Field feedback

After each diagnosis the farmer can answer **Yes**, **No** (and pick the real disease) or **Not sure**, add a comment and choose to share the photo. Answers are stored in `feedback.db` (SQLite) inside `FEEDBACK_DIR`. Shared photos are resized to 1024 px and saved without EXIF metadata, so phone GPS coordinates are never kept.

```bash
# Field accuracy per class        -> reports/tables/feedback_summary.csv
# Shared photos with a known class -> data/feedback_dataset/<class>/ (review before training)
python scripts/analysis/export_feedback.py
```

### Changing the Gemini models

The fallback order is defined by `GEMINI_MODELS` in `src/gemini_client.py`. The app moves to the next model on errors 404, 408, 429, 500, 502, 503 and 504, or on a 60-second timeout. It stops immediately on other errors (such as an invalid API key), since those would affect every model.

## Re-training and evaluation

Run the steps in this order from the project folder. The raw datasets must first be placed in `data/raw/`.

```bash
# 1. Build the dataset (filter_dataset.py clears data/processed first)
python scripts/data_prep/filter_dataset.py
python scripts/data_prep/organize_dataset.py
python scripts/data_prep/fix_dataset_classes.py

# 2. Check and clean image quality
python scripts/data_prep/check_images.py
python scripts/data_prep/clean_dataset.py

# 3. Remove duplicates and make the 70/15/15 train/val/test split
python scripts/data_prep/split_dataset.py
#    Optional: cap images per class into data/processed_pruned
python scripts/data_prep/prune_dataset.py

# 4. Inspect the dataset
python scripts/analysis/count_classes.py
python scripts/analysis/audit_dataset.py

# 5. Train (automatically uses Apple MPS, NVIDIA CUDA or CPU)
python scripts/training/train.py

# 6. Evaluate on the held-out test split and plot results into reports/
python scripts/evaluation/evaluate_model.py
python scripts/evaluation/plot_training_history.py
python scripts/evaluation/visualize_dataset.py
```

The evaluation and data-cleaning scripts also use `matplotlib`, `seaborn`, `scikit-learn` and `cleanvision`; install them with `pip install matplotlib seaborn scikit-learn cleanvision` if needed.

Steps 1–6 produce the stage 1 lab model. Stage 2 (field fine-tuning) was run from a separate working folder, `../plant-field-training/`, which holds the field datasets, the dataset build script and `train_field.py`; it starts from the stage 1 `best.pt` and writes `yolo11m_field_ft/weights/best.pt`. Copy that file to `runs/classify/crop_model_runs/yolo11m_field_ft/weights/` in this project.

### Reliability and system evaluation

Calibration, abstention, unknown-input detection, the field test, RAG retrieval metrics, the A/B/C ablation and the safety test suite are described step by step in [`docs/EVALUATION.md`](docs/EVALUATION.md). For the app's model, one script re-runs everything and the stage 1 vs stage 2 comparison:

```bash
pip install -r requirements-eval.txt
bash scripts/evaluation/evaluate_field_model.sh   # writes models/reliability.json, reports/reliability_field_ft/, reports/model_comparison/
python -m pytest tests -q
```

The app switches on the reliability checks automatically when `models/reliability.json` exists and was calibrated for the model it loads.

## Safety design

Key decisions that keep the advice safe:

- **The LLM never chooses a chemical or dose.** These come only from `src/rule_engine.py`, whose rules were audited against PPQS / CIBRC sources on 22 September 2026.
- **Healthy predictions never trigger pesticide advice.**
- **Partial matches give no automatic dose** and are flagged for manual label checking.
- **Unknown images are refused.** A photo whose features are far from every training image (another crop, an unseen disease, not a leaf) gets no diagnosis (`UNKNOWN_INPUT`).
- **Low confidence means no advice.** Confidence is temperature-calibrated; below the abstention threshold the app shows the likely options and asks for a better photo or an extension officer (`LOW_CONFIDENCE`).
- **No evidence, no report.** If retrieval finds nothing for the detected disease, the report is not generated.
- **Values are protected in translation.** Chemical names, doses, units, numbers and URLs stay exactly as verified in every language.
- **The image result is presented as an initial assessment.** Farmers are told to confirm in the field and to follow the registered product label.

## Limitations

- **The field test is small and uneven for some classes.** The 92.44 % field accuracy is dominated by rice (574 of 847 photos). Maize gray leaf spot has only 3 field test photos and northern leaf blight 10, so their field scores (F1 0.25 and 0.53) are uncertain, and maize healthy and rice neck blast have no field photos at all.
- **Field photos come from public datasets, not from this app's users.** They were taken in India, Indonesia, Ethiopia and elsewhere, by researchers. Accuracy on Indian farmers' own phone photos has not been measured yet; the field feedback feature is designed to measure this.
- **About a third of field photos get no diagnosis.** The abstention threshold keeps answered field photos about 99 % correct, but withholds 30 % of them as low confidence.
- **Unknown inputs are not always caught.** About 1 in 6 photos of other crops or of rice problems the model was not trained on still receive a (wrong) diagnosis.
- **Rice classes are the weakest on lab images** (F1 0.80–0.89 for brown spot, leaf blast and healthy), mainly confusion between the three.
- **Limited scope.** Only 4 crops and 14 classes are supported. The unknown-input gate rejects most other crops, but it is tested only on the image sets in `docs/EVALUATION.md` and will not catch every unseen disease of the four supported crops.
- **Weather risk is not a validated forecast.** The thresholds come from general infection criteria and have not been tested against Indian outbreak records.
- **Rules can go out of date.** Pesticide registrations change; the rule table must be re-audited periodically.
- **Depends on external APIs.** Report generation needs internet access and the Gemini free tier, which has rate limits and changing model availability.
- **Translation quality** has not been formally evaluated by native speakers for every language.

## Technology stack

| Area             | Tools                                                  |
|------------------|--------------------------------------------------------|
| Image model      | Ultralytics YOLO11m-cls, PyTorch                       |
| Retrieval        | LangChain, ChromaDB, BAAI/bge-large-en-v1.5 embeddings |
| Document loading | PyPDF, pandas, openpyxl                                |
| Language model   | Google Gemini (`google-genai` SDK)                     |
| Speech           | gTTS                                                   |
| Interface        | Streamlit, custom HTML / CSS                           |
| Data quality     | CleanVision, scikit-learn, matplotlib, seaborn         |
| Deployment       | Docker, Docker Hub, Azure Container Apps, Azure Files  |

## Acknowledgements and data sources

- **PlantVillage** dataset: maize and potato images.
- **Jain (2026)** dataset: additional Indian crop images. *(Add full citation.)*
- Field photos for stage 2 fine-tuning:
  - **PlantDoc** (Singh et al., 2020; CC BY 4.0): maize and potato diseases.
  - **Shabrina et al. (2024)**, Potato Leaf Disease Dataset in Uncontrolled Environment, Mendeley Data, doi:10.17632/ptz377bwb8.1: potato late blight and healthy (Indonesia).
  - **Arya & Singh (2020)**, IARI, Mendeley Data, doi:10.17632/th422bg4yd.1: wheat leaf rust and healthy (India).
  - **Getachew (2021)**, Mendeley Data, doi:10.17632/wgd66f8n6h.1 (CC BY 4.0): wheat stripe rust and healthy (Ethiopia).
  - **Paddy Doctor** (Kaggle): rice blast, brown spot and healthy (India).
- **Directorate of Plant Protection, Quarantine & Storage (PPQS)** and the **Central Insecticides Board & Registration Committee (CIBRC)**, Government of India: treatment and registration records.
- **ICAR** guidance on integrated pest management.
- [Ultralytics](https://github.com/ultralytics/ultralytics), [LangChain](https://www.langchain.com/), [ChromaDB](https://www.trychroma.com/), [Streamlit](https://streamlit.io/), [Google Gemini](https://ai.google.dev/) and [Open-Meteo](https://open-meteo.com/).

*Please replace the placeholder citations above with the full references used in the dissertation, including the source of the wheat images.*

## Disclaimer

This software is an academic prototype built for a final-year dissertation. It is **not** a substitute for advice from a qualified agronomist or a local agricultural extension officer. Always confirm a diagnosis in the field and read the registered product label before applying any chemical.
