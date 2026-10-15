# 🌿 Plant Disease Safe-RAG Advisory System (India)

An AI-assisted crop disease advisor for Indian farmers. A farmer uploads a photo of a leaf; the system identifies the disease, checks treatment advice against verified Indian regulatory records (PPQS / CIBRC), and explains what to do in plain language, in English or one of ten Indian languages.

> Final-year Computer Science dissertation project by **Ryan K**.

---

## Contents

- [Motivation](#motivation)
- [How it works](#how-it-works)
- [Features](#features)
- [Supported crops and diseases](#supported-crops-and-diseases)
- [Model and results](#model-and-results)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Running the app](#running-the-app)
- [Re-training and evaluation](#re-training-and-evaluation)
- [Safety design](#safety-design)
- [Limitations](#limitations)
- [Technology stack](#technology-stack)
- [Acknowledgements and data sources](#acknowledgements-and-data-sources)
- [Disclaimer](#disclaimer)

---

## Motivation

General-purpose chatbots can give confident but wrong pesticide advice: invented products, incorrect doses, or chemicals not registered for that crop in India. For a farmer, following that advice can damage the crop, harm health, or leave unsafe residues on food.

This project tests a **"Safe-RAG"** approach. The language model is never trusted to decide *what* chemical or dose to recommend. Those values come only from a verified table built from official PPQS / CIBRC sources and are enforced by a deterministic rule engine. The language model's role is limited to *explaining* the verified evidence clearly.

## How it works

```
 Leaf photo
     │
     ▼
┌──────────────────────┐
│ 1. YOLO11m classifier │  → crop + disease label, confidence score
└──────────────────────┘
     │
     ▼
┌──────────────────────┐
│ 2. RAG retrieval      │  → matching evidence from the knowledge base
│    (ChromaDB + BGE)   │     (advisory PDFs + verified treatment tables)
└──────────────────────┘
     │
     ▼
┌──────────────────────┐
│ 3. Rule engine        │  → APPROVED / MANUAL CHECK / REJECTED
│    (deterministic)    │     with exact chemical, dose, PHI, re-entry
└──────────────────────┘
     │
     ▼
┌──────────────────────┐
│ 4. Gemini LLM         │  → farmer-friendly report, grounded only in
│    (with fallback)    │     steps 2 and 3, in the chosen language
└──────────────────────┘
```

1. **Image classification.** A YOLO11m classification model, fine-tuned on 14 crop-disease classes, predicts the disease and a confidence score.
2. **Retrieval.** The prediction is used to query a ChromaDB vector store. Records tagged with the exact disease key are fetched first, then supporting semantic matches; results are filtered so only evidence for the detected crop and disease is kept.
3. **Rule engine.** `rule_engine.py` looks up the disease in a hard-coded table of audited PPQS / CIBRC rules and returns one of three statuses:
   - **APPROVED**: a verified chemical, dose, timing, maximum applications, pre-harvest interval (PHI) and re-entry period, with the source URL. Healthy classes are also approved, with *no chemical advised*.
   - **WARNING (manual check)**: a partial match where the source does not exactly cover the disease; no automatic dose is given.
   - **REJECTED**: no verified rule exists.
4. **Report generation.** Gemini writes an action-focused report using only the retrieved evidence and the rule output. If no evidence was retrieved, no report is generated.

## Features

- 📷 **Leaf image diagnosis** with a visual confidence meter (high / moderate / low).
- 🛡️ **Regulatory safety card** showing status, chemical, dose, timing, PHI, re-entry period and a link to the official source.
- 📚 **Transparent evidence**: the retrieved RAG passages can be expanded and inspected.
- 🌐 **Multilingual reports** in English, Hindi, Tamil, Telugu, Kannada, Malayalam, Marathi, Bengali, Gujarati, Punjabi and Odia. Chemical names, doses, numbers and URLs are kept exactly as verified, so they still match the product label.
- 🌾 **Crop selector**: the farmer can state the crop, which limits the diagnosis to that crop's diseases and blocks advice when the photo looks like a different crop.
- 🌦️ **Weather-based disease risk** *(prototype)*: an optional location gives a 4-day Low / Moderate / High risk outlook and the next suitable spray window, from the free Open-Meteo forecast.
- 🔁 **Automatic model fallback**: if a Gemini model is busy, rate-limited or retired, the app immediately tries the next free-tier model and shows which one produced the report.
- 🎨 **Custom HTML / CSS interface** on top of Streamlit, with a responsive layout for phones.

## Supported crops and diseases

| Crop   | Classes                                                         |
|--------|-----------------------------------------------------------------|
| Maize  | Common rust, Gray leaf spot, Northern leaf blight, Healthy      |
| Potato | Early blight, Late blight, Healthy                              |
| Rice   | Brown spot, Leaf blast, Neck blast, Healthy                     |
| Wheat  | Brown rust, Yellow rust, Healthy                                |

**14 classes** in total.

## Model and results

| Setting            | Value                                         |
|--------------------|-----------------------------------------------|
| Base model         | `yolo11m-cls.pt` (Ultralytics, pretrained)    |
| Image size         | 480 × 480                                     |
| Epochs             | 20 (early-stopping patience 5)                |
| Batch size         | 16                                            |
| Optimiser          | AdamW, lr₀ = 0.0005, cosine schedule          |
| Regularisation     | Dropout 0.1, weight decay 0.0005              |
| Augmentation       | RandAugment, rotation, flips, HSV, erasing    |
| Hardware           | Apple Silicon (MPS), about 8.5 hours          |
| Seed               | 42 (deterministic)                            |

**Dataset:** 11,666 training images and 3,550 validation images.

**Results on the validation set:**

| Metric              | Value                    |
|---------------------|--------------------------|
| Top-1 accuracy      | **97.94 %** (epoch 19)   |
| Top-5 accuracy      | 100 %                    |
| Validation loss     | 0.075                    |

Training curves, the confusion matrix and sample predictions are in `runs/classify/crop_model_runs/yolo11m_staples_480_split70/` and `reports/figures/`.

> **Note:** these figures are measured on the validation split. A separate held-out test split was not created, so the validation set was also used for model selection. See [Limitations](#limitations).

## Project structure

```
.
├── app.py                     # Streamlit entry point: diagnosis → RAG → rules → Gemini report
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/                       # Application modules
│   ├── rag_pipeline.py        # Knowledge-base ingestion and retrieval (ChromaDB)
│   ├── rule_engine.py         # Deterministic PPQS / CIBRC safety rules
│   ├── input_guard.py         # Crop selector check
│   ├── weather_risk.py        # Open-Meteo forecast, disease risk and spray window
│   ├── labels.py              # Shared label parsing
│   └── ui_components.py       # HTML components (hero, cards, sidebar, footer)
├── assets/style.css           # Custom styling for the Streamlit interface
├── .streamlit/
│   ├── config.toml            # Theme
│   └── secrets.toml           # GEMINI_API_KEY (not to be shared)
│
├── knowledge_base/            # Advisory PDFs + verified treatment tables (.xlsx)
├── chroma_db/                 # Persisted vector store (built from knowledge_base/)
│
├── models/yolo11m-cls.pt      # Pretrained base model used for fine-tuning
├── runs/classify/crop_model_runs/
│   ├── yolo11m_staples_480_split70/
│   │   ├── weights/best.pt    # Trained model loaded by the app
│   │   └── results.csv, *.png # Training metrics, confusion matrix, batch samples
│   └── yolo11m_staples_480_split70_test/   # Test-set validation outputs
│
├── scripts/
│   ├── train.py               # Fine-tune YOLO11m on data/processed
│   ├── data_prep/             # Organise, filter, clean, split and prune the dataset
│   └── analysis/              # Count classes and audit the dataset
├── evaluation/                # Test-set evaluation, training-history and dataset charts
├── reports/                   # Generated figures, tables and evaluation results
│
└── data/
    ├── raw/                   # Source datasets (PlantVillage, Jain 2026)
    └── processed/{train,val,test}/  # Final 14-class dataset (70/15/15 split)
```

## Installation

**Requirements:** Python 3.9 or newer, and about 2 GB of free disk space for the models and embeddings.

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

**3. Add a Gemini API key.** Get a free key from [Google AI Studio](https://aistudio.google.com/apikey) and add it to `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-key-here"
```

Alternatively, set it as an environment variable: `export GEMINI_API_KEY="your-key-here"`.

> ⚠️ Never commit `secrets.toml` to a public repository. Add `.streamlit/secrets.toml` to `.gitignore`.

**4. Check the trained model is present** at `runs/classify/crop_model_runs/yolo11m_staples_480_split70/weights/best.pt`. To use a model stored elsewhere, set:

```bash
export YOLO_MODEL_PATH="/full/path/to/best.pt"
```

The app will not start without a trained model; it deliberately refuses to fall back to the untrained base model.

## Running the app

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually <http://localhost:8501>):

1. Choose a report language.
2. Upload a clear photo of a single leaf (JPG or PNG).
3. Click **Run Safe-RAG Analysis**.
4. Review the diagnosis, the regulatory safety check, the retrieved evidence and the advisory report.

On first launch, the BGE embedding model (`BAAI/bge-large-en-v1.5`) is downloaded from Hugging Face, which can take a few minutes.

### Rebuilding the knowledge base

Run this whenever files in `knowledge_base/` change:

```bash
python src/rag_pipeline.py
```

It deletes the old `chroma_db/`, loads every PDF, CSV and XLSX file, splits them into chunks of 900 characters (120 overlap) and stores them in a fresh vector database. Each spreadsheet row is stored as its own record with its `crop_key`, `disease_key`, `regulatory_status` and `source_url`, so exact matches can be retrieved by disease.

### Weather risk

Type a village, town or district under **Location** and pick the match. The app fetches an hourly forecast from [Open-Meteo](https://open-meteo.com/) (free, no API key) and shows:

- **Disease risk** for the next 4 days, from hours of high humidity or rain within each disease's favourable temperature range (`DISEASE_WEATHER` in `src/weather_risk.py`). Potato late blight uses the Hutton criteria, with a mean-temperature ceiling of 25 °C added for Indian conditions.
- **Next spray window** (only when a chemical is advised): daytime, wind under 15 km/h, below 35 °C and no rain for the following 6 hours.

The weather only affects risk and timing advice. It never changes the chemical, dose or rule status. The thresholds are indicative and should be checked against ICAR guidance.

### Changing the Gemini models

The fallback order is defined by `GEMINI_MODELS` near the top of `app.py`. The app moves to the next model on errors 404, 408, 429, 500, 502, 503 and 504, or on a 60-second timeout. It stops immediately on other errors (such as an invalid API key), since those would affect every model.

## Re-training and evaluation

```bash
# Dataset preparation scripts (filter, organise, fix classes, check, clean, prune)
python scripts/data_prep/filter_dataset.py
python scripts/data_prep/organize_dataset.py
python scripts/data_prep/fix_dataset_classes.py
python scripts/data_prep/check_images.py
python scripts/data_prep/clean_dataset.py
python scripts/data_prep/prune_dataset.py

# Inspect the dataset
python scripts/analysis/count_classes.py
python scripts/analysis/audit_dataset.py

# Train (automatically uses Apple MPS, NVIDIA CUDA or CPU)
python scripts/train.py

# Plot results into reports/figures/
python evaluation/plot_training_history.py
python evaluation/visualize_dataset.py
```

The evaluation and data-cleaning scripts also use `matplotlib`, `seaborn`, `scikit-learn` and `cleanvision`; install them with `pip install matplotlib seaborn scikit-learn cleanvision` if needed.

## Safety design

Key decisions that keep the advice safe:

- **The LLM never chooses a chemical or dose.** These come only from `rule_engine.py`, whose rules were audited against PPQS / CIBRC sources on 22 September 2026.
- **Healthy predictions never trigger pesticide advice.**
- **Partial matches give no automatic dose** and are flagged for manual label checking.
- **No evidence, no report.** If retrieval finds nothing for the detected disease, the report is not generated.
- **Values are protected in translation.** Chemical names, doses, units, numbers and URLs stay exactly as verified in every language.
- **The image result is presented as an initial assessment.** Farmers are told to confirm in the field and to follow the registered product label.

## Limitations

- **No held-out test set.** Reported accuracy is on the validation split; real-world accuracy on farmers' phone photos is likely to be lower, especially for images with cluttered backgrounds, multiple leaves or poor lighting.
- **Dataset bias.** Many images come from PlantVillage, which was photographed under controlled conditions.
- **Limited scope.** Only 4 crops and 14 classes are supported. Any other crop or disease will be forced into one of these classes.
- **Weather risk is not a validated forecast.** The thresholds come from general infection criteria and have not been tested against Indian outbreak records.
- **Rules can go out of date.** Pesticide registrations change; the rule table must be re-audited periodically.
- **Depends on an external API.** Report generation needs internet access and the Gemini free tier, which has rate limits and changing model availability.
- **Translation quality** has not been formally evaluated by native speakers for every language.

## Technology stack

| Area               | Tools                                                    |
|--------------------|----------------------------------------------------------|
| Image model        | Ultralytics YOLO11m-cls, PyTorch                         |
| Retrieval          | LangChain, ChromaDB, BAAI/bge-large-en-v1.5 embeddings   |
| Document loading   | PyPDF, pandas, openpyxl                                  |
| Language model     | Google Gemini (`google-genai` SDK)                       |
| Interface          | Streamlit, custom HTML / CSS                             |
| Data quality       | CleanVision, scikit-learn, matplotlib, seaborn           |

## Acknowledgements and data sources

- **PlantVillage** dataset: maize and potato images.
- **Jain (2026)** dataset: additional Indian crop images. *(Add full citation.)*
- **Directorate of Plant Protection, Quarantine & Storage (PPQS)** and the **Central Insecticides Board & Registration Committee (CIBRC)**, Government of India: treatment and registration records.
- **ICAR** guidance on integrated pest management.
- [Ultralytics](https://github.com/ultralytics/ultralytics), [LangChain](https://www.langchain.com/), [ChromaDB](https://www.trychroma.com/), [Streamlit](https://streamlit.io/) and [Google Gemini](https://ai.google.dev/).

*Please replace the placeholder citations above with the full references used in the dissertation, including the source of the wheat images.*

## Disclaimer

This software is an academic prototype built for a final-year dissertation. It is **not** a substitute for advice from a qualified agronomist or a local agricultural extension officer. Always confirm a diagnosis in the field and read the registered product label before applying any chemical.
